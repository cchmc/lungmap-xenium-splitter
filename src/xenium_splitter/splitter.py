from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from xenium_splitter.image_utils import (
    mask_and_crop_region,
    read_masked_cropped_region,
    read_image,
    save_image_like,
    supports_windowed_region_read,
    write_array_as_ome_tiff,
)
from xenium_splitter.io_utils import (
    build_analysis_zarr_from_analysis_dir,
    build_cell_feature_matrix_zarr_from_h5,
    build_tabular_zarr_from_filtered_output,
    classify_file,
    copy_gene_panel,
    detect_xy_columns,
    extract_entity_ids_in_region,
    find_matching_tabular_for_zarr,
    filter_cell_feature_matrix,
    filter_cell_feature_matrix_h5,
    filter_table_by_entity_ids,
    find_boundary_files,
    get_cell_feature_matrix_files,
    get_entity_id_column,
    is_cell_feature_matrix_group,
    is_cell_feature_matrix_h5,
    iter_input_files,
    load_pixel_size_from_experiment,
    rebase_table_coordinates_to_region_crop,
    read_table,
    read_hdf5_table,
    read_zarr_zip_table,
    subset_table_for_region,
    subset_table_for_regions_optimized,
    update_experiment_xenium_for_region,
    write_table,
    write_hdf5_table,
    write_zarr_zip_table,
)
from xenium_splitter.lasso import load_lasso_regions
from xenium_splitter.metadata import build_run_metadata_markdown
from xenium_splitter.models import FileMetric, RunMetrics, SplitConfig
from xenium_splitter.recalculate_diffexp import recalculate_diffexp_for_region

logger = logging.getLogger(__name__)


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _run_logged_task(task_name: str, script_started_perf: float, fn):
    """Run a task while logging start time, end time, and task duration."""
    since_start = time.perf_counter() - script_started_perf
    logger.info("[+%s] Starting task: %s", _format_elapsed(since_start), task_name)
    task_started = time.perf_counter()
    result = fn()
    task_duration = time.perf_counter() - task_started
    since_start_end = time.perf_counter() - script_started_perf
    logger.info(
        "[+%s] Finished task: %s (took %s)",
        _format_elapsed(since_start_end),
        task_name,
        _format_elapsed(task_duration),
    )
    return result, task_duration


def _log_boundary_files(boundary_files: dict[str, Path], input_dir: Path) -> None:
    if boundary_files:
        for entity_type, boundary_file in sorted(boundary_files.items()):
            logger.info(
                "Using %s boundary file: %s",
                entity_type,
                boundary_file.relative_to(input_dir),
            )
        return

    logger.info("No boundary files found under %s", input_dir)


def _write_region_entity_ids_and_counts(
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]],
) -> None:
    all_entity_types = sorted(
        {
            entity_type
            for per_region in region_entity_ids.values()
            for entity_type in per_region.keys()
        }
    )
    if not all_entity_types:
        return

    counts_by_region: dict[str, dict[str, int]] = {}
    totals_by_entity: dict[str, int] = dict.fromkeys(all_entity_types, 0)

    for region_id, per_region in region_entity_ids.items():
        region_counts: dict[str, int] = {}
        region_dir = config.output_dir / f"region_{region_id}" / "entity_ids"
        region_dir.mkdir(parents=True, exist_ok=True)

        for entity_type in all_entity_types:
            entity_ids = per_region.get(entity_type, set())
            count = len(entity_ids)
            region_counts[entity_type] = count
            totals_by_entity[entity_type] += count

            output_path = region_dir / f"{entity_type}_ids.txt"
            output_path.write_text("\n".join(sorted(entity_ids)) + ("\n" if entity_ids else ""), encoding="utf-8")

        counts_by_region[region_id] = region_counts

    metrics.extra["entity_counts_by_region"] = counts_by_region
    metrics.extra["entity_counts_totals"] = totals_by_entity
    metrics.extra["entity_counts_total_all"] = sum(totals_by_entity.values())


def _collect_original_entity_totals(boundary_files: dict[str, Path]) -> dict[str, int]:
    totals_by_entity: dict[str, int] = {}
    for entity_type, boundary_file in boundary_files.items():
        try:
            table = read_table(boundary_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read boundary file %s for totals: %s", boundary_file.name, exc)
            continue

        id_col = get_entity_id_column(table, entity_type)
        if not id_col:
            logger.warning("Could not determine ID column for %s in %s", entity_type, boundary_file.name)
            continue

        totals_by_entity[entity_type] = int(table[id_col].astype(str).nunique())

    return totals_by_entity


def _extract_region_entity_ids(
    regions,
    boundary_files: dict[str, Path],
) -> dict[str, dict[str, set[str]]]:
    region_entity_ids: dict[str, dict[str, set[str]]] = {}
    for region in regions:
        region_entity_ids[region.region_id] = {}
        for entity_type, boundary_file in boundary_files.items():
            ids = extract_entity_ids_in_region(boundary_file, region)
            region_entity_ids[region.region_id][entity_type] = ids
            if ids:
                logger.info("Region %s: %d %s entities in boundaries", region.region_id, len(ids), entity_type)
    return region_entity_ids


def _process_cell_feature_matrix_groups(
    input_dir: Path,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]],
    script_started_perf: float,
) -> list[Path]:
    bundles = _find_cell_feature_matrix_bundles(input_dir)
    processed_files: list[Path] = []
    for cfm_dir, cfm_h5, cfm_zarr_zip in bundles:
        _run_logged_task(
            f"process cell_feature_matrix bundle {cfm_dir.relative_to(input_dir).as_posix()}",
            script_started_perf,
            lambda: _split_cell_feature_matrix_bundle(
                cfm_dir,
                cfm_h5,
                cfm_zarr_zip,
                regions,
                config,
                metrics,
                region_entity_ids,
            ),
        )

        for component in cfm_dir.iterdir():
            if component.is_file():
                processed_files.append(component)
        if cfm_h5 is not None and cfm_h5.is_file():
            processed_files.append(cfm_h5)
        if cfm_zarr_zip is not None and cfm_zarr_zip.is_file():
            processed_files.append(cfm_zarr_zip)
    return processed_files


def _apply_recalculated_diffexp_metric_update(
    metrics: RunMetrics,
    source_path: str,
    region_id: str,
    row_count: int,
) -> None:
    for item in metrics.file_metrics:
        if item.source_path != source_path:
            continue

        previous_status = item.status.lower()
        item.status = "re-calculated"
        item.detail = "Recalculated from region-filtered matrix + clustering labels"
        item.rows_written_by_region[region_id] = row_count
        item.rows_written_total = sum(item.rows_written_by_region.values())

        if previous_status == "skipped":
            metrics.files_skipped = max(0, metrics.files_skipped - 1)
            metrics.files_processed += 1
        return

    metrics.file_metrics.append(
        FileMetric(
            source_path=source_path,
            file_type="tabular",
            status="re-calculated",
            detail="Recalculated from region-filtered matrix + clustering labels",
            rows_written_total=row_count,
            rows_written_by_region={region_id: row_count},
        )
    )
    metrics.files_processed += 1


def _merge_recalculated_diffexp_metrics(
    metrics: RunMetrics,
    region_id: str,
    written: dict[str, int],
) -> None:
    for source_path, row_count in written.items():
        _apply_recalculated_diffexp_metric_update(metrics, source_path, region_id, row_count)


def _recalculate_diffexp_for_regions(config: SplitConfig, regions, metrics: RunMetrics) -> None:
    if not config.recalculate_diffexp:
        return

    for region in regions:
        region_dir = config.output_dir / f"region_{region.region_id}"
        try:
            written = recalculate_diffexp_for_region(region_dir)
            metrics.extra[f"diffexp_recalculated_{region.region_id}"] = len(written)
            logger.info(
                "Region %s: recalculated diffexp for %d clustering outputs",
                region.region_id,
                len(written),
            )
            _merge_recalculated_diffexp_metrics(metrics, region.region_id, written)
        except ValueError as exc:
            logger.warning("Region %s: skipping diffexp recalculation (%s)", region.region_id, exc)


def _find_region_table_path(region_dir: Path, stem: str) -> Path | None:
    candidates = [
        region_dir / f"{stem}.parquet",
        region_dir / f"{stem}.csv.gz",
        region_dir / f"{stem}.tsv.gz",
        region_dir / f"{stem}.csv",
        region_dir / f"{stem}.tsv",
        region_dir / f"{stem}.txt",
    ]
    return next((path for path in candidates if path.exists() and path.is_file()), None)


def _count_region_rows(region_dir: Path, stem: str) -> int:
    table_path = _find_region_table_path(region_dir, stem)
    if table_path is None:
        return 0
    try:
        return int(len(read_table(table_path)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not count rows in %s for %s: %s", table_path.name, stem, exc)
        return 0


def _update_region_metadata_outputs(
    config: SplitConfig,
    regions,
    metrics: RunMetrics,
) -> None:
    for region in regions:
        region_id = str(region.region_id)
        region_dir = config.output_dir / f"region_{region_id}"
        if not region_dir.exists():
            continue

        copy_gene_panel(config.input_dir, region_dir)

        # Prefer row counts from filtered outputs; fallback to entity-id counts if needed.
        num_cells = _count_region_rows(region_dir, "cells")
        if num_cells == 0:
            by_region = metrics.extra.get("entity_counts_by_region", {})
            if isinstance(by_region, dict):
                per_region = by_region.get(region_id, {})
                if isinstance(per_region, dict):
                    num_cells = int(per_region.get("cells", 0) or per_region.get("cell", 0) or 0)

        num_transcripts = _count_region_rows(region_dir, "transcripts")
        region_area_um2 = float(region.polygon.area)

        updated = update_experiment_xenium_for_region(
            config.input_dir,
            region_dir,
            region_id,
            num_cells,
            num_transcripts,
            region_area_um2,
        )
        if updated:
            logger.info(
                "Updated metadata for region %s (cells=%d, transcripts=%d, area_um2=%.2f)",
                region_id,
                num_cells,
                num_transcripts,
                region_area_um2,
            )
        else:
            logger.warning("Skipped metadata update for region %s (experiment.xenium not found)", region_id)


def run_split(config: SplitConfig) -> tuple[RunMetrics, Path]:
    started_at = datetime.now(timezone.utc)
    script_started_perf = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stage_times_s: dict[str, float] = {}

    def _time_stage(stage_name: str, fn):
        result, duration = _run_logged_task(stage_name, script_started_perf, fn)
        stage_times_s[stage_name] = stage_times_s.get(stage_name, 0.0) + duration
        return result

    # Load pixel_size from experiment.xenium if available
    if config.pixel_size_um is None:
        config.pixel_size_um = _time_stage(
            "load_pixel_size",
            lambda: load_pixel_size_from_experiment(config.input_dir),
        )

    regions = _time_stage("load_lasso_regions", lambda: load_lasso_regions(config.lasso_file))
    metrics = RunMetrics(region_count=len(regions))
    
    # Pre-extract entity IDs from boundary files for each region
    boundary_files = _time_stage("find_boundary_files", lambda: find_boundary_files(config.input_dir))
    _log_boundary_files(boundary_files, config.input_dir)

    original_totals = _time_stage(
        "collect_original_entity_totals",
        lambda: _collect_original_entity_totals(boundary_files),
    )
    if original_totals:
        metrics.extra["entity_counts_original_totals"] = original_totals
        metrics.extra["entity_counts_original_total_all"] = sum(original_totals.values())
    
    region_entity_ids = _time_stage(
        "extract_region_entity_ids",
        lambda: _extract_region_entity_ids(regions, boundary_files),
    )

    _time_stage(
        "write_region_entity_ids",
        lambda: _write_region_entity_ids_and_counts(config, metrics, region_entity_ids),
    )

    cfm_processed_files = _time_stage(
        "process_cell_feature_matrix_groups",
        lambda: _process_cell_feature_matrix_groups(
            config.input_dir,
            regions,
            config,
            metrics,
            region_entity_ids,
            script_started_perf,
        ),
    )

    input_files = _time_stage("select_input_files", lambda: _select_input_files(config))
    
    # Exclude cell_feature_matrix artifacts already processed as a single bundle
    cfm_component_files = set(cfm_processed_files)
    input_files = [f for f in input_files if f not in cfm_component_files]

    metrics.files_total = len(input_files) + len(cfm_processed_files)

    # Group known multi-format stems (cells, transcripts, cell_boundaries,
    # nucleus_boundaries) so they are read once and written to all formats.
    multi_format_groups, remainder_files = _group_multi_format_files(input_files)

    main_loop_start = time.perf_counter()

    for stem, fps in sorted(multi_format_groups.items()):
        _run_logged_task(
            f"process grouped {stem} files",
            script_started_perf,
            lambda fps=fps: _split_file_group(fps, regions, config, metrics, region_entity_ids),
        )

        logger.info(
            "File group %s finished in %.2fs",
            stem,
            time.perf_counter() - main_loop_start,
        )

    for file_path in remainder_files:
        relative_path = file_path.relative_to(config.input_dir)
        if file_path.resolve() == config.lasso_file.resolve():
            metrics.files_skipped += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(relative_path),
                    file_type="lasso",
                    status="skipped",
                    detail="Input LASSO file",
                )
            )
            continue

        _process_file(file_path, relative_path, regions, config, metrics, region_entity_ids)
    stage_times_s["process_input_files"] = time.perf_counter() - main_loop_start

    if config.he_image and not config.skip_images:
        _time_stage("split_external_he_image", lambda: _split_external_he_image(config, regions, metrics))

    _time_stage("recalculate_diffexp", lambda: _recalculate_diffexp_for_regions(config, regions, metrics))
    _time_stage("update_region_metadata", lambda: _update_region_metadata_outputs(config, regions, metrics))

    completed_at = datetime.now(timezone.utc)
    metrics.extra["timing_stage_seconds"] = stage_times_s
    for stage_name, seconds in sorted(stage_times_s.items(), key=lambda item: item[1], reverse=True):
        logger.info("Timing: %s = %.2fs", stage_name, seconds)

    file_type_seconds: dict[str, float] = {}
    file_type_counts: dict[str, int] = {}
    for item in metrics.file_metrics:
        if item.duration_s is None:
            continue
        file_type_seconds[item.file_type] = file_type_seconds.get(item.file_type, 0.0) + item.duration_s
        file_type_counts[item.file_type] = file_type_counts.get(item.file_type, 0) + 1

    metrics.extra["timing_file_type_seconds"] = file_type_seconds
    metrics.extra["timing_file_type_counts"] = file_type_counts

    slowest = sorted(
        [
            {
                "source_path": item.source_path,
                "file_type": item.file_type,
                "status": item.status,
                "duration_s": item.duration_s,
            }
            for item in metrics.file_metrics
            if item.duration_s is not None
        ],
        key=lambda x: float(x["duration_s"]),
        reverse=True,
    )[:10]
    metrics.extra["timing_slowest_files"] = slowest
    for entry in slowest[:5]:
        logger.info(
            "Slow file: %s (%s, %s) %.2fs",
            entry["source_path"],
            entry["file_type"],
            entry["status"],
            float(entry["duration_s"]),
        )

    metadata_text = build_run_metadata_markdown(config, metrics, started_at, completed_at)
    metadata_path = config.output_dir / "run_metadata_README.md"
    metadata_path.write_text(metadata_text, encoding="utf-8")
    return metrics, metadata_path


def _select_input_files(config: SplitConfig) -> list[Path]:
    if not config.include_globs:
        return iter_input_files(config.input_dir)

    selected: set[Path] = set()
    for pattern in config.include_globs:
        selected.update(config.input_dir.rglob(pattern))
    return sorted(path for path in selected if path.is_file())


# Stems whose multiple formats (csv.gz, parquet, zarr.zip) should be read once
# and written to all output formats from the single in-memory filtered DataFrame.
_MULTI_FORMAT_STEMS = frozenset(
    ["cells", "transcripts", "cell_boundaries", "nucleus_boundaries"]
)


def _tabular_file_stem(file_path: Path) -> str | None:
    """Return the canonical stem for a known multi-format tabular file, else None."""
    name = file_path.name.lower()
    for stem in _MULTI_FORMAT_STEMS:
        for ext in (".csv.gz", ".tsv.gz", ".parquet", ".csv", ".tsv", ".zarr.zip"):
            if name == stem + ext:
                return stem
    return None


def _group_multi_format_files(
    input_files: list[Path],
) -> tuple[dict[str, list[Path]], list[Path]]:
    """Split input_files into grouped multi-format files and the remainder.

    Returns:
        groups: dict mapping stem -> list of Paths with that stem
        remainder: files that are not part of any group
    """
    groups: dict[str, list[Path]] = {}
    remainder: list[Path] = []
    for file_path in input_files:
        stem = _tabular_file_stem(file_path)
        if stem is not None:
            groups.setdefault(stem, []).append(file_path)
        else:
            remainder.append(file_path)
    return groups, remainder


def _split_file_group(
    file_paths: list[Path],
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]] | None,
) -> None:
    """Read the source data once and write filtered output for every format in the group.

    Preferred source priority: parquet > csv.gz > csv/tsv > zarr.zip.
    All formats in the group are written from the same filtered in-memory DataFrame.
    """
    started = time.perf_counter()
    logger.debug("[_split_file_group] start: files=%d", len(file_paths))

    # Pick the best source to read from (parquet is fastest)
    block_start = time.perf_counter()
    logger.debug("[_split_file_group] block=select_source begin")
    ext_priority = [".parquet", ".csv.gz", ".tsv.gz", ".csv", ".tsv"]
    file_by_ext: dict[str, Path] = {}
    for fp in file_paths:
        n = fp.name.lower()
        for ext in ext_priority:
            if n.endswith(ext):
                file_by_ext[ext] = fp
                break

    source_path = next(
        (file_by_ext[e] for e in ext_priority if e in file_by_ext), file_paths[0]
    )
    logger.debug(
        "[_split_file_group] block=select_source done: source=%s elapsed=%.2fs",
        source_path.name,
        time.perf_counter() - block_start,
    )

    block_start = time.perf_counter()
    logger.debug("[_split_file_group] block=read_table begin: source=%s", source_path.name)
    try:
        table = read_table(source_path)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        logger.debug("[_split_file_group] block=read_table failed: elapsed=%.2fs", elapsed)
        for fp in file_paths:
            rel = fp.relative_to(config.input_dir)
            metrics.files_failed += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(rel),
                    file_type="tabular",
                    status="failed",
                    detail=f"Could not read source {source_path.name}: {exc}",
                    duration_s=elapsed,
                )
            )
        return
    logger.debug(
        "[_split_file_group] block=read_table done: rows=%d elapsed=%.2fs",
        len(table),
        time.perf_counter() - block_start,
    )

    block_start = time.perf_counter()
    logger.debug("[_split_file_group] block=infer_columns begin")
    entity_type = _infer_entity_type_from_filename(source_path.name)
    id_col = get_entity_id_column(table, entity_type) if entity_type else None
    xy_cols = detect_xy_columns(table)
    prefer_coordinates = _prefer_coordinate_filtering(entity_type, xy_cols)
    logger.debug(
        "[_split_file_group] block=infer_columns done: entity_type=%s id_col=%s xy_cols=%s prefer_coordinates=%s elapsed=%.2fs",
        entity_type,
        id_col,
        xy_cols,
        prefer_coordinates,
        time.perf_counter() - block_start,
    )

    if not id_col and not xy_cols:
        elapsed = time.perf_counter() - started
        logger.debug("[_split_file_group] block=early_skip done: elapsed=%.2fs", elapsed)
        for fp in file_paths:
            rel = fp.relative_to(config.input_dir)
            metrics.files_skipped += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(rel),
                    file_type="tabular",
                    status="skipped",
                    detail="No recognized x/y coordinate columns and cannot infer entity type",
                    rows_input=len(table),
                    duration_s=elapsed,
                )
            )
        return

    # Filter per region once, then write every format from that subset
    per_region_subsets: dict[str, pd.DataFrame] = {}
    rows_by_region: dict[str, int] = {}

    optimized_subsets: dict[str, pd.DataFrame] | None = None
    if prefer_coordinates and xy_cols and entity_type == "transcripts":
        block_start = time.perf_counter()
        logger.debug("[_split_file_group] block=build_transcript_optimized_subsets begin")
        optimized_subsets = subset_table_for_regions_optimized(
            table,
            list(regions),
            xy_cols[0],
            xy_cols[1],
            region_cell_ids_by_region={
                region.region_id: region_entity_ids.get(region.region_id, {}).get("cells", set())
                for region in regions
            } if region_entity_ids else None,
            pixel_size_um=config.pixel_size_um,
        )
        logger.debug(
            "[_split_file_group] block=build_transcript_optimized_subsets done: elapsed=%.2fs",
            time.perf_counter() - block_start,
        )

    block_start = time.perf_counter()
    logger.debug("[_split_file_group] block=build_region_subsets begin: regions=%d", len(regions))
    for region in regions:
        region_start = time.perf_counter()
        if optimized_subsets is not None:
            subset = optimized_subsets[region.region_id]
        elif xy_cols and not (id_col and region_entity_ids):
            subset = subset_table_for_region(
                table, region, xy_cols[0], xy_cols[1], pixel_size_um=config.pixel_size_um
            )
        elif id_col and region_entity_ids:
            entity_ids = (
                region_entity_ids.get(region.region_id, {}).get(entity_type, set())
                if entity_type
                else set()
            )
            subset = filter_table_by_entity_ids(table, entity_ids, id_col)
            if xy_cols:
                subset = rebase_table_coordinates_to_region_crop(
                    subset, region, xy_cols[0], xy_cols[1], pixel_size_um=config.pixel_size_um
                )
        else:
            subset = table.iloc[0:0].copy()

        per_region_subsets[region.region_id] = subset
        rows_by_region[region.region_id] = len(subset)
        logger.debug(
            "[_split_file_group] block=build_region_subsets region=%s rows=%d elapsed=%.2fs",
            region.region_id,
            len(subset),
            time.perf_counter() - region_start,
        )
    logger.debug(
        "[_split_file_group] block=build_region_subsets done: elapsed=%.2fs",
        time.perf_counter() - block_start,
    )

    rows_written_total = sum(rows_by_region.values())
    detail_prefix = (
        f"Coordinates detected in columns: {xy_cols[0]}, {xy_cols[1]}"
        if prefer_coordinates or not id_col
        else f"Filtered by entity IDs from boundaries (ID column: {id_col})"
    )
    read_elapsed = time.perf_counter() - started
    logger.debug(
        "[_split_file_group] block=pre_write_summary done: rows_written_total=%d elapsed=%.2fs",
        rows_written_total,
        read_elapsed,
    )

    block_start = time.perf_counter()
    logger.debug("[_split_file_group] block=write_outputs begin: formats=%d", len(file_paths))
    for fp in file_paths:
        rel = fp.relative_to(config.input_dir)
        file_name_lower = fp.name.lower()
        write_start = time.perf_counter()
        file_type = "zarr" if file_name_lower.endswith(".zarr.zip") else "tabular"

        try:
            logger.debug("[_split_file_group] block=write_format begin: output=%s type=%s", rel, file_type)
            for region in regions:
                region_write_start = time.perf_counter()
                subset = per_region_subsets[region.region_id]
                dest = config.output_dir / f"region_{region.region_id}" / rel
                if file_name_lower.endswith(".zarr.zip"):
                    write_zarr_zip_table(subset, dest, dataset_name="data")
                else:
                    write_table(subset, dest)
                logger.debug(
                    "[_split_file_group] block=write_format region=%s output=%s rows=%d elapsed=%.2fs",
                    region.region_id,
                    rel,
                    len(subset),
                    time.perf_counter() - region_write_start,
                )

            metrics.files_processed += 1
            write_elapsed = time.perf_counter() - write_start
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(rel),
                    file_type=file_type,
                    status="processed",
                    detail=f"{detail_prefix} [read once, {len(file_paths)} formats]",
                    rows_input=len(table),
                    rows_written_total=rows_written_total,
                    rows_written_by_region=dict(rows_by_region),
                    duration_s=read_elapsed + write_elapsed,
                )
            )
            logger.debug(
                "[_split_file_group] block=write_format done: output=%s type=%s elapsed=%.2fs",
                rel,
                file_type,
                write_elapsed,
            )
        except Exception as exc:  # noqa: BLE001
            metrics.files_failed += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(rel),
                    file_type=file_type,
                    status="failed",
                    detail=str(exc),
                    duration_s=read_elapsed + (time.perf_counter() - write_start),
                )
            )
            logger.debug(
                "[_split_file_group] block=write_format failed: output=%s type=%s elapsed=%.2fs error=%s",
                rel,
                file_type,
                time.perf_counter() - write_start,
                exc,
            )
    logger.debug(
        "[_split_file_group] block=write_outputs done: elapsed=%.2fs total_elapsed=%.2fs",
        time.perf_counter() - block_start,
        time.perf_counter() - started,
    )


def _process_file(
    file_path: Path,
    relative_path: Path,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]] | None = None,
) -> None:
    """Route file to appropriate handler based on type."""
    file_type = classify_file(file_path)
    started = time.perf_counter()
    before_metrics = len(metrics.file_metrics)
    try:
        if file_type == "tabular":
            _split_tabular(file_path, relative_path, regions, config, metrics, region_entity_ids)
        elif file_type == "hdf5":
            _split_hdf5(file_path, relative_path, regions, config, metrics, region_entity_ids)
        elif file_type == "zarr":
            _split_zarr(file_path, relative_path, regions, config, metrics, region_entity_ids)
        elif file_type == "image" and not config.skip_images:
            _split_image(file_path, relative_path, regions, config, metrics)
        elif file_type == "image" and config.skip_images:
            metrics.files_skipped += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(relative_path),
                    file_type="image",
                    status="skipped",
                    detail="Image processing disabled with --skip-images",
                )
            )
        else:
            metrics.files_skipped += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(relative_path),
                    file_type="unknown",
                    status="skipped",
                    detail="Unsupported file type",
                )
            )
    except Exception as exc:  # noqa: BLE001
        metrics.files_failed += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type=file_type,
                status="failed",
                detail=str(exc),
            )
        )
    finally:
        elapsed = time.perf_counter() - started
        for item in metrics.file_metrics[before_metrics:]:
            if item.duration_s is None:
                item.duration_s = elapsed


def _split_tabular(
    file_path: Path,
    relative_path: Path,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]] | None = None,
) -> None:
    table = read_table(file_path)
    entity_type = _infer_entity_type_for_table(file_path, relative_path, table)
    
    xy_cols = detect_xy_columns(table)
    prefer_coordinates = _prefer_coordinate_filtering(entity_type, xy_cols)

    if xy_cols is None and not entity_type:
        metrics.files_skipped += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="tabular",
                status="skipped",
                detail="No recognized x/y coordinate columns and cannot infer entity type",
                rows_input=len(table),
            )
        )
        return
    
    use_coordinate_filtering = prefer_coordinates or (xy_cols is not None and not (entity_type and region_entity_ids))

    if use_coordinate_filtering:
        _split_tabular_by_coordinates(
            relative_path, table, xy_cols, regions, config, metrics,
            region_entity_ids=region_entity_ids,
            entity_type=entity_type
        )
    elif entity_type and region_entity_ids:
        _split_tabular_by_entity_ids(
            file_path, relative_path, table, regions, config, metrics, 
            region_entity_ids, entity_type
        )
    else:
        metrics.files_skipped += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="tabular",
                status="skipped",
                detail="Cannot filter: no coordinates and no boundary data",
                rows_input=len(table),
            )
        )


def _split_tabular_by_entity_ids(
    file_path: Path,
    relative_path: Path,
    table: pd.DataFrame,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]],
    entity_type: str,
) -> None:
    """Filter tabular file using entity IDs extracted from boundary files.
    
    When boundary files are available, this provides accurate filtering:
    1. Extract region-specific entity IDs from boundary polygon intersections
    2. Filter table rows to matched IDs
    3. If x/y columns exist, rebase coordinates to crop origin (for QC/inspection)
    
    Coordinate Rebasing:
    When xy_cols are detected in the table, they are rebased to align with the
    cropped image origin. This uses pixel_size_um (if available) to compute the
    image-aligned crop origin via floor(bounds/pixel_size) logic.
    
    Args:
        file_path: Source file path (for logging)
        relative_path: Relative path for output organization
        table: Full input table
        regions: List of LASSO regions
        config: Configuration with pixel_size_um
        metrics: Metrics tracking
        region_entity_ids: Pre-extracted boundary-based entity IDs
        entity_type: Type of entity (cells, nucleus, transcripts)
    """
    id_col = get_entity_id_column(table, entity_type)
    if not id_col:
        logger = logging.getLogger(__name__)
        logger.debug(f"Could not find entity ID column in {file_path.name}, using coordinates")
        xy_cols = detect_xy_columns(table)
        if xy_cols:
            _split_tabular_by_coordinates(
                relative_path, table, xy_cols, regions, config, metrics,
                region_entity_ids=region_entity_ids,
                entity_type=entity_type
            )
        return
    
    rows_by_region: dict[str, int] = {}
    rows_written_total = 0
    xy_cols = detect_xy_columns(table)  # Detect for rebasing if present
    
    for region in regions:
        entity_ids = region_entity_ids.get(region.region_id, {}).get(entity_type, set())
        if not entity_ids:
            continue
        
        subset = filter_table_by_entity_ids(table, entity_ids, id_col)
        # Rebase coordinates if columns exist (for data consistency with cropped image)
        if xy_cols is not None:
            subset = rebase_table_coordinates_to_region_crop(
                subset,
                region,
                xy_cols[0],
                xy_cols[1],
                pixel_size_um=config.pixel_size_um,
            )
        rows_by_region[region.region_id] = len(subset)
        rows_written_total += len(subset)
        
        destination = config.output_dir / f"region_{region.region_id}" / relative_path
        write_table(subset, destination)
    
    metrics.files_processed += 1
    metrics.file_metrics.append(
        FileMetric(
            source_path=str(relative_path),
            file_type="tabular",
            status="processed",
            detail=f"Filtered by entity IDs from boundaries (ID column: {id_col})",
            rows_input=len(table),
            rows_written_total=rows_written_total,
            rows_written_by_region=rows_by_region,
        )
    )


def _split_tabular_by_coordinates(
    relative_path: Path,
    table: pd.DataFrame,
    xy_cols: tuple[str, str],
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]] | None = None,
    entity_type: str | None = None,
) -> None:
    """Filter tabular file using coordinate-based polygon containment.
    
    Fallback path when boundary files are unavailable:
    1. Test each row's (x, y) coordinates for containment in region polygon
    2. Keep rows where coordinates fall within the polygon
    3. Rebase coordinates to the cropped image origin
    
    For transcripts: uses optimized filtering (cell assignment + bounding box pre-filter)
    when region_entity_ids available.
    
    Coordinate Rebasing:
    Uses pixel_size_um (if available) to compute image-aligned crop origin.
    This ensures output coordinates align with cropped image pixel layout.
    
    Args:
        relative_path: Relative path for output organization
        table: Full input table
        xy_cols: Tuple of (x_column_name, y_column_name)
        regions: List of LASSO regions
        config: Configuration with pixel_size_um
        metrics: Metrics tracking
        region_entity_ids: Optional pre-extracted boundary entity IDs (for transcripts optimization)
        entity_type: Type of entity (for transcripts optimization)
    """
    x_col, y_col = xy_cols
    rows_by_region: dict[str, int] = {}
    rows_written_total = 0
    use_optimized = entity_type == "transcripts" and region_entity_ids is not None

    optimized_subsets: dict[str, pd.DataFrame] | None = None
    if use_optimized:
        optimized_subsets = subset_table_for_regions_optimized(
            table,
            list(regions),
            x_col,
            y_col,
            region_cell_ids_by_region={
                region.region_id: region_entity_ids.get(region.region_id, {}).get("cells", set())
                for region in regions
            },
            pixel_size_um=config.pixel_size_um,
        )

    for region in regions:
        # Filter by containment + rebase coordinates to crop origin
        if optimized_subsets is not None:
            subset = optimized_subsets[region.region_id]
        else:
            subset = subset_table_for_region(table, region, x_col, y_col, pixel_size_um=config.pixel_size_um)
        rows_by_region[region.region_id] = len(subset)
        rows_written_total += len(subset)

        destination = config.output_dir / f"region_{region.region_id}" / relative_path
        write_table(subset, destination)

    method_detail = f"Coordinates detected in columns: {x_col}, {y_col}"
    if use_optimized:
        method_detail += " [optimized with cell assignment + bbox]"
    
    metrics.files_processed += 1
    metrics.file_metrics.append(
        FileMetric(
            source_path=str(relative_path),
            file_type="tabular",
            status="processed",
            detail=method_detail,
            rows_input=len(table),
            rows_written_total=rows_written_total,
            rows_written_by_region=rows_by_region,
        )
    )


def _split_image(
    file_path: Path,
    relative_path: Path,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
) -> None:
    if supports_windowed_region_read(file_path):
        for region in regions:
            cropped = read_masked_cropped_region(
                file_path,
                region.polygon,
                pixel_size_um=config.pixel_size_um,
                squash_layers=config.squash_layers,
            )
            destination = config.output_dir / f"region_{region.region_id}" / relative_path
            save_image_like(file_path, destination, cropped)
    else:
        image = read_image(file_path, squash_layers=config.squash_layers)
        for region in regions:
            cropped = mask_and_crop_region(image, region.polygon, pixel_size_um=config.pixel_size_um)
            destination = config.output_dir / f"region_{region.region_id}" / relative_path
            save_image_like(file_path, destination, cropped)

    metrics.files_processed += 1
    metrics.file_metrics.append(
        FileMetric(
            source_path=str(relative_path),
            file_type="image",
            status="processed",
            detail="Masked + cropped to region polygon",
        )
    )


def _split_external_he_image(config: SplitConfig, regions, metrics: RunMetrics) -> None:
    assert config.he_image is not None
    he_image = config.he_image
    if supports_windowed_region_read(he_image):
        for region in regions:
            region_dir = config.output_dir / f"region_{region.region_id}"
            region_dir.mkdir(parents=True, exist_ok=True)

            cropped = read_masked_cropped_region(
                he_image,
                region.polygon,
                pixel_size_um=config.pixel_size_um,
                squash_layers=config.squash_layers,
            )
            destination = region_dir / he_image.name
            save_image_like(he_image, destination, cropped)

            if config.convert_svs_to_ome and he_image.suffix.lower() == ".svs":
                ome_output = region_dir / f"{he_image.stem}.ome.tiff"
                write_array_as_ome_tiff(cropped, ome_output)
    else:
        image = read_image(he_image, squash_layers=config.squash_layers)
        for region in regions:
            region_dir = config.output_dir / f"region_{region.region_id}"
            region_dir.mkdir(parents=True, exist_ok=True)

            cropped = mask_and_crop_region(image, region.polygon, pixel_size_um=config.pixel_size_um)
            destination = region_dir / he_image.name
            save_image_like(he_image, destination, cropped)

            if config.convert_svs_to_ome and he_image.suffix.lower() == ".svs":
                ome_output = region_dir / f"{he_image.stem}.ome.tiff"
                write_array_as_ome_tiff(cropped, ome_output)

    metrics.file_metrics.append(
        FileMetric(
            source_path=str(he_image),
            file_type="external_he_image",
            status="processed",
            detail="External H&E image split by regions",
        )
    )


def _infer_entity_type_from_filename(file_name: str) -> str | None:
    lower_name = file_name.lower()
    if "cell" in lower_name:
        return "cells"
    if "nucleus" in lower_name or "nuclei" in lower_name:
        return "nucleus"
    if "transcript" in lower_name:
        return "transcripts"
    return None


def _infer_entity_type_for_table(
    file_path: Path,
    relative_path: Path,
    table: pd.DataFrame,
) -> str | None:
    entity_type = _infer_entity_type_from_filename(file_path.name)
    if entity_type:
        return entity_type

    parts_lower = {part.lower() for part in relative_path.parts}
    columns_lower = {str(col).lower() for col in table.columns}

    # Xenium analysis CSVs commonly use Barcode as a cell identifier.
    if "analysis" in parts_lower and "barcode" in columns_lower:
        return "cells"

    return None


def _prefer_coordinate_filtering(
    entity_type: str | None,
    xy_cols: tuple[str, str] | None,
) -> bool:
    return entity_type == "transcripts" and xy_cols is not None


def _try_split_table_by_entity_ids(
    file_path: Path,
    relative_path: Path,
    table: pd.DataFrame,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]] | None,
    file_type: str,
    write_subset,
) -> bool:
    """Attempt to filter HDF5/Zarr table using boundary-extracted entity IDs.
    
    Shared entity ID filtering for both HDF5 and Zarr file types:
    1. Infer entity type from file path
    2. Get entity ID column (cell_id, nucleus_id, etc.)
    3. For each region, filter rows to matched IDs
    4. If x/y columns exist, rebase to crop origin for consistency
    5. Write filtered subset
    
    Coordinate Rebasing:
    When xy_cols detected in table, they are rebased to align with cropped
    image origin using pixel_size_um (if available).
    
    Args:
        file_path: Source file path (for logging)
        relative_path: Relative path for output organization
        table: Full input table
        regions: List of LASSO regions
        config: Configuration with pixel_size_um
        metrics: Metrics tracking
        region_entity_ids: Pre-extracted boundary-based entity IDs
        file_type: File type string for metrics ("hdf5" or "zarr")
        write_subset: Callable to write filtered subset (write_hdf5_table or write_zarr_zip_table)
    
    Returns:
        True if successfully filtered, False if unable to process
    """
    entity_type = _infer_entity_type_for_table(file_path, relative_path, table)
    if not entity_type or not region_entity_ids:
        return False

    id_col = get_entity_id_column(table, entity_type)
    if not id_col:
        return False

    rows_by_region: dict[str, int] = {}
    rows_written_total = 0
    xy_cols = detect_xy_columns(table)  # Detect for rebasing if present

    if _prefer_coordinate_filtering(entity_type, xy_cols):
        return False

    for region in regions:
        entity_ids = region_entity_ids.get(region.region_id, {}).get(entity_type, set())
        if not entity_ids:
            continue

        # Filter rows to matched entity IDs
        subset = filter_table_by_entity_ids(table, entity_ids, id_col)
        # Rebase coordinates if columns exist (for data consistency with cropped image)
        if xy_cols is not None:
            subset = rebase_table_coordinates_to_region_crop(
                subset,
                region,
                xy_cols[0],
                xy_cols[1],
                pixel_size_um=config.pixel_size_um,
            )
        rows_by_region[region.region_id] = len(subset)
        rows_written_total += len(subset)

        destination = config.output_dir / f"region_{region.region_id}" / relative_path
        write_subset(subset, destination)

    metrics.files_processed += 1
    metrics.file_metrics.append(
        FileMetric(
            source_path=str(relative_path),
            file_type=file_type,
            status="processed",
            detail=f"Filtered by entity IDs from boundaries (ID column: {id_col})",
            rows_input=len(table),
            rows_written_total=rows_written_total,
            rows_written_by_region=rows_by_region,
        )
    )
    return True


def _try_split_cell_feature_matrix_h5(
    file_path: Path,
    relative_path: Path,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]] | None,
) -> bool:
    """Handle 10X-style cell_feature_matrix.h5 as sparse matrix data.

    Returns True when file is a cell_feature_matrix.h5 and has been handled,
    otherwise returns False so caller can continue generic HDF5 flow.
    """
    if not is_cell_feature_matrix_h5(file_path):
        return False

    if not region_entity_ids:
        metrics.files_skipped += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="hdf5",
                status="skipped",
                detail="No boundary IDs available for cell_feature_matrix.h5 filtering",
            )
        )
        return True

    rows_by_region: dict[str, int] = {}
    rows_written_total = 0
    processed_any = False

    for region in regions:
        cell_ids = region_entity_ids.get(region.region_id, {}).get("cells", set())
        destination = config.output_dir / f"region_{region.region_id}" / relative_path

        result = filter_cell_feature_matrix_h5(file_path, destination, cell_ids)
        if result is None:
            rows_by_region[region.region_id] = 0
            continue

        processed_any = True
        _original_count, filtered_count = result
        rows_by_region[region.region_id] = filtered_count
        rows_written_total += filtered_count

    if processed_any:
        metrics.files_processed += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="hdf5",
                status="processed",
                detail="Filtered cell_feature_matrix.h5 by boundary-derived cell IDs",
                rows_written_total=rows_written_total,
                rows_written_by_region=rows_by_region,
            )
        )
    else:
        metrics.files_skipped += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="hdf5",
                status="skipped",
                detail="Could not parse cell_feature_matrix.h5 schema",
            )
        )
    return True


def _split_hdf5(
    file_path: Path,
    relative_path: Path,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]] | None = None,
) -> None:
    """Split HDF5 file using boundary IDs when available, else coordinates.
    
    Processing order:
    1. Try entity ID filtering (if boundaries available)
    2. Fall back to coordinate-based filtering (with rebasing)
    3. Skip if neither method applicable
    
    Coordinate Rebasing in Fallback:
    When boundary data unavailable, coordinates are rebased to crop origin
    using pixel_size_um (if available) to ensure alignment with image layout.
    
    Args:
        file_path: Source HDF5 file path
        relative_path: Relative path for output organization
        regions: List of LASSO regions
        config: Configuration with pixel_size_um
        metrics: Metrics tracking
        region_entity_ids: Pre-extracted boundary-based entity IDs
    """
    if _try_split_cell_feature_matrix_h5(
        file_path,
        relative_path,
        regions,
        config,
        metrics,
        region_entity_ids,
    ):
        return

    table = read_hdf5_table(file_path)
    
    if table is None or table.empty:
        metrics.files_skipped += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="hdf5",
                status="skipped",
                detail="Could not read table from HDF5 file",
            )
        )
        return
    
    # Try boundary-based filtering first
    if _try_split_table_by_entity_ids(
        file_path,
        relative_path,
        table,
        regions,
        config,
        metrics,
        region_entity_ids,
        "hdf5",
        lambda subset, destination: write_hdf5_table(subset, destination, dataset_name="data"),
    ):
        return

    # Fall back to coordinate-based filtering
    xy_cols = detect_xy_columns(table)
    if xy_cols is None:
        metrics.files_skipped += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="hdf5",
                status="skipped",
                detail="No recognized x/y coordinate columns",
                rows_input=len(table),
            )
        )
        return

    x_col, y_col = xy_cols
    rows_by_region: dict[str, int] = {}
    rows_written_total = 0

    for region in regions:
        # Filter by containment + rebase to crop origin
        subset = subset_table_for_region(table, region, x_col, y_col, pixel_size_um=config.pixel_size_um)
        rows_by_region[region.region_id] = len(subset)
        rows_written_total += len(subset)

        destination = config.output_dir / f"region_{region.region_id}" / relative_path
        write_hdf5_table(subset, destination, dataset_name="data")

    metrics.files_processed += 1
    metrics.file_metrics.append(
        FileMetric(
            source_path=str(relative_path),
            file_type="hdf5",
            status="processed",
            detail=f"Coordinates detected in columns: {x_col}, {y_col}",
            rows_input=len(table),
            rows_written_total=rows_written_total,
            rows_written_by_region=rows_by_region,
        )
    )


def _split_zarr(
    file_path: Path,
    relative_path: Path,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]] | None = None,
) -> None:
    """Split Zarr ZIP file using boundary IDs when available, else coordinates.
    
    Processing order:
    1. Try entity ID filtering (if boundaries available)
    2. Fall back to coordinate-based filtering (with rebasing)
    3. Skip if neither method applicable
    
    Coordinate Rebasing in Fallback:
    When boundary data unavailable, coordinates are rebased to crop origin
    using pixel_size_um (if available) to ensure alignment with image layout.
    
    Args:
        file_path: Source Zarr ZIP file path
        relative_path: Relative path for output organization
        regions: List of LASSO regions
        config: Configuration with pixel_size_um
        metrics: Metrics tracking
        region_entity_ids: Pre-extracted boundary-based entity IDs
    """
    lower_name = file_path.name.lower()
    if lower_name == "cell_feature_matrix.zarr.zip" and not config.write_cell_feature_matrix_zarr:
        metrics.files_skipped += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="zarr",
                status="skipped",
                detail="Skipped by --skip-cell-feature-matrix-zarr",
            )
        )
        return

    table = read_zarr_zip_table(file_path)
    
    if table is None or table.empty:
        if lower_name == "cell_feature_matrix.zarr.zip" and region_entity_ids:
            rows_by_region: dict[str, int] = {}
            rows_written_total = 0
            processed_any = False
            for region in regions:
                region_dir = config.output_dir / f"region_{region.region_id}"
                filtered_h5 = region_dir / relative_path.parent / "cell_feature_matrix.h5"
                destination = region_dir / relative_path
                logger.info(
                    "Rebuilding cell_feature_matrix.zarr.zip for region %s from filtered H5: %s",
                    region.region_id,
                    filtered_h5,
                )
                result = build_cell_feature_matrix_zarr_from_h5(filtered_h5, destination)
                if result is None:
                    rows_by_region[region.region_id] = 0
                    continue
                processed_any = True
                _n_features, n_cells = result
                rows_by_region[region.region_id] = n_cells
                rows_written_total += n_cells

            if processed_any:
                metrics.files_processed += 1
                metrics.file_metrics.append(
                    FileMetric(
                        source_path=str(relative_path),
                        file_type="zarr",
                        status="processed",
                        detail="Rebuilt from filtered cell_feature_matrix.h5",
                        rows_written_total=rows_written_total,
                        rows_written_by_region=rows_by_region,
                    )
                )
                return

        if lower_name == "analysis.zarr.zip":
            rows_by_region: dict[str, int] = {}
            rows_written_total = 0
            processed_any = False
            for region in regions:
                region_dir = config.output_dir / f"region_{region.region_id}"
                analysis_dir = region_dir / "analysis"
                destination = region_dir / relative_path
                written = build_analysis_zarr_from_analysis_dir(analysis_dir, destination)
                rows_by_region[region.region_id] = written
                rows_written_total += written
                if written > 0:
                    processed_any = True

            if processed_any:
                metrics.files_processed += 1
                metrics.file_metrics.append(
                    FileMetric(
                        source_path=str(relative_path),
                        file_type="zarr",
                        status="processed",
                        detail="Rebuilt from filtered analysis CSV files",
                        rows_written_total=rows_written_total,
                        rows_written_by_region=rows_by_region,
                    )
                )
                return

        tabular_fallback = find_matching_tabular_for_zarr(file_path)
        if tabular_fallback is not None:
            fallback_table = read_table(tabular_fallback)
            if _try_split_table_by_entity_ids(
                file_path,
                relative_path,
                fallback_table,
                regions,
                config,
                metrics,
                region_entity_ids,
                "zarr",
                lambda subset, destination: write_zarr_zip_table(subset, destination, dataset_name="data"),
            ):
                return

            xy_cols = detect_xy_columns(fallback_table)
            if xy_cols is not None:
                x_col, y_col = xy_cols
                rows_by_region: dict[str, int] = {}
                rows_written_total = 0
                for region in regions:
                    subset = subset_table_for_region(
                        fallback_table,
                        region,
                        x_col,
                        y_col,
                        pixel_size_um=config.pixel_size_um,
                    )
                    rows_by_region[region.region_id] = len(subset)
                    rows_written_total += len(subset)
                    destination = config.output_dir / f"region_{region.region_id}" / relative_path
                    write_zarr_zip_table(subset, destination, dataset_name="data")

                metrics.files_processed += 1
                metrics.file_metrics.append(
                    FileMetric(
                        source_path=str(relative_path),
                        file_type="zarr",
                        status="processed",
                        detail=f"Rebuilt from {tabular_fallback.name} with coordinate/ID filtering",
                        rows_input=len(fallback_table),
                        rows_written_total=rows_written_total,
                        rows_written_by_region=rows_by_region,
                    )
                )
                return

        metrics.files_skipped += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="zarr",
                status="skipped",
                detail="Could not read table from Zarr ZIP file",
            )
        )
        return
    
    # Try boundary-based filtering first
    if _try_split_table_by_entity_ids(
        file_path,
        relative_path,
        table,
        regions,
        config,
        metrics,
        region_entity_ids,
        "zarr",
        lambda subset, destination: write_zarr_zip_table(subset, destination, dataset_name="data"),
    ):
        return

    # Fall back to coordinate-based filtering
    xy_cols = detect_xy_columns(table)
    if xy_cols is None:
        # For known tabular zarr mirrors (cells, transcripts), rebuild from the
        # already-filtered output parquet/csv rather than skipping entirely.
        zarr_stem = file_path.name.lower().split(".zarr.zip")[0]
        if zarr_stem in ("cells", "transcripts"):
            rows_by_region: dict[str, int] = {}
            rows_written_total = 0
            processed_any = False
            for region in regions:
                region_dir = config.output_dir / f"region_{region.region_id}"
                destination = region_dir / relative_path
                n_rows = build_tabular_zarr_from_filtered_output(zarr_stem, region_dir, destination)
                if n_rows is not None:
                    rows_by_region[region.region_id] = n_rows
                    rows_written_total += n_rows
                    processed_any = True
                else:
                    rows_by_region[region.region_id] = 0

            if processed_any:
                metrics.files_processed += 1
                metrics.file_metrics.append(
                    FileMetric(
                        source_path=str(relative_path),
                        file_type="zarr",
                        status="processed",
                        detail=f"Rebuilt from filtered {zarr_stem}.parquet",
                        rows_written_total=rows_written_total,
                        rows_written_by_region=rows_by_region,
                    )
                )
                return

        metrics.files_skipped += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="zarr",
                status="skipped",
                detail="No recognized x/y coordinate columns",
                rows_input=len(table),
            )
        )
        return

    x_col, y_col = xy_cols
    rows_by_region: dict[str, int] = {}
    rows_written_total = 0

    for region in regions:
        # Filter by containment + rebase to crop origin
        subset = subset_table_for_region(table, region, x_col, y_col, pixel_size_um=config.pixel_size_um)
        rows_by_region[region.region_id] = len(subset)
        rows_written_total += len(subset)

        destination = config.output_dir / f"region_{region.region_id}" / relative_path
        write_zarr_zip_table(subset, destination, dataset_name="data")

    metrics.files_processed += 1
    metrics.file_metrics.append(
        FileMetric(
            source_path=str(relative_path),
            file_type="zarr",
            status="processed",
            detail=f"Coordinates detected in columns: {x_col}, {y_col}",
            rows_input=len(table),
            rows_written_total=rows_written_total,
            rows_written_by_region=rows_by_region,
        )
    )


def _find_cell_feature_matrix_bundles(input_dir: Path) -> list[tuple[Path, Path | None, Path | None]]:
    """Locate all cell_feature_matrix bundles in the input.

    A bundle is composed of:
    - required: ``cell_feature_matrix/`` folder
    - optional: sibling ``cell_feature_matrix.h5``
    - optional: sibling ``cell_feature_matrix.zarr.zip``
    """
    bundles: list[tuple[Path, Path | None, Path | None]] = []
    for cfm_dir in sorted(input_dir.rglob("cell_feature_matrix")):
        if not is_cell_feature_matrix_group(cfm_dir):
            continue
        parent = cfm_dir.parent
        h5_path = parent / "cell_feature_matrix.h5"
        zarr_zip_path = parent / "cell_feature_matrix.zarr.zip"
        bundles.append(
            (
                cfm_dir,
                h5_path if h5_path.is_file() else None,
                zarr_zip_path if zarr_zip_path.is_file() else None,
            )
        )
    return bundles


def _region_cell_ids(region_entity_ids: dict[str, dict[str, set[str]]] | None, region_id: str) -> set[str]:
    """Get region cell IDs without altering key names in source data schemas.

    Boundary extraction typically yields ``cells``; some legacy paths may use
    ``cell``. This helper keeps both compatible without mutating source keys.
    """
    if not region_entity_ids:
        return set()
    per_region = region_entity_ids.get(region_id, {})
    return per_region.get("cells") or per_region.get("cell") or set()


def _split_cell_feature_matrix_bundle(
    cfm_dir: Path,
    cfm_h5: Path | None,
    cfm_zarr_zip: Path | None,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]] | None = None,
) -> None:
    """Process cell_feature_matrix folder + h5 + zarr.zip together per region.

    This keeps the original schemas/keys for each artifact type and only changes
    filtered data content. ``cell_feature_matrix.zarr.zip`` is rebuilt from the
    filtered ``cell_feature_matrix.h5`` when available.
    """
    relative_dir = cfm_dir.relative_to(config.input_dir)
    relative_h5 = cfm_h5.relative_to(config.input_dir) if cfm_h5 is not None else None
    relative_zarr = cfm_zarr_zip.relative_to(config.input_dir) if cfm_zarr_zip is not None else None

    files = get_cell_feature_matrix_files(cfm_dir)
    if not files["barcodes"] or not files["matrix"]:
        metrics.files_failed += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_dir),
                file_type="cell_feature_matrix",
                status="failed",
                detail="Missing required barcodes or matrix files",
            )
        )
        return

    dir_rows_by_region: dict[str, int] = {}
    h5_rows_by_region: dict[str, int] = {}
    zarr_rows_by_region: dict[str, int] = {}

    dir_rows_total = 0
    h5_rows_total = 0
    zarr_rows_total = 0

    any_dir_processed = False
    any_h5_processed = False
    any_zarr_processed = False

    for region in regions:
        cell_ids = _region_cell_ids(region_entity_ids, region.region_id)

        if not cell_ids:
            dir_rows_by_region[region.region_id] = 0
            if cfm_h5 is not None:
                h5_rows_by_region[region.region_id] = 0
            if cfm_zarr_zip is not None:
                zarr_rows_by_region[region.region_id] = 0
            continue

        region_dir = config.output_dir / f"region_{region.region_id}"

        # 1) Filter the folder-based sparse matrix bundle
        dir_destination = region_dir / relative_dir
        try:
            _orig_count, filtered_count = filter_cell_feature_matrix(cfm_dir, dir_destination, cell_ids)
            dir_rows_by_region[region.region_id] = filtered_count
            dir_rows_total += filtered_count
            any_dir_processed = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to filter %s for %s: %s", cfm_dir.name, region.region_id, exc)
            dir_rows_by_region[region.region_id] = 0

        # 2) Filter sibling cell_feature_matrix.h5 (if present)
        filtered_h5_path: Path | None = None
        if cfm_h5 is not None and relative_h5 is not None:
            h5_destination = region_dir / relative_h5
            result = filter_cell_feature_matrix_h5(cfm_h5, h5_destination, cell_ids)
            if result is None:
                h5_rows_by_region[region.region_id] = 0
            else:
                _orig_h5_count, filtered_h5_count = result
                h5_rows_by_region[region.region_id] = filtered_h5_count
                h5_rows_total += filtered_h5_count
                any_h5_processed = True
                filtered_h5_path = h5_destination

        # 3) Build sibling cell_feature_matrix.zarr.zip from filtered h5 (if present)
        if (
            cfm_zarr_zip is not None
            and relative_zarr is not None
            and config.write_cell_feature_matrix_zarr
        ):
            zarr_destination = region_dir / relative_zarr
            if filtered_h5_path is None:
                zarr_rows_by_region[region.region_id] = 0
            else:
                logger.info(
                    "Rebuilding cell_feature_matrix.zarr.zip for region %s from filtered H5: %s",
                    region.region_id,
                    filtered_h5_path,
                )
                rebuilt = build_cell_feature_matrix_zarr_from_h5(filtered_h5_path, zarr_destination)
                if rebuilt is None:
                    zarr_rows_by_region[region.region_id] = 0
                else:
                    _n_features, n_cells = rebuilt
                    zarr_rows_by_region[region.region_id] = n_cells
                    zarr_rows_total += n_cells
                    any_zarr_processed = True

    if any_dir_processed:
        metrics.files_processed += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_dir),
                file_type="cell_feature_matrix",
                status="processed",
                detail="Processed with bundled cell_feature_matrix workflow",
                rows_written_total=dir_rows_total,
                rows_written_by_region=dir_rows_by_region,
            )
        )
    else:
        metrics.files_skipped += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_dir),
                file_type="cell_feature_matrix",
                status="skipped",
                detail="No boundary-derived cell IDs available",
                rows_written_total=0,
                rows_written_by_region=dir_rows_by_region,
            )
        )

    if cfm_h5 is not None and relative_h5 is not None:
        if any_h5_processed:
            metrics.files_processed += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(relative_h5),
                    file_type="hdf5",
                    status="processed",
                    detail="Processed with bundled cell_feature_matrix workflow",
                    rows_written_total=h5_rows_total,
                    rows_written_by_region=h5_rows_by_region,
                )
            )
        else:
            metrics.files_skipped += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(relative_h5),
                    file_type="hdf5",
                    status="skipped",
                    detail="No boundary-derived cell IDs available",
                    rows_written_total=0,
                    rows_written_by_region=h5_rows_by_region,
                )
            )

    if cfm_zarr_zip is not None and relative_zarr is not None:
        if not config.write_cell_feature_matrix_zarr:
            metrics.files_skipped += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(relative_zarr),
                    file_type="zarr",
                    status="skipped",
                    detail="Skipped by --skip-cell-feature-matrix-zarr",
                    rows_written_total=0,
                    rows_written_by_region={region.region_id: 0 for region in regions},
                )
            )
        elif any_zarr_processed:
            metrics.files_processed += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(relative_zarr),
                    file_type="zarr",
                    status="processed",
                    detail="Rebuilt from bundled filtered cell_feature_matrix.h5",
                    rows_written_total=zarr_rows_total,
                    rows_written_by_region=zarr_rows_by_region,
                )
            )
        else:
            metrics.files_skipped += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(relative_zarr),
                    file_type="zarr",
                    status="skipped",
                    detail="No filtered cell_feature_matrix.h5 available for rebuild",
                    rows_written_total=0,
                    rows_written_by_region=zarr_rows_by_region,
                )
            )
