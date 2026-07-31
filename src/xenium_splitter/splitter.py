from __future__ import annotations

import logging
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from xenium_splitter.image_utils import (
    generate_morphology_focus,
    generate_morphology_focus_with_stats,
    generate_morphology_mip,
    mask_and_crop_region,
    read_masked_cropped_region,
    read_image,
    render_grid_overlay_image,
    save_image_like,
    supports_windowed_region_read,
    write_array_as_ome_tiff,
)
from xenium_splitter.io_utils import (
    build_cell_feature_matrix_zarr_from_sparse_bundle,
    build_cell_feature_matrix_zarr_from_h5,
    classify_file,
    copy_gene_panel,
    detect_xy_columns,
    extract_entity_ids_in_region,
    find_matching_tabular_for_zarr,
    filter_cell_feature_matrix,
    filter_cell_feature_matrix_h5,
    filter_zarr_zip_by_row_indices_preserve_schema,
    get_fov_dimensions_px_for_sw_version,
    filter_table_by_entity_ids,
    find_boundary_files,
    get_cell_feature_matrix_files,
    get_entity_id_column,
    is_cell_feature_matrix_group,
    is_cell_feature_matrix_h5,
    iter_input_files,
    load_instrument_sw_version_from_experiment,
    load_pixel_size_from_experiment,
    rebase_table_coordinates_to_region_crop,
    read_table,
    read_hdf5_table,
    read_barcodes_file,
    read_zarr_zip_table,
    subset_table_for_region,
    subset_table_for_regions_optimized,
    update_experiment_xenium_for_region,
    write_table,
    write_hdf5_table,
)
from xenium_splitter.lasso import load_lasso_regions
from xenium_splitter.metadata import build_region_readme_markdown, build_run_metadata_markdown
from xenium_splitter.models import FileMetric, RunMetrics, SplitConfig
from xenium_splitter.recalculate_diffexp import recalculate_diffexp_for_region

logger = logging.getLogger(__name__)


def _write_canonical_transcript_sidecars(
    region_dir: Path,
    subset: pd.DataFrame,
    transcript_zarr_path: Path,
) -> None:
    """Write transcript tabular sidecars that match the rebuilt transcript zarr.

    The schema-preserving zarr writer may reassign transcript IDs/FOVs. When it
    does, it writes ``transcripts_id_fov_remap.csv.gz`` alongside the output zarr.
    Re-apply that remap to the filtered transcript table before writing flat
    sidecars so ``transcripts.csv.gz`` / ``transcripts.parquet`` stay consistent
    with ``transcripts.zarr.zip``.
    """
    canonical_subset = subset.reset_index(drop=True).copy()
    remap_path = transcript_zarr_path.with_name("transcripts_id_fov_remap.csv.gz")

    if "transcript_id" in canonical_subset.columns and remap_path.is_file() and not canonical_subset.empty:
        try:
            remap_df = read_table(remap_path)
            if {"old_transcript_id", "new_transcript_id"}.issubset(remap_df.columns):
                join_df = remap_df.loc[:, [c for c in ["old_transcript_id", "new_transcript_id", "new_fov"] if c in remap_df.columns]].copy()
                join_df["_old_transcript_id"] = pd.to_numeric(
                    join_df["old_transcript_id"], errors="coerce"
                ).astype("Int64")

                merged = canonical_subset.copy()
                merged["_old_transcript_id"] = pd.to_numeric(
                    merged["transcript_id"], errors="coerce"
                ).astype("Int64")
                merged = merged.merge(
                    join_df.drop(columns=["old_transcript_id"]),
                    how="left",
                    on="_old_transcript_id",
                )

                remapped_ids = pd.to_numeric(merged.get("new_transcript_id"), errors="coerce")
                original_ids = pd.to_numeric(canonical_subset["transcript_id"], errors="coerce")
                resolved_ids = remapped_ids.where(remapped_ids.notna(), original_ids)
                canonical_subset["transcript_id"] = (
                    resolved_ids.astype("int64") if resolved_ids.notna().all() else resolved_ids
                )

                if "new_fov" in merged.columns:
                    remapped_fov = pd.to_numeric(merged["new_fov"], errors="coerce")
                    for fov_col in ["fov", "fov_id", "fov_index"]:
                        if fov_col not in canonical_subset.columns:
                            continue
                        original_fov = pd.to_numeric(canonical_subset[fov_col], errors="coerce")
                        resolved_fov = remapped_fov.where(remapped_fov.notna(), original_fov)
                        canonical_subset[fov_col] = (
                            resolved_fov.astype("int64") if resolved_fov.notna().all() else resolved_fov
                        )
        except Exception:
            logger.warning(
                "Failed to apply transcript ID remap from %s; writing transcript sidecars with filtered table values",
                remap_path,
                exc_info=True,
            )

    write_table(canonical_subset, region_dir / "transcripts.csv.gz")
    try:
        write_table(canonical_subset, region_dir / "transcripts.parquet")
    except Exception:
        logger.debug(
            "Skipping transcripts.parquet write for %s (optional parquet dependencies unavailable)",
            region_dir,
            exc_info=True,
        )


def _collect_fov_layout_summary(
    config: SplitConfig,
    regions,
) -> dict[str, object] | None:
    """Compute FOV dimensions and potential FOV counts for each region.

    Potential counts are based on region crop size in pixels and FOV stride,
    where stride = FOV dimension - overlap (128 px).
    """
    pixel_size_um = config.pixel_size_um
    if pixel_size_um is None or pixel_size_um <= 0:
        return None

    sw_version = load_instrument_sw_version_from_experiment(config.input_dir)
    fov_rows_px, fov_cols_px = get_fov_dimensions_px_for_sw_version(sw_version)
    overlap_px = 128
    x_stride_px = max(1, int(fov_cols_px) - int(overlap_px))
    y_stride_px = max(1, int(fov_rows_px) - int(overlap_px))

    per_region: list[dict[str, int | str]] = []
    max_x = 0
    max_y = 0
    for region in regions:
        min_x, min_y, max_x_um, max_y_um = region.bounds
        width_um = max(0.0, float(max_x_um) - float(min_x))
        height_um = max(0.0, float(max_y_um) - float(min_y))

        width_px = int(math.ceil(width_um / float(pixel_size_um)))
        height_px = int(math.ceil(height_um / float(pixel_size_um)))
        potential_fov_x = max(1, int(math.ceil(width_px / float(x_stride_px)))) if width_px > 0 else 1
        potential_fov_y = max(1, int(math.ceil(height_px / float(y_stride_px)))) if height_px > 0 else 1

        max_x = max(max_x, potential_fov_x)
        max_y = max(max_y, potential_fov_y)
        per_region.append(
            {
                "region_id": str(region.region_id),
                "width_px": int(width_px),
                "height_px": int(height_px),
                "potential_fov_x": int(potential_fov_x),
                "potential_fov_y": int(potential_fov_y),
            }
        )

    return {
        "instrument_sw_version": sw_version,
        "pixel_size_um": float(pixel_size_um),
        "fov_rows_px": int(fov_rows_px),
        "fov_cols_px": int(fov_cols_px),
        "fov_overlap_px": int(overlap_px),
        "fov_stride_rows_px": int(y_stride_px),
        "fov_stride_cols_px": int(x_stride_px),
        "max_potential_fov_x": int(max_x),
        "max_potential_fov_y": int(max_y),
        "per_region": per_region,
    }


def _drop_negative_rebased_transcripts(
    subset: pd.DataFrame,
    x_col: str,
    y_col: str,
    region_id: str,
) -> pd.DataFrame:
    """Remove transcript rows with negative rebased coordinates.

    Rebasing can produce a small number of slightly negative values near crop
    boundaries. Xenium transcript outputs should not contain negative x/y.
    """
    if subset.empty:
        return subset

    x_vals = pd.to_numeric(subset[x_col], errors="coerce")
    y_vals = pd.to_numeric(subset[y_col], errors="coerce")
    keep_mask = (x_vals >= 0) & (y_vals >= 0)

    dropped = int((~keep_mask).sum())
    if dropped > 0:
        logger.debug(
            "Dropped %d transcripts with negative rebased coordinates for region %s",
            dropped,
            region_id,
        )
        return subset.loc[keep_mask].copy()

    return subset


def _always_skip_rule_for_path(relative_path: Path) -> str | None:
    """Return the always-skip rule name for ``relative_path``, or ``None`` if none applies.

    Currently suppresses anything under an ``aux_outputs/`` directory, which
    contains large Xenium auxiliary files that are not useful for downstream
    analysis of sub-regions.
    """
    parts_lower = {part.lower() for part in relative_path.parts}
    if "aux_outputs" in parts_lower:
        return "aux_outputs/**"
    return None


def _record_always_skipped(metrics: RunMetrics, rule: str, relative_path: Path) -> None:
    """Record ``relative_path`` as skipped under the given hard-coded ``rule`` in metrics."""
    by_rule = metrics.extra.setdefault("always_skipped_by_rule", {})
    if not isinstance(by_rule, dict):
        by_rule = {}
        metrics.extra["always_skipped_by_rule"] = by_rule
    entries = by_rule.setdefault(rule, [])
    if isinstance(entries, list):
        entries.append(str(relative_path))


def _format_elapsed(seconds: float) -> str:
    """Format a duration in seconds as a compact string for log messages."""
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
    """Log the discovered boundary files or a message when none are found."""
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
    """Write per-entity-type ID lists for each region and record counts in metrics.

    Creates ``entity_ids/<entity_type>_ids.txt`` files under each region output
    directory and populates ``metrics.extra`` with per-region and aggregate entity
    counts so they appear in the run metadata README.
    """
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
    """Count the unique entity IDs across all boundary files in the original dataset.

    Used to report what fraction of the original entities are captured across
    all split regions.

    Args:
        boundary_files: Mapping of entity type to boundary file path.

    Returns:
        Mapping of entity type to total unique ID count.
    """
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
    """Extract entity IDs from boundary files for each LASSO region.

    For every combination of region and entity type, finds entities whose boundary
    polygons intersect the region polygon.  Results are used to filter tabular
    outputs by pre-computed IDs rather than coordinate containment tests.

    Args:
        regions: List of :class:`LassoRegion` objects.
        boundary_files: Mapping of entity type to boundary file path.

    Returns:
        Nested mapping: ``{region_id: {entity_type: set_of_id_strings}}``.
    """
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
    """Find and process all cell_feature_matrix bundles, returning their component paths.

    Each bundle is split as a unit (barcodes, features, matrix, zarr) so the
    sparse matrix is filtered consistently from a single barcode list.  Returns
    the list of all component file paths that were handled, allowing the main
    file loop to skip them later.
    """
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
    """Update or create a FileMetric entry to reflect a recalculated diffexp file.

    If an existing metric entry for ``source_path`` is found it is updated
    in-place (status set to ``"re-calculated"``), adjusting skip/process counters
    accordingly.  Otherwise a new entry is appended.
    """
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
    """Merge diffexp recalculation results into ``metrics`` for all written files."""
    for source_path, row_count in written.items():
        _apply_recalculated_diffexp_metric_update(metrics, source_path, region_id, row_count)


def _recalculate_diffexp_for_regions(config: SplitConfig, regions, metrics: RunMetrics) -> None:
    """Re-run differential-expression calculation for every region output directory.

    Skipped when ``config.recalculate_diffexp`` is ``False``.  Warnings are
    logged for regions where the required inputs (cell_feature_matrix and
    clustering outputs) are not yet present.
    """
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
    """Find the first existing tabular file with the given stem in ``region_dir``.

    Checks candidates in priority order: Parquet, then gzipped CSV/TSV, then
    plain CSV/TSV/TXT.
    """
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
    """Return the number of rows in a tabular file within ``region_dir``, or 0 if absent."""
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
    """Finalise per-region metadata after all files have been split.

    For each region this function:
    - Copies ``gene_panel.json`` from the source run.
    - Counts cells and transcripts written.
    - Updates ``experiment.xenium`` and ``_region_metadata.json`` with region-level counts.
    - Writes a ``README.md`` summarising region provenance and processing.
    """
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

        region_readme = build_region_readme_markdown(
            config,
            region_id=region_id,
            region_bounds_um=tuple(float(v) for v in region.bounds),
            region_area_um2=region_area_um2,
            num_cells=num_cells,
            num_transcripts=num_transcripts,
            has_old_fov_to_new_fov=(region_dir / "old_fov_to_new_fov.csv").is_file(),
            has_transcript_id_fov_remap=(region_dir / "transcripts_id_fov_remap.csv.gz").is_file(),
            has_grid_overlays=(region_dir / "grid_overlays").is_dir(),
            focus_selection=(metrics.extra.get("focus_selection_by_region", {}) or {}).get(region_id),
        )
        (region_dir / "README.md").write_text(region_readme, encoding="utf-8")


def run_split(config: SplitConfig) -> tuple[RunMetrics, Path]:
    """Execute a full xenium-splitter run for the given configuration.

    Pipeline stages (in order):
    1. Load pixel size from ``experiment.xenium``.
    2. Parse LASSO regions from the lasso file.
    3. Compute FOV layout summary.
    4. Find boundary files and extract per-region entity ID sets.
    5. Process ``cell_feature_matrix`` bundles as a unit.
    6. Select and filter remaining input files.
    7. Split grouped multi-format files (cells, transcripts, boundaries) read-once.
    8. Process remaining individual files.
    9. Split optional external H&E image.
    10. Generate ``morphology_mip`` / ``morphology_focus`` outputs where missing.
    11. Write morphology grid overlays (when ``config.overlays`` is set).
    12. Recalculate diffexp outputs.
    13. Write per-region and run-level metadata.

    Args:
        config: Fully populated :class:`SplitConfig`.

    Returns:
        ``(metrics, metadata_path)`` where ``metadata_path`` is the written
        ``run_metadata_README.md``.
    """
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
    fov_layout_summary = _time_stage(
        "compute_fov_layout_summary",
        lambda: _collect_fov_layout_summary(config, regions),
    )
    if isinstance(fov_layout_summary, dict) and fov_layout_summary:
        metrics.extra["fov_layout_summary"] = fov_layout_summary
    
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

    filtered_input_files: list[Path] = []
    for file_path in input_files:
        relative_path = file_path.relative_to(config.input_dir)
        skip_rule = _always_skip_rule_for_path(relative_path)
        if skip_rule is not None:
            metrics.files_skipped += 1
            _record_always_skipped(metrics, skip_rule, relative_path)
            continue
        filtered_input_files.append(file_path)

    # Group known multi-format stems (cells, transcripts, cell_boundaries,
    # nucleus_boundaries) so they are read once and written to all formats.
    multi_format_groups, remainder_files = _group_multi_format_files(filtered_input_files)

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

    _time_stage(
        "ensure_morphology_mip_outputs",
        lambda: _ensure_region_morphology_mip_outputs(config, regions, metrics, filtered_input_files),
    )
    _time_stage(
        "ensure_morphology_focus_outputs",
        lambda: _ensure_region_morphology_focus_outputs(config, regions, metrics, filtered_input_files),
    )

    if config.overlays:
        _time_stage("write_morphology_grid_overlays", lambda: _write_morphology_grid_overlays(config, regions, metrics))
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
    """Return the list of input files to process.

    When ``config.include_globs`` is empty all files under ``config.input_dir``
    are returned via :func:`iter_input_files`.  Otherwise only files matching at
    least one of the provided glob patterns are included.
    """
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

    if config.copy_transcripts and entity_type == "transcripts":
        # Only intercept the zarr.zip; let CSV/parquet fall through to normal filtering below.
        zarr_files = [fp for fp in file_paths if fp.name.lower().endswith(".zarr.zip")]
        non_zarr_files = [fp for fp in file_paths if not fp.name.lower().endswith(".zarr.zip")]
        for fp in zarr_files:
            rel = fp.relative_to(config.input_dir)
            write_start = time.perf_counter()
            try:
                for region in regions:
                    dest = config.output_dir / f"region_{region.region_id}" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fp, dest)

                metrics.files_processed += 1
                metrics.file_metrics.append(
                    FileMetric(
                        source_path=str(rel),
                        file_type="zarr",
                        status="processed",
                        detail="Copied verbatim by --copy-transcripts (no filtering/rebasing)",
                        rows_input=len(table),
                        rows_written_total=len(table) * len(regions),
                        rows_written_by_region={r.region_id: len(table) for r in regions},
                        duration_s=time.perf_counter() - write_start,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                metrics.files_failed += 1
                metrics.file_metrics.append(
                    FileMetric(
                        source_path=str(rel),
                        file_type="zarr",
                        status="failed",
                        detail=str(exc),
                        duration_s=time.perf_counter() - write_start,
                    )
                )
        if not non_zarr_files:
            return
        # Continue with normal filtering for any remaining non-zarr transcript formats.
        file_paths = non_zarr_files


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

        if entity_type == "transcripts" and xy_cols:
            subset = _drop_negative_rebased_transcripts(
                subset,
                xy_cols[0],
                xy_cols[1],
                region.region_id,
            )

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
    transcript_zarr_rel = next(
        (
            fp.relative_to(config.input_dir)
            for fp in file_paths
            if fp.name.lower() == "transcripts.zarr.zip"
        ),
        None,
    )
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
                    transcript_id_values = None
                    if file_name_lower == "transcripts.zarr.zip" and "transcript_id" in subset.columns:
                        transcript_id_values = subset["transcript_id"].to_numpy()

                    # Rebase coordinate-bearing zarr outputs to the crop origin.
                    if xy_cols is not None or file_name_lower in {"cells.zarr.zip", "transcripts.zarr.zip"}:
                        _rebase_region_for_zarr = region
                    else:
                        _rebase_region_for_zarr = None
                    ok = filter_zarr_zip_by_row_indices_preserve_schema(
                        fp,
                        dest,
                        subset.index.to_numpy(dtype=int),
                        base_row_count=len(table),
                        rebase_region=_rebase_region_for_zarr,
                        pixel_size_um=config.pixel_size_um,
                        transcript_id_values=transcript_id_values,
                        transcript_table=subset if file_name_lower == "transcripts.zarr.zip" else None,
                    )
                    if not ok:
                        if file_name_lower == "transcripts.zarr.zip":
                            raise RuntimeError(
                                "Schema-preserving filter failed for transcripts.zarr.zip"
                            )
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(fp, dest)
                        logger.warning(
                            "Schema-preserving filter failed for %s; copied source archive verbatim for compatibility.",
                            fp.name,
                        )
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
            metric_detail = f"{detail_prefix} [read once, {len(file_paths)} formats]"
            metric_rows_input = len(table)
            metric_rows_written_total = rows_written_total
            metric_rows_by_region = dict(rows_by_region)
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(rel),
                    file_type=file_type,
                    status="processed",
                    detail=metric_detail,
                    rows_input=metric_rows_input,
                    rows_written_total=metric_rows_written_total,
                    rows_written_by_region=metric_rows_by_region,
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

    if transcript_zarr_rel is not None:
        for region in regions:
            region_dir = config.output_dir / f"region_{region.region_id}"
            _write_canonical_transcript_sidecars(
                region_dir,
                per_region_subsets[region.region_id],
                region_dir / transcript_zarr_rel,
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


def _is_analysis_model_metadata(relative_path: Path) -> bool:
    """Detect if file is analysis model metadata (components, variance, etc.)
    that should be copied as-is to all regions (not filtered by cells)."""
    metadata_files = {
        "components.csv",
        "variance.csv",
        "dispersion.csv",
        "features_selected.csv",
        "stdev.csv",
    }
    parts_lower = [part.lower() for part in relative_path.parts]
    if "analysis" not in parts_lower:
        return False
    return relative_path.name.lower() in metadata_files


def _copy_analysis_model_metadata(
    file_path: Path,
    relative_path: Path,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
) -> None:
    """Copy analysis model metadata files to each region output directory.
    
    Model metadata files (PCA components, variance, UMAP components, etc.) describe
    the full dataset and are immutable. They are copied as-is to each region's
    corresponding analysis subdirectory.
    """
    try:
        for region in regions:
            region_output = config.output_dir / f"region_{region.region_id}"
            dest_path = region_output / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # Preserve source bytes and formatting exactly (no parsing/re-write).
            shutil.copy2(file_path, dest_path)
        
        metrics.files_processed += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="tabular",
                status="processed",
                detail="Copied verbatim to all regions (analysis model metadata)",
            )
        )
    except Exception as e:
        logger.error(f"Failed to copy analysis model metadata {relative_path}: {e}")
        metrics.files_failed += 1
        metrics.file_metrics.append(
            FileMetric(
                source_path=str(relative_path),
                file_type="tabular",
                status="failed",
                detail=f"Failed to copy model metadata: {e}",
            )
        )


def _split_tabular(
    file_path: Path,
    relative_path: Path,
    regions,
    config: SplitConfig,
    metrics: RunMetrics,
    region_entity_ids: dict[str, dict[str, set[str]]] | None = None,
) -> None:
    # Check if this is analysis model metadata (PCA/UMAP components, variance, etc.)
    # These are immutable model artifacts describing the full dataset.
    if _is_analysis_model_metadata(relative_path):
        _copy_analysis_model_metadata(file_path, relative_path, regions, config, metrics)
        return

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

        if entity_type == "transcripts":
            subset = _drop_negative_rebased_transcripts(
                subset,
                x_col,
                y_col,
                region.region_id,
            )

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
    """Crop and mask an image file to each region and write the outputs.

    For formats that support windowed reads (TIFF, OME-TIFF, SVS) the image is
    read once per region using windowed I/O to minimise memory usage.  For all
    other formats the full image is read once and then cropped in memory.
    """
    if supports_windowed_region_read(file_path):
        for region in regions:
            cropped = read_masked_cropped_region(
                file_path,
                region.polygon,
                pixel_size_um=config.pixel_size_um,
                squash_layers=config.squash_layers,
            )
            destination = config.output_dir / f"region_{region.region_id}" / relative_path
            save_image_like(file_path, destination, cropped, pixel_size_um=config.pixel_size_um)
    else:
        image = read_image(file_path, squash_layers=config.squash_layers)
        for region in regions:
            cropped = mask_and_crop_region(image, region.polygon, pixel_size_um=config.pixel_size_um)
            destination = config.output_dir / f"region_{region.region_id}" / relative_path
            save_image_like(file_path, destination, cropped, pixel_size_um=config.pixel_size_um)

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
    """Crop and write the externally supplied H&E image to each region output directory.

    The output is always written as an OME-TIFF (``<stem>.ome.tif``) regardless
    of the source format.  Windowed reads are used for TIFF and SVS files;
    other formats are read fully into memory and then cropped.
    """
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
            destination = region_dir / f"{he_image.stem}.ome.tif"
            write_array_as_ome_tiff(cropped, destination, pixel_size_um=config.pixel_size_um)
    else:
        image = read_image(he_image, squash_layers=config.squash_layers)
        for region in regions:
            region_dir = config.output_dir / f"region_{region.region_id}"
            region_dir.mkdir(parents=True, exist_ok=True)

            cropped = mask_and_crop_region(image, region.polygon, pixel_size_um=config.pixel_size_um)
            destination = region_dir / f"{he_image.stem}.ome.tif"
            write_array_as_ome_tiff(cropped, destination, pixel_size_um=config.pixel_size_um)

    metrics.file_metrics.append(
        FileMetric(
            source_path=str(he_image),
            file_type="external_he_image",
            status="processed",
            detail="External H&E image split by regions",
        )
    )


def _existing_region_morphology_mip(region_dir: Path) -> Path | None:
    for file_name in ("morphology_mip.ome.tif", "morphology_mip.ome.tiff"):
        candidate = region_dir / file_name
        if candidate.is_file():
            return candidate
    return None


def _existing_region_morphology(region_dir: Path) -> Path | None:
    for file_name in ("morphology.ome.tif", "morphology.ome.tiff"):
        candidate = region_dir / file_name
        if candidate.is_file():
            return candidate
    return None


def _existing_region_morphology_focus(region_dir: Path) -> Path | None:
    for file_name in ("morphology_focus.ome.tif", "morphology_focus.ome.tiff"):
        candidate = region_dir / file_name
        if candidate.is_file():
            return candidate
    return None


def _has_original_morphology_mip_input(input_files: list[Path]) -> bool:
    for path in input_files:
        if path.name.lower() in {"morphology_mip.ome.tif", "morphology_mip.ome.tiff"}:
            return True
    return False


def _has_original_morphology_focus_input(input_files: list[Path]) -> bool:
    for path in input_files:
        if path.name.lower() in {"morphology_focus.ome.tif", "morphology_focus.ome.tiff"}:
            return True
    return False


def _ensure_region_morphology_mip_outputs(
    config: SplitConfig,
    regions,
    metrics: RunMetrics,
    input_files: list[Path],
) -> None:
    if config.skip_images:
        return
    if _has_original_morphology_mip_input(input_files):
        return

    for region in regions:
        region_dir = config.output_dir / f"region_{region.region_id}"
        if _existing_region_morphology_mip(region_dir) is not None:
            continue

        morphology_path = _existing_region_morphology(region_dir)
        if morphology_path is None:
            continue

        mip_image = generate_morphology_mip(read_image(morphology_path, squash_layers=True))
        mip_path = region_dir / "morphology_mip.ome.tif"
        write_array_as_ome_tiff(mip_image, mip_path, pixel_size_um=config.pixel_size_um)

        metrics.file_metrics.append(
            FileMetric(
                source_path=str(morphology_path.relative_to(config.output_dir)),
                file_type="image",
                status="processed",
                detail="Generated morphology_mip.ome.tif from cropped morphology.ome.tif",
            )
        )


def _ensure_region_morphology_focus_outputs(
    config: SplitConfig,
    regions,
    metrics: RunMetrics,
    input_files: list[Path],
) -> None:
    if config.skip_images:
        return
    if _has_original_morphology_focus_input(input_files):
        return

    for region in regions:
        region_dir = config.output_dir / f"region_{region.region_id}"
        if _existing_region_morphology_focus(region_dir) is not None:
            continue

        morphology_path = _existing_region_morphology(region_dir)
        if morphology_path is None:
            continue

        morphology_image = read_image(morphology_path, squash_layers=False)
        focus_image, focus_stats = generate_morphology_focus_with_stats(morphology_image, morphology_path)
        focus_path = region_dir / "morphology_focus.ome.tif"
        write_array_as_ome_tiff(focus_image, focus_path, pixel_size_um=config.pixel_size_um)

        if focus_stats is not None:
            focus_by_region = metrics.extra.setdefault("focus_selection_by_region", {})
            if isinstance(focus_by_region, dict):
                focus_by_region[str(region.region_id)] = focus_stats

        metrics.file_metrics.append(
            FileMetric(
                source_path=str(morphology_path.relative_to(config.output_dir)),
                file_type="image",
                status="processed",
                detail="Generated morphology_focus.ome.tif from cropped morphology.ome.tif",
            )
        )


def _read_transcript_grid_spec(transcripts_zarr_zip: Path) -> tuple[list[str], float] | None:
    try:
        import tempfile
        import zipfile
        import zarr
    except ImportError:
        logger.warning("zarr not available; cannot build morphology grid overlays")
        return None

    if not transcripts_zarr_zip.is_file():
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="grid_overlay_zarr_") as tmpdir:
            with zipfile.ZipFile(transcripts_zarr_zip, "r") as zf:
                zf.extractall(tmpdir)
            root = zarr.open_group(tmpdir, mode="r")
            if "grids" not in root:
                return None
            grids = root["grids"]
            level_names = sorted(
                (str(name) for name in grids.keys()),
                key=lambda name: int(name) if str(name).isdigit() else str(name),
            )
            if not level_names:
                return None

            base_grid_size = 250.0
            grid_size = grids.attrs.get("grid_size", [base_grid_size])
            if isinstance(grid_size, (list, tuple)) and grid_size:
                base_grid_size = float(grid_size[0])
            elif grid_size is not None:
                base_grid_size = float(grid_size)

            return level_names, base_grid_size
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read grid spec from %s: %s", transcripts_zarr_zip, exc)
        return None


def _write_morphology_grid_overlays(
    config: SplitConfig,
    regions,
    metrics: RunMetrics,
) -> None:
    if config.pixel_size_um is None or config.pixel_size_um <= 0:
        logger.info("Skipping morphology grid overlays: pixel_size_um unavailable")
        return

    sw_version = load_instrument_sw_version_from_experiment(config.input_dir)
    fov_rows_px, fov_cols_px = get_fov_dimensions_px_for_sw_version(sw_version)
    fov_overlap_px = 128
    fov_stride_um = (
        max(1, int(fov_cols_px) - int(fov_overlap_px)) * float(config.pixel_size_um),
        max(1, int(fov_rows_px) - int(fov_overlap_px)) * float(config.pixel_size_um),
    )
    fov_size_um = (
        float(fov_cols_px) * float(config.pixel_size_um),
        float(fov_rows_px) * float(config.pixel_size_um),
    )

    for region in regions:
        region_dir = config.output_dir / f"region_{region.region_id}"
        morphology_path = _existing_region_morphology_mip(region_dir)
        transcripts_path = region_dir / "transcripts.zarr.zip"

        if morphology_path is None:
            logger.info(
                "Skipping morphology grid overlays for region %s: no existing morphology_mip.ome.tif output",
                region.region_id,
            )
            continue

        grid_spec = _read_transcript_grid_spec(transcripts_path)
        if grid_spec is None:
            logger.info(
                "Skipping morphology grid overlays for region %s: transcript grid metadata unavailable",
                region.region_id,
            )
            continue

        level_names, base_grid_size_um = grid_spec
        image = read_image(morphology_path, squash_layers=config.squash_layers)
        overlay_dir = region_dir / "grid_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        for level_name in level_names:
            level_index = int(level_name) if str(level_name).isdigit() else 0
            tile_size_um = float(base_grid_size_um) * (2.0 ** float(level_index))
            overlay, overlay_pixel_size_um = render_grid_overlay_image(
                image,
                level_name=str(level_name),
                tile_size_um=tile_size_um,
                pixel_size_um=config.pixel_size_um,
                fov_stride_um=fov_stride_um,
                fov_size_um=fov_size_um,
            )
            output_path = overlay_dir / f"morphology_mip_grid_overlay_level_{level_name}.ome.tif"
            write_array_as_ome_tiff(overlay, output_path, pixel_size_um=overlay_pixel_size_um)

        metrics.file_metrics.append(
            FileMetric(
                source_path=str(morphology_path.relative_to(config.output_dir)),
                file_type="image_overlay",
                status="processed",
                detail=f"Wrote morphology grid overlays for levels: {', '.join(level_names)}",
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

    def _write_schema_preserving_subset(
        source_table: pd.DataFrame,
        subset: pd.DataFrame,
        destination: Path,
        *,
        region=None,
    ) -> None:
        transcript_id_values = None
        if lower_name == "transcripts.zarr.zip" and "transcript_id" in subset.columns:
            transcript_id_values = subset["transcript_id"].to_numpy()

        # Rebase coordinate-bearing zarr outputs to the crop origin.
        if lower_name in {"cells.zarr.zip", "transcripts.zarr.zip"}:
            rebase_region = region
        else:
            rebase_region = None

        ok = filter_zarr_zip_by_row_indices_preserve_schema(
            file_path,
            destination,
            subset.index.to_numpy(dtype=int),
            base_row_count=len(source_table),
            rebase_region=rebase_region,
            pixel_size_um=config.pixel_size_um,
            transcript_id_values=transcript_id_values,
            transcript_table=subset if lower_name == "transcripts.zarr.zip" else None,
        )
        if not ok:
            if lower_name == "transcripts.zarr.zip":
                raise RuntimeError(
                    "Schema-preserving filter failed for transcripts.zarr.zip"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, destination)
            logger.warning(
                "Schema-preserving filter failed for %s; copied source archive verbatim for compatibility.",
                file_path.name,
            )
        elif lower_name == "transcripts.zarr.zip":
            _write_canonical_transcript_sidecars(destination.parent, subset, destination)

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
        # Some Xenium zarr archives (notably analysis.zarr.zip and
        # cell_feature_matrix.zarr.zip) are not simple row tables. Rebuilding
        # them from CSV/H5 changes schema/version and breaks Explorer.
        # Preserve exact source schema by copying the archive verbatim.
        if lower_name in {"analysis.zarr.zip", "cell_feature_matrix.zarr.zip"}:
            for region in regions:
                region_dir = config.output_dir / f"region_{region.region_id}"
                destination = region_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, destination)

            metrics.files_processed += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(relative_path),
                    file_type="zarr",
                    status="processed",
                    detail="Copied verbatim to preserve original Xenium zarr schema",
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
                lambda subset, destination, fallback_table=fallback_table: _write_schema_preserving_subset(
                    fallback_table,
                    subset,
                    destination,
                    region=region,
                ),
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
                    _write_schema_preserving_subset(
                        fallback_table,
                        subset,
                        destination,
                        region=region,
                    )

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
        lambda subset, destination, table=table: _write_schema_preserving_subset(
            table,
            subset,
            destination,
            region=region,
        ),
    ):
        return

    # Fall back to coordinate-based filtering
    xy_cols = detect_xy_columns(table)
    if xy_cols is None:
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
                lambda subset, destination, fallback_table=fallback_table: _write_schema_preserving_subset(
                    fallback_table,
                    subset,
                    destination,
                    region=region,
                ),
            ):
                return

            fallback_xy_cols = detect_xy_columns(fallback_table)
            if fallback_xy_cols is not None:
                x_col, y_col = fallback_xy_cols
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
                    _write_schema_preserving_subset(
                        fallback_table,
                        subset,
                        destination,
                        region=region,
                    )

                metrics.files_processed += 1
                metrics.file_metrics.append(
                    FileMetric(
                        source_path=str(relative_path),
                        file_type="zarr",
                        status="processed",
                        detail=(
                            f"Schema-preserving rebuild from {tabular_fallback.name} "
                            f"with coordinates: {x_col}, {y_col}"
                        ),
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
        _write_schema_preserving_subset(
            table,
            subset,
            destination,
            region=region,
        )

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


def _try_filter_cfm_zarr_by_cell_ids(
    cfm_zarr_zip: Path,
    output_zarr_zip: Path,
    cell_ids: set[str],
) -> int | None:
    """Filter cell_feature_matrix.zarr.zip by cell_ids, preserving zarr v2 schema.
    
    The CFM zarr contains one row per cell, indexed sequentially. We need to:
    1. Try to match cell_ids from the cell_features group
    2. If that fails, try index-based matching (assume order matches cells.zarr.zip)
    
    Args:
        cfm_zarr_zip: Source cell_feature_matrix.zarr.zip path
        output_zarr_zip: Destination zarr.zip path
        cell_ids: Set of cell IDs to keep
        
    Returns:
        Number of cells in filtered output, or None if filtering failed.
    """
    try:
        import tempfile
        import zipfile
        import zarr
        import numpy as np
    except ImportError:
        logger.warning("zarr/numpy not available; cannot filter CFM zarr")
        return None

    if not cell_ids or not cfm_zarr_zip.exists():
        logger.debug(
            "[CFM zarr] Cannot filter: cell_ids=%d, cfm_exists=%s",
            len(cell_ids) if cell_ids else 0,
            cfm_zarr_zip.exists(),
        )
        return None

    def _normalize_cell_id_value(value) -> str:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            return value.decode("utf-8", errors="replace").strip()
        if isinstance(value, np.ndarray):
            # Xenium CFM cell_id can be uint32[N,2]; first column matches cells.csv cell_id.
            if value.ndim == 1 and value.size >= 1:
                return str(int(value[0]))
            return str(value.tolist()).strip()
        if isinstance(value, (tuple, list)) and value:
            return str(value[0]).strip()
        return str(value).strip()

    try:
        with tempfile.TemporaryDirectory(prefix="cfm_zarr_in_") as tmpdir_in:
            with tempfile.TemporaryDirectory(prefix="cfm_zarr_out_") as tmpdir_out:
                with zipfile.ZipFile(cfm_zarr_zip, "r") as zf:
                    zf.extractall(tmpdir_in)

                src_root = zarr.open_group(tmpdir_in, mode="r")
                if "cell_features" not in src_root:
                    logger.warning(
                        "[CFM zarr] cell_features group not found in %s; available=%s",
                        cfm_zarr_zip.name,
                        list(src_root.keys()),
                    )
                    return None

                src_cf = src_root["cell_features"]
                if not all(k in src_cf for k in ["cell_id", "data", "indices", "indptr"]):
                    logger.warning(
                        "[CFM zarr] missing required arrays in cell_features; found=%s",
                        list(src_cf.keys()),
                    )
                    return None

                src_cell_id = src_cf["cell_id"][:]
                src_data = src_cf["data"][:]
                src_indices = src_cf["indices"][:]
                src_indptr = src_cf["indptr"][:]

                requested = {_normalize_cell_id_value(v) for v in cell_ids}
                keep_indices = [
                    i for i, v in enumerate(src_cell_id)
                    if _normalize_cell_id_value(v) in requested
                ]

                # Xenium boundaries provide barcode-like IDs (e.g. aaaaacik-1), while
                # CFM zarr cell_id may store numeric [index, 1]. Translate via sibling
                # cell_feature_matrix/barcodes file when direct matching yields no hits.
                if not keep_indices:
                    cfm_dir = cfm_zarr_zip.parent / "cell_feature_matrix"
                    try:
                        barcodes_files = get_cell_feature_matrix_files(cfm_dir)
                        barcodes_path = barcodes_files.get("barcodes")
                        if barcodes_path is not None and barcodes_path.exists():
                            all_barcodes = read_barcodes_file(barcodes_path)
                            req_barcodes = {str(v).strip() for v in cell_ids}
                            keep_indices = [i for i, bc in enumerate(all_barcodes) if bc in req_barcodes]
                            logger.debug(
                                "[CFM zarr] Fallback barcode->index mapping matched %d columns",
                                len(keep_indices),
                            )
                    except Exception as map_exc:  # noqa: BLE001
                        logger.debug("[CFM zarr] barcode mapping fallback failed: %s", map_exc)

                if not keep_indices:
                    logger.warning("[CFM zarr] No matching cell IDs found in source cell_id array")
                    return None

                keep_indices_arr = np.asarray(sorted(set(keep_indices)), dtype=np.int64)
                logger.debug(
                    "[CFM zarr] Matched %d requested IDs -> %d columns",
                    len(requested),
                    len(keep_indices_arr),
                )

                # Filter CSR matrix columns (indices), preserving feature rows (indptr length).
                old_to_new = np.full(src_cell_id.shape[0], -1, dtype=np.int64)
                old_to_new[keep_indices_arr] = np.arange(len(keep_indices_arr), dtype=np.int64)

                new_data_parts: list[np.ndarray] = []
                new_indices_parts: list[np.ndarray] = []
                new_indptr = np.zeros_like(src_indptr)
                running = 0
                for r in range(len(src_indptr) - 1):
                    start = int(src_indptr[r])
                    end = int(src_indptr[r + 1])
                    cols = src_indices[start:end]
                    vals = src_data[start:end]
                    mapped = old_to_new[cols.astype(np.int64, copy=False)]
                    mask = mapped >= 0
                    kept_n = int(mask.sum())
                    if kept_n:
                        new_data_parts.append(vals[mask])
                        new_indices_parts.append(mapped[mask].astype(src_indices.dtype, copy=False))
                        running += kept_n
                    new_indptr[r + 1] = running

                if new_data_parts:
                    new_data = np.concatenate(new_data_parts).astype(src_data.dtype, copy=False)
                    new_indices = np.concatenate(new_indices_parts).astype(src_indices.dtype, copy=False)
                else:
                    new_data = np.array([], dtype=src_data.dtype)
                    new_indices = np.array([], dtype=src_indices.dtype)

                new_cell_id = src_cell_id[keep_indices_arr]

                def _create_like(dst_group, key: str, data_arr, src_arr) -> None:
                    kwargs = {
                        "shape": data_arr.shape,
                        "dtype": data_arr.dtype,
                        "data": data_arr,
                    }
                    chunks = getattr(src_arr, "chunks", None)
                    if chunks is not None and len(chunks) == len(data_arr.shape):
                        kwargs["chunks"] = tuple(min(int(c), int(s)) for c, s in zip(chunks, data_arr.shape))
                    compressors = getattr(src_arr, "compressors", None)
                    compressor = getattr(src_arr, "compressor", None)
                    if compressors is not None:
                        kwargs["compressors"] = compressors
                    elif compressor is not None:
                        kwargs["compressor"] = compressor

                    try:
                        arr = dst_group.create_dataset(key, **kwargs)
                    except TypeError:
                        kwargs.pop("compressors", None)
                        kwargs.pop("compressor", None)
                        arr = dst_group.create_dataset(key, **kwargs)

                    for attr_key, attr_val in dict(src_arr.attrs).items():
                        arr.attrs[attr_key] = attr_val

                try:
                    dst_root = zarr.open_group(tmpdir_out, mode="w", zarr_format=2)
                except TypeError:
                    dst_root = zarr.open_group(tmpdir_out, mode="w")

                for attr_key, attr_val in dict(src_root.attrs).items():
                    dst_root.attrs[attr_key] = attr_val

                dst_cf = dst_root.require_group("cell_features")
                for attr_key, attr_val in dict(src_cf.attrs).items():
                    dst_cf.attrs[attr_key] = attr_val
                dst_cf.attrs["number_cells"] = int(len(keep_indices_arr))

                _create_like(dst_cf, "cell_id", new_cell_id, src_cf["cell_id"])
                _create_like(dst_cf, "data", new_data, src_cf["data"])
                _create_like(dst_cf, "indices", new_indices, src_cf["indices"])
                _create_like(dst_cf, "indptr", new_indptr.astype(src_indptr.dtype, copy=False), src_cf["indptr"])

                with zipfile.ZipFile(output_zarr_zip, "w", zipfile.ZIP_STORED) as zf:
                    for fpath in Path(tmpdir_out).rglob("*"):
                        if fpath.is_file():
                            zf.write(fpath, arcname=fpath.relative_to(tmpdir_out))

                logger.debug("[CFM zarr] SUCCESS: wrote %s with %d cells", output_zarr_zip.name, len(keep_indices_arr))
                return int(len(keep_indices_arr))

    except Exception as e:  # noqa: BLE001
        logger.error("[CFM zarr] FAILED: %s", e, exc_info=True)
        return None


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
            logger.debug(
                "[_split_cfm_bundle] region=%s: no cell_ids from boundaries, skipping",
                region.region_id,
            )
            continue

        logger.debug(
            "[_split_cfm_bundle] region=%s: %d cell_ids from boundaries",
            region.region_id,
            len(cell_ids),
        )
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

        # 3) Filter sibling cell_feature_matrix.zarr.zip directly (preserve schema)
        # Fall back to rebuild from H5 only if direct filter fails
        if (
            cfm_zarr_zip is not None
            and relative_zarr is not None
            and config.write_cell_feature_matrix_zarr
        ):
            zarr_destination = region_dir / relative_zarr
            if not cell_ids:
                zarr_rows_by_region[region.region_id] = 0
                logger.debug(
                    "[_split_cfm_bundle] region=%s: no cell_ids for zarr, skipping",
                    region.region_id,
                )
            else:
                logger.debug(
                    "[_split_cfm_bundle] region=%s: attempting to filter zarr (%s exists=%s)",
                    region.region_id,
                    cfm_zarr_zip.name,
                    cfm_zarr_zip.exists(),
                )
                # Try to filter the zarr file directly using schema-preserving approach
                # to maintain zarr v2 format for Xenium Explorer compatibility
                zarr_rows = _try_filter_cfm_zarr_by_cell_ids(
                    cfm_zarr_zip,
                    zarr_destination,
                    cell_ids,
                )
                
                if zarr_rows is None:
                    logger.debug(
                        "[_split_cfm_bundle] region=%s: direct zarr filter returned None",
                        region.region_id,
                    )
                    # Fallback 1: rebuild from already-filtered sparse bundle.
                    rebuilt_from_sparse = None
                    if dir_destination.exists():
                        logger.info(
                            "Direct zarr filter failed; rebuilding cell_feature_matrix.zarr.zip for region %s from filtered sparse bundle",
                            region.region_id,
                        )
                        rebuilt_from_sparse = build_cell_feature_matrix_zarr_from_sparse_bundle(
                            dir_destination,
                            zarr_destination,
                            source_cfm_dir=cfm_dir,
                            source_zarr_zip_path=cfm_zarr_zip,
                        )

                    if rebuilt_from_sparse is not None:
                        _n_features_sparse, n_cells_sparse = rebuilt_from_sparse
                        zarr_rows = n_cells_sparse
                        logger.debug(
                            "[_split_cfm_bundle] region=%s: rebuilt zarr from sparse bundle: %d cells",
                            region.region_id,
                            n_cells_sparse,
                        )
                    elif filtered_h5_path is not None:
                        # Fallback 2: rebuild from filtered H5 if sparse-bundle rebuild failed
                        logger.info(
                            "Direct zarr/sparse rebuild failed; rebuilding cell_feature_matrix.zarr.zip for region %s from filtered H5",
                            region.region_id,
                        )
                        rebuilt = build_cell_feature_matrix_zarr_from_h5(
                            filtered_h5_path,
                            zarr_destination,
                            source_zarr_zip_path=cfm_zarr_zip,
                        )
                        if rebuilt is None:
                            logger.warning(
                                "[_split_cfm_bundle] region=%s: rebuild from H5 also failed",
                                region.region_id,
                            )
                            zarr_rows = 0
                        else:
                            _n_features, n_cells = rebuilt
                            zarr_rows = n_cells
                            logger.debug(
                                "[_split_cfm_bundle] region=%s: rebuilt zarr from H5: %d cells",
                                region.region_id,
                                n_cells,
                            )
                    else:
                        logger.debug(
                            "[_split_cfm_bundle] region=%s: no filtered_h5_path available for fallback",
                            region.region_id,
                        )
                        zarr_rows = 0
                
                if zarr_rows is not None and zarr_rows > 0:
                    zarr_rows_by_region[region.region_id] = zarr_rows
                    zarr_rows_total += zarr_rows
                    any_zarr_processed = True
                    logger.debug(
                        "[_split_cfm_bundle] region=%s: zarr processing successful: %d cells",
                        region.region_id,
                        zarr_rows,
                    )
                else:
                    zarr_rows_by_region[region.region_id] = 0
                    logger.debug(
                        "[_split_cfm_bundle] region=%s: zarr processing failed",
                        region.region_id,
                    )

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
            skip_detail = (
                "Handled by bundled cell_feature_matrix workflow (standalone H5 not emitted)"
                if any_dir_processed
                else "No boundary-derived cell IDs available"
            )
            metrics.files_skipped += 1
            metrics.file_metrics.append(
                FileMetric(
                    source_path=str(relative_h5),
                    file_type="hdf5",
                    status="skipped",
                    detail=skip_detail,
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
                    detail="Filtered with schema preservation for Xenium Explorer v2 compatibility",
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
                    detail="No cell IDs available or filtering failed",
                    rows_written_total=0,
                    rows_written_by_region=zarr_rows_by_region,
                )
            )
