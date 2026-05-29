from __future__ import annotations

from datetime import datetime

from xenium_splitter.models import RunMetrics, SplitConfig


def _append_boundary_entity_counts(lines: list[str], metrics: RunMetrics) -> None:
    counts_by_region = metrics.extra.get("entity_counts_by_region")
    totals_by_entity = metrics.extra.get("entity_counts_totals")
    total_all_entities = metrics.extra.get("entity_counts_total_all")
    original_totals_by_entity = metrics.extra.get("entity_counts_original_totals")
    original_total_all_entities = metrics.extra.get("entity_counts_original_total_all")
    if not isinstance(counts_by_region, dict) or not counts_by_region:
        return

    lines.append("")
    lines.append("## Boundary Entity Counts")
    lines.append("")

    entity_types = sorted(
        {
            entity_type
            for per_region in counts_by_region.values()
            if isinstance(per_region, dict)
            for entity_type in per_region.keys()
        }
    )

    if entity_types:
        header = "| Region | " + " | ".join(entity_types) + " | Total |"
        separator = "|---|" + "|".join(["---:" for _ in entity_types]) + "|---:|"
        lines.append(header)
        lines.append(separator)

        for region_id in sorted(counts_by_region.keys()):
            per_region = counts_by_region.get(region_id, {})
            row_counts = [int(per_region.get(entity_type, 0)) for entity_type in entity_types]
            region_total = sum(row_counts)
            counts_text = " | ".join(str(count) for count in row_counts)
            lines.append(f"| {region_id} | {counts_text} | {region_total} |")

    if isinstance(totals_by_entity, dict) and totals_by_entity:
        lines.append("")
        lines.append("- Totals by entity type across selected regions:")
        for entity_type in sorted(totals_by_entity.keys()):
            lines.append(f"  - {entity_type}: {int(totals_by_entity[entity_type])}")
    if isinstance(total_all_entities, int):
        lines.append(f"- Total entity count across selected regions: {total_all_entities}")

    if isinstance(original_totals_by_entity, dict) and original_totals_by_entity:
        lines.append("- Totals by entity type in original data:")
        for entity_type in sorted(original_totals_by_entity.keys()):
            lines.append(f"  - {entity_type}: {int(original_totals_by_entity[entity_type])}")
    if isinstance(original_total_all_entities, int):
        lines.append(f"- Total entity count in original data: {original_total_all_entities}")


def _append_timing_breakdown(lines: list[str], metrics: RunMetrics) -> None:
    stage_times = metrics.extra.get("timing_stage_seconds")
    file_type_times = metrics.extra.get("timing_file_type_seconds")
    file_type_counts = metrics.extra.get("timing_file_type_counts")
    slowest_files = metrics.extra.get("timing_slowest_files")

    if not any(
        [
            isinstance(stage_times, dict) and stage_times,
            isinstance(file_type_times, dict) and file_type_times,
            isinstance(slowest_files, list) and slowest_files,
        ]
    ):
        return

    lines.append("")
    lines.append("## Timing Breakdown")
    lines.append("")

    if isinstance(stage_times, dict) and stage_times:
        lines.append("### Pipeline Stages")
        lines.append("")
        for stage_name, seconds in sorted(stage_times.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {stage_name}: {float(seconds):.2f}s")
        lines.append("")

    if isinstance(file_type_times, dict) and file_type_times:
        lines.append("### File Types")
        lines.append("")
        lines.append("| File Type | Files Timed | Total Time (s) | Avg Time (s) |")
        lines.append("|---|---:|---:|---:|")
        for file_type, seconds in sorted(file_type_times.items(), key=lambda item: item[1], reverse=True):
            count = 0
            if isinstance(file_type_counts, dict):
                count = int(file_type_counts.get(file_type, 0))
            avg = float(seconds) / count if count > 0 else 0.0
            lines.append(f"| {file_type} | {count} | {float(seconds):.2f} | {avg:.2f} |")
        lines.append("")

    if isinstance(slowest_files, list) and slowest_files:
        lines.append("### Slowest Files")
        lines.append("")
        lines.append("| Source | Type | Status | Time (s) |")
        lines.append("|---|---|---|---:|")
        for entry in slowest_files:
            source = str(entry.get("source_path", ""))
            file_type = str(entry.get("file_type", ""))
            status = str(entry.get("status", ""))
            duration_s = float(entry.get("duration_s", 0.0))
            lines.append(f"| {source} | {file_type} | {status} | {duration_s:.2f} |")


def _append_fov_layout_section(lines: list[str], metrics: RunMetrics) -> None:
    summary = metrics.extra.get("fov_layout_summary")
    if not isinstance(summary, dict) or not summary:
        return

    fov_rows_px = int(summary.get("fov_rows_px", 0) or 0)
    fov_cols_px = int(summary.get("fov_cols_px", 0) or 0)
    fov_overlap_px = int(summary.get("fov_overlap_px", 0) or 0)
    stride_rows_px = int(summary.get("fov_stride_rows_px", 0) or 0)
    stride_cols_px = int(summary.get("fov_stride_cols_px", 0) or 0)
    max_fov_x = int(summary.get("max_potential_fov_x", 0) or 0)
    max_fov_y = int(summary.get("max_potential_fov_y", 0) or 0)
    sw_version = summary.get("instrument_sw_version")
    pixel_size_um = summary.get("pixel_size_um")

    lines.append("")
    lines.append("## FOV Layout")
    lines.append("")
    lines.append(f"- Instrument SW version: {sw_version if sw_version else 'Unknown'}")
    if pixel_size_um is not None:
        lines.append(f"- Pixel size (um): {float(pixel_size_um):.6f}")
    lines.append(f"- FOV dimensions (rows x cols, px): {fov_rows_px} x {fov_cols_px}")
    lines.append(f"- FOV overlap (px): {fov_overlap_px}")
    lines.append(f"- FOV stride (Y rows px): {stride_rows_px}")
    lines.append(f"- FOV stride (X cols px): {stride_cols_px}")
    lines.append(f"- Max potential FOV in X direction: {max_fov_x}")
    lines.append(f"- Max potential FOV in Y direction: {max_fov_y}")

    per_region = summary.get("per_region")
    if isinstance(per_region, list) and per_region:
        lines.append("")
        lines.append("### Per-Region Potential FOV")
        lines.append("")
        lines.append("| Region | Width (px) | Height (px) | Potential FOV X | Potential FOV Y |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in sorted(per_region, key=lambda r: str(r.get("region_id", ""))):
            lines.append(
                f"| {row.get('region_id', '')} | {int(row.get('width_px', 0) or 0)} | {int(row.get('height_px', 0) or 0)} | {int(row.get('potential_fov_x', 0) or 0)} | {int(row.get('potential_fov_y', 0) or 0)} |"
            )


def _append_always_skipped_section(lines: list[str], metrics: RunMetrics) -> None:
    by_rule = metrics.extra.get("always_skipped_by_rule")
    if not isinstance(by_rule, dict) or not by_rule:
        return

    lines.append("")
    lines.append("## Always-Skipped Files")
    lines.append("")
    lines.append("| Rule | Count | Examples |")
    lines.append("|---|---:|---|")

    for rule in sorted(by_rule.keys()):
        raw_items = by_rule.get(rule, [])
        if not isinstance(raw_items, list):
            continue
        items = sorted({str(item) for item in raw_items})
        if not items:
            continue
        examples = ", ".join(items[:3])
        if len(items) > 3:
            examples += ", ..."
        lines.append(f"| {rule} | {len(items)} | {examples} |")


def _append_per_region_sections(lines: list[str], metrics: RunMetrics) -> None:
    region_ids: set[str] = set()
    counts_by_region = metrics.extra.get("entity_counts_by_region")
    if isinstance(counts_by_region, dict):
        region_ids.update(str(region_id) for region_id in counts_by_region.keys())

    for item in metrics.file_metrics:
        for region_id in item.rows_written_by_region.keys():
            region_ids.add(str(region_id))

    if not region_ids:
        return

    lines.append("")
    lines.append("## Per-Region Summary")
    lines.append("")

    for region_id in sorted(region_ids):
        files_with_region = 0
        rows_written_total = 0
        processing_time_s = 0.0

        for item in metrics.file_metrics:
            if region_id not in item.rows_written_by_region:
                continue

            files_with_region += 1
            rows_written_total += int(item.rows_written_by_region.get(region_id, 0))

            if item.duration_s is not None:
                region_count_for_item = max(1, len(item.rows_written_by_region))
                processing_time_s += float(item.duration_s) / region_count_for_item

        lines.append(f"### Region {region_id}")
        lines.append("")
        lines.append(f"- Total processing time (s): {processing_time_s:.2f}")
        lines.append(f"- Files with region output: {files_with_region}")
        lines.append(f"- Total rows written: {rows_written_total}")

        if isinstance(counts_by_region, dict):
            per_region_counts = counts_by_region.get(region_id)
            if isinstance(per_region_counts, dict) and per_region_counts:
                entity_parts = [
                    f"{entity_type}={int(per_region_counts.get(entity_type, 0))}"
                    for entity_type in sorted(per_region_counts.keys())
                ]
                lines.append(f"- Boundary entity counts: {', '.join(entity_parts)}")

        lines.append("")


def build_run_metadata_markdown(
    config: SplitConfig,
    metrics: RunMetrics,
    started_at: datetime,
    completed_at: datetime,
) -> str:
    duration_s = (completed_at - started_at).total_seconds()

    lines: list[str] = []
    lines.append("# xenium-splitter run metadata")
    lines.append("")
    lines.append("## Timing")
    lines.append("")
    lines.append(f"- Started (UTC): {started_at.isoformat()}")
    lines.append(f"- Completed (UTC): {completed_at.isoformat()}")
    lines.append(f"- Duration (s): {duration_s:.2f}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Input directory: {config.input_dir}")
    lines.append(f"- LASSO file: {config.lasso_file}")
    lines.append(f"- H&E image: {config.he_image if config.he_image else 'None'}")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    lines.append(f"- Output directory: {config.output_dir}")
    lines.append(f"- Convert SVS to OME-TIFF: {config.convert_svs_to_ome}")
    lines.append(f"- Squash multi-layer image stacks: {config.squash_layers}")
    lines.append(f"- Write grid overlays: {config.overlays}")
    lines.append(f"- Include globs: {config.include_globs if config.include_globs else 'None'}")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append(f"- Regions: {metrics.region_count}")
    lines.append(f"- Files discovered: {metrics.files_total}")
    lines.append(f"- Files processed: {metrics.files_processed}")
    lines.append(f"- Files skipped: {metrics.files_skipped}")
    lines.append(f"- Files failed: {metrics.files_failed}")
    _append_fov_layout_section(lines, metrics)
    _append_boundary_entity_counts(lines, metrics)
    _append_always_skipped_section(lines, metrics)

    lines.append("")
    lines.append("## Per-file Results")
    lines.append("")
    lines.append("| Source | Type | Status | Detail | Time (s) | Rows in | Rows out |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    sorted_file_metrics = sorted(
        metrics.file_metrics,
        key=lambda item: (item.status.lower(), item.file_type.lower(), item.source_path.lower()),
    )
    for item in sorted_file_metrics:
        rows_in = "" if item.rows_input is None else str(item.rows_input)
        rows_out = "" if item.rows_written_total is None else str(item.rows_written_total)
        duration_s = "" if item.duration_s is None else f"{item.duration_s:.2f}"
        lines.append(
            f"| {item.source_path} | {item.file_type} | {item.status} | {item.detail} | {duration_s} | {rows_in} | {rows_out} |"
        )

    _append_per_region_sections(lines, metrics)
    _append_timing_breakdown(lines, metrics)

    return "\n".join(lines) + "\n"


def build_region_readme_markdown(
    config: SplitConfig,
    *,
    region_id: str,
    region_area_um2: float,
    num_cells: int,
    num_transcripts: int,
    has_old_fov_to_new_fov: bool,
    has_transcript_id_fov_remap: bool,
    has_grid_overlays: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# Region {region_id}")
    lines.append("")
    lines.append("## Creation")
    lines.append("")
    lines.append(
        f"This directory was created by xenium-splitter from `{config.input_dir}` using LASSO file `{config.lasso_file}` for region `{region_id}`."
    )
    lines.append("")
    lines.append("## Processing Summary")
    lines.append("")
    lines.append(f"- Cells written: {num_cells}")
    lines.append(f"- Transcripts written: {num_transcripts}")
    lines.append(f"- Region area (um^2): {region_area_um2:.2f}")
    lines.append(f"- Images skipped during run: {config.skip_images}")
    lines.append(f"- Grid overlays requested: {config.overlays}")
    lines.append(f"- Transcript archives copied verbatim: {config.copy_transcripts}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Filtered transcript, cell, and boundary outputs are region-specific subsets of the original Xenium outputs.")
    lines.append("- When transcript coordinates are rebased, locations in region outputs are crop-local so they align with region image crops.")
    lines.append("- `experiment.xenium` and `_region_metadata.json` are updated to reflect region-level counts and area.")

    explained_files: list[tuple[str, str]] = []
    if has_old_fov_to_new_fov:
        explained_files.append(
            (
                "old_fov_to_new_fov.csv",
                "Maps original FOV indices to compacted FOV indices used in the region transcript zarr after rebuilding sparse FOV spaces.",
            )
        )
    if has_transcript_id_fov_remap:
        explained_files.append(
            (
                "transcripts_id_fov_remap.csv.gz",
                "Lists old and new transcript ID/FOV assignments after transcript grid rebuilding so transcript identity changes can be audited.",
            )
        )
    if has_grid_overlays:
        explained_files.append(
            (
                "grid_overlays/",
                "Contains per-level morphology overlay images with transcript grid lines, labels, and FOV guides for spatial debugging.",
            )
        )

    if explained_files:
        lines.append("")
        lines.append("## Additional Files")
        lines.append("")
        for name, description in explained_files:
            lines.append(f"- `{name}`: {description}")

    return "\n".join(lines) + "\n"
