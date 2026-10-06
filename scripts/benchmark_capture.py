#!/usr/bin/env python3
"""Capture source, image, and splitter metrics on the compute node."""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import benchmark_report
except ImportError:
    import benchmark_report


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def _directory_inventory(root: Path) -> tuple[int | None, list[dict]]:
    if not root.is_dir():
        return None, []

    total_bytes = 0
    files: list[dict] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        total_bytes += size_bytes
        item = {
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "size_bytes": size_bytes,
        }
        if path.suffix.lower() in benchmark_report._IMAGE_SUFFIXES:
            image_metadata = benchmark_report._he_image_metadata(str(path))
            item["image"] = {
                "format": image_metadata["he_format"],
                "width_px": image_metadata["he_width_px"],
                "height_px": image_metadata["he_height_px"],
                "pixel_count": image_metadata["he_pixel_count"],
                "compression": image_metadata["he_compression"],
                **benchmark_report._tiff_spatial_metadata(path),
            }
        files.append(item)
    return total_bytes, files


def _capture(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    lasso_file = Path(args.lasso_file)
    output_dir = Path(args.output_dir)
    he_image = Path(args.he_image) if args.he_image else None

    input_size_bytes, input_files = _directory_inventory(input_dir)
    run_metadata_path = output_dir / "run_metadata_README.md"
    run_metrics = benchmark_report._parse_run_metadata(str(run_metadata_path))
    region_counts = benchmark_report._parse_region_counts_from_log(args.err_log)

    cells_written_total = (
        sum(int(region["cells"]) for region in region_counts) if region_counts else None
    )
    run_metrics = {
        "regions": len(region_counts) if region_counts else run_metrics.get("meta_regions"),
        "cells_total": run_metrics.get("meta_cells") or cells_written_total,
        "cells_written_total": cells_written_total,
        "transcripts_total": (
            sum(int(region["transcripts"]) for region in region_counts)
            if region_counts
            else run_metrics.get("meta_transcripts")
        ),
        "total_entities": run_metrics.get("meta_total_entities"),
        "files_processed": run_metrics.get("meta_files_processed"),
        "files_skipped": run_metrics.get("meta_files_skipped"),
        "files_failed": run_metrics.get("meta_files_failed"),
        "files_discovered": run_metrics.get("meta_files_discovered"),
        "splitter_duration_s": run_metrics.get("meta_duration_s"),
        "slowest_file": run_metrics.get("meta_slowest_file"),
        "slowest_file_s": run_metrics.get("meta_slowest_file_s"),
    }

    return {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "job_name": args.job_name,
        "mode": args.mode,
        "exit_code": args.exit_code,
        "host": platform.node(),
        "source_paths": {
            "input_dir": str(input_dir),
            "lasso_file": str(lasso_file),
            "he_image": str(he_image) if he_image else "",
            "output_dir": str(output_dir),
        },
        "file_sizes_bytes": {
            "input_dir": input_size_bytes,
            "lasso_file": _file_size(lasso_file),
            "he_image": _file_size(he_image) if he_image else None,
        },
        "he_image": benchmark_report._he_image_metadata(str(he_image) if he_image else None),
        "input_files": input_files,
        "region_counts": region_counts,
        "run_metrics": run_metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--lasso-file", required=True)
    parser.add_argument("--he-image", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--err-log", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(_capture(args), indent=2), encoding="utf-8")
    temporary_path.replace(output_path)


if __name__ == "__main__":
    main()
