#!/usr/bin/env python3
"""Process xenium-splitter benchmark log files and produce a summary report.

Reads LSF output logs, /usr/bin/time -v stderr output, per-job metrics JSON
sidecars, and xenium-splitter run_metadata_README.md files to assemble per-run
metrics.

Reported metrics
----------------
  Status         - SUCCESS / FAILED / TIMEOUT / MEMLIMIT / RUNNING / MISSING
  Wall time      - Elapsed wall-clock time (seconds and HH:MM:SS)
  CPU time       - Total CPU seconds consumed
  Peak RAM (GB)  - Maximum resident set size (/usr/bin/time -v or LSF)
  Avg RAM (GB)   - Average RSS (LSF resource summary)
  Regions        - Number of LASSO regions split
  Cells total    - Total cells across all regions (from entity counts)
    Transcripts    - Sum of completed region counts from sidecar/log metadata
  Files ok/skip/fail  - xenium-splitter file processing summary
  Duration (s)   - Wall time reported by xenium-splitter itself
    Input data GB  - Recursive input directory size captured on the compute node
  H&E size GB    - Size of the H&E image file if provided
    H&E dimensions - Format, width, height, and total pixels from image metadata
    Image compression - TIFF codecs from page metadata; standard raster codecs by format
    Source paths   - Input, LASSO, H&E, and output paths from manifest/sidecar
    File inventory - Source file paths and byte sizes from the compute-node sidecar
  Output GB      - Total size of the xenium-splitter output directory
  Slowest file   - Slowest individual file processed and its time

Usage
-----
  # After jobs finish:
  python benchmark_report.py --log-dir /path/to/logs

  # Save a CSV copy:
  python benchmark_report.py --log-dir /path/to/logs --csv report.csv

  # Specify manifest explicitly:
  python benchmark_report.py --log-dir /path/to/logs \\
      --manifest /path/to/logs/benchmark_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from math import prod
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Report on xenium-splitter benchmark runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--log-dir", "-l", required=True,
        help="Directory containing LSF .out / .err log files.",
    )
    p.add_argument(
        "--manifest", "-m", default=None,
        help=(
            "Path to benchmark_manifest.csv. "
            "If omitted, benchmark_manifest.csv inside --log-dir is used when "
            "present; otherwise logs are discovered by scanning --log-dir for "
            "files matching xsplit_*.out."
        ),
    )
    p.add_argument(
        "--csv", default=None, metavar="FILE",
        help="Write the summary table to this CSV file in addition to stdout.",
    )
    p.add_argument(
        "--no-detail", action="store_true",
        help="Suppress the per-run detail sections; print only the summary table.",
    )
    p.add_argument(
        "--metrics-only", action="store_true",
        help="Hide the Status column from the summary table and CSV.",
    )
    p.add_argument(
        "--sort-by",
        choices=["name", "mode", "status", "wall_s", "peak_ram_gb"],
        default="name",
        help="Sort the summary table by this column (default: name).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# LSF .out log parser
# ---------------------------------------------------------------------------

# Patterns for the LSF resource usage section appended to .out logs.
_RE_LSF_MAX_MEM = re.compile(r"Max Memory\s*:\s*([\d.]+)\s*(MB|GB|KB)", re.IGNORECASE)
_RE_LSF_AVG_MEM = re.compile(r"Average Memory\s*:\s*([\d.]+)\s*(MB|GB|KB)", re.IGNORECASE)
_RE_LSF_CPU_TIME = re.compile(r"CPU time\s*:\s*([\d.]+)\s*sec", re.IGNORECASE)
_RE_LSF_RUN_TIME = re.compile(r"Run time\s*:\s*([\d.]+)\s*sec", re.IGNORECASE)
_RE_LSF_SUCCESS = re.compile(r"Successfully completed\.", re.IGNORECASE)
_RE_LSF_EXIT = re.compile(r"Exited with exit code\s+(\d+)", re.IGNORECASE)
_RE_LSF_TERM_RUN = re.compile(r"TERM_RUNLIMIT", re.IGNORECASE)
_RE_LSF_TERM_MEM = re.compile(r"TERM_MEMLIMIT", re.IGNORECASE)
_RE_BENCH_EXIT = re.compile(r"Exit code\s*:\s*(\d+)")
_RE_BENCH_START = re.compile(r"Start UTC\s*:\s*(\S+)")
_RE_BENCH_END = re.compile(r"End UTC\s*:\s*(\S+)")
_RE_BENCH_OUTPUT_DIR = re.compile(r"Output dir\s*:\s*(.+)")


def _to_gb(value: float, unit: str) -> float:
    unit = unit.upper()
    if unit == "KB":
        return value / (1024 ** 2)
    if unit == "MB":
        return value / 1024
    return value  # GB


def _parse_out_log(path: str) -> dict:
    result: dict = {
        "lsf_status": "MISSING",
        "exit_code": None,
        "lsf_max_ram_gb": None,
        "lsf_avg_ram_gb": None,
        "lsf_cpu_s": None,
        "lsf_run_s": None,
        "bench_start_utc": None,
        "bench_end_utc": None,
        "output_dir_from_log": None,
    }
    if not Path(path).is_file():
        return result

    text = Path(path).read_text(errors="replace")

    # Determine LSF job completion status
    if _RE_LSF_TERM_MEM.search(text):
        result["lsf_status"] = "MEMLIMIT"
    elif _RE_LSF_TERM_RUN.search(text):
        result["lsf_status"] = "TIMEOUT"
    elif _RE_LSF_SUCCESS.search(text):
        result["lsf_status"] = "SUCCESS"
    else:
        m_exit = _RE_LSF_EXIT.search(text)
        if m_exit:
            result["lsf_status"] = "FAILED"
            result["exit_code"] = int(m_exit.group(1))
        else:
            # Log exists but no completion marker → job likely still running
            result["lsf_status"] = "RUNNING"

    # LSF resource summary
    m = _RE_LSF_MAX_MEM.search(text)
    if m:
        result["lsf_max_ram_gb"] = round(_to_gb(float(m.group(1)), m.group(2)), 2)
    m = _RE_LSF_AVG_MEM.search(text)
    if m:
        result["lsf_avg_ram_gb"] = round(_to_gb(float(m.group(1)), m.group(2)), 2)
    m = _RE_LSF_CPU_TIME.search(text)
    if m:
        result["lsf_cpu_s"] = float(m.group(1))
    m = _RE_LSF_RUN_TIME.search(text)
    if m:
        result["lsf_run_s"] = float(m.group(1))

    # Benchmark markers written by the job script itself
    m = _RE_BENCH_EXIT.search(text)
    if m:
        result["exit_code"] = int(m.group(1))
    m = _RE_BENCH_START.search(text)
    if m:
        result["bench_start_utc"] = m.group(1)
    m = _RE_BENCH_END.search(text)
    if m:
        result["bench_end_utc"] = m.group(1)
    m = _RE_BENCH_OUTPUT_DIR.search(text)
    if m:
        result["output_dir_from_log"] = m.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# /usr/bin/time -v stderr parser
# ---------------------------------------------------------------------------

_RE_TIME_WALL = re.compile(
    r"Elapsed \(wall clock\) time.*?:\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)"
)
_RE_TIME_MAX_RSS = re.compile(r"Maximum resident set size \(kbytes\)\s*:\s*(\d+)")
_RE_TIME_AVG_RSS = re.compile(r"Average resident set size \(kbytes\)\s*:\s*(\d+)")
_RE_TIME_USER = re.compile(r"User time \(seconds\)\s*:\s*([\d.]+)")
_RE_TIME_SYS = re.compile(r"System time \(seconds\)\s*:\s*([\d.]+)")
_RE_TIME_EXIT = re.compile(r"Exit status\s*:\s*(\d+)")
_RE_TIME_VOL_CTX = re.compile(r"Voluntary context switches\s*:\s*(\d+)")
_RE_TIME_INVOL_CTX = re.compile(r"Involuntary context switches\s*:\s*(\d+)")


def _wall_to_seconds(h: str | None, m: str, s: str) -> float:
    hours = int(h) if h else 0
    return hours * 3600 + int(m) * 60 + float(s)


def _parse_err_log(path: str) -> dict:
    result: dict = {
        "time_wall_s": None,
        "time_peak_ram_gb": None,
        "time_avg_ram_gb": None,
        "time_cpu_s": None,
        "time_exit_code": None,
    }
    if not Path(path).is_file():
        return result

    text = Path(path).read_text(errors="replace")

    m = _RE_TIME_WALL.search(text)
    if m:
        result["time_wall_s"] = round(_wall_to_seconds(m.group(1), m.group(2), m.group(3)), 1)

    m = _RE_TIME_MAX_RSS.search(text)
    if m:
        result["time_peak_ram_gb"] = round(int(m.group(1)) / (1024 ** 2), 2)

    m = _RE_TIME_AVG_RSS.search(text)
    if m:
        result["time_avg_ram_gb"] = round(int(m.group(1)) / (1024 ** 2), 2)

    m = _RE_TIME_USER.search(text)
    m2 = _RE_TIME_SYS.search(text)
    if m and m2:
        result["time_cpu_s"] = round(float(m.group(1)) + float(m2.group(1)), 1)

    m = _RE_TIME_EXIT.search(text)
    if m:
        result["time_exit_code"] = int(m.group(1))

    return result


# ---------------------------------------------------------------------------
# run_metadata_README.md parser
# ---------------------------------------------------------------------------

_RE_META_REGIONS = re.compile(r"^- Regions:\s*(\d+)", re.MULTILINE)
_RE_META_DURATION = re.compile(r"^- Duration \(s\):\s*([\d.]+)", re.MULTILINE)
_RE_META_FILES_PROC = re.compile(r"^- Files processed:\s*(\d+)", re.MULTILINE)
_RE_META_FILES_SKIP = re.compile(r"^- Files skipped:\s*(\d+)", re.MULTILINE)
_RE_META_FILES_FAIL = re.compile(r"^- Files failed:\s*(\d+)", re.MULTILINE)
_RE_META_FILES_DISC = re.compile(r"^- Files discovered:\s*(\d+)", re.MULTILINE)
_RE_META_TOTAL_ENT = re.compile(
    r"^- Total entity count across selected regions:\s*(\d+)", re.MULTILINE
)
_RE_META_ENTITY_ROW = re.compile(
    r"^\|\s*(\S+)\s*\|\s*([\d,]+)\s*\|.*?\|\s*([\d,]+)\s*\|", re.MULTILINE
)
# Cells specifically in entity table header
_RE_META_ENTITY_HEADER = re.compile(r"^\|\s*Region\s*\|\s*(.*?)\s*\|.*?\|\s*Total\s*\|", re.MULTILINE)
# Per-region row count
_RE_META_ROWS_WRITTEN = re.compile(r"^- Total rows written:\s*(\d+)", re.MULTILINE)
_RE_REGION_COUNTS = re.compile(
    r"Updated metadata for region (.+?) \(cells=(\d+), transcripts=(\d+), area_um2="
)

# Slowest file: first data row from timing table
_RE_META_SLOWEST = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*\w+\s*\|\s*\w+\s*\|\s*([\d.]+)\s*\|", re.MULTILINE
)


def _parse_run_metadata(path: str) -> dict:
    result: dict = {
        "meta_regions": None,
        "meta_duration_s": None,
        "meta_files_processed": None,
        "meta_files_skipped": None,
        "meta_files_failed": None,
        "meta_files_discovered": None,
        "meta_total_entities": None,
        "meta_cells": None,
        "meta_transcripts": None,
        "meta_slowest_file": None,
        "meta_slowest_file_s": None,
        "meta_entity_types": [],
    }
    if not Path(path).is_file():
        return result

    text = Path(path).read_text(errors="replace")

    def _int(pattern: re.Pattern) -> int | None:
        m = pattern.search(text)
        return int(m.group(1)) if m else None

    def _float(pattern: re.Pattern) -> float | None:
        m = pattern.search(text)
        return float(m.group(1)) if m else None

    result["meta_regions"] = _int(_RE_META_REGIONS)
    result["meta_duration_s"] = _float(_RE_META_DURATION)
    result["meta_files_processed"] = _int(_RE_META_FILES_PROC)
    result["meta_files_skipped"] = _int(_RE_META_FILES_SKIP)
    result["meta_files_failed"] = _int(_RE_META_FILES_FAIL)
    result["meta_files_discovered"] = _int(_RE_META_FILES_DISC)
    result["meta_total_entities"] = _int(_RE_META_TOTAL_ENT)

    # Try to extract cell and transcript counts from entity type columns
    m_header = _RE_META_ENTITY_HEADER.search(text)
    if m_header:
        entity_types = [t.strip().lower() for t in m_header.group(1).split("|")]
        result["meta_entity_types"] = entity_types

        # Sum totals column for each entity type across all region rows
        type_totals: dict[str, int] = {t: 0 for t in entity_types}
        for m_row in _RE_META_ENTITY_ROW.finditer(text):
            region_label = m_row.group(1)
            if region_label.lower() in ("region", "---"):
                continue
            # Re-parse the full row to get per-column counts
            row_text = m_row.group(0)
            cols = [c.strip().replace(",", "") for c in row_text.split("|") if c.strip()]
            if len(cols) >= len(entity_types) + 2:  # Region + types + Total
                for i, etype in enumerate(entity_types):
                    try:
                        type_totals[etype] += int(cols[i + 1])
                    except (ValueError, IndexError):
                        pass

        for etype, total in type_totals.items():
            if total > 0:
                if "cell" in etype:
                    result["meta_cells"] = total
                if "transcript" in etype:
                    result["meta_transcripts"] = total

    # Try to get transcript count from per-region rows written (fallback)
    if result["meta_transcripts"] is None:
        # Look for "Total rows written:" under regions that appear to be transcripts
        # This is a heuristic: sum rows_written from per-region sections
        pass  # Leave as None; entity table is the primary source

    # Slowest file from timing breakdown table
    # The table appears after "### Slowest Files"
    slowest_section = text.find("### Slowest Files")
    if slowest_section != -1:
        table_text = text[slowest_section:]
        matches = list(_RE_META_SLOWEST.finditer(table_text))
        # First match after the header separator
        for m_slow in matches:
            source = m_slow.group(1).strip()
            if source.startswith("---") or source.lower() == "source":
                continue
            result["meta_slowest_file"] = source
            try:
                result["meta_slowest_file_s"] = float(m_slow.group(2))
            except ValueError:
                pass
            break

    return result


def _parse_region_counts_from_log(path: str) -> list[dict[str, int | str]]:
    """Extract completed per-region cell and transcript counts from splitter logs."""
    log_path = Path(path)
    if not log_path.is_file():
        return []

    counts_by_region: dict[str, dict[str, int | str]] = {}
    for match in _RE_REGION_COUNTS.finditer(log_path.read_text(errors="replace")):
        region_id = match.group(1).strip()
        counts_by_region[region_id] = {
            "region_id": region_id,
            "cells": int(match.group(2)),
            "transcripts": int(match.group(3)),
        }
    return list(counts_by_region.values())


def _load_metrics_sidecar(path: str | Path) -> dict:
    """Load the optional per-job JSON metadata sidecar."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# File size helpers
# ---------------------------------------------------------------------------

def _directory_size_bytes(path: str | Path) -> int | None:
    """Recursively sum file sizes in a directory, or return None if missing."""
    p = Path(path)
    if not p.is_dir():
        return None
    total = 0
    for file_path in p.rglob("*"):
        try:
            if file_path.is_file():
                total += file_path.stat().st_size
        except OSError:
            continue
    return total


def _bytes_to_gb(size_bytes: int | None) -> float | None:
    return round(size_bytes / (1024 ** 3), 2) if size_bytes is not None else None


def _dir_size_gb(path: str) -> float | None:
    """Recursively sum file sizes in a directory and return GB, or None if missing."""
    return _bytes_to_gb(_directory_size_bytes(path))


def _file_size_gb(path: str | None) -> float | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return round(p.stat().st_size / (1024 ** 3), 2)


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs"}


def _image_compression(path: Path) -> str | None:
    """Read image compression metadata without decoding pixel data."""
    if not path.is_file():
        return None

    if path.suffix.lower() in {".tif", ".tiff", ".svs"}:
        try:
            import tifffile

            with tifffile.TiffFile(path) as image_file:
                codecs = list(dict.fromkeys(page.compression.name for page in image_file.pages))
            return ", ".join(codecs) if codecs else None
        except Exception:
            return None

    try:
        from PIL import Image

        with Image.open(path) as image:
            image_format = (image.format or "").upper()
            if image_format == "PNG":
                return "DEFLATE"
            if image_format in {"JPEG", "WEBP"}:
                return image_format
            if image_format == "BMP":
                return "UNCOMPRESSED"
            compression = image.info.get("compression")
            return str(compression) if compression is not None else None
    except Exception:
        return None


def _input_image_compressions(input_dir: str, he_image: str | None) -> list[str]:
    """Return compression descriptions for recognized images under input_dir."""
    root = Path(input_dir)
    if not root.is_dir():
        return []

    he_path = Path(he_image).resolve() if he_image else None
    details = []
    for image_path in sorted(path for path in root.rglob("*") if path.is_file()):
        if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if he_path is not None and image_path.resolve() == he_path:
            continue
        codec = _image_compression(image_path) or "unknown"
        details.append(f"{image_path.relative_to(root)}={codec}")
    return details


def _input_file_inventory(input_dir: str) -> list[dict]:
    """Collect source paths and byte sizes, annotating recognized image files."""
    root = Path(input_dir)
    if not root.is_dir():
        return []

    inventory = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        item = {
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "size_bytes": size_bytes,
        }
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            image_metadata = _he_image_metadata(str(path))
            item["image"] = {
                "format": image_metadata["he_format"],
                "width_px": image_metadata["he_width_px"],
                "height_px": image_metadata["he_height_px"],
                "pixel_count": image_metadata["he_pixel_count"],
                "compression": image_metadata["he_compression"],
            }
            item["image"].update(_tiff_spatial_metadata(path))
        inventory.append(item)
    return inventory


def _he_image_metadata(path: str | None) -> dict[str, str | int | None]:
    """Read H&E format and base image dimensions without decoding pixel data."""
    metadata: dict[str, str | int | None] = {
        "he_format": None,
        "he_width_px": None,
        "he_height_px": None,
        "he_pixel_count": None,
        "he_compression": None,
    }
    if not path:
        return metadata

    image_path = Path(path)
    lower_name = image_path.name.lower()
    if lower_name.endswith(".svs"):
        metadata["he_format"] = "SVS"
    elif lower_name.endswith((".ome.tif", ".ome.tiff")):
        metadata["he_format"] = "OME-TIFF"
    elif image_path.suffix.lower() in (".tif", ".tiff"):
        metadata["he_format"] = "TIFF"
    elif image_path.suffix:
        metadata["he_format"] = image_path.suffix[1:].upper()

    if not image_path.is_file():
        return metadata

    metadata["he_compression"] = _image_compression(image_path)
    try:
        if lower_name.endswith(".svs"):
            try:
                import openslide

                with openslide.OpenSlide(str(image_path)) as slide:
                    width, height = slide.dimensions
            except Exception:
                import tifffile

                with tifffile.TiffFile(image_path) as image_file:
                    page = image_file.pages[0]
                    width, height = int(page.imagewidth), int(page.imagelength)
        elif lower_name.endswith((".tif", ".tiff")):
            import tifffile

            with tifffile.TiffFile(image_path) as image_file:
                page = image_file.pages[0]
                width, height = int(page.imagewidth), int(page.imagelength)
                if image_file.ome_metadata:
                    metadata["he_format"] = "OME-TIFF"
        else:
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
                if image.format:
                    metadata["he_format"] = image.format.upper()
    except Exception:
        return metadata

    metadata["he_width_px"] = width
    metadata["he_height_px"] = height
    metadata["he_pixel_count"] = width * height
    return metadata


def _tiff_spatial_metadata(path: Path) -> dict[str, object | None]:
    """Read TIFF series shape and spatial/stack dimensions without decoding pixels."""
    metadata: dict[str, object | None] = {
        "axes": None,
        "shape": None,
        "plane_count": None,
        "sample_count": None,
    }
    if path.suffix.lower() not in {".tif", ".tiff", ".svs"} or not path.is_file():
        return metadata

    try:
        import tifffile

        with tifffile.TiffFile(path) as image_file:
            if not image_file.series:
                return metadata
            series = image_file.series[0]
            axes = str(series.axes).upper()
            shape = tuple(int(size) for size in series.shape)
            if len(shape) != len(axes):
                return metadata
            if "X" not in axes or "Y" not in axes:
                return metadata
            width = shape[axes.index("X")]
            height = shape[axes.index("Y")]
            nonspatial_shape = [
                size for axis, size in zip(axes, shape) if axis not in {"X", "Y", "C", "S"}
            ]
            metadata.update(
                {
                    "axes": axes,
                    "shape": list(shape),
                    "width_px": width,
                    "height_px": height,
                    "spatial_pixel_count": width * height,
                    "plane_count": prod(nonspatial_shape) if nonspatial_shape else 1,
                    "sample_count": prod(shape),
                }
            )
    except Exception:
        return metadata
    return metadata


def _he_dimensions_from_log(log_path: str, he_image: str) -> dict[str, int | None]:
    """Recover H&E dimensions from the logged image array shape if the source is offline."""
    dimensions: dict[str, int | None] = {
        "he_width_px": None,
        "he_height_px": None,
        "he_pixel_count": None,
    }
    if not he_image or not Path(log_path).is_file():
        return dimensions

    image_name = Path(he_image).name
    shape_pattern = re.compile(
        rf"Read .*{re.escape(image_name)} crop via .*source_shape=\(([^)]*)\)"
    )
    for line in Path(log_path).read_text(errors="replace").splitlines():
        match = shape_pattern.search(line)
        if not match:
            continue
        shape = [int(value.strip()) for value in match.group(1).split(",") if value.strip()]
        if len(shape) < 2:
            continue
        if len(shape) >= 3 and shape[-1] in (3, 4):
            height, width = shape[-3], shape[-2]
        else:
            height, width = shape[-2], shape[-1]
        dimensions["he_width_px"] = width
        dimensions["he_height_px"] = height
        dimensions["he_pixel_count"] = width * height
        break
    return dimensions


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def _fmt_seconds(secs: float | None) -> str:
    if secs is None:
        return "-"
    h = int(secs) // 3600
    m = (int(secs) % 3600) // 60
    s = int(secs) % 60
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _fmt_gb(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.0f}K"
    return str(value)


# ---------------------------------------------------------------------------
# Status resolution
# ---------------------------------------------------------------------------

def _resolve_status(out: dict, err: dict) -> str:
    """Determine final status, preferring the most specific signal available."""
    lsf = out.get("lsf_status", "MISSING")
    if lsf in ("TIMEOUT", "MEMLIMIT"):
        return lsf
    if lsf == "RUNNING":
        return "RUNNING"
    if lsf == "MISSING":
        return "MISSING"

    # Check exit codes
    exit_code = out.get("exit_code") or err.get("time_exit_code")
    if exit_code is not None and exit_code != 0:
        return "FAILED"
    if lsf == "SUCCESS":
        return "SUCCESS"
    return "FAILED"


# ---------------------------------------------------------------------------
# Manifest discovery
# ---------------------------------------------------------------------------

def _load_manifest(log_dir: str, manifest_path: str | None) -> list[dict] | None:
    candidate = Path(manifest_path) if manifest_path else Path(log_dir) / "benchmark_manifest.csv"
    if not candidate.is_file():
        return None
    with open(candidate, newline="") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        for field in ("log_out", "log_err", "job_file", "metrics_json"):
            value = row.get(field)
            if not value:
                continue
            source_path = Path(value)
            if source_path.is_file():
                continue
            bundled_path = candidate.parent / source_path.name
            if bundled_path.is_file():
                row[field] = str(bundled_path)
    return rows


def _discover_from_logs(log_dir: str) -> list[dict]:
    """Build a minimal manifest by scanning for xsplit_*.out files."""
    rows = []
    for out_file in sorted(Path(log_dir).glob("xsplit_*.out")):
        job_name = out_file.stem  # xsplit_<name>_<mode>
        # Best-effort: extract mode suffix
        if job_name.endswith("_data_only"):
            mode = "data_only"
            name = job_name[len("xsplit_"):-len("_data_only")]
        elif job_name.endswith("_with_images"):
            mode = "with_images"
            name = job_name[len("xsplit_"):-len("_with_images")]
        elif job_name.endswith("_images_only"):
            mode = "images_only"
            name = job_name[len("xsplit_"):-len("_images_only")]
        else:
            mode = "unknown"
            name = job_name[len("xsplit_"):]

        rows.append({
            "name": name,
            "mode": mode,
            "job_name": job_name,
            "job_id": "",
            "input_dir": "",
            "he_image": "",
            "lasso_file": "",
            "output_dir": "",  # will try to read from log
            "ram_gb": "",
            "walltime": "",
            "log_out": str(out_file),
            "log_err": str(out_file.with_suffix(".err")),
            "job_file": str(out_file.with_suffix(".job")),
            "metrics_json": str(out_file.with_suffix(".metrics.json")),
        })
    return rows


# ---------------------------------------------------------------------------
# Assemble per-run record
# ---------------------------------------------------------------------------

def _archive_size_mb(input_dir: str, inventory: list[dict], archive_name: str) -> float | None:
    """Read a root-level input archive's encoded size in decimal MB."""
    for item in inventory:
        if str(item.get("relative_path", "")).replace("\\", "/") == archive_name:
            size_bytes = item.get("size_bytes")
            if size_bytes is not None:
                return int(size_bytes) / 1_000_000
    if not input_dir:
        return None
    archive_path = Path(input_dir) / archive_name
    try:
        return archive_path.stat().st_size / 1_000_000 if archive_path.is_file() else None
    except OSError:
        return None


def _assemble_run(row: dict) -> dict:
    """Parse all available log/metadata sources and return a flat metrics dict."""
    out = _parse_out_log(row["log_out"])
    err = _parse_err_log(row["log_err"])
    metrics_json_path = row.get("metrics_json") or str(
        Path(row["log_err"]).with_suffix(".metrics.json")
    )
    sidecar = _load_metrics_sidecar(metrics_json_path)
    sidecar_paths = sidecar.get("source_paths", {})
    sidecar_run_metrics = sidecar.get("run_metrics", {})
    region_counts = sidecar.get("region_counts") or _parse_region_counts_from_log(
        row["log_err"]
    )

    # Output dir: manifest > log
    output_dir = (
        row.get("output_dir")
        or sidecar_paths.get("output_dir")
        or out.get("output_dir_from_log")
        or ""
    )
    metadata_path = str(Path(output_dir) / "run_metadata_README.md") if output_dir else ""
    meta = _parse_run_metadata(metadata_path)

    status = _resolve_status(out, err)

    # Prefer /usr/bin/time -v values over LSF (more precise)
    peak_ram_gb = err.get("time_peak_ram_gb") or out.get("lsf_max_ram_gb")
    avg_ram_gb = out.get("lsf_avg_ram_gb")  # only available from LSF
    wall_s = err.get("time_wall_s") or out.get("lsf_run_s") or meta.get("meta_duration_s")
    cpu_s = err.get("time_cpu_s") or out.get("lsf_cpu_s")

    input_dir = row.get("input_dir") or sidecar_paths.get("input_dir") or ""
    lasso_file = row.get("lasso_file") or sidecar_paths.get("lasso_file") or ""
    he_image = row.get("he_image") or sidecar_paths.get("he_image") or ""
    file_sizes = sidecar.get("file_sizes_bytes", {})
    file_inventory = sidecar.get("input_files")
    if file_inventory is None:
        file_inventory = _input_file_inventory(input_dir)

    sidecar_he_metadata = sidecar.get("he_image")
    he_metadata = sidecar_he_metadata or _he_image_metadata(he_image or None)
    if he_metadata.get("he_width_px") is None:
        he_metadata.update(_he_dimensions_from_log(row["log_err"], he_image))
    he_size_gb = (
        _bytes_to_gb(file_sizes.get("he_image"))
        if "he_image" in file_sizes
        else _file_size_gb(he_image or None)
    )
    input_dir_size_gb = (
        _bytes_to_gb(file_sizes.get("input_dir"))
        if "input_dir" in file_sizes
        else _dir_size_gb(input_dir) if input_dir else None
    )
    lasso_size_gb = (
        _bytes_to_gb(file_sizes.get("lasso_file"))
        if "lasso_file" in file_sizes
        else _file_size_gb(lasso_file or None)
    )
    input_images = [item for item in file_inventory if isinstance(item.get("image"), dict)]
    input_image_compressions = [
        f"{item.get('relative_path', item.get('path', ''))}="
        f"{item['image'].get('compression') or 'unknown'}"
        for item in input_images
    ]
    if not file_inventory and input_dir:
        input_image_compressions = _input_image_compressions(input_dir, he_image or None)

    morphology_item = next(
        (
            item
            for item in file_inventory
            if Path(str(item.get("relative_path", item.get("path", "")))).name.lower()
            in {"morphology.ome.tif", "morphology.ome.tiff"}
        ),
        None,
    )
    morphology_image = morphology_item.get("image", {}) if morphology_item else {}
    if morphology_item and not morphology_image.get("spatial_pixel_count"):
        morphology_image = {
            **morphology_image,
            **_tiff_spatial_metadata(Path(str(morphology_item.get("path", "")))),
        }
    morphology_file_size_gb = (
        _bytes_to_gb(int(morphology_item.get("size_bytes", 0)))
        if morphology_item and morphology_item.get("size_bytes") is not None
        else None
    )

    # Combined size of the morphology, focus, and MIP OME-TIFF inputs.
    morphology_size_bytes = sum(
        int(item.get("size_bytes", 0))
        for item in file_inventory
        if Path(str(item.get("relative_path", item.get("path", "")))).name.lower().startswith(
            "morphology"
        )
        and Path(str(item.get("relative_path", item.get("path", "")))).name.lower().endswith(
            (".ome.tif", ".ome.tiff")
        )
    )
    if morphology_size_bytes:
        morphology_size_gb = _bytes_to_gb(morphology_size_bytes)
    elif input_dir and Path(input_dir).is_dir():
        morph_files = (
            list(Path(input_dir).rglob("morphology.ome.tif"))
            + list(Path(input_dir).rglob("morphology.ome.tiff"))
        )
        morphology_size_gb = _bytes_to_gb(sum(f.stat().st_size for f in morph_files)) if morph_files else None
    else:
        morphology_size_gb = None

    regions = sidecar_run_metrics.get("regions")
    if regions is None:
        regions = meta.get("meta_regions")
    if regions is None and region_counts:
        regions = len(region_counts)

    cells_written_total = (
        sum(int(region["cells"]) for region in region_counts) if region_counts else None
    )
    cells_total = sidecar_run_metrics.get("cells_total")
    if cells_total is None:
        cells_total = meta.get("meta_cells")
    if cells_total is None:
        cells_total = cells_written_total

    transcripts_total = sidecar_run_metrics.get("transcripts_total")
    if transcripts_total is None:
        transcripts_total = meta.get("meta_transcripts")
    if transcripts_total is None and region_counts:
        transcripts_total = sum(int(region["transcripts"]) for region in region_counts)

    def _run_metric(sidecar_key: str, metadata_key: str):
        value = sidecar_run_metrics.get(sidecar_key)
        return value if value is not None else meta.get(metadata_key)

    return {
        # Identity
        "name": row["name"],
        "mode": row["mode"],
        "job_name": row["job_name"],
        "job_id": row.get("job_id") or "-",
        # Status
        "status": status,
        "exit_code": out.get("exit_code"),
        "input_dir": input_dir,
        "lasso_file": lasso_file,
        "he_image": he_image,
        "metrics_json": metrics_json_path,
        # Timing
        "wall_s": wall_s,
        "wall_fmt": _fmt_seconds(wall_s),
        "cpu_s": cpu_s,
        "cpu_fmt": _fmt_seconds(cpu_s),
        # Memory
        "peak_ram_gb": peak_ram_gb,
        "peak_ram_fmt": _fmt_gb(peak_ram_gb),
        "avg_ram_gb": avg_ram_gb,
        "avg_ram_fmt": _fmt_gb(avg_ram_gb),
        "requested_ram_gb": row.get("ram_gb") or "-",
        # xenium-splitter metrics
        "regions": regions,
        "cells_total": cells_total,
        "cells_written_total": cells_written_total,
        "transcripts_total": transcripts_total,
        "total_entities": _run_metric("total_entities", "meta_total_entities"),
        "files_processed": _run_metric("files_processed", "meta_files_processed"),
        "files_skipped": _run_metric("files_skipped", "meta_files_skipped"),
        "files_failed": _run_metric("files_failed", "meta_files_failed"),
        "files_discovered": _run_metric("files_discovered", "meta_files_discovered"),
        "splitter_duration_s": _run_metric("splitter_duration_s", "meta_duration_s"),
        "slowest_file": _run_metric("slowest_file", "meta_slowest_file") or "-",
        "slowest_file_s": _run_metric("slowest_file_s", "meta_slowest_file_s"),
        # File sizes
        "he_size_gb": he_size_gb,
        **he_metadata,
        "input_dir_size_gb": input_dir_size_gb,
        "lasso_size_gb": lasso_size_gb,
        "transcript_zarr_mb": _archive_size_mb(input_dir, file_inventory, "transcripts.zarr.zip"),
        "cell_zarr_mb": _archive_size_mb(input_dir, file_inventory, "cells.zarr.zip"),
        "input_files": file_inventory,
        "input_file_count": len(file_inventory),
        "region_counts": region_counts,
        "input_image_compressions": input_image_compressions,
        "input_image_compression": "; ".join(input_image_compressions),
        "morphology_size_gb": morphology_size_gb,
        "morphology_file_size_gb": morphology_file_size_gb,
        "morphology_width_px": morphology_image.get("width_px"),
        "morphology_height_px": morphology_image.get("height_px"),
        "morphology_pixel_count": morphology_image.get("spatial_pixel_count"),
        "morphology_plane_count": morphology_image.get("plane_count"),
        "morphology_sample_count": morphology_image.get("sample_count"),
        # Paths
        "output_dir": output_dir,
        "log_out": row["log_out"],
        "log_err": row["log_err"],
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_STATUS_LABEL = {
    "SUCCESS":  "SUCCESS ",
    "FAILED":   "FAILED  ",
    "TIMEOUT":  "TIMEOUT ",
    "MEMLIMIT": "MEMLIMIT",
    "RUNNING":  "RUNNING ",
    "MISSING":  "MISSING ",
}


def _fmt_status(s: str) -> str:
    return _STATUS_LABEL.get(s, s.ljust(8))


def _col(value, width: int) -> str:
    return str(value)[:width].ljust(width)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_TABLE_HEADERS = [
    ("Name",          20),
    ("Mode",          12),
    ("Status",         9),
    ("Wall",           9),
    ("CPU",            9),
    ("PeakRAM(GB)",   12),
    ("AvgRAM(GB)",    11),
    ("Regions",        8),
    ("Cells",          9),
    ("Cells written", 14),
    ("Transcripts",   13),
    ("TranscriptZarr(MB)", 20),
    ("CellZarr(MB)",   14),
    ("H&E(GB)",        9),
    ("H&E format",    12),
    ("H&E WxH(px)",   16),
    ("Morph(GB)",     11),
    ("Morph WxH(px)", 16),
    ("Morph planes",  13),
]

_SEPARATOR = "-" * sum(w for _, w in _TABLE_HEADERS)


def _print_summary_table(
    records: list[dict],
    sort_by: str,
    metrics_only: bool = False,
) -> None:
    # Sort
    def _sort_key(r: dict):
        v = r.get(sort_by)
        if v is None:
            return (1, 0)
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (0, str(v))

    records = sorted(records, key=_sort_key)

    table_headers = [
        (header, width)
        for header, width in _TABLE_HEADERS
        if not (metrics_only and header == "Status")
    ]
    separator = "-" * sum(width for _, width in table_headers)
    header = "".join(_col(h, w) for h, w in table_headers)
    print(separator)
    print(header)
    print(separator)

    for r in records:
        row_vals = [
            r["name"],
            r["mode"],
            _fmt_status(r["status"]),
            r["wall_fmt"],
            r["cpu_fmt"],
            _fmt_gb(r["peak_ram_gb"]),
            _fmt_gb(r["avg_ram_gb"]),
            _fmt_int(r["regions"]),
            _fmt_int(r["cells_total"]),
            _fmt_int(r["cells_written_total"]),
            _fmt_int(r["transcripts_total"]),
            f"{r['transcript_zarr_mb']:.2f}" if r["transcript_zarr_mb"] is not None else "-",
            f"{r['cell_zarr_mb']:.2f}" if r["cell_zarr_mb"] is not None else "-",
            _fmt_gb(r["he_size_gb"]),
            r["he_format"] or "-",
            (
                f"{r['he_width_px']}x{r['he_height_px']}"
                if r["he_width_px"] is not None and r["he_height_px"] is not None
                else "-"
            ),
            _fmt_gb(r["morphology_file_size_gb"]),
            (
                f"{r['morphology_width_px']}x{r['morphology_height_px']}"
                if r["morphology_width_px"] is not None
                and r["morphology_height_px"] is not None
                else "-"
            ),
            _fmt_int(r["morphology_plane_count"]),
        ]
        if metrics_only:
            row_vals.pop(2)
        print("".join(_col(v, w) for v, w in zip(row_vals, (w for _, w in table_headers))))

    print(separator)


# ---------------------------------------------------------------------------
# Per-run detail sections
# ---------------------------------------------------------------------------

def _print_run_detail(r: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {r['job_name']}  [{r['status']}]")
    print(f"{'='*60}")
    print(f"  Dataset      : {r['name']}  (mode={r['mode']})")
    print(f"  Job ID       : {r['job_id']}")
    print(f"  Input dir    : {r['input_dir'] or '-'}")
    print(f"  LASSO file   : {r['lasso_file'] or '-'}")
    print(f"  H&E source   : {r['he_image'] or '-'}")
    print(f"  Output dir   : {r['output_dir'] or '(unknown)'}")

    print(f"\n  Timing")
    print(f"    Wall clock : {r['wall_fmt']}  ({r['wall_s']:.0f}s)" if r["wall_s"] else "    Wall clock : -")
    print(f"    CPU time   : {r['cpu_fmt']}  ({r['cpu_s']:.0f}s)" if r["cpu_s"] else "    CPU time   : -")
    if r.get("splitter_duration_s"):
        print(f"    Splitter   : {_fmt_seconds(r['splitter_duration_s'])} (internal timer)")

    print(f"\n  Memory")
    print(f"    Peak RAM   : {r['peak_ram_fmt']} GB  (requested {r['requested_ram_gb']} GB)")
    print(f"    Avg  RAM   : {r['avg_ram_fmt']} GB")

    print(f"\n  xenium-splitter")
    print(f"    Regions    : {_fmt_int(r['regions'])}")
    print(f"    Cells      : {_fmt_int(r['cells_total'])}")
    if r["cells_written_total"] is not None:
        print(f"    Cells written in regions: {_fmt_int(r['cells_written_total'])}")
    print(f"    Transcripts: {_fmt_int(r['transcripts_total'])}")
    print(f"    Files ok   : {_fmt_int(r['files_processed'])}  "
          f"skipped={_fmt_int(r['files_skipped'])}  "
          f"failed={_fmt_int(r['files_failed'])}  "
          f"(discovered={_fmt_int(r['files_discovered'])})")
    if r["slowest_file"] != "-":
        s_s = f"  ({r['slowest_file_s']:.1f}s)" if r["slowest_file_s"] else ""
        print(f"    Slowest    : {r['slowest_file']}{s_s}")

    print(f"\n  File sizes")
    print(
        f"    Transcript Zarr: {r['transcript_zarr_mb']:.2f} MB"
        if r["transcript_zarr_mb"] is not None else "    Transcript Zarr: -"
    )
    print(
        f"    Cell Zarr      : {r['cell_zarr_mb']:.2f} MB"
        if r["cell_zarr_mb"] is not None else "    Cell Zarr      : -"
    )
    print(f"    H&E image  : {_fmt_gb(r['he_size_gb'])} GB")
    print(f"    H&E format : {r['he_format'] or '-'}")
    print(f"    H&E codec  : {r['he_compression'] or '-'}")
    if r["he_width_px"] is not None and r["he_height_px"] is not None:
        print(f"    H&E size   : {r['he_width_px']} x {r['he_height_px']} px")
    else:
        print("    H&E size   : -")
    print(f"    Input dir  : {_fmt_gb(r['input_dir_size_gb'])} GB")
    print(f"    LASSO file : {_fmt_gb(r['lasso_size_gb'])} GB")
    print(f"    Morphology file: {_fmt_gb(r['morphology_file_size_gb'])} GB")
    if r["morphology_width_px"] is not None and r["morphology_height_px"] is not None:
        print(
            f"    Morphology size: {r['morphology_width_px']} x "
            f"{r['morphology_height_px']} px"
        )
    else:
        print("    Morphology size: -")
    print(f"    Morphology planes: {_fmt_int(r['morphology_plane_count'])}")
    print(f"    Morphology family: {_fmt_gb(r['morphology_size_gb'])} GB")
    print(f"    Input files: {r['input_file_count']}")
    if r["input_image_compressions"]:
        print("    Input image compression:")
        for image_compression in r["input_image_compressions"]:
            print(f"      {image_compression}")
    if r["region_counts"]:
        print("    Per-region counts:")
        for region in r["region_counts"]:
            print(
                f"      {region['region_id']}: cells={region['cells']}, "
                f"transcripts={region['transcripts']}"
            )
    print(f"\n  Logs")
    print(f"    stdout     : {r['log_out']}")
    print(f"    stderr     : {r['log_err']}")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "name", "mode", "status", "job_name", "job_id",
    "input_dir", "lasso_file", "he_image", "metrics_json",
    "wall_s", "cpu_s", "peak_ram_gb", "avg_ram_gb", "requested_ram_gb",
    "regions", "cells_total", "cells_written_total", "transcripts_total", "total_entities",
    "transcript_zarr_mb", "cell_zarr_mb",
    "splitter_duration_s", "slowest_file", "slowest_file_s",
    "he_size_gb", "he_format", "he_width_px", "he_height_px",
    "he_compression", "input_image_compression",
    "input_dir_size_gb", "lasso_size_gb", "input_file_count",
    "input_files_json", "region_counts_json",
    "morphology_file_size_gb", "morphology_width_px", "morphology_height_px",
    "morphology_plane_count", "morphology_size_gb",
    "output_dir", "log_out", "log_err",
]


def _write_csv(records: list[dict], path: str, metrics_only: bool = False) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field for field in _CSV_FIELDS if not (metrics_only and field == "status")]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        serialized_records = []
        for record in records:
            csv_record = dict(record)
            csv_record["input_files_json"] = json.dumps(record.get("input_files", []))
            csv_record["region_counts_json"] = json.dumps(record.get("region_counts", []))
            serialized_records.append(csv_record)
        writer.writerows(serialized_records)
    print(f"\nCSV written to: {csv_path}")


# ---------------------------------------------------------------------------
# Quick statistics block
# ---------------------------------------------------------------------------

def _print_statistics(records: list[dict]) -> None:
    success = [r for r in records if r["status"] == "SUCCESS"]
    failed = [r for r in records if r["status"] == "FAILED"]
    other = [r for r in records if r["status"] not in ("SUCCESS", "FAILED")]

    print(f"\n{'='*60}")
    print("  STATISTICS")
    print(f"{'='*60}")
    print(f"  Total runs     : {len(records)}")
    print(f"  Successful     : {len(success)}")
    print(f"  Failed         : {len(failed)}")
    print(f"  Other (running/timeout/memlimit/missing): {len(other)}")

    for mode in ("data_only", "with_images", "images_only"):
        mode_success = [r for r in success if r["mode"] == mode]
        if not mode_success:
            continue
        walls = [r["wall_s"] for r in mode_success if r["wall_s"] is not None]
        peaks = [r["peak_ram_gb"] for r in mode_success if r["peak_ram_gb"] is not None]
        print(f"\n  Mode: {mode} ({len(mode_success)} successful run(s))")
        if walls:
            print(f"    Wall time  : min={_fmt_seconds(min(walls))}  "
                  f"max={_fmt_seconds(max(walls))}  "
                  f"avg={_fmt_seconds(sum(walls)/len(walls))}")
        if peaks:
            print(f"    Peak RAM   : min={min(peaks):.1f} GB  "
                  f"max={max(peaks):.1f} GB  "
                  f"avg={sum(peaks)/len(peaks):.1f} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    manifest = _load_manifest(args.log_dir, args.manifest)
    if manifest:
        print(f"Loaded manifest: {args.manifest or (Path(args.log_dir) / 'benchmark_manifest.csv')}")
    else:
        print(f"No manifest found; scanning {args.log_dir} for xsplit_*.out files...")
        manifest = _discover_from_logs(args.log_dir)
        if not manifest:
            print("No log files found.  Nothing to report.", file=sys.stderr)
            sys.exit(1)

    print(f"Processing {len(manifest)} run record(s)...\n")

    records = [_assemble_run(row) for row in manifest]

    from datetime import datetime
    print(f"xenium-splitter Benchmark Report")
    print(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log dir   : {args.log_dir}\n")

    _print_summary_table(
        records,
        sort_by=args.sort_by,
        metrics_only=args.metrics_only,
    )
    _print_statistics(records)

    if not args.no_detail:
        print(f"\n\n{'='*60}")
        print("  PER-RUN DETAILS")
        print(f"{'='*60}")
        for r in records:
            _print_run_detail(r)

    if args.csv:
        _write_csv(records, args.csv, metrics_only=args.metrics_only)


if __name__ == "__main__":
    main()
