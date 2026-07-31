#!/usr/bin/env python3
"""Process xenium-splitter benchmark log files and produce a summary report.

Reads LSF output logs, /usr/bin/time -v stderr output, and xenium-splitter
run_metadata_README.md files to assemble per-run metrics.

Reported metrics
----------------
  Status         - SUCCESS / FAILED / TIMEOUT / MEMLIMIT / RUNNING / MISSING
  Wall time      - Elapsed wall-clock time (seconds and HH:MM:SS)
  CPU time       - Total CPU seconds consumed
  Peak RAM (GB)  - Maximum resident set size (/usr/bin/time -v or LSF)
  Avg RAM (GB)   - Average RSS (LSF resource summary)
  Regions        - Number of LASSO regions split
  Cells total    - Total cells across all regions (from entity counts)
  Transcripts    - Estimated from per-region row counts in metadata
  Files ok/skip/fail  - xenium-splitter file processing summary
  Duration (s)   - Wall time reported by xenium-splitter itself
  Input data GB  - Size of key input files (input_dir recursively)
  H&E size GB    - Size of the H&E image file if provided
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
import os
import re
import sys
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


# ---------------------------------------------------------------------------
# File size helpers
# ---------------------------------------------------------------------------

def _dir_size_gb(path: str) -> float | None:
    """Recursively sum file sizes in a directory and return GB, or None if missing."""
    p = Path(path)
    if not p.is_dir():
        return None
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / (1024 ** 3), 2)


def _file_size_gb(path: str | None) -> float | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return round(p.stat().st_size / (1024 ** 3), 2)


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
        return list(csv.DictReader(fh))


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
        })
    return rows


# ---------------------------------------------------------------------------
# Assemble per-run record
# ---------------------------------------------------------------------------

def _assemble_run(row: dict) -> dict:
    """Parse all available log/metadata sources and return a flat metrics dict."""
    out = _parse_out_log(row["log_out"])
    err = _parse_err_log(row["log_err"])

    # Output dir: manifest > log
    output_dir = row.get("output_dir") or out.get("output_dir_from_log") or ""
    metadata_path = str(Path(output_dir) / "run_metadata_README.md") if output_dir else ""
    meta = _parse_run_metadata(metadata_path)

    status = _resolve_status(out, err)

    # Prefer /usr/bin/time -v values over LSF (more precise)
    peak_ram_gb = err.get("time_peak_ram_gb") or out.get("lsf_max_ram_gb")
    avg_ram_gb = out.get("lsf_avg_ram_gb")  # only available from LSF
    wall_s = err.get("time_wall_s") or out.get("lsf_run_s") or meta.get("meta_duration_s")
    cpu_s = err.get("time_cpu_s") or out.get("lsf_cpu_s")

    # File sizes
    he_size_gb = _file_size_gb(row.get("he_image") or None)
    input_dir = row.get("input_dir") or ""
    output_size_gb = _dir_size_gb(output_dir) if output_dir else None

    # Input image sizes (morphology files in input_dir)
    morphology_size_gb: float | None = None
    if input_dir and Path(input_dir).is_dir():
        morph_files = (
            list(Path(input_dir).rglob("morphology*.ome.tif"))
            + list(Path(input_dir).rglob("morphology*.ome.tiff"))
        )
        if morph_files:
            morphology_size_gb = round(
                sum(f.stat().st_size for f in morph_files) / (1024 ** 3), 2
            )

    return {
        # Identity
        "name": row["name"],
        "mode": row["mode"],
        "job_name": row["job_name"],
        "job_id": row.get("job_id") or "-",
        # Status
        "status": status,
        "exit_code": out.get("exit_code"),
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
        "regions": meta.get("meta_regions"),
        "cells_total": meta.get("meta_cells"),
        "transcripts_total": meta.get("meta_transcripts"),
        "total_entities": meta.get("meta_total_entities"),
        "files_processed": meta.get("meta_files_processed"),
        "files_skipped": meta.get("meta_files_skipped"),
        "files_failed": meta.get("meta_files_failed"),
        "files_discovered": meta.get("meta_files_discovered"),
        "splitter_duration_s": meta.get("meta_duration_s"),
        "slowest_file": meta.get("meta_slowest_file") or "-",
        "slowest_file_s": meta.get("meta_slowest_file_s"),
        # File sizes
        "he_size_gb": he_size_gb,
        "morphology_size_gb": morphology_size_gb,
        "output_size_gb": output_size_gb,
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
    ("Transcripts",   13),
    ("Files ok",       9),
    ("Files fail",    10),
    ("H&E(GB)",        9),
    ("Output(GB)",    11),
]

_SEPARATOR = "-" * sum(w for _, w in _TABLE_HEADERS)


def _print_summary_table(records: list[dict], sort_by: str) -> None:
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

    header = "".join(_col(h, w) for h, w in _TABLE_HEADERS)
    print(_SEPARATOR)
    print(header)
    print(_SEPARATOR)

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
            _fmt_int(r["transcripts_total"]),
            _fmt_int(r["files_processed"]),
            _fmt_int(r["files_failed"]),
            _fmt_gb(r["he_size_gb"]),
            _fmt_gb(r["output_size_gb"]),
        ]
        print("".join(_col(v, w) for v, w in zip(row_vals, (w for _, w in _TABLE_HEADERS))))

    print(_SEPARATOR)


# ---------------------------------------------------------------------------
# Per-run detail sections
# ---------------------------------------------------------------------------

def _print_run_detail(r: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {r['job_name']}  [{r['status']}]")
    print(f"{'='*60}")
    print(f"  Dataset      : {r['name']}  (mode={r['mode']})")
    print(f"  Job ID       : {r['job_id']}")
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
    print(f"    Transcripts: {_fmt_int(r['transcripts_total'])}")
    print(f"    Files ok   : {_fmt_int(r['files_processed'])}  "
          f"skipped={_fmt_int(r['files_skipped'])}  "
          f"failed={_fmt_int(r['files_failed'])}  "
          f"(discovered={_fmt_int(r['files_discovered'])})")
    if r["slowest_file"] != "-":
        s_s = f"  ({r['slowest_file_s']:.1f}s)" if r["slowest_file_s"] else ""
        print(f"    Slowest    : {r['slowest_file']}{s_s}")

    print(f"\n  File sizes")
    print(f"    H&E image  : {_fmt_gb(r['he_size_gb'])} GB")
    print(f"    Morphology : {_fmt_gb(r['morphology_size_gb'])} GB")
    print(f"    Output dir : {_fmt_gb(r['output_size_gb'])} GB")

    print(f"\n  Logs")
    print(f"    stdout     : {r['log_out']}")
    print(f"    stderr     : {r['log_err']}")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "name", "mode", "job_name", "job_id", "status", "exit_code",
    "wall_s", "cpu_s", "peak_ram_gb", "avg_ram_gb", "requested_ram_gb",
    "regions", "cells_total", "transcripts_total", "total_entities",
    "files_processed", "files_skipped", "files_failed", "files_discovered",
    "splitter_duration_s", "slowest_file", "slowest_file_s",
    "he_size_gb", "morphology_size_gb", "output_size_gb",
    "output_dir", "log_out", "log_err",
]


def _write_csv(records: list[dict], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCSV written to: {path}")


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

    for mode in ("data_only", "with_images"):
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

    _print_summary_table(records, sort_by=args.sort_by)
    _print_statistics(records)

    if not args.no_detail:
        print(f"\n\n{'='*60}")
        print("  PER-RUN DETAILS")
        print(f"{'='*60}")
        for r in records:
            _print_run_detail(r)

    if args.csv:
        _write_csv(records, args.csv)


if __name__ == "__main__":
    main()
