#!/usr/bin/env python3
"""Submit LSF benchmark jobs for xenium-splitter.

Reads a CSV config file listing Xenium datasets and submits two LSF jobs per
dataset:

  <name>_data_only    -- xenium-splitter with --skip-images (measures data
                          processing speed at lower RAM)
  <name>_with_images  -- xenium-splitter full run including all image processing
                          (measures peak RAM and image-processing throughput)

Each job:
  - Wraps the xenium-splitter call with /usr/bin/time -v to capture peak RSS
    and elapsed wall time in the .err log.
  - Writes START / END markers and the exit code to the .out log.
  - Saves the rendered job script to <log_dir>/<job_name>.job for inspection
    and manual resubmission.

After all jobs are submitted a manifest CSV is written to <log_dir>/
benchmark_manifest.csv, which benchmark_report.py uses to locate logs and
xenium-splitter outputs.

Usage
-----
  python benchmark_submit.py \\
    --config benchmark_config_example.csv \\
    --log-dir /path/to/logs \\
    --output-base /path/to/benchmark_outputs \\
    --queue long \\
    --project myproject \\
    --env-script /path/to/activate_env.sh \\
    --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit LSF benchmark jobs for xenium-splitter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument(
        "--config", "-c", required=True,
        help="Path to benchmark config CSV (see benchmark_config_example.csv).",
    )
    p.add_argument(
        "--log-dir", "-l", required=True,
        help="Directory for LSF stdout/stderr logs and rendered job scripts.",
    )
    p.add_argument(
        "--output-base", "-o", required=True,
        help=(
            "Base directory for xenium-splitter outputs. "
            "Each run writes to <output-base>/<name>_<mode>/."
        ),
    )

    # LSF resource settings
    p.add_argument(
        "--queue", "-q", default="normal",
        help="LSF queue name (default: normal).",
    )
    p.add_argument(
        "--project", "-P", default=None,
        help="LSF project / account name passed to -P.",
    )
    p.add_argument(
        "--cores", type=int, default=1,
        help="Number of CPU cores to request per job (default: 1).",
    )

    # Environment
    p.add_argument(
        "--env-script", default=None,
        help=(
            "Path to a shell script that is sourced at the start of every LSF "
            "job to set up the Python environment (e.g. conda activate, "
            "module load, or a virtualenv activate script). "
            "If omitted the job inherits the submitting shell environment."
        ),
    )
    p.add_argument(
        "--python", default="xenium-splitter",
        help=(
            "Command used to invoke xenium-splitter inside the job. "
            "Use 'xenium-splitter' (default) when it is on PATH, or supply "
            "a full path like '/path/to/venv/bin/xenium-splitter', or "
            "'python3 -m xenium_splitter.cli' for a non-installed checkout."
        ),
    )

    # Run selection
    p.add_argument(
        "--modes",
        choices=["both", "data_only", "with_images"],
        default="both",
        help=(
            "Which job variants to submit: 'both' (default), "
            "'data_only' (--skip-images only), or 'with_images' (images only)."
        ),
    )

    # Output management
    p.add_argument(
        "--timestamp-outputs", action="store_true",
        help=(
            "Append a UTC timestamp to each output directory name so that "
            "repeated benchmark runs do not overwrite previous results."
        ),
    )

    # Extra xenium-splitter arguments
    p.add_argument(
        "--extra-args", default="",
        help=(
            "Additional xenium-splitter split arguments appended to every job, "
            "e.g. '--overlays --skip-diffexp-recalc'."
        ),
    )
    p.add_argument(
        "--verbose-splitter", action="store_true",
        help="Pass -v to xenium-splitter inside each job for debug logging.",
    )

    # Dry run
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print rendered job scripts and manifest without submitting or writing anything.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config file reader
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = {
    "name", "input_dir", "lasso_file",
    "walltime_data_only", "ram_gb_data_only",
    "walltime_with_images", "ram_gb_with_images",
}


def _read_config(path: str) -> list[dict]:
    """Read the benchmark config CSV and validate required columns."""
    rows: list[dict] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(row for row in fh if not row.lstrip().startswith("#"))
        if reader.fieldnames is None:
            raise ValueError(f"Config file appears to be empty: {path}")

        headers = {h.strip() for h in reader.fieldnames}
        missing = _REQUIRED_COLUMNS - headers
        if missing:
            raise ValueError(
                f"Config file is missing required columns: {', '.join(sorted(missing))}"
            )

        for raw_row in reader:
            row = {k.strip(): (v or "").strip() for k, v in raw_row.items()}
            if not row.get("name"):
                continue  # skip blank rows
            rows.append(row)

    if not rows:
        raise ValueError(f"No dataset rows found in config file: {path}")
    return rows


# ---------------------------------------------------------------------------
# RAM conversion helpers
# ---------------------------------------------------------------------------

def _ram_gb_to_mb(value: str) -> int:
    """Convert a RAM string like '32', '32GB', '32G', '32gb' to integer MB."""
    cleaned = value.strip().upper().replace("GB", "").replace("G", "").replace("B", "").strip()
    return int(float(cleaned) * 1024)


# ---------------------------------------------------------------------------
# Job script builder
# ---------------------------------------------------------------------------

_JOB_TEMPLATE = """\
#!/bin/bash
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -n {cores}
#BSUB -M {ram_mb}
#BSUB -R "rusage[mem={ram_mb}MB] span[hosts=1]"
#BSUB -W {walltime}
#BSUB -o {log_out}
#BSUB -e {log_err}
{project_line}

set -euo pipefail

{env_block}

echo "=== BENCHMARK JOB START ==="
echo "Job name   : {job_name}"
echo "Dataset    : {dataset_name}"
echo "Mode       : {mode}"
echo "Start UTC  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host       : $(hostname)"
echo "Input dir  : {input_dir}"
echo "Output dir : {output_dir}"
echo "Lasso file : {lasso_file}"
echo "H&E image  : {he_image_display}"
echo ""

# Create output directory explicitly so its mtime can be used as start time
mkdir -p '{output_dir}'

# /usr/bin/time -v writes detailed resource usage (peak RSS, wall time, etc.)
# to stderr at job completion, captured in the .err log file.
/usr/bin/time -v {splitter_cmd} split \\
  --input-dir '{input_dir}' \\
  --lasso-file '{lasso_file}' \\
  --output-dir '{output_dir}' \\
{he_image_arg}  {mode_flags}{verbose_flag}{extra_args_block}

SPLITTER_EXIT=$?

echo ""
echo "=== BENCHMARK JOB END ==="
echo "End UTC    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Exit code  : $SPLITTER_EXIT"

exit $SPLITTER_EXIT
"""


def _build_job_script(
    *,
    job_name: str,
    dataset_name: str,
    mode: str,
    queue: str,
    project: str | None,
    cores: int,
    ram_gb: str,
    walltime: str,
    log_out: str,
    log_err: str,
    env_script: str | None,
    splitter_cmd: str,
    input_dir: str,
    lasso_file: str,
    output_dir: str,
    he_image: str | None,
    mode_flags: str,
    extra_args: str,
    verbose_splitter: bool,
) -> str:
    ram_mb = _ram_gb_to_mb(ram_gb)
    project_line = f"#BSUB -P {project}" if project else ""

    if env_script:
        env_block = f"# Load environment\n# shellcheck source=/dev/null\nsource '{env_script}'\n"
    else:
        env_block = "# No --env-script provided; using inherited environment."

    he_image_display = he_image if he_image else "(none)"
    he_image_arg = f"  --he-image '{he_image}' \\\n" if he_image else ""

    verbose_flag = " \\\n  -v" if verbose_splitter else ""

    if extra_args.strip():
        extra_args_block = f" \\\n  {extra_args.strip()}"
    else:
        extra_args_block = ""

    mode_flags_str = f"  {mode_flags} \\" if mode_flags.strip() else "\\"

    return _JOB_TEMPLATE.format(
        job_name=job_name,
        dataset_name=dataset_name,
        mode=mode,
        queue=queue,
        cores=cores,
        ram_mb=ram_mb,
        walltime=walltime,
        log_out=log_out,
        log_err=log_err,
        project_line=project_line,
        env_block=env_block,
        input_dir=input_dir,
        output_dir=output_dir,
        lasso_file=lasso_file,
        he_image_display=he_image_display,
        he_image_arg=he_image_arg,
        splitter_cmd=splitter_cmd,
        mode_flags=mode_flags_str,
        verbose_flag=verbose_flag,
        extra_args_block=extra_args_block,
    )


# ---------------------------------------------------------------------------
# LSF submission
# ---------------------------------------------------------------------------

def _submit_job(script: str, dry_run: bool) -> str | None:
    """Pipe the job script to bsub and return the numeric job ID, or None."""
    if dry_run:
        print(script)
        print()
        return None

    result = subprocess.run(
        ["bsub"],
        input=script,
        text=True,
        capture_output=True,
    )

    output = result.stdout.strip()
    if result.returncode != 0:
        print(
            f"  ERROR: bsub failed (rc={result.returncode})\n"
            f"  stderr: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return None

    # Typical bsub output: "Job <12345> is submitted to queue <normal>."
    print(f"  {output}")
    match = re.search(r"Job <(\d+)>", output)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    log_dir = Path(args.log_dir)
    output_base = Path(args.output_base)

    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        output_base.mkdir(parents=True, exist_ok=True)

    datasets = _read_config(args.config)
    print(f"Loaded {len(datasets)} dataset(s) from {args.config}")

    # Determine timestamp suffix if requested
    from datetime import datetime, timezone
    ts_suffix = (
        "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        if args.timestamp_outputs
        else ""
    )

    # Decide which modes to run
    run_data_only = args.modes in ("both", "data_only")
    run_with_images = args.modes in ("both", "with_images")

    mode_specs: list[dict] = []
    if run_data_only:
        mode_specs.append({
            "suffix": "data_only",
            "flags": "--skip-images",
            "walltime_key": "walltime_data_only",
            "ram_key": "ram_gb_data_only",
        })
    if run_with_images:
        mode_specs.append({
            "suffix": "with_images",
            "flags": "",
            "walltime_key": "walltime_with_images",
            "ram_key": "ram_gb_with_images",
        })

    manifest_rows: list[dict] = []

    for row in datasets:
        name = row["name"]
        input_dir = row["input_dir"]
        lasso_file = row["lasso_file"]
        he_image = row.get("he_image") or None

        for spec in mode_specs:
            job_name = f"xsplit_{name}_{spec['suffix']}"
            output_dir = str(output_base / f"{name}_{spec['suffix']}{ts_suffix}")
            log_out = str(log_dir / f"{job_name}.out")
            log_err = str(log_dir / f"{job_name}.err")
            job_file = str(log_dir / f"{job_name}.job")
            ram_gb = row[spec["ram_key"]]
            walltime = row[spec["walltime_key"]]

            script = _build_job_script(
                job_name=job_name,
                dataset_name=name,
                mode=spec["suffix"],
                queue=args.queue,
                project=args.project,
                cores=args.cores,
                ram_gb=ram_gb,
                walltime=walltime,
                log_out=log_out,
                log_err=log_err,
                env_script=args.env_script,
                splitter_cmd=args.python,
                input_dir=input_dir,
                lasso_file=lasso_file,
                output_dir=output_dir,
                he_image=he_image,
                mode_flags=spec["flags"],
                extra_args=args.extra_args,
                verbose_splitter=args.verbose_splitter,
            )

            if args.dry_run:
                print(f"--- {job_name} ---")
                print(script)
                print()
            else:
                # Save rendered job script for inspection / resubmission
                Path(job_file).write_text(script)

            print(f"Submitting: {job_name}  (RAM={ram_gb}GB  walltime={walltime})")
            job_id = _submit_job(script, args.dry_run)

            manifest_rows.append({
                "name": name,
                "mode": spec["suffix"],
                "job_name": job_name,
                "job_id": job_id or "",
                "input_dir": input_dir,
                "he_image": he_image or "",
                "lasso_file": lasso_file,
                "output_dir": output_dir,
                "ram_gb": ram_gb,
                "walltime": walltime,
                "log_out": log_out,
                "log_err": log_err,
                "job_file": job_file,
            })

    # Write manifest
    manifest_path = log_dir / "benchmark_manifest.csv"
    manifest_fields = [
        "name", "mode", "job_name", "job_id",
        "input_dir", "he_image", "lasso_file", "output_dir",
        "ram_gb", "walltime", "log_out", "log_err", "job_file",
    ]

    if args.dry_run:
        print("--- MANIFEST (dry run, not written) ---")
        print(",".join(manifest_fields))
        for r in manifest_rows:
            print(",".join(str(r[f]) for f in manifest_fields))
    else:
        with open(manifest_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=manifest_fields)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nManifest written to: {manifest_path}")
        print(f"Submitted {len(manifest_rows)} job(s) total.")
        print(
            "\nTo generate a report after jobs complete:\n"
            f"  python benchmark_report.py --log-dir {args.log_dir}"
        )


if __name__ == "__main__":
    main()
