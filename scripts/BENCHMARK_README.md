# xenium-splitter Benchmarking Guide

This directory contains two scripts for systematically benchmarking xenium-splitter
across different datasets, data sizes, and processing modes on an LSF-based HPC cluster.

---

## Files

| File                           | Purpose                                                            |
| ------------------------------ | ------------------------------------------------------------------ |
| `benchmark_submit.py`          | Build and submit LSF jobs; write manifest and rendered job scripts |
| `benchmark_report.py`          | Parse logs after jobs finish and print a formatted metrics report  |
| `benchmark_config_example.csv` | Example benchmark configuration showing required columns           |
| `benchmark_example_output.txt` | Example of the complete report output for reference                |

---

## Prerequisites

- **LSF cluster access** with `bsub` on PATH.
- **xenium-splitter installed** in a Python environment reachable from compute nodes
  (conda environment, virtualenv, or system install).
- **`/usr/bin/time`** — used inside each job to capture precise peak RSS and wall time.
  Present on Linux; may require `time` package on some minimal images.
- Python ≥ 3.10 on the submit host (for the scripts themselves; no third-party dependencies).

---

## Quick Start

### 1. Create a config file

Copy and edit `benchmark_config_example.csv`:

```csv
name,input_dir,he_image,lasso_file,walltime_data_only,ram_gb_data_only,walltime_with_images,ram_gb_with_images
my_dataset,/path/to/xenium/outs,,/path/to/lasso.csv,2:00,32,6:00,64
```

**Required columns:**

| Column                 | Format                 | Description                                            |
| ---------------------- | ---------------------- | ------------------------------------------------------ |
| `name`                 | string, no spaces      | Short identifier used in job names and output paths    |
| `input_dir`            | absolute path          | Xenium output directory (`outs/` folder or equivalent) |
| `he_image`             | absolute path or blank | H&E image file; leave blank if no H&E                  |
| `lasso_file`           | absolute path          | LASSO region file (GeoJSON or CSV/TSV)                 |
| `walltime_data_only`   | `HH:MM`                | LSF wall-time limit for `--skip-images` run            |
| `ram_gb_data_only`     | integer GB             | RAM to request for `--skip-images` run                 |
| `walltime_with_images` | `HH:MM`                | LSF wall-time limit for full run                       |
| `ram_gb_with_images`   | integer GB             | RAM to request for full run                            |

> **Sizing guidance:** See the RAM Requirements section in the main README.md.
> As a starting point, request 1.5–2× the estimated peak RAM so the job is not killed
> by LSF memory limits. Adjust based on what the report shows.

### 2. Write an environment setup script (recommended)

Create a shell script that activates your environment so each job uses the correct
Python and xenium-splitter installation:

```bash
# /path/to/activate_env.sh
module load python/3.11
source /path/to/venv/bin/activate
# or: conda activate xenium_env
```

### 3. Submit benchmark jobs

```bash
python benchmark_submit.py \
    --config benchmark_config.csv \
    --log-dir /hpc/logs/xenium_benchmark \
    --output-base /hpc/benchmark_outputs \
    --queue long \
    --project myproject \
    --env-script /path/to/activate_env.sh
```

This submits two LSF jobs per dataset row (one `data_only`, one `with_images`) and
writes a `benchmark_manifest.csv` to the log directory.

### 4. Wait for jobs to finish

Monitor with standard LSF commands:

```bash
bjobs -u $USER
```

Or watch for specific jobs by name:

```bash
bjobs -J "xsplit_*"
```

### 5. Generate the report

```bash
python benchmark_report.py \
    --log-dir /hpc/logs/xenium_benchmark
```

To save a CSV copy alongside the terminal output:

```bash
python benchmark_report.py \
    --log-dir /hpc/logs/xenium_benchmark \
    --csv /hpc/logs/xenium_benchmark/report.csv
```

---

## benchmark_submit.py — All Options

```
Required:
  --config / -c       Benchmark config CSV file path
  --log-dir / -l      Directory for .out, .err, and .job log files
  --output-base / -o  Base directory for xenium-splitter outputs

LSF resources:
  --queue / -q        LSF queue name (default: normal)
  --project / -P      LSF project / account (-P flag)
  --cores             CPU cores per job (default: 1)

Environment:
  --env-script        Shell script sourced at job start to set up Python environment
  --python            xenium-splitter command; default 'xenium-splitter'.
                      Use a full path or 'python3 -m xenium_splitter.cli' for
                      a non-installed checkout.

Run selection:
  --modes             both (default), data_only, or with_images

Output:
  --timestamp-outputs Append UTC timestamp to output dirs to preserve prior runs

xenium-splitter arguments:
  --extra-args        Extra flags appended to every job (e.g. '--overlays')
  --verbose-splitter  Pass -v to xenium-splitter in each job

Utility:
  --dry-run           Print scripts and manifest without submitting or writing
```

### Output per job

Each submitted job produces three files in `--log-dir`:

| File                       | Contents                                                              |
| -------------------------- | --------------------------------------------------------------------- |
| `xsplit_<name>_<mode>.out` | LSF stdout: job markers, xenium-splitter output, LSF resource summary |
| `xsplit_<name>_<mode>.err` | LSF stderr: `/usr/bin/time -v` resource detail, xenium-splitter logs  |
| `xsplit_<name>_<mode>.job` | Rendered job script (can be resubmitted with `bsub < file.job`)       |

---

## benchmark_report.py — All Options

```
Required:
  --log-dir / -l      Directory containing LSF .out / .err log files

Optional:
  --manifest / -m     Path to benchmark_manifest.csv (default: auto-detected in --log-dir)
  --csv FILE          Write summary table to this CSV file
  --no-detail         Suppress per-run detail sections; print summary table only
  --sort-by           Sort summary table by: name (default), mode, status,
                      wall_s, or peak_ram_gb
```

### Data sources parsed per run

The report assembles metrics from three sources in priority order:

1. **`.err` log — `/usr/bin/time -v` output** (most precise):
   - Peak RSS (maximum resident set size, kbytes → GB)
   - Wall-clock elapsed time
   - User + system CPU seconds

2. **`.out` log — LSF resource summary** (fallback):
   - `Max Memory` and `Average Memory` (MB → GB)
   - `Run time` and `CPU time`
   - Job completion status (`Successfully completed.` / `Exited with exit code N.` /
     `TERM_RUNLIMIT` / `TERM_MEMLIMIT`)

3. **`run_metadata_README.md`** in the xenium-splitter output directory:
   - Number of regions, cells, total entities
   - Files processed / skipped / failed / discovered
   - xenium-splitter internal duration (seconds)
   - Slowest individual file and its processing time

Additionally, at report time:

- H&E image file size (GB) — measured from the path in the manifest
- Morphology image size (GB) — `morphology*.ome.tif` files in `input_dir`
- xenium-splitter output directory size (GB) — recursive `du`

### Report status values

| Status     | Meaning                                                         |
| ---------- | --------------------------------------------------------------- |
| `SUCCESS`  | Job exited 0 and LSF reported successful completion             |
| `FAILED`   | Non-zero exit code from xenium-splitter or the job script       |
| `TIMEOUT`  | LSF killed the job (`TERM_RUNLIMIT`) — increase `walltime`      |
| `MEMLIMIT` | LSF killed the job (`TERM_MEMLIMIT`) — increase `ram_gb`        |
| `RUNNING`  | Log file exists but no completion marker yet — job still active |
| `MISSING`  | No `.out` log file found — job not yet started or wrong path    |

---

## Interpreting Results

### Reading the summary table

```
Name                Mode        Status   Wall     CPU      PeakRAM(GB) AvgRAM(GB) ...
GSM7990532_small    data_only   SUCCESS  2m30s    2m15s    12.4        8.2        ...
GSM7990532_small    with_images SUCCESS  8m20s    6m45s    27.8        18.3       ...
large_slide_A       with_images MEMLIMIT 1h23m20s -        118.4       -          ...
```

- **Comparing `data_only` vs `with_images`** for the same dataset shows the isolated
  cost of image processing (wall time, RAM). If `with_images` times out or hits
  memory limits while `data_only` succeeds, you know image RAM is the bottleneck.

- **`PeakRAM(GB)` close to requested RAM** is a warning sign. If peak ≥ 80% of
  requested, increase the `ram_gb` allocation before resubmitting.

- **`Slowest` file in per-run detail** identifies the pipeline bottleneck for each run.
  `transcripts.zarr.zip` dominating is expected for large datasets.
  `morphology.ome.tif` dominating indicates large image loads — consider SVS format.

- **`TIMEOUT` with `wall_s` equal to the walltime limit** confirms the job hit the
  LSF wall limit cleanly. Increase `walltime_with_images` (e.g., double it) and
  resubmit with `bsub < xsplit_<name>_with_images.job`.

- **`MEMLIMIT` with `PeakRAM` showing a value** tells you approximately how much RAM
  was in use when the job was killed. Set `ram_gb_with_images` to at least
  `PeakRAM × 1.5` and resubmit.

### Common patterns

| Observation                                    | Likely cause                               | Action                                             |
| ---------------------------------------------- | ------------------------------------------ | -------------------------------------------------- |
| `data_only` fast, `with_images` TIMEOUT        | Large TIFF H&E — full image loaded         | Convert H&E to SVS; use `--images-only` pass       |
| Both modes MEMLIMIT                            | Transcript table too large                 | Increase RAM or split lasso into smaller regions   |
| `files_failed` > 0                             | Unsupported file type or format error      | Check `.err` log for stack traces                  |
| `transcripts.zarr.zip` slowest by large margin | Normal for dense datasets                  | Consider `--copy-transcripts` to skip zarr rebuild |
| `Splitter` duration ≈ `Wall clock`             | Healthy — little overhead outside splitter | No action needed                                   |
| `Splitter` much less than `Wall clock`         | Long Python startup or disk I/O wait       | Check node load; try dedicated queue               |

---

## Resubmitting Failed or Timed-Out Jobs

Each submitted job script is saved to the log directory. Resubmit without re-running
the full submit script:

```bash
# Resubmit a single job after adjusting nothing (e.g., to a higher-RAM queue):
bsub < /hpc/logs/xenium_benchmark/xsplit_large_slide_A_with_images.job

# Or edit the .job file first to change memory / walltime, then resubmit:
nano /hpc/logs/xenium_benchmark/xsplit_large_slide_A_with_images.job
bsub < /hpc/logs/xenium_benchmark/xsplit_large_slide_A_with_images.job
```

To resubmit with a timestamp to avoid overwriting prior output, add
`--timestamp-outputs` when re-running `benchmark_submit.py`.

---

## Tips

- Run `--dry-run` first to verify the generated job scripts before submitting to the
  cluster.
- Use `--modes data_only` for an initial quick pass to measure data processing speed
  and RAM without the image overhead.
- Use `--modes with_images` with `--extra-args '--images-only'` to benchmark image
  processing in isolation (no transcript/cell data loaded).
- The `--sort-by peak_ram_gb` option is useful for identifying which datasets are
  closest to running out of memory.
- The generated CSV (`--csv`) can be opened in Excel or loaded into pandas for
  further analysis across multiple benchmark runs.
