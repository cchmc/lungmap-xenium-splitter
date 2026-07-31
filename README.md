# xenium-splitter

`xenium-splitter` splits Xenium output files by user-defined LASSO regions and writes one output
directory per region. It can also process a matching H&E image, applying polygon masks and
cropping to each region.

## Features

- CLI-first workflow.
- LASSO region parsing from GeoJSON/JSON or CSV/TSV.
- Tabular Xenium split (CSV/TSV/TXT/Parquet) using detected coordinate columns.
- Coordinate re-basing in each region output (region origin becomes `(0, 0)`).
- Image masking + cropping for raster images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.ome.tif`, `.ome.tiff`).
- Optional SVS handling for H&E: load, split, and optionally convert to OME-TIFF.
- Metadata README generation with run parameters and file-level metrics.

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[parquet,svs,hdf5,zarr,dev]"
```

## CLI Usage

### Options

**Required:**

| Option         | Description                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| `--input-dir`  | Xenium output directory (the `outs/` folder or equivalent).                                              |
| `--lasso-file` | LASSO region file — GeoJSON/JSON or CSV/TSV (see [LASSO Input Expectations](#lasso-input-expectations)). |
| `--output-dir` | Destination directory; one `region_<id>/` subfolder is created per region.                               |

**Optional — H&E image:**

| Option                 | Default | Description                                                                            |
| ---------------------- | ------- | -------------------------------------------------------------------------------------- |
| `--he-image`           | None    | External H&E image to split alongside Xenium outputs (TIFF, OME-TIFF, SVS, PNG, JPEG). |
| `--convert-svs-to-ome` | off     | When `--he-image` is an SVS file, also write a full-slide OME-TIFF per region.         |

> **Note:** xenium-splitter does **not** perform H&E registration. The H&E image is assumed to already be spatially aligned with the Xenium coordinate space. No registration file, affine transform, or warping is applied. If your H&E requires registration to the morphology image, that step must be completed before running this tool. The tool also assumes the H&E image uses the **same pixel size** as the Xenium morphology data — it reads `pixel_size_um` from `experiment.xenium` and uses that value to convert polygon coordinates to pixel coordinates for both. If your H&E has a different pixel resolution, the crop will be incorrectly positioned.

**Optional — image processing:**

| Option                                   | Default            | Description                                                                                                                                                                                                                             |
| ---------------------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--squash-layers` / `--no-squash-layers` | `--squash-layers`  | Flatten multi-layer TIFF stacks to 2D/RGB before cropping.                                                                                                                                                                              |
| `--skip-images` / `--process-images`     | `--process-images` | Skip all image cropping and masking (useful for fast data-only runs).                                                                                                                                                                   |
| `--images-only`                          | off                | Process **only** image files; skip all tabular data, zarr, CFM, and diffexp stages. Useful when running images separately to keep RAM available, or for a re-run after a prior data-only pass. Mutually exclusive with `--skip-images`. |
| `--overlays` / `--no-overlays`           | `--no-overlays`    | Write annotated FOV grid overlay images alongside each region `morphology_mip.ome.tif`.                                                                                                                                                 |

**Optional — data filtering:**

| Option               | Default          | Description                                                                        |
| -------------------- | ---------------- | ---------------------------------------------------------------------------------- |
| `--include-glob`     | None (all files) | Glob pattern to restrict which files are read from `--input-dir`; repeatable.      |
| `--copy-transcripts` | off              | Copy `transcripts.zarr.zip` verbatim to each region instead of filtering/rebasing. |

**Optional — analysis outputs:**

| Option                                                                 | Default                            | Description                                                                       |
| ---------------------------------------------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------- |
| `--recalculate-diffexp` / `--skip-diffexp-recalc`                      | `--recalculate-diffexp`            | Recompute `analysis/diffexp` CSVs from the filtered matrix and clustering labels. |
| `--write-cell-feature-matrix-zarr` / `--skip-cell-feature-matrix-zarr` | `--write-cell-feature-matrix-zarr` | Write `cell_feature_matrix.zarr.zip` in each region output.                       |

**Optional — logging:**

| Option             | Default | Description                           |
| ------------------ | ------- | ------------------------------------- |
| `-v` / `--verbose` | off     | Enable debug-level logging to stderr. |

### Examples

**Minimal split — tabular data and morphology only:**

```bash
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file regions.geojson \
  --output-dir /path/to/output
```

**Fast data-only run — skip all image processing:**

```bash
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file regions.csv \
  --output-dir /path/to/output \
  --skip-images -v
```

**Include a pre-aligned H&E TIFF alongside Xenium outputs:**

```bash
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file regions.geojson \
  --output-dir /path/to/output \
  --he-image /path/to/aligned_he.ome.tif
```

**Include an SVS H&E and also produce per-region OME-TIFF exports:**

```bash
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file regions.geojson \
  --output-dir /path/to/output \
  --he-image /path/to/he.svs \
  --convert-svs-to-ome
```

**Write FOV grid overlays for visual QC of region alignment:**

```bash
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file regions.geojson \
  --output-dir /path/to/output \
  --overlays
```

**Process only a subset of files matching a glob pattern:**

```bash
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file regions.geojson \
  --output-dir /path/to/output \
  --include-glob "cells*" \
  --include-glob "transcripts*" \
  --skip-images
```

**Copy transcripts verbatim (no filtering or coordinate rebasing):**

```bash
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file regions.geojson \
  --output-dir /path/to/output \
  --copy-transcripts
```

**Images-only re-run after a prior data-only pass (minimise peak RAM):**

```bash
# Step 1 — data only (low RAM):
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file regions.geojson \
  --output-dir /path/to/output \
  --skip-images

# Step 2 — images only (reserves all RAM for image loading):
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file regions.geojson \
  --output-dir /path/to/output \
  --images-only
```

**Images-only with a large SVS H&E:**

```bash
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file regions.geojson \
  --output-dir /path/to/output \
  --images-only \
  --he-image /path/to/he.svs
```

### Temp Cleanup

Remove temporary working directories created under the OS temp folder
(`.../xenium_splitter`):

```bash
xenium-splitter clean-temp
```

## LASSO Input Expectations

### GeoJSON / JSON

- `FeatureCollection` with polygon-like geometries (`Polygon` or `MultiPolygon`).
- Region id is read from `id`, `region_id`, `name`, or generated automatically.

### CSV / TSV

Two supported layouts:

1. Point list with columns: `region_id`, `x`, `y`.
2. WKT with columns: `region_id`, `polygon_wkt`.

## Output Layout

For each region, a directory is created:

```text
output/
  region_<id>/
    ...split files mirroring input names...
  run_metadata_README.md
```

## RAM Requirements

Memory usage is driven by three independent factors: transcript table size, image size, and zarr processing. They do not all peak simultaneously, but on large datasets they can overlap significantly. Note, official benchmarking has not been done on this yet. Once benchmarks are available, they will be added to this section.

### Without image processing (`--skip-images`)

The dominant cost is the full transcript table, which is loaded entirely into RAM as a pandas DataFrame before being filtered per region.

| Dataset scale    | Approx. transcripts | Recommended RAM |
| ---------------- | ------------------- | --------------- |
| Small (< 1 mm²)  | ~500 K              | 4 GB            |
| Medium (~10 mm²) | ~5 M                | 8 GB            |
| Large (~50 mm²)  | ~50 M               | 16–32 GB        |

The cell/barcode tables and CFM zarr add a modest additional cost (typically < 1 GB).

### With image processing (default)

Image memory cost depends on format and adds on top of the transcript cost above.

**Morphology TIFF / OME-TIFF — always full-image load:**
The entire morphology image is read into RAM before cropping; windowed reads are not used for TIFF. A temporary copy is created during Z-stack squashing (max-projection), so peak is approximately 2× the raw array size.

| Image dimensions | Channels / planes  | RAM (peak ~2×) |
| ---------------- | ------------------ | -------------- |
| 5 K × 5 K        | 1 plane, uint16    | ~100 MB        |
| 10 K × 10 K      | 1 plane, uint16    | ~400 MB        |
| 20 K × 20 K      | 7 Z-planes, uint16 | ~11 GB         |
| 30 K × 30 K      | 7 Z-planes, uint16 | ~25 GB         |

**H&E as TIFF / OME-TIFF — full-image load:**
Same behaviour as morphology: the complete image is loaded before any cropping.

| Image dimensions | Channels  | RAM (peak ~2×) |
| ---------------- | --------- | -------------- |
| 10 K × 20 K      | RGB uint8 | ~1.2 GB        |
| 25 K × 45 K      | RGB uint8 | ~6.8 GB        |

**H&E as SVS — windowed, memory-efficient:**
OpenSlide reads only the polygon bounding box per region; the full slide is never loaded. A typical crop (5 K × 5 K px) uses ~100 MB regardless of source file size. **SVS is the recommended format for large H&E images.**

### Practical guidance

| Scenario                                      | Recommended minimum RAM |
| --------------------------------------------- | ----------------------- |
| Data only, small dataset                      | 8 GB                    |
| Data only, large dataset (≥ 50 M transcripts) | 32–64 GB                |
| Data + morphology TIFF, medium dataset        | 16 GB                   |
| Data + H&E TIFF (25 K × 45 K), medium dataset | 32 GB                   |
| Data + H&E TIFF (25 K × 45 K), large dataset  | 64 GB                   |
| Data + H&E SVS (any size), medium dataset     | 16 GB                   |
| Data + H&E SVS (any size), large dataset      | 32–48 GB                |

> **Tip:** Run `--skip-images` first for a fast data-only pass. Re-run with images only if needed. For large H&E files, convert to SVS or use a tiled OME-TIFF with a viewer that supports windowed reads before passing to this tool.

## Current Scope and Notes

- Files without recognized coordinate columns (for tabular) are skipped and documented in metadata.
- Unknown binary formats are currently skipped.
- SVS requires `openslide-python` and system OpenSlide libraries.

## FOV Recalculation Rules

When rebuilding transcript FOV assignments, `xenium-splitter` reads
`instrument_sw_version` from `experiment.xenium` and uses these FOV dimensions
(in pixels):

- `< 1.2`: `4240` rows x `2960` cols
- `>= 1.2`: `3520` rows x `2960` cols

The FOV overlap is fixed at `128` pixels.

- X stride: `2960 - 128 = 2832`
- Y stride (`< 1.2`): `4240 - 128 = 4112`
- Y stride (`>= 1.2`): `3520 - 128 = 3392`

Potential FOV counts in each direction are computed from rebased transcript
pixel coordinates (`x_px`, `y_px`) as:

- `potential_fov_x = floor(max(x_px) / x_stride) + 1`
- `potential_fov_y = floor(max(y_px) / y_stride) + 1`

where `x_px = floor(x_um / pixel_size_um)` and `y_px = floor(y_um / pixel_size_um)`.

For example, with `instrument_sw_version: 1.1.2.4`, the splitter uses the
`< 1.2` rule: `4240 x 2960` FOVs, overlap `128`, strides `4112` (Y) and `2832` (X).

## Development

```bash
pytest
ruff check .
```
