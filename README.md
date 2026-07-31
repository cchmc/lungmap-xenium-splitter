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

> **Note:** xenium-splitter does **not** perform H&E registration. The H&E image is assumed to already be spatially aligned with the Xenium coordinate space. No registration file, affine transform, or warping is applied. If your H&E requires registration to the morphology image, that step must be completed before running this tool.

**Optional — image processing:**

| Option                                   | Default            | Description                                                                             |
| ---------------------------------------- | ------------------ | --------------------------------------------------------------------------------------- |
| `--squash-layers` / `--no-squash-layers` | `--squash-layers`  | Flatten multi-layer TIFF stacks to 2D/RGB before cropping.                              |
| `--skip-images` / `--process-images`     | `--process-images` | Skip all image cropping and masking (useful for fast data-only runs).                   |
| `--overlays` / `--no-overlays`           | `--no-overlays`    | Write annotated FOV grid overlay images alongside each region `morphology_mip.ome.tif`. |

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
