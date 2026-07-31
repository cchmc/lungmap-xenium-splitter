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

```bash
xenium-splitter split \
  --input-dir /path/to/xenium_output \
  --lasso-file /path/to/lasso_regions.geojson \
  --output-dir /path/to/output \
  --he-image /path/to/he_image.svs \
  --convert-svs-to-ome
```

### Important Options

- `--input-dir`: Xenium output directory.
- `--lasso-file`: LASSO definition file (GeoJSON/JSON/CSV/TSV).
- `--output-dir`: Target directory for region folders.
- `--he-image`: Optional external H&E image path.
- `--convert-svs-to-ome`: If H&E is SVS, also emit an OME-TIFF per region.
- `--squash-layers/--no-squash-layers`: Flatten multi-layer images when needed.
- `--include-glob`: Glob pattern(s) to limit which files are processed from the input directory (repeatable).
- `--skip-images/--process-images`: Skip image extraction to reduce memory usage.
- `--overlays/--no-overlays`: Write region grid overlay images when a `morphology_mip.ome.tif` output is available.
- `--recalculate-diffexp/--skip-diffexp-recalc`: Recompute `analysis/diffexp` files from the region-filtered matrix and clustering outputs (default: on).
- `--write-cell-feature-matrix-zarr/--skip-cell-feature-matrix-zarr`: Write `cell_feature_matrix.zarr.zip` outputs (default: on).
- `--copy-transcripts`: Copy transcript files verbatim to each region output without filtering or rebasing.
- `-v/--verbose`: Enable debug logging.

Example — skip images for a quick run:

```bash
xenium-splitter split \
  --input-dir data/GSM7990532/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400/outs \
  --output-dir data_out/GSM7990532_slim \
  --lasso-file data/GSM7990532/GSM7990532_lasso_slim.csv \
  --skip-images -v
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
