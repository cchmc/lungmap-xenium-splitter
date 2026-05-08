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

## Development

```bash
pytest
ruff check .
```
