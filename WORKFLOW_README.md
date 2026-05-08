# Xenium Splitter Workflow (Human-Readable)

This workflow is organized by practical processing steps. Each step explains what happens, why it happens, and the key functions involved.

## Quick Flow Diagram

```text
CLI split command
	-> build SplitConfig
	-> run_split(config)
			-> identify files to process
			-> load LASSO regions + pixel size
			-> find boundary files
			-> extract entity IDs by region intersection
			-> write region entity ID summaries
			-> process cell_feature_matrix groups first
			-> route each remaining file by type
					 -> images: crop + mask per region
					 -> tabular/hdf5/zarr: ID-filter first, coordinate fallback second
			-> rebase x/y coordinates to region crop origin
			-> optionally split external H&E image
			-> write run metadata
	-> done
```

Decision summary:

- Preferred filtering path: boundary-ID intersection.
- Fallback filtering path: coordinate containment.
- Coordinate system rule: rebased table coordinates must align with image crop origin.

## Step 1: Start the split job

What happens:

- The CLI parses options and builds runtime config.
- The splitter entrypoint begins orchestration.

Why:

- Establishes all inputs, flags, and output paths before any file work begins.

Key functions:

- `split_command()`
- `run_split(config)`

## Step 2: Identify files to process

What happens:

- Collect all candidate files from the input directory (or include globs).
- Detect grouped cell feature matrix directories first.
- Exclude CFM component files from the generic per-file loop.

Why:

- Prevents duplicate processing and ensures CFM files are handled as a single logical unit.

Key functions:

- `_select_input_files(config)`
- `iter_input_files(input_dir)`
- `_find_cell_feature_matrix_groups(input_dir)`
- `is_cell_feature_matrix_group(path)`

## Step 3: Load region geometry and crop semantics

What happens:

- Parse LASSO regions from GeoJSON or tabular format.
- Load pixel size from `experiment.xenium` when available.
- Use region bounds to define image crop origin and coordinate rebasing behavior.

Why:

- Region polygons drive filtering, and pixel size keeps table coordinates aligned with image crops.

Key functions:

- `load_lasso_regions(lasso_path)`
- `_load_geojson_regions(...)`
- `_load_tabular_regions(...)`
- `load_pixel_size_from_experiment(input_dir)`
- `_region_crop_origin_um(region, pixel_size_um)`

## Step 4: Filter entities by region intersection

What happens:

- Discover boundary files (cells, nucleus, transcripts).
- For each region, intersect boundary geometry with region polygon.
- Build per-region sets of entity IDs and write them to `entity_ids/*.txt`.

Why:

- Boundary-based ID filtering is the most accurate way to assign entities to regions.

Key functions:

- `find_boundary_files(input_dir)`
- `extract_entity_ids_in_region(boundary_file, region, ...)`
- `_extract_entity_ids_from_vertices(...)`
- `_extract_entity_ids_from_wkt(...)`
- `_write_region_entity_ids_and_counts(...)`

## Step 5: Process images per region

What happens:

- Route image files to image splitter.
- If format supports windowed reads, crop/mask only the needed region.
- Otherwise read full image, then crop/mask.
- Optionally process external H&E image and SVS->OME output.

Why:

- Produces per-region image outputs that visually match the same spatial frame used by filtered entities.

Key functions:

- `_split_image(...)`
- `supports_windowed_region_read(path)`
- `read_masked_cropped_region(...)`
- `read_image(path, squash_layers=...)`
- `mask_and_crop_region(image, polygon, pixel_size_um)`
- `save_image_like(...)`
- `_split_external_he_image(...)`
- `write_array_as_ome_tiff(...)`

## Step 6: Route each non-image file to the correct handler

What happens:

- Classify each file as tabular, HDF5, Zarr, image, or unsupported.
- Dispatch to the appropriate split function.

Why:

- Keeps the workflow modular and format-aware.

Key functions:

- `_process_file(...)`
- `classify_file(path)`
- `_split_tabular(...)`
- `_split_hdf5(...)`
- `_split_zarr(...)`

## Step 7: Filter tables and archives by region

What happens:

- For tabular/HDF5/Zarr, try ID-based filtering first (from boundary IDs).
- If that is not possible, fall back to coordinate containment.

Why:

- Prioritizes biologically accurate boundary intersections while still supporting datasets without usable ID columns.

Key functions:

- `_split_tabular(...)`
- `_split_tabular_by_entity_ids(...)`
- `_split_tabular_by_coordinates(...)`
- `_try_split_table_by_entity_ids(...)`
- `get_entity_id_column(table, entity_type)`
- `filter_table_by_entity_ids(table, entity_ids, id_col)`
- `detect_xy_columns(df)`
- `subset_table_for_region(df, region, x_col, y_col, pixel_size_um)`
- `write_table(...)`
- `read_hdf5_table(...)`, `write_hdf5_table(...)`
- `read_zarr_zip_table(...)`, `write_zarr_zip_table(...)`

## Step 8: Shift entity coordinates to the new region coordinate frame

What happens:

- After region filtering, x/y coordinates are rebased to region crop origin.
- Origin computation mirrors image crop math (`floor(bounds/pixel_size)` + clamp + scale back).

Why:

- Ensures filtered tables and cropped images share the same `(0,0)` origin.

Key functions:

- `rebase_table_coordinates_to_region_crop(df, region, x_col, y_col, pixel_size_um)`
- `_region_crop_origin_um(region, pixel_size_um)`
- `subset_table_for_region(...)` (contains filter + rebase path)

## Step 9: Process cell feature matrix as a grouped dataset

What happens:

- For each region, filter CFM by matched cell IDs.
- Keep features unchanged, filter barcodes, rewrite MTX columns, and filter optional Zarr.

Why:

- Maintains consistency across barcodes, matrix, and zarr representations.

Key functions:

- `_split_cell_feature_matrix(...)`
- `get_cell_feature_matrix_files(group_dir)`
- `filter_cell_feature_matrix(group_dir, output_dir, cell_ids)`
- `read_barcodes_file(...)`, `write_barcodes_file(...)`
- `read_mtx_file(...)`, `write_mtx_file(...)`
- `filter_cell_feature_matrix_zarr(...)`
- `_filter_zarr_dataset(...)`

## Step 10: Write run metadata and finish

What happens:

- Summarize counts, statuses, and outputs.
- Write final run metadata markdown.

Why:

- Provides a reproducible audit trail for each split execution.

Key functions:

- `build_run_metadata_markdown(...)`
- metadata write in `run_split(...)`
