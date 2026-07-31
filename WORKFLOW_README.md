# Xenium Splitter Workflow (Human-Readable)

This workflow is organized by practical processing steps. Each step explains what happens, why it happens, and the key functions involved.

## Quick Flow Diagram

```text
CLI split command
	-> build SplitConfig
	-> run_split(config)
			-> load pixel size from experiment.xenium
			-> load LASSO regions
			-> compute FOV layout summary
			-> find boundary files
			-> extract entity IDs by region intersection
			-> write region entity ID summaries
			-> process cell_feature_matrix bundles (before main loop)
			-> select remaining input files
			-> process grouped multi-format files (cells, transcripts, boundaries)
			-> route each remaining file by type
					 -> images: crop + mask per region
					 -> tabular/hdf5/zarr: ID-filter first, coordinate fallback second
			-> rebase x/y coordinates to region crop origin
			-> optionally split external H&E image
			-> generate morphology MIP and focus outputs (if missing)
			-> write morphology grid overlays (if --overlays)
			-> recalculate diffexp (if --recalculate-diffexp)
			-> update per-region metadata (experiment.xenium, README.md)
			-> write run metadata README
	-> done
```

Decision summary:

- Preferred filtering path: boundary-ID intersection.
- Fallback filtering path: coordinate containment.
- Coordinate system rule: rebased table coordinates must align with image crop origin.
- Multi-format grouping: cells/transcripts/boundaries are read once and written to all formats from one in-memory filtered DataFrame.

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
- Detect cell feature matrix bundles (CFM directory + sibling `.h5` + `.zarr.zip`) and process them first as a logical unit.
- Exclude CFM component files from the generic per-file loop.
- Group remaining files by known multi-format stems (`cells`, `transcripts`, `cell_boundaries`, `nucleus_boundaries`) so each group is read once and written to all output formats.

Why:

- Prevents duplicate processing, ensures CFM files are handled atomically, and avoids re-reading large files for each output format.

Key functions:

- `_select_input_files(config)`
- `iter_input_files(input_dir)`
- `_find_cell_feature_matrix_bundles(input_dir)`
- `is_cell_feature_matrix_group(path)`
- `_group_multi_format_files(input_files)`
- `_tabular_file_stem(file_path)`

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

- For each region, filter the CFM bundle by matched cell IDs.
- Keep features unchanged, filter barcodes, rewrite MTX columns, and optionally rebuild the Zarr.
- Rebuild `cell_feature_matrix.zarr.zip` from the filtered H5 or MTX bundle when enabled.

Why:

- Maintains consistency across barcodes, matrix, and zarr representations.

Note: CFM bundles are processed **before** the main file loop (Step 2 / Step 6 in `run_split`).

Key functions:

- `_process_cell_feature_matrix_groups(...)`
- `_split_cell_feature_matrix_bundle(...)`
- `get_cell_feature_matrix_files(group_dir)`
- `filter_cell_feature_matrix(group_dir, output_dir, cell_ids)`
- `filter_cell_feature_matrix_h5(...)`
- `read_barcodes_file(...)`, `write_barcodes_file(...)`
- `read_mtx_file(...)`, `write_mtx_file(...)`
- `filter_cell_feature_matrix_zarr(...)`
- `build_cell_feature_matrix_zarr_from_h5(...)`
- `build_cell_feature_matrix_zarr_from_sparse_bundle(...)`
- `_filter_zarr_dataset(...)`

## Step 10: Generate morphology outputs and overlays

What happens:

- If `morphology_mip.ome.tif` or `morphology_focus.ome.tif` were not present in the source run, generate them from the cropped `morphology.ome.tif`.
- Morphology MIP uses max-projection across Z-planes.
- Morphology focus selects the sharpest Z-plane using Laplacian variance scoring.
- If `--overlays` is set, render annotated grid overlay PNG images over the MIP output for each region.

Why:

- Ensures all downstream tools have the expected morphology output regardless of source run configuration.
- Overlays aid visual QC of FOV grid alignment per region.

Key functions:

- `_ensure_region_morphology_mip_outputs(...)`
- `_ensure_region_morphology_focus_outputs(...)`
- `generate_morphology_mip(...)`
- `generate_morphology_focus_with_stats(...)`
- `_write_morphology_grid_overlays(...)`
- `render_grid_overlay_image(...)`

## Step 11: Recalculate diffexp and write metadata

What happens:

- If `--recalculate-diffexp` is set, re-run differential expression calculation for each region using the filtered `cell_feature_matrix` and clustering outputs.
- Copy `gene_panel.json` to each region directory.
- Update `experiment.xenium` with region-level cell counts, transcript counts, and area.
- Write a per-region `README.md` summarising provenance and processing.
- Write the top-level `run_metadata_README.md` with timing, metrics, and per-file results.

Why:

- Keeps analysis outputs consistent with the filtered region data.
- Provides a reproducible audit trail for each split execution.

Key functions:

- `_recalculate_diffexp_for_regions(...)`
- `recalculate_diffexp_for_region(...)`
- `_update_region_metadata_outputs(...)`
- `copy_gene_panel(...)`
- `update_experiment_xenium_for_region(...)`
- `build_region_readme_markdown(...)`
- `build_run_metadata_markdown(...)`
