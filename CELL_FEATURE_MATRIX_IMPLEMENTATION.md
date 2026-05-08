# Cell Feature Matrix Implementation Summary

## Overview

Implemented support for processing cell_feature_matrix files as a grouped entity with boundary-based cell ID filtering.

## Files Modified

### 1. io_utils.py

Added functions to detect, read, and filter cell_feature_matrix components:

- **`is_cell_feature_matrix_group(path)`**: Checks if a path is the cell_feature_matrix directory
- **`get_cell_feature_matrix_files(group_dir)`**: Finds all component files (barcodes, features, matrix, zarr)
- **`read_mtx_file(path)`**: Reads Matrix Market sparse matrix format (MTX/GZ)
- **`write_mtx_file(path, data_rows, num_features, num_barcodes)`**: Writes filtered MTX file
- **`read_barcodes_file(path)`**: Reads barcode list from TSV/CSV/GZ
- **`write_barcodes_file(path, barcodes)`**: Writes barcodes to file
- **`filter_cell_feature_matrix(group_dir, output_dir, cell_ids)`**: Main filtering function
- **`_filter_zarr_dataset(src_arr, key, cell_id_to_new_idx)`**: Filters individual Zarr arrays (1D/2D)
- **`filter_cell_feature_matrix_zarr(zarr_zip_path, output_path, cell_id_to_new_idx, num_cells)`**: Filters Zarr archive
- **`_copy_zarr_attributes()` / `_copy_root_attributes()` / `_rezip_zarr()`**: Helper functions for Zarr processing

### 2. splitter.py

Updated to detect and process cell_feature_matrix groups:

- Added imports for new io_utils functions
- **`_find_cell_feature_matrix_groups(input_dir)`**: Discovers all cell_feature_matrix directories
- **`_split_cell_feature_matrix(cfm_dir, regions, ...)`**: Processes each CFM group for each region
- Updated `run_split()` to:
  - Find and process cell_feature_matrix groups before main file loop
  - Exclude CFM component files from general file processing
  - Track CFM processing metrics

### 3. lasso.py

Fixed LASSO file loading to handle quoted metadata lines:

- Enhanced `_load_tabular_regions()` to properly skip metadata lines in CSV files
- Now correctly identifies header lines even with quoted comment markers

## Implementation Details

### Cell Feature Matrix Structure

Standard 10X Genomics format:

- **barcodes.tsv.gz**: Cell barcodes (one per line)
- **features.tsv.gz**: Gene metadata (Ensembl ID, Name, Type)
- **matrix.mtx.gz**: Sparse matrix in Matrix Market format (genes × cells)
- **cell_feature_matrix.zarr.zip**: Optional Zarr archive version (also filtered per region)

### Filtering Algorithm

1. Read original barcodes from file
2. Identify which barcodes are in the region's extracted cell IDs
3. Create 1-indexed mapping (MTX format uses 1-based indexing)
4. Filter matrix rows to keep only those with cell indices in the mapping
5. Reindex cell columns in the matrix
6. Write filtered files:
   - Filtered barcodes (subset of original)
   - Features file (unchanged)
   - Matrix file (filtered rows, renumbered columns)
   - Zarr file: Extract → filter arrays → copy attributes → re-zip
     - 1D barcode arrays: keep matching indices
     - 2D expression arrays: filter cell dimension (usually axis 1)
     - Higher-dimensional arrays: copied unchanged

### Integration Points

- **Boundary extraction**: Existing region_entity_ids["cell"] provides filtered cell IDs
- **File routing**: Cell_feature_matrix groups detected before main file loop
- **Per-region processing**: Each region gets its own filtered cell_feature_matrix
- **Metrics tracking**: FilterCount and rows_written_by_region tracked for reporting

## Usage

No CLI changes needed. Cell_feature_matrix files are automatically:

1. Detected in the input directory
2. Processed alongside other region data
3. Filtered based on boundary-extracted cell IDs
4. Written to region_X/ output directories

## Testing

Created test files:

- `test_cfm.py`: Full integration test with LASSO loading
- `test_cfm_filter.py`: Isolated CFM filtering test

## Future Enhancements

- [ ] Add optional output format conversion (e.g., MTX to HDF5)
- [ ] Optimize for very large matrices (stream processing for MTX)
- [ ] Add progress reporting for large files
- [ ] Support for additional sparse matrix formats (COO, CSR, CSC)
