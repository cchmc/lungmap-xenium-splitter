#!/usr/bin/env python3
"""Quick test of cell_feature_matrix filtering."""

import sys
from pathlib import Path

# Test 1: Check LASSO file
print("Test 1: Checking LASSO file...")
lasso_file = Path("./data/GSE297945/lasso_files/slide1_coordinates.csv/slide1_coordinates.csv")
import pandas as pd
df = pd.read_csv(lasso_file, sep=",", comment="#")
print(f"  Columns: {list(df.columns)}")
print(f"  Shape: {df.shape}")
cols_lower = {c.lower(): c for c in df.columns}
print(f"  Has 'selection': {'selection' in cols_lower}")
print(f"  Has 'x': {'x' in cols_lower}")
print(f"  Has 'y': {'y' in cols_lower}")

# Test 2: Check cell_feature_matrix files exist
print("\nTest 2: Checking cell_feature_matrix files...")
cfm_dir = Path("./data/GSM7990532/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400/outs/cell_feature_matrix")
for f in cfm_dir.iterdir():
    print(f"  {f.name}")

# Test 3: Test the filtering function
print("\nTest 3: Testing cell_feature_matrix filtering...")
from src.xenium_splitter.io_utils import (
    read_barcodes_file,
    read_mtx_file,
    filter_cell_feature_matrix,
)

barcodes_file = cfm_dir / "barcodes.tsv.gz"
barcodes = read_barcodes_file(barcodes_file)
print(f"  Total barcodes: {len(barcodes)}")
print(f"  First 5 barcodes: {barcodes[:5]}")

# Test reading matrix
matrix_file = cfm_dir / "matrix.mtx.gz"
data_rows, dims = read_mtx_file(matrix_file)
print(f"  Matrix dimensions: {dims[0]} features × {dims[1]} cells, {len(data_rows)} non-zero values")

# Test filtering with a subset
test_cell_ids = set(barcodes[:100])  # First 100 barcodes
output_dir = Path("./data_out/cfm_test")
try:
    orig_count, filt_count = filter_cell_feature_matrix(cfm_dir, output_dir, test_cell_ids)
    print(f"  Filtered: {orig_count} → {filt_count} cells")
    
    # Verify output files
    for fname in ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]:
        out_file = output_dir / fname
        if out_file.exists():
            print(f"    ✓ {fname}")
        else:
            print(f"    ✗ {fname} (MISSING)")
            
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")
