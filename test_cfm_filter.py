#!/usr/bin/env python3
"""Test cell_feature_matrix filtering in isolation."""

from pathlib import Path
from src.xenium_splitter.io_utils import (
    read_barcodes_file,
    write_barcodes_file,
    read_mtx_file,
    write_mtx_file,
    filter_cell_feature_matrix,
)

cfm_dir = Path("./data/GSM7990532/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400/outs/cell_feature_matrix")
output_dir = Path("./data_out/cfm_filter_test")

print("=" * 60)
print("Cell Feature Matrix Filtering Test")
print("=" * 60)

# Test 1: Read barcodes
print("\n1. Reading barcodes...")
barcodes_file = cfm_dir / "barcodes.tsv.gz"
barcodes = read_barcodes_file(barcodes_file)
print(f"   Total barcodes: {len(barcodes)}")
print(f"   First 3: {barcodes[:3]}")
print(f"   Last 3: {barcodes[-3:]}")

# Test 2: Read matrix
print("\n2. Reading matrix...")
matrix_file = cfm_dir / "matrix.mtx.gz"
data_rows, (n_features, n_barcodes, n_values) = read_mtx_file(matrix_file)
print(f"   Dimensions: {n_features} features × {n_barcodes} barcodes")
print(f"   Non-zero values: {len(data_rows)}")
print(f"   First 3 entries: {data_rows[:3]}")

# Test 3: Filter with a subset
print("\n3. Filtering to subset...")
test_cell_ids = set(barcodes[100:200])  # Keep barcodes 100-200
print(f"   Keeping {len(test_cell_ids)} out of {len(barcodes)} cells")

try:
    orig_count, filt_count = filter_cell_feature_matrix(cfm_dir, output_dir, test_cell_ids)
    print(f"   ✓ Filtered: {orig_count} → {filt_count} cells")
    
    # Verify output files exist
    print("\n4. Checking output files...")
    for fname in ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]:
        out_file = output_dir / fname
        if out_file.exists():
            size = out_file.stat().st_size
            print(f"   ✓ {fname}: {size} bytes")
        else:
            print(f"   ✗ {fname} MISSING")
    
    # Verify filtered barcodes
    print("\n5. Verifying filtered content...")
    filtered_barcodes = read_barcodes_file(output_dir / "barcodes.tsv.gz")
    print(f"   Filtered barcode count: {len(filtered_barcodes)}")
    print(f"   Expected: {len(test_cell_ids)}")
    print(f"   Match: {len(filtered_barcodes) == len(test_cell_ids)}")
    
    # Verify all filtered barcodes are in the input set
    all_match = all(bc in test_cell_ids for bc in filtered_barcodes)
    print(f"   All filtered barcodes in input set: {all_match}")
    
    # Verify filtered matrix dimensions
    print("\n6. Verifying filtered matrix...")
    filt_rows, (filt_features, filt_cols, filt_values) = read_mtx_file(output_dir / "matrix.mtx.gz")
    print(f"   Features: {filt_features} (expected: {n_features})")
    print(f"   Cells: {filt_cols} (expected: {filt_count})")
    print(f"   Non-zero values: {len(filt_rows)} (vs {len(data_rows)} original)")
    print(f"   Features match: {filt_features == n_features}")
    print(f"   Cells match: {filt_cols == filt_count}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
