from __future__ import annotations

import gzip
import json
import logging
import math
from pathlib import Path
import shutil
import tempfile
import time
import warnings
import zipfile

import pandas as pd
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads

from xenium_splitter.models import LassoRegion

logger = logging.getLogger(__name__)

TABULAR_EXTS = {".csv", ".tsv", ".txt", ".parquet"}
COMPRESSED_TABULAR_EXTS = {".csv.gz", ".tsv.gz"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs"}
ARCHIVE_EXTS = {".zarr.zip", ".h5", ".hdf5"}


def _suppress_zipstore_duplicate_name_warning() -> None:
    """Suppress harmless zipfile duplicate-name warnings from ZipStore metadata rewrites."""
    warnings.filterwarnings(
        "ignore",
        message=r"Duplicate name: '.*zarr\\.json'",
        category=UserWarning,
        module=r"zipfile",
    )


def _xenium_temp_root() -> Path:
    """Return/create the dedicated temp root for xenium-splitter scratch files."""
    root = Path(tempfile.gettempdir()) / "xenium_splitter"
    root.mkdir(parents=True, exist_ok=True)
    return root


def iter_input_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in input_dir.rglob("*"):
        if path.is_file():
            files.append(path)
    return files


def classify_file(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith(".ome.tif") or lower_name.endswith(".ome.tiff"):
        return "image"
    if path.suffix.lower() in IMAGE_EXTS:
        return "image"
    
    # Check for compressed and archive formats
    if lower_name.endswith(".csv.gz") or lower_name.endswith(".tsv.gz"):
        return "tabular"
    if lower_name.endswith(".zarr.zip"):
        return "zarr"
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return "hdf5"
    if path.suffix.lower() in TABULAR_EXTS:
        return "tabular"
    return "unknown"


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    lower_name = path.name.lower()
    
    if lower_name.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if lower_name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", compression="gzip")
    if suffix == ".csv":
        return pd.read_csv(path, comment="#")
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", comment="#")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table extension: {path.name}")


def write_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    lower_name = output_path.name.lower()
    
    if lower_name.endswith(".csv.gz"):
        df.to_csv(output_path, index=False, compression="gzip")
        return
    if lower_name.endswith(".tsv.gz"):
        df.to_csv(output_path, index=False, sep="\t", compression="gzip")
        return
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    if suffix in {".tsv", ".txt"}:
        df.to_csv(output_path, index=False, sep="\t")
        return
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        return
    raise ValueError(f"Unsupported output table extension: {output_path.name}")


def detect_xy_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    columns_by_lower = {str(c).lower(): c for c in df.columns}

    candidate_pairs = [
        ("x", "y"),
        ("x_location", "y_location"),
        ("x_centroid", "y_centroid"),
        ("global_x", "global_y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ]
    for x_candidate, y_candidate in candidate_pairs:
        if x_candidate in columns_by_lower and y_candidate in columns_by_lower:
            return columns_by_lower[x_candidate], columns_by_lower[y_candidate]
    return None


def subset_table_for_region(
    df: pd.DataFrame,
    region: LassoRegion,
    x_col: str,
    y_col: str,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """Filter table rows to those within region polygon, and rebase coordinates to crop origin.
    
    Uses Shapely's contains_xy() for efficient polygon containment testing.
    Coordinates are rebased so that the crop bounding box top-left corner becomes (0, 0),
    aligning with the image crop origin computation.
    
    Args:
        df: Input DataFrame with coordinate columns
        region: LASSO region with polygon definition
        x_col: Name of x-coordinate column
        y_col: Name of y-coordinate column
        pixel_size_um: Pixel size in micrometers (optional); when provided, crop origin is
                       computed using floor(bounds/pixel_size) to align with image cropping
    
    Returns:
        Subset of df with rows in region, coordinates rebased to crop origin
    """
    x_vals = pd.to_numeric(df[x_col], errors="coerce")
    y_vals = pd.to_numeric(df[y_col], errors="coerce")
    mask = contains_xy(region.polygon, x_vals.to_numpy(), y_vals.to_numpy())

    subset = df.loc[mask].copy()
    if subset.empty:
        return subset

    return rebase_table_coordinates_to_region_crop(
        subset,
        region,
        x_col,
        y_col,
        pixel_size_um=pixel_size_um,
    )


def subset_table_for_region_optimized(
    df: pd.DataFrame,
    region: LassoRegion,
    x_col: str,
    y_col: str,
    region_entity_ids: dict[str, set[str]] | None = None,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """Fast filtered subset for transcripts using cell assignment + bounding box.

    Optimization strategy for transcript tables:
    1. If transcript has cell_id/barcode assignment -> include if cell is in region
    2. If unassigned -> bounding box pre-filter, then polygon containment

    This is much faster than full polygon containment for all rows when many
    transcripts have cell assignments.

    Args:
        df: Input DataFrame with coordinate columns and optional cell_id column
        region: LASSO region with polygon definition
        x_col: Name of x-coordinate column
        y_col: Name of y-coordinate column
        region_entity_ids: Set of cell IDs in this region (optional; when provided,
                          assigned transcripts are fast-filtered by cell membership)
        pixel_size_um: Pixel size in micrometers (optional); when provided, crop origin is
                       computed using floor(bounds/pixel_size) to align with image cropping

    Returns:
        Subset of df with rows in region, coordinates rebased to crop origin
    """
    subsets = subset_table_for_regions_optimized(
        df,
        [region],
        x_col,
        y_col,
        region_cell_ids_by_region={region.region_id: region_entity_ids or set()},
        pixel_size_um=pixel_size_um,
    )
    return subsets[region.region_id]


def _get_transcript_cell_id_column(
    df: pd.DataFrame,
    region_cell_ids_by_region: dict[str, set[str]] | None,
) -> str | None:
    if region_cell_ids_by_region is None:
        return None
    for candidate in ["cell_id", "barcode", "cell", "transcript_cell_id"]:
        if candidate in df.columns:
            return candidate
    return None


def _build_transcript_region_mask(
    region: LassoRegion,
    x_vals: pd.Series,
    y_vals: pd.Series,
    base_false_mask: pd.Series,
    cell_ids: pd.Series | None,
    assigned_known: pd.Series | None,
    region_cells: set[str] | None,
) -> pd.Series:
    if cell_ids is not None and assigned_known is not None:
        # Trust assignment when present:
        # - assigned + in-region cell -> keep
        # - assigned + out-of-region cell -> drop
        # - only unassigned transcripts take spatial path
        final_mask = assigned_known & cell_ids.isin(region_cells or set())
        spatial_candidates = ~assigned_known
    else:
        spatial_candidates = ~base_false_mask
        final_mask = base_false_mask.copy()

    min_x, min_y, max_x, max_y = region.bounds
    bbox_mask = spatial_candidates & (
        (x_vals >= min_x)
        & (x_vals <= max_x)
        & (y_vals >= min_y)
        & (y_vals <= max_y)
    )
    if not bbox_mask.any():
        return final_mask

    polygon_hits = base_false_mask.copy()
    polygon_hits.loc[bbox_mask] = contains_xy(
        region.polygon,
        x_vals.loc[bbox_mask].to_numpy(),
        y_vals.loc[bbox_mask].to_numpy(),
    )
    return final_mask | polygon_hits


def subset_table_for_regions_optimized(
    df: pd.DataFrame,
    regions: list[LassoRegion],
    x_col: str,
    y_col: str,
    region_cell_ids_by_region: dict[str, set[str]] | None = None,
    pixel_size_um: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Build region subsets for transcripts while reusing precomputed columns.

    This keeps the same semantics as per-region filtering, but avoids repeated
    numeric conversion and cell-id normalization for every region.

    Assignment policy:
    - If transcript has a valid cell assignment, trust that assignment.
    - Only transcripts without assignment are evaluated spatially.
    """
    if df.empty:
        return {region.region_id: df.copy() for region in regions}

    cell_id_col = _get_transcript_cell_id_column(df, region_cell_ids_by_region)
    x_vals = pd.to_numeric(df[x_col], errors="coerce")
    y_vals = pd.to_numeric(df[y_col], errors="coerce")
    base_empty = df.iloc[0:0].copy()
    cell_ids = None
    assigned_known = None
    if cell_id_col:
        raw_cell_ids = df[cell_id_col]
        cell_ids = raw_cell_ids.astype("string").str.strip()
        assigned_known = ~(
            raw_cell_ids.isna()
            | cell_ids.isin(["", "-1", "<NA>", "nan", "None", "none", "NULL", "null"])
        )
    base_false_mask = pd.Series(False, index=df.index)
    subsets: dict[str, pd.DataFrame] = {}

    for region in regions:
        final_mask = _build_transcript_region_mask(
            region,
            x_vals,
            y_vals,
            base_false_mask,
            cell_ids,
            assigned_known,
            region_cell_ids_by_region.get(region.region_id, set()) if region_cell_ids_by_region else None,
        )
        subset = df.loc[final_mask].copy()
        if subset.empty:
            subsets[region.region_id] = base_empty.copy()
            continue

        subsets[region.region_id] = rebase_table_coordinates_to_region_crop(
            subset,
            region,
            x_col,
            y_col,
            pixel_size_um=pixel_size_um,
        )

    return subsets


def _region_crop_origin_um(region: LassoRegion, pixel_size_um: float | None = None) -> tuple[float, float]:
    """Compute crop origin in the same coordinate frame used by image crop bbox.

    CRITICAL: This function ensures coordinate rebasing matches image crop behavior.
    
    When pixel_size_um is provided (from experiment.xenium), the origin computation uses:
    1. Convert polygon bounds from coordinate space to pixel space (divide by pixel_size_um)
    2. Floor the result to get integer pixel coordinates (matches image crop _bbox_int)
    3. Clamp to >= 0 to handle negative bounds
    4. Convert back to coordinate space (multiply by pixel_size_um)
    
    When no pixel_size is provided, returns raw polygon min bounds (legacy fallback).
    
    This ensures:
    - Entities at the crop top-left corner have coordinates close to (0, 0)
    - Entity coordinates align with cropped image pixel layout
    - Consistency between image masking and table rebasing
    
    Args:
        region: LASSO region with polygon bounds
        pixel_size_um: Pixel size in micrometers (optional)
    
    Returns:
        (origin_x, origin_y) tuple in coordinate units
    """
    min_x, min_y, _, _ = region.bounds
    if pixel_size_um is not None and pixel_size_um > 0:
        # Convert to pixel space, floor to get integer pixel coordinates
        min_x_px = max(int(math.floor(min_x / pixel_size_um)), 0)
        min_y_px = max(int(math.floor(min_y / pixel_size_um)), 0)
        # Convert back to coordinate space
        return min_x_px * pixel_size_um, min_y_px * pixel_size_um
    # Fallback: use raw polygon bounds
    return min_x, min_y


def rebase_table_coordinates_to_region_crop(
    df: pd.DataFrame,
    region: LassoRegion,
    x_col: str,
    y_col: str,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """Shift x/y columns so output coordinates align to the cropped image origin.
    
    This transformation ensures that entity data (cells, transcripts, nucleus)
    written to region output has coordinates that match the cropped/masked image.
    The result is that an entity at image pixel (0, 0) will have table coordinates ≈ (0, 0).
    
    Used by:
    - Coordinate-filtered tabular/HDF5/Zarr outputs (fallback when no boundary data)
    - ID-filtered tabular/HDF5/Zarr outputs (when x/y columns are present for QC)
    
    Args:
        df: DataFrame with rows in the region (pre-filtered by containment or ID)
        region: LASSO region defining the crop bounding box
        x_col: Name of x-coordinate column
        y_col: Name of y-coordinate column
        pixel_size_um: Pixel size for image-aligned origin computation
    
    Returns:
        DataFrame with coordinates shifted so crop origin is (0, 0)
    """
    if df.empty:
        return df

    # Compute crop origin using image-aligned logic
    origin_x, origin_y = _region_crop_origin_um(region, pixel_size_um=pixel_size_um)
    
    # Subtract origin from all coordinates
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce") - origin_x
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce") - origin_y
    return df


def load_pixel_size_from_experiment(input_dir: Path) -> float | None:
    """Extract pixel_size_um from experiment.xenium JSON if it exists in input directory.
    
    Returns the pixel size in micrometers, or None if file not found or field missing.
    """
    exp_path = input_dir / "experiment.xenium"
    if not exp_path.exists():
        return None
    
    try:
        payload = json.loads(exp_path.read_text(encoding="utf-8"))
        return payload.get("pixel_size")
    except (json.JSONDecodeError, OSError):
        return None


def list_hdf5_datasets(h5_path: Path) -> list[str]:
    """List all datasets in an HDF5 file."""
    try:
        import h5py
    except ImportError:
        logger.warning("h5py not installed; HDF5 support disabled.")
        return []
    
    datasets = []
    try:
        with h5py.File(h5_path, "r") as f:
            def visit_func(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append(name)
            f.visititems(visit_func)
    except Exception as e:
        logger.error(f"Failed to list HDF5 datasets in {h5_path.name}: {e}")
    return datasets


def read_hdf5_table(h5_path: Path, dataset_path: str = None) -> pd.DataFrame | None:
    """Read a table-like dataset from an HDF5 file.
    
    If dataset_path is None, try common table-like dataset names (cells, transcripts, barcodes).
    """
    try:
        import h5py
    except ImportError:
        logger.warning("h5py not installed; HDF5 support disabled.")
        return None
    
    candidates = dataset_path or ["cells", "transcripts", "barcodes", "data"]
    if isinstance(candidates, str):
        candidates = [candidates]
    
    try:
        with h5py.File(h5_path, "r") as f:
            for cand in candidates:
                if cand in f:
                    ds = f[cand]
                    if ds.ndim == 1:
                        return pd.DataFrame({cand: ds[()]})
                    elif ds.ndim == 2:
                        cols = ds.attrs.get("column_names", None)
                        data = ds[()]
                        if cols is not None:
                            return pd.DataFrame(data, columns=cols)
                        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Failed to read HDF5 table from {h5_path.name}: {e}")
    
    return None


def read_zarr_zip_table(zarr_zip_path: Path) -> pd.DataFrame | None:
    """Extract and read a Zarr ZIP archive, looking for table-like arrays."""
    try:
        import zarr
    except ImportError:
        logger.warning("zarr not installed; Zarr ZIP support disabled.")
        return None
    
    with tempfile.TemporaryDirectory(dir=_xenium_temp_root(), prefix="zarr_read_") as tmpdir:
        try:
            with zipfile.ZipFile(zarr_zip_path, "r") as zf:
                zf.extractall(tmpdir)
            
            root = zarr.open(tmpdir, mode="r")
            
            for key in root.keys():
                arr = root[key]
                if hasattr(arr, "shape") and len(arr.shape) in (1, 2):
                    try:
                        data = arr[:]
                    except Exception:
                        continue
                    try:
                        if len(data.shape) == 1:
                            return pd.DataFrame({key: data})
                        else:
                            return pd.DataFrame(data)
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Failed to read Zarr ZIP from {zarr_zip_path.name}: {e}")
    
    return None


def write_hdf5_table(df: pd.DataFrame, output_path: Path, dataset_name: str = "data") -> None:
    """Write a DataFrame to an HDF5 file."""
    try:
        import h5py
    except ImportError:
        logger.error("h5py not installed; cannot write HDF5 files.")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(output_path, "w") as f:
            ds = f.create_dataset(dataset_name, data=df.values)
            ds.attrs["column_names"] = list(df.columns)
    except Exception as e:
        logger.error(f"Failed to write HDF5 to {output_path.name}: {e}")


def write_zarr_zip_table(df: pd.DataFrame, output_path: Path, dataset_name: str = "data") -> None:
    """Write a DataFrame to a Zarr ZIP archive.

    Uses a compatibility ZipStore helper to write zarr chunks directly into the zip file, avoiding
    a temp directory and a second compression pass.  ZIP_STORED is used at the
    outer zip layer.  No zarr-level compression is applied either — compression at
    both layers wastes CPU with no size benefit when ZIP_STORED is used.

    Only numeric and bool columns are written, to avoid zarr v3 ambiguous-dtype
    errors with pandas StringDtype / object columns.  Non-numeric columns are
    preserved in the CSV/parquet counterparts that are always written alongside.
    """
    try:
        import zarr
    except ImportError:
        logger.error("zarr not installed; cannot write Zarr ZIP files.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Determine version-appropriate no-compression kwargs.
        # zarr v3 uses compressors=[], v2 uses compressor=None.
        zarr_major = int(str(getattr(zarr, "__version__", "2")).split(".")[0])
        _no_compress: dict = {"compressors": []} if zarr_major >= 3 else {"compressor": None}

        # ZipStore writes zarr chunks directly into the zip - no temp dir, no re-archiving.
        with warnings.catch_warnings():
            _suppress_zipstore_duplicate_name_warning()
            store = _open_zarr_zip_store(output_path, mode="w")
            try:
                root = zarr.open_group(store=store, mode="w")
                data_group = root.require_group(dataset_name)
                numeric_cols: list[str] = []
                for col_name in df.columns:
                    series = df[col_name]
                    if not (
                        pd.api.types.is_numeric_dtype(series)
                        or pd.api.types.is_bool_dtype(series)
                    ):
                        continue
                    # Preserve native dtype (float32 stays float32, int64 stays int64).
                    # Converting everything to float64 doubles data size for float32 columns.
                    native_dtype = series.dtype
                    if pd.api.types.is_float_dtype(native_dtype):
                        arr = series.to_numpy(dtype=native_dtype, na_value=0.0)
                    elif pd.api.types.is_integer_dtype(native_dtype) or pd.api.types.is_bool_dtype(native_dtype):
                        arr = series.to_numpy(dtype=native_dtype, na_value=0)
                    else:
                        arr = series.to_numpy(dtype=float, na_value=0.0)
                    # One chunk per column → one zip entry; minimises zip central-directory
                    # overhead for files with millions of rows and dozens of columns.
                    chunk_size = min(len(arr), 1_000_000) if len(arr) > 0 else 1
                    data_group.create_dataset(
                        str(col_name),
                        shape=arr.shape,
                        dtype=arr.dtype,
                        data=arr,
                        chunks=(chunk_size,),
                        **_no_compress,
                    )
                    numeric_cols.append(str(col_name))
                root[dataset_name].attrs["column_names"] = numeric_cols
            finally:
                store.close()
    except Exception as e:
        logger.error(f"Failed to write Zarr ZIP to {output_path.name}: {e}")


def find_boundary_files(input_dir: Path) -> dict[str, Path]:
    """Find boundary files for cells, nucleus, transcripts.
    
    Returns dict mapping entity type to boundary file path.
    E.g., {"cells": Path("cell_boundaries.csv.gz"), "nucleus": Path("nucleus_boundaries.csv.gz")}
    """
    boundaries: dict[str, Path] = {}
    
    entity_patterns = {
        "cells": ["cell_boundaries", "cells_boundary"],
        "nucleus": ["nucleus_boundaries", "nuclei_boundaries", "nucleus_boundary"],
        "transcripts": ["transcript_boundaries", "transcripts_boundary"],
    }
    
    for entity_type, patterns in entity_patterns.items():
        for file_path in input_dir.rglob("*"):
            if not file_path.is_file():
                continue
            lower_name = file_path.name.lower()
            if any(p.lower() in lower_name for p in patterns):
                if lower_name.endswith(".parquet"):
                    boundaries[entity_type] = file_path
                    break
    
    return boundaries


def extract_entity_ids_in_region(
    boundary_file: Path,
    region: LassoRegion,
    id_col: str = "cell_id",
    polygon_col: str = "polygon_wkt",
) -> set[str]:
    """Extract entity IDs from boundary file that intersect the region polygon.
    
    Args:
        boundary_file: CSV/Parquet with entity boundaries
        region: LASSO region polygon
        id_col: Column name for entity IDs
        polygon_col: Column name for WKT polygon coordinates when present
    
    Returns:
        Set of entity IDs within the region
    """
    try:
        table = read_table(boundary_file)
    except Exception as e:
        logger.warning(f"Could not read boundary file {boundary_file.name}: {e}")
        return set()

    if id_col not in table.columns:
        logger.warning(f"Boundary file {boundary_file.name} missing {id_col} column")
        return set()

    if {"vertex_x", "vertex_y"}.issubset(table.columns):
        return _extract_entity_ids_from_vertices(table, region, id_col)

    if polygon_col not in table.columns:
        logger.warning(
            f"Boundary file {boundary_file.name} missing either vertex_x/vertex_y or {polygon_col} columns"
        )
        return set()

    return _extract_entity_ids_from_wkt(table, region, id_col, polygon_col)


def _extract_entity_ids_from_vertices(
    table: pd.DataFrame,
    region: LassoRegion,
    id_col: str,
) -> set[str]:
    ids_in_region: set[str] = set()
    for entity_id, group in table.groupby(id_col, sort=False):
        try:
            entity_polygon = _polygon_from_vertex_rows(group)
            if entity_polygon is None:
                continue

            if entity_polygon.intersects(region.polygon):
                ids_in_region.add(str(entity_id))
        except Exception as e:
            logger.debug(f"Error processing entity {entity_id}: {e}")
    return ids_in_region


def _polygon_from_vertex_rows(group: pd.DataFrame) -> Polygon | None:
    vertices = list(
        zip(
            pd.to_numeric(group["vertex_x"], errors="coerce"),
            pd.to_numeric(group["vertex_y"], errors="coerce"),
        )
    )
    vertices = [(x, y) for x, y in vertices if pd.notna(x) and pd.notna(y)]
    if len(vertices) < 3:
        return None

    entity_polygon = Polygon(vertices)
    if not entity_polygon.is_valid:
        entity_polygon = entity_polygon.buffer(0)
    if entity_polygon.is_empty:
        return None
    return entity_polygon


def _extract_entity_ids_from_wkt(
    table: pd.DataFrame,
    region: LassoRegion,
    id_col: str,
    polygon_col: str,
) -> set[str]:
    ids_in_region: set[str] = set()
    for _, row in table.iterrows():
        try:
            entity_id = str(row[id_col])
            wkt_str = str(row[polygon_col])
            entity_polygon = wkt_loads(wkt_str)

            if entity_polygon.intersects(region.polygon):
                ids_in_region.add(entity_id)
        except Exception as e:
            logger.debug(f"Error processing entity {row.get(id_col, '?')}: {e}")
    return ids_in_region


def filter_table_by_entity_ids(
    table: pd.DataFrame,
    entity_ids: set[str],
    id_col: str = "cell_id",
) -> pd.DataFrame:
    """Filter table to only rows matching entity IDs.
    
    Args:
        table: DataFrame to filter
        entity_ids: Set of entity IDs to keep
        id_col: Column name for entity IDs
    
    Returns:
        Filtered DataFrame
    """
    if id_col not in table.columns:
        logger.warning(f"Column {id_col} not in table, cannot filter by entity IDs")
        return table
    
    mask = table[id_col].astype(str).isin(entity_ids)
    return table[mask].copy()


def get_entity_id_column(table: pd.DataFrame, entity_type: str = "cell") -> str | None:
    """Detect entity ID column name for a given entity type.
    
    Tries common naming patterns: cell_id, cellID, cell ID, etc.
    """
    entity_aliases = {
        "cells": "cell",
        "cell": "cell",
        "nuclei": "nucleus",
        "nucleus": "nucleus",
        "transcripts": "transcript",
        "transcript": "transcript",
    }
    normalized_entity_type = entity_aliases.get(entity_type.lower(), entity_type.lower())

    candidates = {
        "cell": ["cell_id", "cellid", "cell_iid", "barcode"],
        "nucleus": ["cell_id", "nucleus_id", "nucleusid", "nuc_id", "barcode"],
        "transcript": ["transcript_id", "transcriptid", "tx_id"],
    }
    
    cols_lower = {str(c).lower(): c for c in table.columns}
    for cand in candidates.get(normalized_entity_type, []):
        if cand in cols_lower:
            return cols_lower[cand]
    
    return None


def is_cell_feature_matrix_group(path: Path) -> bool:
    """Check if a path is the cell_feature_matrix directory."""
    return path.is_dir() and path.name.lower() == "cell_feature_matrix"


def get_cell_feature_matrix_files(group_dir: Path) -> dict[str, Path | None]:
    """Get all cell_feature_matrix component files from a group directory.
    
    Returns:
        Dict with keys: barcodes, features, matrix, zarr
        Values are Path objects if file exists, None otherwise
    """
    files = {
        "barcodes": None,
        "features": None,
        "matrix": None,
        "zarr": None,
    }
    
    for f in group_dir.iterdir():
        fname = f.name.lower()
        if "barcodes" in fname:
            files["barcodes"] = f
        elif "features" in fname:
            files["features"] = f
        elif "matrix" in fname:
            files["matrix"] = f
        elif "cell_feature_matrix.zarr.zip" == fname:
            files["zarr"] = f
    
    return files


def read_mtx_file(path: Path) -> tuple[list[list[int]], tuple[int, int, int]]:
    """Read Matrix Market format file (MTX).
    
    MTX is the 10X Genomics sparse matrix format:
    - Header: %%MatrixMarket matrix coordinate integer general
    - Metadata: %metadata_json: {...}
    - Dimensions: num_features num_barcodes num_values
    - Data: row col value (1-indexed, where row=feature, col=barcode)
    
    Args:
        path: Path to .mtx or .mtx.gz file
    
    Returns:
        (data_rows, dimensions) where:
        - data_rows: list of [row, col, value] (1-indexed)
        - dimensions: (num_features, num_barcodes, num_values)
    """
    open_fn = gzip.open if str(path).endswith(".gz") else open
    
    with open_fn(path, "rt") as f:
        f.readline()  # Skip header: %%MatrixMarket matrix coordinate integer general
        f.readline()  # Skip metadata line
        
        dims_line = f.readline().strip()
        num_features, num_barcodes, num_values = map(int, dims_line.split())
        
        data_rows = []
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                row, col, val = int(parts[0]), int(parts[1]), int(parts[2])
                data_rows.append([row, col, val])
    
    return data_rows, (num_features, num_barcodes, num_values)


def write_mtx_file(
    path: Path,
    data_rows: list[list[int]],
    num_features: int,
    num_barcodes: int,
) -> None:
    """Write Matrix Market format file (MTX) in 10X Genomics format.
    
    Writes sparse matrix with 1-based indexing per MTX specification.
    
    Args:
        path: Output path (.mtx or .mtx.gz)
        data_rows: List of [row, col, value] entries (1-indexed)
        num_features: Number of genes/features in matrix
        num_barcodes: Number of cells/barcodes (columns) in matrix
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    num_values = len(data_rows)
    
    open_fn = gzip.open if str(path).endswith(".gz") else open
    
    with open_fn(path, "wt") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write('%metadata_json: {"software_version": "xenium unknown", "format_version": 2}\n')
        f.write(f"{num_features} {num_barcodes} {num_values}\n")
        
        for row, col, val in data_rows:
            f.write(f"{row} {col} {val}\n")


def read_barcodes_file(path: Path) -> list[str]:
    """Read barcode IDs from file (TSV, CSV, or gzipped).
    
    Expected format: one barcode per line.
    
    Args:
        path: Path to barcodes file (.tsv, .txt, .tsv.gz, etc.)
    
    Returns:
        List of barcode strings in original order
    """
    open_fn = gzip.open if str(path).endswith(".gz") else open
    
    barcodes = []
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                barcodes.append(line)
    
    return barcodes


def write_barcodes_file(path: Path, barcodes: list[str]) -> None:
    """Write barcode IDs to file (TSV format, optionally gzipped).
    
    Args:
        path: Output path (.tsv, .txt, .tsv.gz, etc.)
        barcodes: List of barcode strings to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    open_fn = gzip.open if str(path).endswith(".gz") else open
    
    with open_fn(path, "wt") as f:
        for bc in barcodes:
            f.write(f"{bc}\n")


def _filter_zarr_dataset(
    src_arr,
    key: str,
    cell_id_to_new_idx: dict[int, int],
) -> tuple[object, bool]:
    """Filter a single Zarr array to a subset of cells.
    
    Handles both 1D arrays (barcodes) and 2D arrays (expression matrix):
    - 1D: Subset by filtered indices (used for barcode lists)
    - 2D: Subset columns (barcode dimension), keep all rows (features)
    
    Args:
        src_arr: Source Zarr array
        key: Dataset name (used to detect array type)
        cell_id_to_new_idx: Mapping from old cell index (1-based) to new (1-based)
    
    Returns:
        (filtered_data, should_write): Data to write and whether it was modified
    """
    if not hasattr(src_arr, "shape"):
        return src_arr, False
    
    # Convert old indices (1-based) to array indices (0-based)
    filtered_indices = sorted({idx - 1 for idx in cell_id_to_new_idx.keys()})
    
    if len(src_arr.shape) == 1:
        # 1D array (e.g., barcodes)
        if "barcode" in key.lower():
            return src_arr[filtered_indices], True
        return src_arr[:], False
    
    if len(src_arr.shape) == 2:
        # 2D array (e.g., expression matrix: genes × cells)
        # Keep all rows (genes), filter columns (barcodes)
        rows, cols = src_arr.shape
        data = src_arr[:, filtered_indices]
        logger.debug(f"Filtered {key}: {rows}×{cols} → {rows}×{len(filtered_indices)}")
        return data, True
    
    # Higher-dimensional arrays - return as-is
    logger.warning(f"Copying {key} unchanged (unsupported dimension: {len(src_arr.shape)})")
    return src_arr[:], False


def _copy_zarr_attributes(src_arr, dst_arr, key: str) -> None:
    """Copy attributes from source array to destination."""
    if not hasattr(src_arr, "attrs"):
        return
    
    for attr_key, attr_val in src_arr.attrs.items():
        try:
            dst_arr.attrs[attr_key] = attr_val
        except TypeError:
            logger.debug(f"Could not copy attribute {attr_key} for {key}")


def _copy_root_attributes(src_root, dst_root) -> None:
    """Copy root-level attributes from source to destination."""
    for attr_key, attr_val in src_root.attrs.items():
        try:
            dst_root.attrs[attr_key] = attr_val
        except TypeError:
            logger.debug(f"Could not copy root attribute {attr_key}")


def _rezip_zarr(src_path: Path, dst_path: Path) -> None:
    """Re-zip a Zarr directory into an archive using ZIP_STORED.

    Zarr chunks are already Blosc-compressed; using ZIP_STORED avoids a
    second compression pass and is much faster.
    """
    with zipfile.ZipFile(dst_path, "w", zipfile.ZIP_STORED) as zf:
        for fpath in src_path.rglob("*"):
            if fpath.is_file():
                zf.write(fpath, arcname=fpath.relative_to(src_path))


def _open_zarr_zip_store(path: Path, mode: str = "w"):
    """Open a zarr ZipStore, handling API differences across zarr v2 and v3.

    - zarr v2: zarr.ZipStore(path, mode=mode, compression=ZIP_STORED)
    - zarr v3: zarr.storage.ZipStore(path, mode=mode)  (compression kwarg removed)
    """
    import zarr as _zarr

    # zarr v3 moved ZipStore to zarr.storage
    ZipStore = getattr(_zarr, "ZipStore", None) or getattr(
        getattr(_zarr, "storage", None), "ZipStore", None
    )
    if ZipStore is None:
        raise ImportError("Could not locate zarr.ZipStore or zarr.storage.ZipStore")

    try:
        return ZipStore(str(path), mode=mode, compression=zipfile.ZIP_STORED)
    except TypeError:
        # zarr v3 dropped the compression kwarg
        return ZipStore(str(path), mode=mode)


def filter_cell_feature_matrix_zarr(
    zarr_zip_path: Path,
    output_path: Path,
    cell_id_to_new_idx: dict[int, int],
    num_cells: int,
) -> bool:
    """Filter a cell_feature_matrix.zarr.zip file to a subset of cells.
    
    Workflow:
    1. Extract zarr.zip to temporary directory
    2. Open with zarr library
    3. Filter each dataset: subset 1D arrays (barcodes), filter columns from 2D (genes×cells)
    4. Copy root-level attributes
    5. Re-zip filtered zarr to output
    
    Args:
        zarr_zip_path: Path to the source zarr.zip file
        output_path: Path to write the filtered zarr.zip
        cell_id_to_new_idx: Mapping from old cell index (1-based) to new (1-based)
        num_cells: Total number of cells after filtering (for logging)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import zarr
    except ImportError:
        logger.warning("zarr not installed; cannot filter Zarr ZIP files.")
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory(dir=_xenium_temp_root(), prefix="cfm_zarr_in_") as tmpdir_in:
        with tempfile.TemporaryDirectory(dir=_xenium_temp_root(), prefix="cfm_zarr_out_") as tmpdir_out:
            try:
                # Extract and filter
                with zipfile.ZipFile(zarr_zip_path, "r") as zf:
                    zf.extractall(tmpdir_in)
                
                src_root = zarr.open(tmpdir_in, mode="r")
                dst_root = zarr.open(tmpdir_out, mode="w")
                
                # Filter datasets
                for key in src_root.keys():
                    src_arr = src_root[key]
                    filtered_data, _ = _filter_zarr_dataset(src_arr, key, cell_id_to_new_idx)
                    
                    if hasattr(src_arr, "shape"):
                        dst_arr = dst_root.create_dataset(key, data=filtered_data, chunks=True)
                        _copy_zarr_attributes(src_arr, dst_arr, key)
                    else:
                        dst_root.attrs[key] = filtered_data
                
                _copy_root_attributes(src_root, dst_root)
                _rezip_zarr(Path(tmpdir_out), output_path)
                logger.info(f"Filtered {zarr_zip_path.name} to {num_cells} cells")
                return True
                
            except Exception as e:
                logger.error(f"Failed to filter Zarr ZIP {zarr_zip_path.name}: {e}")
                return False


def filter_cell_feature_matrix(
    group_dir: Path,
    output_dir: Path,
    cell_ids: set[str],
) -> tuple[int, int]:
    """Filter cell_feature_matrix files to a subset of cell IDs.
    
    Processes all four 10X Genomics cell_feature_matrix components:
    - Barcodes: Filtered to matched cell IDs (subset + reindex)
    - Features: Copied unchanged (all features kept)
    - Matrix (MTX): Filtered to matched columns, reindexed (1-based)
    - Zarr: Filtered to matched cells (if present)
    
    Barcode Index Handling:
    MTX format uses 1-based column indices. This function:
    1. Builds mapping: old_barcode_idx (1-based) → new (1-based)
    2. Filters matrix rows: keep entries where col_idx in mapping
    3. Rewrites matrix columns with consecutive 1-based indices
    
    Args:
        group_dir: Directory containing barcodes/features/matrix/zarr files
        output_dir: Output directory for filtered files
        cell_ids: Set of cell barcodes to keep
    
    Returns:
        (original_cell_count, filtered_cell_count) for metrics reporting
    
    Raises:
        ValueError: If required files (barcodes, matrix) missing
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all component files
    files = get_cell_feature_matrix_files(group_dir)
    
    if not files["barcodes"] or not files["matrix"]:
        raise ValueError(f"cell_feature_matrix missing required files: {group_dir}")
    
    # Read original barcodes
    original_barcodes = read_barcodes_file(files["barcodes"])
    original_count = len(original_barcodes)
    
    # Create mapping from old barcode index to new (filtered)
    # MTX uses 1-based indexing, so old_idx + 1 is the MTX column index
    barcode_index_mapping = {}
    filtered_barcodes = []
    
    for old_idx, bc in enumerate(original_barcodes):
        if bc in cell_ids:
            new_idx = len(filtered_barcodes)
            barcode_index_mapping[old_idx + 1] = new_idx + 1  # Convert to 1-based
            filtered_barcodes.append(bc)
    
    filtered_count = len(filtered_barcodes)
    logger.info(f"Filtered cell_feature_matrix from {original_count} to {filtered_count} cells")
    
    # Write filtered barcodes
    write_barcodes_file(output_dir / files["barcodes"].name, filtered_barcodes)
    
    # Copy features file unchanged
    if files["features"]:
        shutil.copy2(files["features"], output_dir / files["features"].name)
    
    # Filter and rewrite matrix file
    if files["matrix"]:
        data_rows, (num_features, _, _) = read_mtx_file(files["matrix"])
        
        # Filter matrix rows: keep only entries where col_idx (barcode) is in mapping
        # MTX indices are 1-based, so col directly corresponds to old barcode index
        filtered_rows = [
            [row, barcode_index_mapping[col], val]
            for row, col, val in data_rows
            if col in barcode_index_mapping
        ]
        
        write_mtx_file(
            output_dir / files["matrix"].name,
            filtered_rows,
            num_features,
            filtered_count,
        )
    
    # For zarr file, filter to matched cells
    if files["zarr"]:
        filter_cell_feature_matrix_zarr(
            files["zarr"],
            output_dir / files["zarr"].name,
            barcode_index_mapping,
            filtered_count,
        )
    
    return original_count, filtered_count


def is_cell_feature_matrix_h5(path: Path) -> bool:
    """Return True when path looks like a 10X cell_feature_matrix.h5 file."""
    return path.is_file() and path.name.lower() == "cell_feature_matrix.h5"


def filter_cell_feature_matrix_h5(
    h5_path: Path,
    output_path: Path,
    cell_ids: set[str],
) -> tuple[int, int] | None:
    """Filter 10X-style cell_feature_matrix.h5 by barcode IDs.

    Expected schema:
    - matrix/barcodes
    - matrix/data
    - matrix/indices
    - matrix/indptr
    - matrix/shape
    - matrix/features/*

    The sparse matrix is stored in CSC format where columns correspond to barcodes.
    We filter selected barcode columns and rebuild data/indices/indptr.
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        logger.warning("h5py/numpy not installed; cannot filter cell_feature_matrix.h5")
        return None

    try:
        with h5py.File(h5_path, "r") as src:
            if "matrix" not in src:
                return None
            matrix = src["matrix"]
            required = {"barcodes", "data", "indices", "indptr", "shape", "features"}
            if not required.issubset(set(matrix.keys())):
                return None

            barcodes_raw = matrix["barcodes"][:]
            barcodes: list[str] = []
            for value in barcodes_raw:
                if isinstance(value, (bytes, bytearray)):
                    barcodes.append(value.decode("utf-8"))
                else:
                    barcodes.append(str(value))

            keep_cols = [i for i, barcode in enumerate(barcodes) if barcode in cell_ids]
            original_count = len(barcodes)
            filtered_count = len(keep_cols)

            data = matrix["data"][:]
            indices = matrix["indices"][:]
            indptr = matrix["indptr"][:]
            shape = matrix["shape"][:]

            new_data_chunks = []
            new_indices_chunks = []
            new_indptr = [0]

            for col_idx in keep_cols:
                start = int(indptr[col_idx])
                end = int(indptr[col_idx + 1])
                chunk_data = data[start:end]
                chunk_indices = indices[start:end]
                new_data_chunks.append(chunk_data)
                new_indices_chunks.append(chunk_indices)
                new_indptr.append(new_indptr[-1] + int(end - start))

            if new_data_chunks:
                new_data = np.concatenate(new_data_chunks).astype(data.dtype, copy=False)
                new_indices = np.concatenate(new_indices_chunks).astype(indices.dtype, copy=False)
            else:
                new_data = np.array([], dtype=data.dtype)
                new_indices = np.array([], dtype=indices.dtype)

            new_indptr_arr = np.asarray(new_indptr, dtype=indptr.dtype)
            new_shape = np.asarray([int(shape[0]), filtered_count], dtype=shape.dtype)
            kept_barcodes = barcodes_raw[keep_cols]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(output_path, "w") as dst:
                # Copy root attributes for compatibility
                for attr_key, attr_val in src.attrs.items():
                    dst.attrs[attr_key] = attr_val

                dst_matrix = dst.create_group("matrix")
                for attr_key, attr_val in matrix.attrs.items():
                    dst_matrix.attrs[attr_key] = attr_val

                dst_matrix.create_dataset("barcodes", data=kept_barcodes)
                dst_matrix.create_dataset("data", data=new_data)
                dst_matrix.create_dataset("indices", data=new_indices)
                dst_matrix.create_dataset("indptr", data=new_indptr_arr)
                dst_matrix.create_dataset("shape", data=new_shape)

                # Copy features subtree unchanged (rows/features are not filtered)
                src.copy(matrix["features"], dst_matrix, name="features")

        return original_count, filtered_count
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to filter %s: %s", h5_path.name, exc)
        return None


def find_matching_tabular_for_zarr(zarr_zip_path: Path) -> Path | None:
    """Find a sibling tabular file that corresponds to a *.zarr.zip file.

    Example: cells.zarr.zip -> cells.parquet (preferred) or cells.csv.gz/csv/tsv.
    """
    name = zarr_zip_path.name
    if not name.lower().endswith(".zarr.zip"):
        return None

    stem = name[: -len(".zarr.zip")]
    parent = zarr_zip_path.parent
    candidates = [
        parent / f"{stem}.parquet",
        parent / f"{stem}.csv.gz",
        parent / f"{stem}.tsv.gz",
        parent / f"{stem}.csv",
        parent / f"{stem}.tsv",
        parent / f"{stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def build_cell_feature_matrix_zarr_from_h5(
    filtered_h5_path: Path,
    output_zarr_zip_path: Path,
) -> tuple[int, int] | None:
    """Rebuild cell_feature_matrix.zarr.zip from a filtered cell_feature_matrix.h5.

    Returns:
        (num_features, num_cells) when successful, else None.
    """
    try:
        import h5py
        import zarr
    except ImportError:
        logger.warning("h5py/zarr not installed; cannot rebuild cell_feature_matrix.zarr.zip")
        return None

    t0 = time.perf_counter()
    logger.info(
        "Rebuilding %s from %s",
        output_zarr_zip_path.name,
        filtered_h5_path.name,
    )
    try:
        t_read = time.perf_counter()
        with h5py.File(filtered_h5_path, "r") as h5f:
            if "matrix" not in h5f:
                return None
            matrix = h5f["matrix"]
            required = {"barcodes", "data", "indices", "indptr", "shape", "features"}
            if not required.issubset(set(matrix.keys())):
                return None

            barcodes = matrix["barcodes"][:]
            data = matrix["data"][:]
            indices = matrix["indices"][:]
            indptr = matrix["indptr"][:]
            shape = matrix["shape"][:]
            feature_ids = matrix["features"]["id"][:]

            # Read H5 chunk shapes so zarr can use aligned chunking, avoiding
            # misaligned reads that force re-chunking at write time.
            def _h5_chunk(ds, arr) -> tuple[int, ...]:
                if ds.chunks:
                    return ds.chunks
                n = arr.shape[0] if arr.ndim > 0 else 1
                return (min(n, 1_048_576),)

            chunks_cell_id = _h5_chunk(matrix["barcodes"], barcodes)
            chunks_data    = _h5_chunk(matrix["data"],     data)
            chunks_indices = _h5_chunk(matrix["indices"],  indices)
            chunks_indptr  = _h5_chunk(matrix["indptr"],   indptr)

        logger.info(
            "  H5 read: %.2fs  (%d features, %d cells, data chunks=%s)",
            time.perf_counter() - t_read,
            int(shape[0]),
            int(shape[1]),
            chunks_data,
        )

        # Disable zarr's internal compression — the data is ZIP-compressed
        # at the store layer already, so double-compression just wastes time.
        zarr_major = int(str(getattr(zarr, "__version__", "2")).split(".")[0])
        _no_compress: dict = {"compressors": []} if zarr_major >= 3 else {"compressor": None}

        output_zarr_zip_path.parent.mkdir(parents=True, exist_ok=True)
        t_zarr = time.perf_counter()
        with warnings.catch_warnings():
            _suppress_zipstore_duplicate_name_warning()
            store = _open_zarr_zip_store(output_zarr_zip_path, mode="w")
            try:
                root = zarr.open_group(store=store, mode="w")
                g = root.require_group("cell_features")
                g.create_dataset("cell_id", shape=barcodes.shape, dtype=barcodes.dtype, data=barcodes, chunks=chunks_cell_id, **_no_compress)
                g.create_dataset("data",    shape=data.shape,     dtype=data.dtype,     data=data,    chunks=chunks_data,    **_no_compress)
                g.create_dataset("indices", shape=indices.shape,  dtype=indices.dtype,  data=indices, chunks=chunks_indices, **_no_compress)
                g.create_dataset("indptr",  shape=indptr.shape,   dtype=indptr.dtype,   data=indptr,  chunks=chunks_indptr,  **_no_compress)
                g.attrs["feature_ids"] = [
                    x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                    for x in feature_ids
                ]
                g.attrs["shape"] = [int(shape[0]), int(shape[1])]
            finally:
                store.close()
        logger.info("  Zarr ZipStore write: %.2fs", time.perf_counter() - t_zarr)

        logger.info(
            "Rebuilt %s in %.2fs total",
            output_zarr_zip_path.name,
            time.perf_counter() - t0,
        )
        return int(shape[0]), int(shape[1])
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed rebuilding %s from %s: %s", output_zarr_zip_path.name, filtered_h5_path.name, exc)
        return None


def build_tabular_zarr_from_filtered_output(
    zarr_stem: str,
    region_dir: Path,
    output_zarr_zip_path: Path,
) -> int | None:
    """Rebuild a *.zarr.zip from an already-filtered tabular file in the output region dir.

    Looks for ``<zarr_stem>.parquet``, ``.csv.gz``, ``.csv`` (in that preference
    order) under *region_dir*.  The found table is written as a zarr dataset named
    ``data``.

    Returns the row count on success, or None if no matching file was found.
    """
    candidates = [
        region_dir / f"{zarr_stem}.parquet",
        region_dir / f"{zarr_stem}.csv.gz",
        region_dir / f"{zarr_stem}.csv",
    ]
    source = next((p for p in candidates if p.exists() and p.is_file()), None)
    if source is None:
        return None

    try:
        df = read_table(source)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed reading %s for zarr rebuild: %s", source.name, exc)
        return None

    output_zarr_zip_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import zarr
    except ImportError:
        return None

    with warnings.catch_warnings():
        _suppress_zipstore_duplicate_name_warning()
        store = _open_zarr_zip_store(output_zarr_zip_path, mode="w")
        try:
            root = zarr.open_group(store=store, mode="w")
            data_group = root.require_group("data")

            for col_name in df.columns:
                series = df[col_name]
                if not (
                    pd.api.types.is_numeric_dtype(series)
                    or pd.api.types.is_bool_dtype(series)
                ):
                    continue

                arr = series.to_numpy(dtype=float, na_value=0.0)
                data_group.create_dataset(
                    str(col_name),
                    shape=arr.shape,
                    dtype=arr.dtype,
                    data=arr,
                    chunks=True,
                )
        finally:
            store.close()

    return len(df)


def build_analysis_zarr_from_analysis_dir(
    analysis_dir: Path,
    output_zarr_zip_path: Path,
) -> int:
    """Build a simplified analysis.zarr.zip from filtered analysis CSV files.

    Each column of each CSV is written as its own 1D zarr dataset under
    ``tables/<relative_path_stem>/<column_name>``.  Writing per-column avoids
    mixed-dtype arrays that zarr v3 cannot resolve.

    Returns the number of CSV files whose columns were written.
    """
    import numpy as np

    try:
        import zarr
    except ImportError:
        logger.warning("zarr not installed; cannot rebuild analysis.zarr.zip")
        return 0

    if not analysis_dir.exists():
        return 0

    csv_paths = sorted(p for p in analysis_dir.rglob("*.csv") if p.is_file())
    if not csv_paths:
        return 0

    output_zarr_zip_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with warnings.catch_warnings():
        _suppress_zipstore_duplicate_name_warning()
        store = _open_zarr_zip_store(output_zarr_zip_path, mode="w")

        try:
            root = zarr.open_group(store=store, mode="w")
            tables_group = root.require_group("tables")

            for csv_path in csv_paths:
                try:
                    df = read_table(csv_path)
                except Exception:
                    continue

                rel = csv_path.relative_to(analysis_dir)
                table_key = rel.with_suffix("").as_posix()
                col_count = 0

                for col_name in df.columns:
                    series = df[col_name]
                    # Keep numeric/bool columns only — use pd.api.types to handle pandas
                    # extension types (e.g. StringDtype) that np.issubdtype cannot interpret.
                    if not (
                        pd.api.types.is_numeric_dtype(series)
                        or pd.api.types.is_bool_dtype(series)
                    ):
                        continue

                    arr = series.to_numpy(dtype=float, na_value=0.0)

                    col_key = f"{table_key}/{col_name}"
                    try:
                        tables_group.create_dataset(
                            col_key,
                            shape=arr.shape,
                            dtype=arr.dtype,
                            data=arr,
                            chunks=True,
                        )
                        col_count += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Skipping column %s/%s: %s", table_key, col_name, exc)

                if col_count > 0:
                    numeric_cols = [
                        str(c)
                        for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
                    ]
                    tables_group[table_key].attrs["column_names"] = numeric_cols
                    written += 1
        finally:
            store.close()

    return written


def copy_gene_panel(input_dir: Path, output_dir: Path) -> bool:
    """Copy gene_panel.json from input to region output directory.

    Gene panel is unchanged across regions, so it is simply copied.

    Args:
        input_dir: Source directory containing gene_panel.json
        output_dir: Target region output directory

    Returns:
        True if copied, False if source file not found
    """
    source_path = input_dir / "gene_panel.json"
    if not source_path.exists():
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "gene_panel.json"
    try:
        shutil.copy2(source_path, target_path)
        logger.debug("Copied gene_panel.json to %s", output_dir)
        return True
    except Exception as e:
        logger.warning("Failed to copy gene_panel.json: %s", e)
        return False


def update_experiment_xenium_for_region(
    input_dir: Path,
    output_dir: Path,
    region_id: str,
    num_cells: int,
    num_transcripts: int,
    region_area_um2: float,
) -> bool:
    """Update experiment.xenium with region-specific metadata.

    Reads the original experiment.xenium, updates:
    - region_name: New region ID
    - num_cells: Filtered cell count
    - transcripts_per_cell: Calculated from num_transcripts / num_cells
    - transcripts_per_100um: Calculated from num_transcripts / area_um2 * 100

    Stores region_area_um2 in a _region_metadata.json file for reference.

    Args:
        input_dir: Source directory containing experiment.xenium
        output_dir: Target region output directory
        region_id: New region identifier
        num_cells: Number of cells in this region
        num_transcripts: Total number of transcripts in this region
        region_area_um2: Area of region polygon in square micrometers

    Returns:
        True if successfully updated, False on error
    """
    source_path = input_dir / "experiment.xenium"
    if not source_path.exists():
        return False

    try:
        # Load original metadata
        payload = json.loads(source_path.read_text(encoding="utf-8"))

        # Update region-specific fields
        payload["region_name"] = str(region_id)
        payload["num_cells"] = int(num_cells)

        # Calculate transcripts_per_cell and transcripts_per_100um
        if num_cells > 0:
            payload["transcripts_per_cell"] = int(num_transcripts / num_cells)
        else:
            payload["transcripts_per_cell"] = 0

        if region_area_um2 > 0:
            payload["transcripts_per_100um"] = (num_transcripts / region_area_um2) * 100
        else:
            payload["transcripts_per_100um"] = 0.0

        # Write updated experiment.xenium to region output
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "experiment.xenium"
        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        # Store region area in metadata file for reference
        metadata_path = output_dir / "_region_metadata.json"
        metadata = {
            "region_id": str(region_id),
            "region_area_um2": float(region_area_um2),
            "num_cells": int(num_cells),
            "num_transcripts": int(num_transcripts),
            "transcripts_per_cell": int(num_transcripts / num_cells) if num_cells > 0 else 0,
            "transcripts_per_100um": (num_transcripts / region_area_um2 * 100) if region_area_um2 > 0 else 0.0,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        logger.debug(
            "Updated experiment.xenium for region %s: %d cells, %.2f transcripts/100um",
            region_id,
            num_cells,
            metadata["transcripts_per_100um"],
        )
        return True

    except Exception as e:
        logger.error("Failed to update experiment.xenium for region %s: %s", region_id, e)
        return False
