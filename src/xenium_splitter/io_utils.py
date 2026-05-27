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
import numpy as np
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
        message=r"Duplicate name: '.*'",
        category=UserWarning,
        module=r"zipfile",
    )


def _xenium_temp_root() -> Path:
    """Return/create the dedicated temp root for xenium-splitter scratch files."""
    root = Path(tempfile.gettempdir()) / "xenium_splitter"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _zarr_create_dataset(zarr_mod, group, name, **kwargs):
    """Create a zarr dataset with write_empty_chunks=True regardless of zarr version.

    - zarr v2: pass write_empty_chunks=True as a kwarg to create_dataset.
    - zarr v3: zarr.config is available; use it as a context manager instead
      (the kwarg is silently ignored in v3).
    """
    if hasattr(zarr_mod, "config"):
        # zarr v3
        with zarr_mod.config.set({"array.write_empty_chunks": True}):
            return group.create_dataset(name, **kwargs)
    else:
        # zarr v2
        kwargs["write_empty_chunks"] = True
        return group.create_dataset(name, **kwargs)


def iter_input_files(input_dir: Path) -> list[Path]:
    """Recursively list all files in input_dir, excluding metadata files that are
    handled separately or are summary/artifact files."""
    files: list[Path] = []
    excluded_names = {
        "gene_panel.json",
        "experiment.xenium",
        "metrics_summary.csv",
        "analysis_summary.html",
    }
    for path in input_dir.rglob("*"):
        if path.is_file() and path.name not in excluded_names:
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


def _index_to_alpha_label(index: int) -> str:
    """Convert 0-based row index to Excel-like labels: 0->A, 25->Z, 26->AA."""
    value = max(0, int(index))
    label = ""
    while True:
        value, rem = divmod(value, 26)
        label = chr(ord("A") + rem) + label
        if value == 0:
            break
        value -= 1
    return label


def _generate_alpha_numeric_fov_names(total_fovs: int, grid_cols: int | None = None) -> list[str]:
    """Generate names in index order as A1,A2,...,B1,B2,...."""
    total = max(0, int(total_fovs))
    if total == 0:
        return []
    cols = int(grid_cols) if grid_cols is not None and int(grid_cols) > 0 else total
    return [f"{_index_to_alpha_label(i // cols)}{(i % cols) + 1}" for i in range(total)]


def _generate_legacy_fov_names(total_fovs: int) -> list[str]:
    """Generate legacy Xenium-style names as fov_<index>."""
    total = max(0, int(total_fovs))
    return [f"fov_{i}" for i in range(total)]


def _generate_fov_names(total_fovs: int, grid_cols: int | None = None) -> list[str]:
    """Generate Letter+Number names in index order: A1,A2,...,B1,B2,...

    Keep _generate_legacy_fov_names() available so the legacy naming
    scheme can be restored quickly if needed.
    """
    return _generate_alpha_numeric_fov_names(total_fovs, grid_cols)


def validate_transcripts_id_uuid_schema(root) -> dict[str, int | bool]:
    """Validate Xenium transcript identity schema in a transcripts zarr root.

    Required invariant for each row where both arrays are present:
    - id[:, 0] == uuid[:, 0]
    - uuid[:, 1] == 65536 + id[:, 1]
    """
    if "grids" not in root or "0" not in root["grids"]:
        return {"checked_rows": 0, "checked_tiles": 0, "max_id1": -1, "ok": True}

    level0 = root["grids"]["0"]
    checked_rows = 0
    checked_tiles = 0
    max_id1 = -1
    canonical_pairs: list[np.ndarray] = []
    used_fov_indices: set[int] = set()

    for tile_key in level0.keys():
        tile = level0[tile_key]
        if "id" not in tile or "uuid" not in tile:
            continue

        id_arr = tile["id"][:]
        uuid_arr = tile["uuid"][:]

        if getattr(id_arr, "ndim", 0) != 2 or id_arr.shape[1] < 2:
            raise ValueError(f"Invalid id array shape at grids/0/{tile_key}: {getattr(id_arr, 'shape', None)}")
        if getattr(uuid_arr, "ndim", 0) != 2 or uuid_arr.shape[1] < 2:
            raise ValueError(
                f"Invalid uuid array shape at grids/0/{tile_key}: {getattr(uuid_arr, 'shape', None)}"
            )
        if id_arr.shape[0] != uuid_arr.shape[0]:
            raise ValueError(
                f"Row mismatch at grids/0/{tile_key}: id_rows={id_arr.shape[0]} uuid_rows={uuid_arr.shape[0]}"
            )

        if id_arr.shape[0] == 0:
            checked_tiles += 1
            continue

        id0 = id_arr[:, 0].astype(np.uint64, copy=False)
        id1 = id_arr[:, 1].astype(np.uint64, copy=False)
        uuid0 = uuid_arr[:, 0].astype(np.uint64, copy=False)
        uuid1 = uuid_arr[:, 1].astype(np.uint64, copy=False)

        bad0 = np.where(uuid0 != id0)[0]
        if bad0.size > 0:
            i = int(bad0[0])
            raise ValueError(
                f"id/uuid low-word mismatch at grids/0/{tile_key} row={i}: "
                f"id0={int(id0[i])} uuid0={int(uuid0[i])}"
            )

        expected_uuid1 = np.uint64(65536) + id1
        bad1 = np.where(uuid1 != expected_uuid1)[0]
        if bad1.size > 0:
            i = int(bad1[0])
            raise ValueError(
                f"id/uuid high-word mismatch at grids/0/{tile_key} row={i}: "
                f"id1={int(id1[i])} uuid1={int(uuid1[i])} expected={int(expected_uuid1[i])}"
            )

        canonical_pairs.append((id1 << np.uint64(32)) | id0)
        used_fov_indices.update(int(v) for v in np.unique(id1).tolist())

        tile_max_id1 = int(np.max(id1))
        max_id1 = max(max_id1, tile_max_id1)
        checked_rows += int(id_arr.shape[0])
        checked_tiles += 1

    if canonical_pairs:
        canonical_all = np.concatenate(canonical_pairs)
        if np.unique(canonical_all).size != canonical_all.size:
            raise ValueError("Duplicate (fov_index, transcript_id) pairs detected in grids/0")

    # Note: FOV indices may be sparse in input files. Output generation applies
    # compatibility remapping to densify indices for loader compatibility.

    number_fovs = root.attrs.get("number_fovs")
    if number_fovs is not None:
        try:
            n_fovs = int(number_fovs)
        except Exception:
            n_fovs = 0
        if n_fovs > 0 and max_id1 >= n_fovs:
            raise ValueError(
                f"FOV index out of bounds: max(id[:,1])={max_id1}, number_fovs={n_fovs}"
            )

    return {
        "checked_rows": int(checked_rows),
        "checked_tiles": int(checked_tiles),
        "max_id1": int(max_id1),
        "ok": True,
    }


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
            x_col: Name of x-coordinate column
            y_col: Name of y-coordinate column

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
    """Compute crop origin in micrometers aligned to image crop pixel boundaries.

    With pixel_size_um, this matches image crop logic by flooring polygon minima
    in pixel space, then converting back to micrometers. Without pixel_size_um,
    falls back to raw polygon minima.
    """
    min_x, min_y, _, _ = region.bounds
    if pixel_size_um is not None and pixel_size_um > 0:
        min_x_px = max(int(math.floor(min_x / pixel_size_um)), 0)
        min_y_px = max(int(math.floor(min_y / pixel_size_um)), 0)
        return min_x_px * pixel_size_um, min_y_px * pixel_size_um
    return min_x, min_y


def rebase_table_coordinates_to_region_crop(
    df: pd.DataFrame,
    region: LassoRegion,
    x_col: str,
    y_col: str,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """Shift x/y columns so output coordinates align to the cropped image origin.

    Uses a pixel-boundary-aligned origin when pixel_size_um is provided, so that
    entity coordinate (0, 0) corresponds to image pixel (0, 0) in the cropped image.
    Coordinates remain in micrometers — only the origin offset is subtracted.

    Args:
        df: DataFrame with rows in the region (pre-filtered by containment or ID)
        region: LASSO region defining the crop bounding box
        x_col: Name of x-coordinate column
        y_col: Name of y-coordinate column
        pixel_size_um: Pixel size in micrometers when image alignment is needed

    Returns:
        DataFrame with rebased x/y coordinates
    """

    origin_x, origin_y = _region_crop_origin_um(region, pixel_size_um=pixel_size_um)
    x_before = pd.to_numeric(df[x_col], errors="coerce")
    y_before = pd.to_numeric(df[y_col], errors="coerce")
    df[x_col] = x_before - origin_x
    df[y_col] = y_before - origin_y

    try:
        logger.debug(
            "Rebasing coordinates: region=%s origin_um=(x:%.6f,y:%.6f) x_before=(%.6f,%.6f) y_before=(%.6f,%.6f) x_after=(%.6f,%.6f) y_after=(%.6f,%.6f)",
            region.region_id,
            origin_x,
            origin_y,
            float(x_before.min(skipna=True)),
            float(x_before.max(skipna=True)),
            float(y_before.min(skipna=True)),
            float(y_before.max(skipna=True)),
            float(df[x_col].min(skipna=True)),
            float(df[x_col].max(skipna=True)),
            float(df[y_col].min(skipna=True)),
            float(df[y_col].max(skipna=True)),
        )
    except Exception:
        logger.debug(
            "Rebasing coordinates: region=%s origin_um=(x:%.6f,y:%.6f)",
            region.region_id,
            origin_x,
            origin_y,
            exc_info=True,
        )
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


def load_instrument_sw_version_from_experiment(input_dir: Path) -> str | None:
    """Extract instrument_sw_version from experiment.xenium if present."""
    exp_path = input_dir / "experiment.xenium"
    if not exp_path.exists():
        return None

    try:
        payload = json.loads(exp_path.read_text(encoding="utf-8"))
        version = payload.get("instrument_sw_version")
        return str(version) if version is not None else None
    except (json.JSONDecodeError, OSError):
        return None


def resolve_experiment_dir_for_path(start_path: Path, max_up: int = 6) -> Path:
    """Find the nearest directory containing experiment.xenium.

    Searches start_path (or its parent if start_path is a file) and then walks
    up parent directories up to max_up levels. Returns the first match, or the
    starting directory when no experiment.xenium is found.
    """
    current = start_path if start_path.is_dir() else start_path.parent
    checked = 0
    while True:
        if (current / "experiment.xenium").exists():
            return current
        if checked >= max_up or current.parent == current:
            return start_path if start_path.is_dir() else start_path.parent
        current = current.parent
        checked += 1


def get_fov_dimensions_px_for_sw_version(instrument_sw_version: str | None) -> tuple[int, int]:
    """Return (rows_px, cols_px) for Xenium FOV based on instrument software version.

    Rules:
    - < 1.2  -> 4240 rows x 2960 cols
    - >= 1.2 -> 3520 rows x 2960 cols
    """
    default_rows, default_cols = 3520, 2960
    if not instrument_sw_version:
        return default_rows, default_cols

    try:
        parts = [int(p) for p in str(instrument_sw_version).split(".") if p.strip() != ""]
    except Exception:
        return default_rows, default_cols

    major = parts[0] if len(parts) > 0 else 0
    minor = parts[1] if len(parts) > 1 else 0
    if (major, minor) < (1, 2):
        return 4240, 2960
    return 3520, 2960


def calculate_fov_layout_and_assignments(
    x_um,
    y_um,
    pixel_size_um: float,
    fov_rows_px: int,
    fov_cols_px: int,
    overlap_px: int = 128,
) -> tuple[pd.Series, dict[str, int]]:
    """Compute transcript FOV indices from rebased coordinates and FOV geometry.

    Coordinates are in micrometers and converted to pixel coordinates using
    pixel_size_um. FOV placement uses stride=(dimension-overlap) on each axis.
    Returns row-major FOV indices and a metadata summary.
    """
    import numpy as np

    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be > 0")

    step_x = max(1, int(fov_cols_px) - int(overlap_px))
    step_y = max(1, int(fov_rows_px) - int(overlap_px))

    x_px = np.floor(np.maximum(np.asarray(x_um, dtype=np.float64), 0.0) / float(pixel_size_um)).astype(np.int64)
    y_px = np.floor(np.maximum(np.asarray(y_um, dtype=np.float64), 0.0) / float(pixel_size_um)).astype(np.int64)

    fov_x = x_px // step_x
    fov_y = y_px // step_y

    # Overlap tie-break: if a point falls into an overlap strip, assign it to
    # the first FOV (top-left / lower row-major index) among valid candidates.
    overlap_x = np.int64(max(0, int(overlap_px)))
    overlap_y = np.int64(max(0, int(overlap_px)))
    local_x = x_px - (fov_x * np.int64(step_x))
    local_y = y_px - (fov_y * np.int64(step_y))
    choose_prev_x = (fov_x > 0) & (local_x < overlap_x)
    choose_prev_y = (fov_y > 0) & (local_y < overlap_y)
    fov_x = np.where(choose_prev_x, fov_x - 1, fov_x)
    fov_y = np.where(choose_prev_y, fov_y - 1, fov_y)

    grid_cols = int(np.max(fov_x)) + 1 if fov_x.size > 0 else 1
    grid_rows = int(np.max(fov_y)) + 1 if fov_y.size > 0 else 1
    fov_idx = (fov_y * grid_cols + fov_x).astype(np.uint32, copy=False)

    metadata = {
        "fov_rows_px": int(fov_rows_px),
        "fov_cols_px": int(fov_cols_px),
        "fov_overlap_px": int(overlap_px),
        "fov_stride_rows_px": int(step_y),
        "fov_stride_cols_px": int(step_x),
        "fov_grid_rows": int(grid_rows),
        "fov_grid_cols": int(grid_cols),
        "number_fovs": int(grid_rows * grid_cols),
    }
    return pd.Series(fov_idx), metadata


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
    outer zip layer.  No zarr-level compression is applied either - compression at
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
            # Deduplicate any .zattrs files at the top level that ZipStore may have created
            _dedupe_zip_entries_keep_last(output_path)
    except Exception as e:
        logger.error(f"Failed to write Zarr ZIP to {output_path.name}: {e}")


def filter_zarr_zip_by_row_indices_preserve_schema(
    zarr_zip_path: Path,
    output_path: Path,
    row_indices,
    base_row_count: int,
    rebase_region: LassoRegion | None = None,
    pixel_size_um: float | None = None,
    transcript_id_values=None,
    transcript_table=None,
) -> bool:
    """Filter a Zarr ZIP while preserving its original hierarchy/schema.

    Any array whose first dimension equals ``base_row_count`` is subset using
    ``row_indices``. Other arrays/groups/attributes are copied unchanged.
    """
    try:
        import numpy as np
        import zarr
    except ImportError:
        logger.warning("zarr or numpy not installed; cannot preserve zarr schema.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(row_indices, dtype=np.int64)
    transcript_ids = None
    transcript_ids_sorted = None
    transcript_id_mode: dict[str, str | None] = {"kind": None}
    if transcript_id_values is not None:
        try:
            transcript_ids = np.asarray(list(transcript_id_values), dtype=np.int64)
            transcript_ids = transcript_ids[np.isfinite(transcript_ids)] if transcript_ids.size else transcript_ids
            transcript_ids = np.unique(transcript_ids.astype(np.int64, copy=False))
            transcript_ids_sorted = transcript_ids

            # Infer the transcript ID encoding once so matching is strict.
            one_shift_48 = (np.int64(1) << np.int64(48))
            two_shift_32 = (np.int64(1) << np.int64(32))
            if transcript_ids_sorted.size > 0:
                min_id = int(transcript_ids_sorted[0])
                max_id = int(transcript_ids_sorted[-1])
                if min_id >= int(one_shift_48):
                    shifted = transcript_ids_sorted - one_shift_48
                    if np.all((shifted >= 0) & (shifted < two_shift_32)):
                        transcript_id_mode["kind"] = "low_plus_2pow48"
                    else:
                        transcript_id_mode["kind"] = "hi_and_low_plus_2pow48"
                elif min_id >= 0 and max_id < int(two_shift_32):
                    transcript_id_mode["kind"] = "low"
                else:
                    transcript_id_mode["kind"] = "pair_u32"

                logger.debug(
                    "[filter_zarr_zip_preserve_schema] transcript_id_mode=%s ids=%d min=%d max=%d",
                    transcript_id_mode["kind"],
                    int(transcript_ids_sorted.size),
                    min_id,
                    max_id,
                )
        except Exception:
            transcript_ids = None
            transcript_ids_sorted = None
            transcript_id_mode["kind"] = None
    origin_xy = _region_crop_origin_um(rebase_region, pixel_size_um=pixel_size_um) if rebase_region else None
    experiment_dir = resolve_experiment_dir_for_path(zarr_zip_path)
    instrument_sw_version = load_instrument_sw_version_from_experiment(experiment_dir)
    fov_rows_px, fov_cols_px = get_fov_dimensions_px_for_sw_version(instrument_sw_version)
    fov_overlap_px = 128
    effective_pixel_size_um = pixel_size_um if pixel_size_um is not None else load_pixel_size_from_experiment(experiment_dir)
    recomputed_fov_metadata: dict[str, int] | None = None

    if effective_pixel_size_um is None or effective_pixel_size_um <= 0:
        logger.warning(
            "Could not determine pixel_size for FOV reassignment in %s; using 1.0 um/px fallback",
            zarr_zip_path.name,
        )
        effective_pixel_size_um = 1.0
    
    logger.debug(
        "[filter_zarr_zip_preserve_schema] START: input=%s output=%s rows_before=%d rows_after=%d sw=%s fov_rows=%d fov_cols=%d overlap=%d",
        zarr_zip_path.name,
        output_path.name,
        base_row_count,
        len(indices),
        instrument_sw_version,
        fov_rows_px,
        fov_cols_px,
        fov_overlap_px,
    )

    with tempfile.TemporaryDirectory(dir=_xenium_temp_root(), prefix="zarr_schema_in_") as tmpdir_in:
        tmp_output_zip: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix="zarr_schema_out_",
                suffix=".zip",
                delete=False,
            ) as tmpf:
                tmp_output_zip = Path(tmpf.name)

            with zipfile.ZipFile(zarr_zip_path, "r") as zf:
                zf.extractall(tmpdir_in)

            # Capture source .zarray schema key presence so output can match it.
            source_zarray_has_dim_sep: dict[str, bool] = {}
            for zarray_path in Path(tmpdir_in).rglob(".zarray"):
                try:
                    rel = zarray_path.relative_to(Path(tmpdir_in)).as_posix()
                    payload = json.loads(zarray_path.read_text(encoding="utf-8"))
                    source_zarray_has_dim_sep[rel] = "dimension_separator" in payload
                except Exception:
                    continue
            source_uses_dim_sep = any(source_zarray_has_dim_sep.values())

            src_root = zarr.open_group(tmpdir_in, mode="r")

            # Force zarr v2 format for Xenium Explorer compatibility
            zarr_format = 2
            if not (Path(tmpdir_in) / ".zgroup").exists():
                # Source is v3, but we still output v2 for compatibility
                zarr_format = 2

            logger.debug(
                "[filter_zarr_zip_preserve_schema] detected zarr_format=%d, forcing output to v2",
                zarr_format,
            )

            dst_store = None
            with warnings.catch_warnings():
                _suppress_zipstore_duplicate_name_warning()
                dst_store = _open_zarr_zip_store(tmp_output_zip, mode="w")
                try:
                    dst_root = zarr.open_group(dst_store, mode="w", zarr_format=2)
                except TypeError:
                    # Older zarr versions don't support zarr_format parameter
                    dst_root = zarr.open_group(dst_store, mode="w")
                    logger.warning("zarr_format parameter not supported; using default format")

                def _rebase_known_arrays(path_key: str, data_arr, node_attrs: dict):
                    if origin_xy is None or not hasattr(data_arr, "shape"):
                        return data_arr

                    x0, y0 = origin_xy
                    leaf = path_key.split("/")[-1].lower()

                    def _x_shift(values):
                        return values - x0

                    def _y_shift(values):
                        return values - y0

                    try:
                        if leaf == "location" and getattr(data_arr, "ndim", 0) == 2 and data_arr.shape[1] >= 2:
                            out = data_arr.copy()
                            out[:, 0] = _x_shift(out[:, 0])
                            out[:, 1] = _y_shift(out[:, 1])
                            return out

                        if leaf == "polygon_vertices" and getattr(data_arr, "ndim", 0) == 3 and data_arr.shape[0] == 2:
                            out = data_arr.copy()
                            # Shape is (2, N, 26): dim0=[cell_poly, nucleus_poly],
                            # dim2 stores interleaved x,y pairs: [x0, y0, x1, y1, ...]
                            # Apply origin shift and optional um->px scaling.
                            out[:, :, 0::2] = _x_shift(out[:, :, 0::2])
                            out[:, :, 1::2] = _y_shift(out[:, :, 1::2])
                            return out

                        if leaf == "cell_summary" and getattr(data_arr, "ndim", 0) == 2:
                            out = data_arr.copy()
                            cols = node_attrs.get("column_names", [])
                            cols_l = [str(c).lower() for c in cols]
                            for i, col in enumerate(cols_l):
                                if col in {"cell_centroid_x", "nucleus_centroid_x", "x", "x_location", "global_x"}:
                                    out[:, i] = _x_shift(out[:, i])
                                elif col in {"cell_centroid_y", "nucleus_centroid_y", "y", "y_location", "global_y"}:
                                    out[:, i] = _y_shift(out[:, i])
                            return out

                        if getattr(data_arr, "ndim", 0) == 1:
                            if leaf in {"x", "x_location", "global_x", "cell_centroid_x", "nucleus_centroid_x"} or leaf.endswith("_x"):
                                out = data_arr.copy()
                                out[...] = _x_shift(out[...])
                                return out
                            if leaf in {"y", "y_location", "global_y", "cell_centroid_y", "nucleus_centroid_y"} or leaf.endswith("_y"):
                                out = data_arr.copy()
                                out[...] = _y_shift(out[...])
                                return out
                    except Exception:
                        return data_arr

                    return data_arr

                def _build_transcript_group_mask(id_arr) -> np.ndarray | None:
                    if transcript_ids is None:
                        return None
                    if not hasattr(id_arr, "shape") or getattr(id_arr, "ndim", 0) != 2 or id_arr.shape[1] < 1:
                        return None

                    low = id_arr[:, 0].astype(np.int64, copy=False)
                    one_shift_48 = (np.int64(1) << np.int64(48))
                    hi = id_arr[:, 1].astype(np.int64, copy=False) if id_arr.shape[1] > 1 else np.zeros_like(low)

                    def _mode_values(kind: str) -> np.ndarray:
                        if kind == "hi_and_low_plus_2pow48":
                            # Correct Xenium encoding: 2^48 + (fov << 32) + local_index
                            # hi encodes the FOV number; low encodes the per-FOV transcript index.
                            return one_shift_48 + (hi << np.int64(32)) + low
                        if kind == "low_plus_2pow48":
                            return low + one_shift_48
                        if kind == "low":
                            return low
                        if kind == "pair_u32":
                            return (hi << np.int64(32)) + low
                        return low

                    if transcript_ids_sorted is None or transcript_ids_sorted.size == 0:
                        return np.zeros(low.shape[0], dtype=bool)

                    kind = transcript_id_mode.get("kind")
                    if kind is None:
                        kind = "hi_and_low_plus_2pow48"

                    vals = _mode_values(kind)
                    pos = np.searchsorted(transcript_ids_sorted, vals)
                    in_bounds = (pos >= 0) & (pos < transcript_ids_sorted.size)
                    out = np.zeros(low.shape[0], dtype=bool)
                    if np.any(in_bounds):
                        out[in_bounds] = transcript_ids_sorted[pos[in_bounds]] == vals[in_bounds]
                    return out

                def _copy_group(src_group, dst_group, path_prefix: str = "") -> None:
                    preserve_unfiltered_arrays = {
                        "gene_category",
                        "codeword_category",
                    }
                    for attr_key, attr_val in dict(src_group.attrs).items():
                        dst_group.attrs[attr_key] = attr_val

                    # Fix number_cells to reflect the filtered count, not the original.
                    if "number_cells" in dict(dst_group.attrs):
                        dst_group.attrs["number_cells"] = len(indices)

                    group_row_mask = None
                    if transcript_ids is not None and "id" in src_group:
                        try:
                            id_node = src_group["id"]
                            if hasattr(id_node, "shape") and getattr(id_node, "ndim", 0) == 2 and id_node.shape[1] >= 1:
                                group_row_mask = _build_transcript_group_mask(id_node[:])
                        except Exception as e:
                            logger.debug("[_copy_group] Exception building mask: %s", e)
                            group_row_mask = None

                    for key in src_group.keys():
                        node = src_group[key]
                        key_path = f"{path_prefix}/{key}" if path_prefix else str(key)
                        if hasattr(node, "shape") and hasattr(node, "dtype"):
                            data = node[:]
                            if (
                                group_row_mask is not None
                                and getattr(node, "ndim", 0) >= 1
                                and len(data.shape) >= 1
                                and data.shape[0] == group_row_mask.shape[0]
                            ):
                                data = data[group_row_mask, ...]
                            elif transcript_ids is not None and getattr(node, "ndim", 0) >= 1 and len(data.shape) >= 1:
                                # Filtering by transcript_id but don't have a mask for this group/tile.
                                # Check if array spans the full input table; if so, use row indices.
                                # Otherwise skip (don't copy) since we can't filter it without a mask.
                                match_axes = [ax for ax, sz in enumerate(node.shape) if int(sz) == int(base_row_count)]
                                if match_axes:
                                    data = np.take(data, indices, axis=match_axes[0])
                                    logger.debug(
                                        "[_copy_group] path=%s/%s filtered via row_indices (no_mask_spanning_full_table)",
                                        path_prefix, key,
                                    )
                                else:
                                    # Preserve known root-level category arrays unchanged.
                                    if key_path in preserve_unfiltered_arrays:
                                        logger.debug(
                                            "[_copy_group] path=%s COPIED unchanged (preserve_unfiltered_array) shape=%s",
                                            key_path,
                                            node.shape,
                                        )
                                    else:
                                        # Array doesn't span full input table and we don't have a mask for it.
                                        # Skip copying this array entirely.
                                        logger.debug(
                                            "[_copy_group] path=%s/%s SKIPPED (no_mask_not_spanning_full_table) shape=%s",
                                            path_prefix, key, node.shape,
                                        )
                                        continue
                            elif getattr(node, "ndim", 0) >= 1:
                                # Not filtering by transcript_id: use row indices as before.
                                match_axes = [ax for ax, sz in enumerate(node.shape) if int(sz) == int(base_row_count)]
                                if match_axes:
                                    data = np.take(data, indices, axis=match_axes[0])

                            data = _rebase_known_arrays(key_path, data, dict(node.attrs))

                            chunks = getattr(node, "chunks", None)
                            if chunks is not None and isinstance(data.shape, tuple) and len(chunks) == len(data.shape):
                                # Keep chunk axes >= 1 even when filtered data has zero rows.
                                # zarr may raise division-by-zero if any chunk axis is 0.
                                chunks = tuple(
                                    max(1, min(int(c), int(s)) if int(s) > 0 else 1)
                                    for c, s in zip(chunks, data.shape)
                                )

                            create_kwargs = {
                                "shape": data.shape,
                                "dtype": data.dtype,
                                "data": data,
                            }
                            if chunks is not None:
                                create_kwargs["chunks"] = chunks

                            compressors = getattr(node, "compressors", None)
                            compressor = getattr(node, "compressor", None)
                            if compressors is not None:
                                create_kwargs["compressors"] = compressors
                            elif compressor is not None:
                                create_kwargs["compressor"] = compressor

                            try:
                                dst_arr = _zarr_create_dataset(zarr, dst_group, key, **create_kwargs)
                            except TypeError:
                                create_kwargs.pop("compressors", None)
                                create_kwargs.pop("compressor", None)
                                dst_arr = _zarr_create_dataset(zarr, dst_group, key, **create_kwargs)

                            for attr_key, attr_val in dict(node.attrs).items():
                                dst_arr.attrs[attr_key] = attr_val
                        else:
                            # Skip the grids group entirely when filtering transcripts: it will be
                            # rebuilt from scratch by _rebuild_transcript_grids_and_density.
                            # This avoids writing stale tile data into the ZipStore (which is
                            # write-once and cannot have keys deleted or overwritten).
                            if key == "grids" and (transcript_ids is not None or origin_xy is not None) and not path_prefix:
                                logger.debug(
                                    "[_copy_group] path=%s SKIPPED (grids will be rebuilt fresh)",
                                    key_path,
                                )
                                continue
                            dst_sub = dst_group.require_group(key)
                            _copy_group(node, dst_sub, key_path)

                def _rebuild_transcript_grids_and_density(dst_root) -> None:
                    """Re-tile transcript grids after coordinate rebasing and recompute density/gene.

                    transcripts.zarr stores transcript rows inside grids/[level]/[tile].
                    After rebasing locations, rows must be moved into new tiles based on the
                    grid size for each level; otherwise tile keys and locations are inconsistent.
                    """
                    if (transcript_ids is None and origin_xy is None) or "grids" not in src_root:
                        return

                    # Grids was skipped during _copy_group (to avoid writing stale tile data into
                    # the write-once ZipStore). Read all metadata directly from the source.
                    grids_group = src_root["grids"]
                    source_grid_attrs = dict(grids_group.attrs)
                    source_level_names = sorted(
                        list(grids_group.keys()),
                        key=lambda s: int(str(s)) if str(s).isdigit() else str(s),
                    )
                    if not source_level_names:
                        return

                    # Keep output pyramid depth aligned with the input store.
                    level_names = source_level_names
                    try:
                        src_level_count = int(source_grid_attrs.get("number_levels", len(level_names)))
                        if src_level_count > 0:
                            canonical_levels = [str(i) for i in range(src_level_count)]
                            if all(name in level_names for name in canonical_levels):
                                level_names = canonical_levels
                    except Exception:
                        level_names = source_level_names

                    base_grid_size = 250.0
                    try:
                        gs = source_grid_attrs.get("grid_size", [250.0])
                        if isinstance(gs, (list, tuple)) and len(gs) > 0:
                            base_grid_size = float(gs[0])
                    except Exception:
                        pass

                    rebuilt_levels: list[tuple[str, list[tuple[str, dict[str, np.ndarray], dict[str, dict]]]]] = []
                    grid_keys_by_level: list[list[str]] = []
                    grid_counts_by_level: list[list[int]] = []
                    level0_total = 0

                    def _tile_size_for_level(level_name: str, level_index: int) -> float:
                        """Return tile size for a level, preferring explicit sizes from attrs.

                        Some datasets store a per-level grid_size list. Preserve that exactly
                        when available; otherwise fall back to the legacy base*2^level behavior.
                        """
                        try:
                            gs = source_grid_attrs.get("grid_size", [base_grid_size])
                            if isinstance(gs, (list, tuple)) and len(gs) > 0:
                                if len(gs) >= len(level_names):
                                    return float(gs[level_index])
                                if len(gs) >= 1:
                                    lvl_num = int(str(level_name)) if str(level_name).isdigit() else level_index
                                    return float(gs[0]) * (2.0 ** float(lvl_num))
                        except Exception:
                            pass
                        lvl_num = int(str(level_name)) if str(level_name).isdigit() else level_index
                        return float(base_grid_size) * (2.0 ** float(lvl_num))

                    def _detect_xy_cols(df) -> tuple[str | None, str | None]:
                        x_candidates = ["x_location", "x", "global_x", "x_position"]
                        y_candidates = ["y_location", "y", "global_y", "y_position"]
                        x_col = next((c for c in x_candidates if c in df.columns), None)
                        y_col = next((c for c in y_candidates if c in df.columns), None)
                        return x_col, y_col

                    def _normalize_fov_name(value) -> str:
                        if isinstance(value, (bytes, np.bytes_)):
                            try:
                                return value.decode("utf-8")
                            except Exception:
                                return str(value)
                        return str(value)

                    def _build_from_transcript_table(table_df, template_level_group):
                        nonlocal recomputed_fov_metadata
                        if table_df is None or getattr(table_df, "empty", True):
                            return None, None
                        x_col, y_col = _detect_xy_cols(table_df)
                        if x_col is None or y_col is None:
                            return None, None

                        if transcript_ids is not None and "transcript_id" in table_df.columns:
                            ids_arr = pd.to_numeric(table_df["transcript_id"], errors="coerce")
                            ids_arr = ids_arr.to_numpy(dtype=np.float64)
                            finite = np.isfinite(ids_arr)
                            ids64 = ids_arr[finite].astype(np.int64, copy=False)
                            keep = np.searchsorted(transcript_ids_sorted, ids64)
                            inb = (keep >= 0) & (keep < transcript_ids_sorted.size)
                            mask = np.zeros(len(table_df), dtype=bool)
                            if np.any(inb):
                                sel = np.zeros(ids64.shape[0], dtype=bool)
                                sel[inb] = transcript_ids_sorted[keep[inb]] == ids64[inb]
                                mask[np.where(finite)[0]] = sel
                            table_df = table_df.loc[mask].copy()
                            if table_df.empty:
                                return None, None

                        x = pd.to_numeric(table_df[x_col], errors="coerce").to_numpy(dtype=np.float64)
                        y = pd.to_numeric(table_df[y_col], errors="coerce").to_numpy(dtype=np.float64)
                        z_col = "z_location" if "z_location" in table_df.columns else ("z" if "z" in table_df.columns else None)
                        if z_col is not None:
                            z = pd.to_numeric(table_df[z_col], errors="coerce").to_numpy(dtype=np.float64)
                            # Keep rows with valid x/y even when z is missing; Xenium transcripts are effectively 2D.
                            z = np.nan_to_num(z, nan=0.0)
                        else:
                            z = np.zeros_like(x)
                        finite_loc = np.isfinite(x) & np.isfinite(y)
                        x = x[finite_loc]
                        y = y[finite_loc]
                        z = z[finite_loc]
                        if x.size == 0:
                            return None, None

                        n = x.shape[0]
                        merged = {
                            "location": np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
                        }
                        new_tid64 = None

                        # Build id from transcript_id where available.
                        # Recompute FOV assignment from rebased coordinates and assign
                        # per-FOV local transcript indices.
                        if "transcript_id" in table_df.columns:
                            tid = pd.to_numeric(table_df.loc[finite_loc, "transcript_id"], errors="coerce").to_numpy(dtype=np.float64)
                            tid = np.nan_to_num(tid, nan=0.0).astype(np.int64, copy=False)
                            one_shift_48 = (np.int64(1) << np.int64(48))
                            payload = np.where(tid >= one_shift_48, tid - one_shift_48, tid).astype(np.int64, copy=False)
                            old_low = (payload & np.int64(0xFFFFFFFF)).astype(np.uint32, copy=False)
                            old_fov = ((payload >> np.int64(32)) & np.int64(0xFFFFFFFF)).astype(np.uint32, copy=False)

                            recomputed_fov_series, fov_meta = calculate_fov_layout_and_assignments(
                                x,
                                y,
                                pixel_size_um=float(effective_pixel_size_um),
                                fov_rows_px=int(fov_rows_px),
                                fov_cols_px=int(fov_cols_px),
                                overlap_px=int(fov_overlap_px),
                            )
                            recomputed_fov_metadata = {
                                str(k): int(v) for k, v in fov_meta.items()
                            }
                            new_fov = recomputed_fov_series.to_numpy(dtype=np.uint32, copy=False)

                            # Renumber low-word transcript identifiers within each
                            # reassigned FOV. Start at 1 and allow reuse across FOVs.
                            new_low = np.zeros(new_fov.shape[0], dtype=np.uint32)
                            for fov_val in np.unique(new_fov.astype(np.int64, copy=False)):
                                mask_fov = new_fov == np.uint32(fov_val)
                                count = int(np.count_nonzero(mask_fov))
                                if count > 0:
                                    new_low[mask_fov] = np.arange(1, count + 1, dtype=np.uint32)

                            new_tid64 = (
                                one_shift_48
                                + (new_fov.astype(np.int64, copy=False) << np.int64(32))
                                + new_low.astype(np.int64, copy=False)
                            ).astype(np.int64, copy=False)

                            if output_path.name.lower() == "transcripts.zarr.zip":
                                remap_path = output_path.with_name("transcripts_id_fov_remap.csv.gz")
                                remap_df = pd.DataFrame(
                                    {
                                        "old_transcript_id": tid,
                                        "old_fov": old_fov,
                                        "old_local_id": old_low,
                                        "new_transcript_id": new_tid64,
                                        "new_fov": new_fov,
                                        "new_local_id": new_low,
                                    }
                                )
                                remap_df.to_csv(remap_path, index=False, compression="gzip")
                                logger.info(
                                    "Wrote transcript ID/FOV remap log: %s (%d rows)",
                                    remap_path,
                                    len(remap_df),
                                )

                            merged["id"] = np.stack(
                                [new_low, new_fov],
                                axis=1,
                            )

                        gene_index_map = dst_root.attrs.get("gene_index_map", {})
                        if "feature_name" in table_df.columns and isinstance(gene_index_map, dict):
                            gene_names = table_df.loc[finite_loc, "feature_name"].astype(str).to_numpy()
                            gene_idx = np.array([gene_index_map.get(g, 65535) for g in gene_names], dtype=np.uint16)
                        elif "gene" in table_df.columns:
                            gene_idx = pd.to_numeric(table_df.loc[finite_loc, "gene"], errors="coerce").fillna(65535).astype(np.int64).to_numpy()
                            gene_idx = np.clip(gene_idx, 0, 65535).astype(np.uint16)
                        else:
                            gene_idx = np.full(n, 65535, dtype=np.uint16)
                        merged["gene_identity"] = gene_idx.reshape(-1, 1)

                        # Best-effort codeword assignment from codeword->gene mapping.
                        codeword_map = dst_root.attrs.get("codeword_gene_mapping", [])
                        codeword_identity = np.zeros(n, dtype=np.uint16)
                        if isinstance(codeword_map, (list, tuple)) and len(codeword_map) > 0:
                            first_by_gene: dict[int, int] = {}
                            for cw, gi in enumerate(codeword_map):
                                gi_int = int(gi)
                                if gi_int not in first_by_gene:
                                    first_by_gene[gi_int] = int(cw)
                            for i, gi in enumerate(gene_idx.astype(np.int64, copy=False)):
                                codeword_identity[i] = np.uint16(first_by_gene.get(int(gi), 0))
                        merged["codeword_identity"] = np.stack(
                            [codeword_identity, np.full(n, np.iinfo(np.uint16).max, dtype=np.uint16)],
                            axis=1,
                        )

                        merged["valid"] = np.ones((n, 1), dtype=np.uint8)
                        merged["status"] = np.zeros((n, 1), dtype=np.uint8)

                        # quality_score: populate from parquet 'qv' column (Phred-scaled quality value)
                        if "qv" in table_df.columns:
                            qv = pd.to_numeric(table_df.loc[finite_loc, "qv"], errors="coerce").to_numpy(dtype=np.float64)
                            qv = np.nan_to_num(qv, nan=0.0).astype(np.float32, copy=False)
                            merged["quality_score"] = qv.reshape(-1, 1)
                        else:
                            merged["quality_score"] = np.zeros((n, 1), dtype=np.float32)

                        # uuid: encode transcript_id-like 64-bit identifier as two uint32 words.
                        # When transcript IDs are rebuilt, keep uuid in sync with rebuilt IDs.
                        if "transcript_id" in table_df.columns:
                            if new_tid64 is not None:
                                tid64 = new_tid64
                            else:
                                tid64 = pd.to_numeric(table_df.loc[finite_loc, "transcript_id"], errors="coerce").to_numpy(dtype=np.float64)
                                tid64 = np.nan_to_num(tid64, nan=0.0).astype(np.int64, copy=False)
                            uuid_lo = (tid64 & np.int64(0xFFFFFFFF)).astype(np.uint32, copy=False)
                            uuid_hi = ((tid64 >> np.int64(32)) & np.int64(0xFFFFFFFF)).astype(np.uint32, copy=False)
                            merged["uuid"] = np.stack([uuid_lo, uuid_hi], axis=1)
                        else:
                            merged["uuid"] = np.zeros((n, 2), dtype=np.uint32)

                        # Template metadata from first tile arrays
                        template_meta: dict[str, dict] = {}
                        try:
                            for tname in template_level_group.keys():
                                tg = template_level_group[tname]
                                for aname in tg.keys():
                                    anode = tg[aname]
                                    if not hasattr(anode, "shape") or not hasattr(anode, "dtype"):
                                        continue
                                    if aname not in template_meta:
                                        template_meta[aname] = {
                                            "dtype": anode.dtype,
                                            "chunks": getattr(anode, "chunks", None),
                                            "compressor": getattr(anode, "compressor", None),
                                            "compressors": getattr(anode, "compressors", None),
                                            "attrs": dict(anode.attrs),
                                            "shape": tuple(anode.shape),
                                        }
                                break
                        except Exception:
                            pass

                        for aname, meta in template_meta.items():
                            if aname in merged:
                                continue
                            shp = meta.get("shape", ())
                            if len(shp) == 0:
                                continue
                            tail = tuple(int(s) for s in shp[1:])
                            arr_shape = (n,) + tail
                            merged[aname] = np.zeros(arr_shape, dtype=meta.get("dtype", np.float32))

                        return merged, template_meta

                    def _collect_level_arrays(level_group):
                        level_arrays: dict[str, list[np.ndarray]] = {}
                        array_meta: dict[str, dict] = {}
                        for tile_name in list(level_group.keys()):
                            tile_group = level_group[tile_name]
                            if "location" not in tile_group:
                                continue
                            try:
                                location = tile_group["location"][:]
                            except Exception:
                                continue
                            if getattr(location, "ndim", 0) != 2 or location.shape[0] == 0:
                                continue

                            row_count = int(location.shape[0])
                            
                            tile_mask = None
                            if transcript_ids is not None:
                                if "id" not in tile_group:
                                    continue
                                try:
                                    tile_mask = _build_transcript_group_mask(tile_group["id"][:])
                                except Exception:
                                    tile_mask = None
                                if tile_mask is None or tile_mask.shape[0] != row_count:
                                    continue

                            for arr_name in tile_group.keys():
                                arr_node = tile_group[arr_name]
                                if not hasattr(arr_node, "shape") or not hasattr(arr_node, "dtype"):
                                    continue
                                try:
                                    arr_data = arr_node[:]
                                except Exception:
                                    continue
                                if getattr(arr_data, "ndim", 0) < 1 or int(arr_data.shape[0]) != row_count:
                                    continue

                                if tile_mask is not None:
                                    arr_data = arr_data[tile_mask, ...]
                                    if arr_data.shape[0] == 0:
                                        continue

                                level_arrays.setdefault(arr_name, []).append(arr_data)
                                if arr_name not in array_meta:
                                    array_meta[arr_name] = {
                                        "dtype": arr_data.dtype,
                                        "chunks": getattr(arr_node, "chunks", None),
                                        "compressor": getattr(arr_node, "compressor", None),
                                        "compressors": getattr(arr_node, "compressors", None),
                                        "attrs": dict(arr_node.attrs),
                                    }
                        if "location" not in level_arrays or len(level_arrays["location"]) == 0:
                            return None, None
                        merged = {name: np.concatenate(parts, axis=0) for name, parts in level_arrays.items()}
                        return merged, array_meta

                    def _prepare_level_for_rebuild(
                        level_arrays: dict[str, np.ndarray],
                        *,
                        write_remap: bool = False,
                    ) -> dict[str, np.ndarray] | None:
                        nonlocal recomputed_fov_metadata
                        if level_arrays is None or "location" not in level_arrays:
                            return None

                        location = level_arrays["location"]
                        if getattr(location, "ndim", 0) != 2 or location.shape[0] == 0:
                            return None

                        x = location[:, 0].astype(np.float64, copy=False)
                        y = location[:, 1].astype(np.float64, copy=False)
                        finite_mask = np.isfinite(x) & np.isfinite(y)
                        if not np.any(finite_mask):
                            return None

                        prepared = {
                            name: arr[finite_mask, ...]
                            for name, arr in level_arrays.items()
                        }

                        loc = prepared.get("location")
                        if loc is None or loc.shape[0] == 0:
                            return None
                        xx = loc[:, 0].astype(np.float64, copy=False)
                        yy = loc[:, 1].astype(np.float64, copy=False)
                        fov_xx = xx
                        fov_yy = yy
                        if origin_xy is not None:
                            x0, y0 = origin_xy
                            fov_xx = xx - float(x0)
                            fov_yy = yy - float(y0)

                        if "id" in prepared:
                            recomputed_fov_series, fov_meta = calculate_fov_layout_and_assignments(
                                fov_xx,
                                fov_yy,
                                pixel_size_um=float(effective_pixel_size_um),
                                fov_rows_px=int(fov_rows_px),
                                fov_cols_px=int(fov_cols_px),
                                overlap_px=int(fov_overlap_px),
                            )
                            candidate_fov_metadata = {
                                str(k): int(v) for k, v in fov_meta.items()
                            }
                            if (
                                recomputed_fov_metadata is None
                                or int(candidate_fov_metadata.get("number_fovs", 0))
                                >= int(recomputed_fov_metadata.get("number_fovs", 0))
                            ):
                                recomputed_fov_metadata = candidate_fov_metadata
                            new_fov = recomputed_fov_series.to_numpy(dtype=np.uint32, copy=False)

                            new_low = np.zeros(new_fov.shape[0], dtype=np.uint32)
                            for fov_val in np.unique(new_fov.astype(np.int64, copy=False)):
                                mask_fov = new_fov == np.uint32(fov_val)
                                count = int(np.count_nonzero(mask_fov))
                                if count > 0:
                                    new_low[mask_fov] = np.arange(1, count + 1, dtype=np.uint32)

                            old_id = prepared["id"].astype(np.uint32, copy=False)
                            if old_id.ndim == 2 and old_id.shape[1] >= 2 and write_remap and output_path.name.lower() == "transcripts.zarr.zip":
                                one_shift_48 = (np.int64(1) << np.int64(48))
                                old_low = old_id[:, 0].astype(np.int64, copy=False)
                                old_fov = old_id[:, 1].astype(np.int64, copy=False)
                                old_tid64 = (
                                    one_shift_48
                                    + (old_fov << np.int64(32))
                                    + old_low
                                ).astype(np.int64, copy=False)
                                new_tid64 = (
                                    one_shift_48
                                    + (new_fov.astype(np.int64, copy=False) << np.int64(32))
                                    + new_low.astype(np.int64, copy=False)
                                ).astype(np.int64, copy=False)
                                remap_path = output_path.with_name("transcripts_id_fov_remap.csv.gz")
                                remap_df = pd.DataFrame(
                                    {
                                        "old_transcript_id": old_tid64,
                                        "old_fov": old_id[:, 1].astype(np.uint32, copy=False),
                                        "old_local_id": old_id[:, 0].astype(np.uint32, copy=False),
                                        "new_transcript_id": new_tid64,
                                        "new_fov": new_fov,
                                        "new_local_id": new_low,
                                    }
                                )
                                remap_df.to_csv(remap_path, index=False, compression="gzip")
                                logger.info(
                                    "Wrote transcript ID/FOV remap log: %s (%d rows)",
                                    remap_path,
                                    len(remap_df),
                                )

                            prepared["id"] = np.stack([new_low, new_fov], axis=1)
                            canonical_u64 = (
                                np.uint64(1 << 48)
                                + (new_fov.astype(np.uint64, copy=False) << np.uint64(32))
                                + new_low.astype(np.uint64, copy=False)
                            )
                            prepared["uuid"] = np.stack(
                                [
                                    (canonical_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32, copy=False),
                                    ((canonical_u64 >> np.uint64(32)) & np.uint64(0xFFFFFFFF)).astype(np.uint32, copy=False),
                                ],
                                axis=1,
                            )

                        return prepared

                    level_inputs: dict[str, tuple[dict[str, np.ndarray], dict[str, dict]]] = {}
                    for level_name in level_names:
                        src_level = str(level_name)
                        if src_level not in grids_group:
                            continue
                        merged_level, array_meta_level = _collect_level_arrays(grids_group[src_level])
                        if merged_level is None or array_meta_level is None:
                            continue
                        prepared_level = _prepare_level_for_rebuild(
                            merged_level,
                            write_remap=(src_level == "0"),
                        )
                        if prepared_level is None:
                            continue
                        level_inputs[src_level] = (prepared_level, array_meta_level)

                    if not level_inputs:
                        return

                    for level_idx, level_name in enumerate(level_names):
                        level_name = str(level_name)
                        if level_name not in level_inputs:
                            rebuilt_levels.append((level_name, []))
                            grid_keys_by_level.append([])
                            grid_counts_by_level.append([])
                            continue

                        level_source, level_meta = level_inputs[level_name]
                        tile_size = _tile_size_for_level(level_name, level_idx)

                        locations_for_tiling = level_source["location"].copy()
                        if origin_xy is not None:
                            x0, y0 = origin_xy
                            locations_for_tiling[:, 0] -= x0
                            locations_for_tiling[:, 1] -= y0

                        gx = np.floor(locations_for_tiling[:, 0] / tile_size).astype(np.int64, copy=False)
                        gy = np.floor(locations_for_tiling[:, 1] / tile_size).astype(np.int64, copy=False)
                        gx = np.maximum(gx, 0)
                        gy = np.maximum(gy, 0)

                        coords = np.stack([gx, gy], axis=1)
                        uniq_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

                        level_tiles: list[tuple[str, dict[str, np.ndarray], dict[str, dict]]] = []
                        level_keys: list[str] = []
                        level_counts: list[int] = []

                        for i, (tx, ty) in enumerate(uniq_coords):
                            row_mask = inverse == i
                            tile_key = f"{int(tx)},{int(ty)}"
                            tile_arrays = {
                                name: arr[row_mask, ...]
                                for name, arr in level_source.items()
                            }
                            tile_count = int(tile_arrays["location"].shape[0])
                            if tile_count > 0:
                                if origin_xy is not None and "location" in tile_arrays:
                                    x0, y0 = origin_xy
                                    loc = tile_arrays["location"].copy()
                                    loc[:, 0] -= x0
                                    loc[:, 1] -= y0
                                    tile_arrays["location"] = loc

                                loc = tile_arrays.get("location")
                                if loc is not None and getattr(loc, "ndim", 0) == 2 and loc.shape[0] > 1:
                                    if loc.shape[1] >= 3:
                                        order = np.lexsort((loc[:, 2], loc[:, 0], loc[:, 1]))
                                    else:
                                        order = np.lexsort((loc[:, 0], loc[:, 1]))
                                    if not np.array_equal(order, np.arange(loc.shape[0])):
                                        tile_arrays = {
                                            name: arr[order, ...]
                                            for name, arr in tile_arrays.items()
                                        }

                                level_tiles.append((tile_key, tile_arrays, level_meta))
                                level_keys.append(tile_key)
                                level_counts.append(tile_count)

                        if level_name == "0":
                            level0_total = int(sum(level_counts))

                        rebuilt_levels.append((level_name, level_tiles))
                        grid_keys_by_level.append(level_keys)
                        grid_counts_by_level.append(level_counts)

                    # Grids was not copied (skipped in _copy_group), so create it fresh.
                    # No delete needed — ZipStore is write-once and can't delete, but since
                    # we never wrote grids keys, we can create them without conflict.
                    # Compatibility normalization: remap active FOV ids to dense [0..N-1]
                    # across all rebuilt levels so loaders that expect dense spaces can
                    # render all transcripts reliably.
                    applied_fov_remap: dict[int, int] = {}
                    if len(rebuilt_levels) > 0:
                        level0_tiles = rebuilt_levels[0][1]
                        used_fov_sorted: list[int] = []
                        for _tile_key, tile_arrays, _meta in level0_tiles:
                            id_arr = tile_arrays.get("id")
                            if (
                                id_arr is None
                                or getattr(id_arr, "ndim", 0) != 2
                                or id_arr.shape[1] < 2
                                or id_arr.shape[0] == 0
                            ):
                                continue
                            used_vals = np.unique(id_arr[:, 1].astype(np.int64, copy=False))
                            used_fov_sorted.extend(int(v) for v in used_vals.tolist() if int(v) >= 0)

                        if used_fov_sorted:
                            used_fov_sorted = sorted(set(used_fov_sorted))
                            fov_remap = {old: new for new, old in enumerate(used_fov_sorted)}
                            applied_fov_remap = dict(fov_remap)
                            max_old_fov = int(used_fov_sorted[-1])
                            fov_lut = np.full(max_old_fov + 1, -1, dtype=np.int64)
                            for old, new in fov_remap.items():
                                fov_lut[int(old)] = int(new)

                            for _level_name, level_tiles in rebuilt_levels:
                                for _tile_key, tile_arrays, _meta in level_tiles:
                                    id_arr = tile_arrays.get("id")
                                    if (
                                        id_arr is None
                                        or getattr(id_arr, "ndim", 0) != 2
                                        or id_arr.shape[1] < 2
                                        or id_arr.shape[0] == 0
                                    ):
                                        continue

                                    id_arr = id_arr.copy()
                                    fov_old = id_arr[:, 1].astype(np.int64, copy=False)
                                    clipped = np.clip(fov_old, 0, max_old_fov)
                                    mapped = np.where(
                                        (fov_old >= 0) & (fov_old <= max_old_fov) & (fov_lut[clipped] >= 0),
                                        fov_lut[clipped],
                                        fov_old,
                                    ).astype(np.uint32, copy=False)
                                    id_arr[:, 1] = mapped
                                    tile_arrays["id"] = id_arr

                                    uuid_arr = tile_arrays.get("uuid")
                                    if (
                                        uuid_arr is not None
                                        and getattr(uuid_arr, "ndim", 0) == 2
                                        and uuid_arr.shape[1] >= 2
                                        and uuid_arr.shape[0] == id_arr.shape[0]
                                    ):
                                        low_u64 = id_arr[:, 0].astype(np.uint64, copy=False)
                                        fov_u64 = id_arr[:, 1].astype(np.uint64, copy=False)
                                        canonical_u64 = (
                                            np.uint64(1 << 48)
                                            + (fov_u64 << np.uint64(32))
                                            + low_u64
                                        )
                                        tile_arrays["uuid"] = np.stack(
                                            [
                                                (canonical_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32, copy=False),
                                                ((canonical_u64 >> np.uint64(32)) & np.uint64(0xFFFFFFFF)).astype(np.uint32, copy=False),
                                            ],
                                            axis=1,
                                        )

                            # Always emit remap sidecar for traceability (identity or compacted).
                            remap_path = output_path.with_name("old_fov_to_new_fov.csv")
                            old_name_attr = dst_root.attrs.get("fov_names", [])
                            if isinstance(old_name_attr, (list, tuple)):
                                old_name_list = [str(v) for v in old_name_attr]
                            else:
                                old_name_list = []

                            remap_df = pd.DataFrame(
                                {
                                    "old_fov": used_fov_sorted,
                                    "new_fov": [fov_remap[v] for v in used_fov_sorted],
                                    "old_fov_name": [old_name_list[v] if 0 <= int(v) < len(old_name_list) else "" for v in used_fov_sorted],
                                }
                            )
                            remap_df.to_csv(remap_path, index=False)
                            logger.info(
                                "Wrote FOV remap log: %s (%d rows)",
                                remap_path,
                                len(remap_df),
                            )

                    new_grids = dst_root.require_group("grids")

                    for level_name, level_tiles in rebuilt_levels:
                        level_group = new_grids.require_group(level_name)
                        for tile_key, tile_arrays, meta in level_tiles:
                            # Safety invariant: keep uuid consistent with id for every rebuilt tile.
                            # canonical64 = (1<<48) + (id[:,1]<<32) + id[:,0]
                            id_arr = tile_arrays.get("id")
                            uuid_arr = tile_arrays.get("uuid")
                            if (
                                id_arr is not None
                                and uuid_arr is not None
                                and getattr(id_arr, "ndim", 0) == 2
                                and getattr(uuid_arr, "ndim", 0) == 2
                                and id_arr.shape[1] >= 2
                                and uuid_arr.shape[1] >= 2
                                and id_arr.shape[0] == uuid_arr.shape[0]
                            ):
                                low_u64 = id_arr[:, 0].astype(np.uint64, copy=False)
                                fov_u64 = id_arr[:, 1].astype(np.uint64, copy=False)
                                canonical_u64 = (
                                    np.uint64(1 << 48)
                                    + (fov_u64 << np.uint64(32))
                                    + low_u64
                                )
                                tile_arrays["uuid"] = np.stack(
                                    [
                                        (canonical_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32, copy=False),
                                        ((canonical_u64 >> np.uint64(32)) & np.uint64(0xFFFFFFFF)).astype(np.uint32, copy=False),
                                    ],
                                    axis=1,
                                )

                            tile_group = level_group.require_group(tile_key)
                            for arr_name, arr_data in tile_arrays.items():
                                m = meta.get(arr_name, {})
                                create_kwargs = {
                                    "shape": arr_data.shape,
                                    "dtype": arr_data.dtype,
                                    "data": arr_data,
                                }
                                if getattr(arr_data, "ndim", 0) >= 1:
                                    create_kwargs["chunks"] = (int(arr_data.shape[0]),) + tuple(
                                        int(s) for s in arr_data.shape[1:]
                                    )
                                compressors = m.get("compressors")
                                compressor = m.get("compressor")
                                if compressors is not None:
                                    create_kwargs["compressors"] = compressors
                                elif compressor is not None:
                                    create_kwargs["compressor"] = compressor

                                try:
                                    dst_arr = _zarr_create_dataset(zarr, tile_group, arr_name, **create_kwargs)
                                except TypeError:
                                    create_kwargs.pop("compressors", None)
                                    create_kwargs.pop("compressor", None)
                                    dst_arr = _zarr_create_dataset(zarr, tile_group, arr_name, **create_kwargs)

                                for ak, av in m.get("attrs", {}).items():
                                    dst_arr.attrs[ak] = av



                    # Recompute grids attrs
                    new_grid_attrs = dict(source_grid_attrs)
                    new_grid_attrs["number_levels"] = len(rebuilt_levels)
                    new_grid_attrs["grid_keys"] = grid_keys_by_level
                    new_grid_attrs["grid_number_objects"] = grid_counts_by_level
                    new_grid_attrs["grid_array_shapes"] = [[{} for _ in keys] for keys in grid_keys_by_level]
                    new_grid_attrs.setdefault("grid_key_names", ["grid_x_loc", "grid_y_loc"])
                    new_grid_attrs.setdefault("grid_size", [base_grid_size])
                    new_grid_attrs.setdefault("grid_zip", False)
                    for ak, av in new_grid_attrs.items():
                        new_grids.attrs[ak] = av

                    # Update root-level FOV metadata.
                    # Use the actually referenced level-0 id[:,1] indices as source of truth.
                    # This trims unnecessary trailing fov_names while preserving index mapping.
                    new_fov_metadata = {}
                    used_fov_indices: set[int] = set()
                    level0_grid_cols: int | None = None
                    if len(rebuilt_levels) > 0:
                        level0_tiles = rebuilt_levels[0][1]  # (level_name, level_tiles) -> level_tiles
                        level0_coords: list[tuple[int, int]] = []
                        for _tile_key, tile_arrays, _meta in level0_tiles:
                            try:
                                tx, ty = map(int, str(_tile_key).split(","))
                                level0_coords.append((tx, ty))
                            except Exception:
                                pass
                            id_arr = tile_arrays.get("id")
                            if id_arr is None or getattr(id_arr, "ndim", 0) != 2 or id_arr.shape[1] < 2 or id_arr.shape[0] == 0:
                                continue
                            vals = np.unique(id_arr[:, 1].astype(np.int64, copy=False))
                            used_fov_indices.update(int(v) for v in vals.tolist() if int(v) >= 0)
                        if level0_coords:
                            level0_grid_cols = max((c[0] for c in level0_coords), default=0) + 1

                    if used_fov_indices:
                        max_used = max(used_fov_indices)
                        total_fovs = int(max_used + 1)
                        raw_fov_names = dst_root.attrs.get("fov_names", [])
                        if isinstance(raw_fov_names, (list, tuple)):
                            old_fov_names = [str(v) for v in raw_fov_names]
                        else:
                            old_fov_names = []
                        generated_fov_names = _generate_fov_names(total_fovs, level0_grid_cols)
                        new_fov_names = list(generated_fov_names)

                        # If sparse old FOV ids were compacted to dense ids, remap the names
                        # to keep name semantics aligned with the rewritten id[:,1] values.
                        if applied_fov_remap and old_fov_names:
                            for old_idx, new_idx in applied_fov_remap.items():
                                if 0 <= int(new_idx) < total_fovs and 0 <= int(old_idx) < len(old_fov_names):
                                    new_fov_names[int(new_idx)] = old_fov_names[int(old_idx)]
                        elif old_fov_names:
                            for i in range(min(total_fovs, len(old_fov_names))):
                                new_fov_names[i] = old_fov_names[i]

                        dst_root.attrs["number_fovs"] = total_fovs
                        dst_root.attrs["fov_names"] = new_fov_names
                        new_fov_metadata["number_fovs"] = total_fovs
                        new_fov_metadata["fov_names"] = new_fov_names
                    elif recomputed_fov_metadata is not None:
                        total_fovs = int(recomputed_fov_metadata.get("number_fovs", 0))
                        if total_fovs > 0:
                            grid_cols = int(recomputed_fov_metadata.get("fov_grid_cols", 0) or 0)
                            fov_names = _generate_fov_names(total_fovs, grid_cols)
                            dst_root.attrs["number_fovs"] = total_fovs
                            dst_root.attrs["fov_names"] = fov_names
                            new_fov_metadata["number_fovs"] = total_fovs
                            new_fov_metadata["fov_names"] = fov_names
                    elif len(rebuilt_levels) > 0:
                        level0_tiles = rebuilt_levels[0][1]  # (level_name, level_tiles) -> level_tiles
                        if len(level0_tiles) > 0:
                            level0_coords = [tuple(map(int, tile_key.split(","))) for tile_key, _, _ in level0_tiles]
                            max_x = max((c[0] for c in level0_coords), default=0)
                            max_y = max((c[1] for c in level0_coords), default=0)
                            total_fovs = (max_y + 1) * (max_x + 1)
                            fov_names = _generate_fov_names(int(total_fovs), int(max_x + 1))
                            dst_root.attrs["number_fovs"] = int(total_fovs)
                            dst_root.attrs["fov_names"] = fov_names
                            new_fov_metadata["number_fovs"] = int(total_fovs)
                            new_fov_metadata["fov_names"] = fov_names
                    

                    # FOV compatibility normalization is applied above before writing tiles.

                    # Recompute density/gene CSR from level 0 locations and gene identities.
                    if "density" in dst_root and "gene" in dst_root["density"] and "0" in new_grids:
                        try:
                            density_gene = dst_root["density"]["gene"]
                            d_attrs = dict(density_gene.attrs)
                            gene_names = d_attrs.get("gene_names", [])
                            n_genes = int(len(gene_names))
                            grid_sz = d_attrs.get("grid_size", [10.0, 10.0])
                            dx = float(grid_sz[0]) if isinstance(grid_sz, (list, tuple)) and len(grid_sz) > 0 else 10.0
                            dy = float(grid_sz[1]) if isinstance(grid_sz, (list, tuple)) and len(grid_sz) > 1 else dx
                            origin = d_attrs.get("origin", {"x": 0.0, "y": 0.0})
                            ox = float(origin.get("x", 0.0)) if isinstance(origin, dict) else 0.0
                            oy = float(origin.get("y", 0.0)) if isinstance(origin, dict) else 0.0

                            # After coordinate rebasing into crop-local space, density origin should be local as well.
                            if rebase_region is not None:
                                ox = 0.0
                                oy = 0.0

                            target_rows = None
                            target_cols = None
                            if rebase_region is not None:
                                min_x, min_y, max_x, max_y = rebase_region.bounds
                                if pixel_size_um is not None and pixel_size_um > 0:
                                    min_x = max(int(math.floor(min_x / pixel_size_um)), 0) * pixel_size_um
                                    min_y = max(int(math.floor(min_y / pixel_size_um)), 0) * pixel_size_um
                                    max_x = int(math.ceil(max_x / pixel_size_um)) * pixel_size_um
                                    max_y = int(math.ceil(max_y / pixel_size_um)) * pixel_size_um
                                width_um = max(0.0, float(max_x) - float(min_x))
                                height_um = max(0.0, float(max_y) - float(min_y))
                                target_cols = max(1, int(math.ceil(width_um / dx))) if dx > 0 else None
                                target_rows = max(1, int(math.ceil(height_um / dy))) if dy > 0 else None

                            x_parts: list[np.ndarray] = []
                            y_parts: list[np.ndarray] = []
                            g_parts: list[np.ndarray] = []
                            v_parts: list[np.ndarray] = []
                            s_parts: list[np.ndarray] = []

                            for tile_key in new_grids["0"].keys():
                                tg = new_grids["0"][tile_key]
                                if "location" not in tg or "gene_identity" not in tg:
                                    continue
                                loc = tg["location"][:]
                                gi = tg["gene_identity"][:]
                                if loc.shape[0] == 0:
                                    continue
                                x_parts.append(loc[:, 0])
                                y_parts.append(loc[:, 1])
                                g_parts.append(gi[:, 0] if gi.ndim == 2 else gi)
                                if "valid" in tg:
                                    vv = tg["valid"][:]
                                    v_parts.append(vv[:, 0] if vv.ndim == 2 else vv)
                                if "status" in tg:
                                    st = tg["status"][:]
                                    s_parts.append(st[:, 0] if st.ndim == 2 else st)

                            if x_parts and n_genes > 0:
                                x = np.concatenate(x_parts).astype(np.float64, copy=False)
                                y = np.concatenate(y_parts).astype(np.float64, copy=False)
                                gene = np.concatenate(g_parts).astype(np.int64, copy=False)
                                valid_mask = np.isfinite(x) & np.isfinite(y) & (gene >= 0) & (gene < n_genes) & (gene != 65535)
                                if v_parts:
                                    v = np.concatenate(v_parts)
                                    valid_mask &= (v != 0)
                                if s_parts:
                                    st = np.concatenate(s_parts)
                                    valid_mask &= (st == 0)

                                x = x[valid_mask]
                                y = y[valid_mask]
                                gene = gene[valid_mask]

                                if x.size > 0:
                                    gx = np.floor((x - ox) / dx).astype(np.int64, copy=False)
                                    gy = np.floor((y - oy) / dy).astype(np.int64, copy=False)

                                    if target_cols is not None and target_rows is not None:
                                        cols = int(target_cols)
                                        rows = int(target_rows)
                                        keep = (gx >= 0) & (gy >= 0) & (gx < cols) & (gy < rows)
                                    else:
                                        keep = (gx >= 0) & (gy >= 0)

                                    gx = gx[keep]
                                    gy = gy[keep]
                                    gene = gene[keep]

                                    if gx.size > 0:
                                        if target_cols is None or target_rows is None:
                                            cols = int(np.max(gx)) + 1
                                            rows = int(np.max(gy)) + 1

                                        row_index = gene * rows + gy
                                        packed = row_index * cols + gx
                                        uniq, counts = np.unique(packed, return_counts=True)
                                        row_u = (uniq // cols).astype(np.int64, copy=False)
                                        col_u = (uniq % cols).astype(np.int64, copy=False)

                                        n_rows_total = int(n_genes * rows)
                                        binc = np.bincount(row_u, minlength=n_rows_total)
                                        indptr = np.empty(n_rows_total + 1, dtype=np.uint32)
                                        indptr[0] = 0
                                        indptr[1:] = np.cumsum(binc, dtype=np.uint64).astype(np.uint32, copy=False)

                                        idx_dtype = np.uint16 if cols <= np.iinfo(np.uint16).max else np.uint32
                                        data_dtype = np.uint16 if int(np.max(counts)) <= np.iinfo(np.uint16).max else np.uint32
                                        indices = col_u.astype(idx_dtype, copy=False)
                                        data = counts.astype(data_dtype, copy=False)
                                    else:
                                        rows = int(target_rows) if target_rows is not None else 1
                                        cols = int(target_cols) if target_cols is not None else 1
                                        indptr = np.zeros(n_genes * rows + 1, dtype=np.uint32)
                                        indices = np.zeros(0, dtype=np.uint16)
                                        data = np.zeros(0, dtype=np.uint16)
                                else:
                                    rows, cols = 1, 1
                                    indptr = np.zeros(n_genes * rows + 1, dtype=np.uint32)
                                    indices = np.zeros(0, dtype=np.uint16)
                                    data = np.zeros(0, dtype=np.uint16)
                            else:
                                rows, cols = 1, 1
                                indptr = np.zeros(max(1, n_genes) + 1, dtype=np.uint32)
                                indices = np.zeros(0, dtype=np.uint16)
                                data = np.zeros(0, dtype=np.uint16)

                            for arr_name in ["data", "indices", "indptr"]:
                                if arr_name in density_gene:
                                    del density_gene[arr_name]

                            _zarr_create_dataset(zarr, density_gene, "data", shape=data.shape, dtype=data.dtype, data=data, chunks=(min(max(1, data.shape[0]), 2_000_000),))
                            _zarr_create_dataset(zarr, density_gene, "indices", shape=indices.shape, dtype=indices.dtype, data=indices, chunks=(min(max(1, indices.shape[0]), 2_000_000),))
                            _zarr_create_dataset(zarr, density_gene, "indptr", shape=indptr.shape, dtype=indptr.dtype, data=indptr, chunks=(indptr.shape[0],))

                            density_gene.attrs["rows"] = int(rows)
                            density_gene.attrs["cols"] = int(cols)
                            density_gene.attrs["origin"] = {"x": float(ox), "y": float(oy)}
                        except Exception:
                            logger.debug("Failed to rebuild transcript density/gene; keeping copied values", exc_info=True)

                _copy_group(src_root, dst_root)
                if transcript_ids is not None:
                    try:
                        dst_root.attrs["number_rnas"] = int(transcript_ids.size)
                    except Exception:
                        logger.debug("Failed to update number_rnas for filtered transcript zarr", exc_info=True)
                if rebase_region is not None:
                    try:
                        # Keep coordinate_space unchanged from source metadata.
                        # Rebased coordinates are still written in micron units.
                        dst_root.attrs["spatial_units"] = "micron"
                    except Exception:
                        logger.debug("Failed to update coordinate space metadata after rebasing", exc_info=True)
                # Rebuild transcript grids after filtering so tile keys and density
                # align with the written transcript coordinates at every level.
                if transcript_table is not None or transcript_ids is not None:
                    try:
                        _rebuild_transcript_grids_and_density(dst_root)
                    except Exception:
                        logger.debug("Failed to rebuild transcript grids and density", exc_info=True)


                # Re-apply FOV metadata after rebuild.
                # Prefer actual written level-0 transcript ids when available.
                try:
                    if "grids" in dst_root and "0" in dst_root["grids"]:
                        level0_tiles = sorted(dst_root["grids"]["0"].keys())
                        if len(level0_tiles) > 0:
                            used_fov_indices: set[int] = set()
                            for tile_key in level0_tiles:
                                tile = dst_root["grids"]["0"][tile_key]
                                if "id" not in tile:
                                    continue
                                id_arr = tile["id"][:]
                                if getattr(id_arr, "ndim", 0) != 2 or id_arr.shape[1] < 2 or id_arr.shape[0] == 0:
                                    continue
                                used_fov_indices.update(
                                    int(v)
                                    for v in np.unique(id_arr[:, 1].astype(np.int64, copy=False)).tolist()
                                    if int(v) >= 0
                                )
                            if used_fov_indices:
                                total_fovs = int(max(used_fov_indices) + 1)
                                dst_root.attrs["number_fovs"] = int(total_fovs)
                                dst_root.attrs["fov_names"] = _generate_fov_names(int(total_fovs))
                    elif recomputed_fov_metadata is not None and int(recomputed_fov_metadata.get("number_fovs", 0)) > 0:
                        total_fovs = int(recomputed_fov_metadata["number_fovs"])
                        grid_cols = int(recomputed_fov_metadata.get("fov_grid_cols", 0) or 0)
                        dst_root.attrs["number_fovs"] = total_fovs
                        dst_root.attrs["fov_names"] = _generate_fov_names(total_fovs, grid_cols)
                except Exception:
                    logger.debug("Failed to re-apply FOV metadata after rebuild", exc_info=True)

                # Ensure all required root attributes for Xenium Explorer are present
                try:
                    # Set default experiment metadata if not present
                    if "experiment_name" not in dst_root.attrs:
                        dst_root.attrs["experiment_name"] = "RnaDataset"
                    if "major_version" not in dst_root.attrs:
                        dst_root.attrs["major_version"] = 4
                    if "minor_version" not in dst_root.attrs:
                        dst_root.attrs["minor_version"] = 1
                    if "name" not in dst_root.attrs:
                        dst_root.attrs["name"] = "RnaDataset"
                    if "dataset_uuid" not in dst_root.attrs and "dataset_uuid" in src_root.attrs:
                        dst_root.attrs["dataset_uuid"] = src_root.attrs["dataset_uuid"]
                    # Ensure coordinate_space is set properly
                    if "coordinate_space" not in dst_root.attrs:
                        dst_root.attrs["coordinate_space"] = "refined-final_global_micron"
                except Exception:
                    logger.debug("Failed to set experiment metadata", exc_info=True)

                # Enforce transcript id/uuid schema invariants in output transcripts zarr.
                if output_path.name.lower() == "transcripts.zarr.zip":
                    schema_summary = validate_transcripts_id_uuid_schema(dst_root)
                    logger.debug(
                        "[filter_zarr_zip_preserve_schema] validated id/uuid schema: rows=%d tiles=%d max_id1=%d",
                        int(schema_summary.get("checked_rows", 0)),
                        int(schema_summary.get("checked_tiles", 0)),
                        int(schema_summary.get("max_id1", -1)),
                    )

            if dst_store is not None:
                try:
                    dst_store.close()
                except Exception:
                    logger.debug("Failed to close destination zarr ZipStore", exc_info=True)

            # ZipStore can contain duplicate member names after repeated metadata
            # rewrites; repack to a canonical ZIP with unique names.
            _dedupe_zip_entries_keep_last(tmp_output_zip)

            if output_path.exists():
                output_path.unlink()
            shutil.move(str(tmp_output_zip), str(output_path))
            tmp_output_zip = None
            logger.debug(
                "[filter_zarr_zip_preserve_schema] SUCCESS: output=%s size=%dB",
                output_path.name,
                output_path.stat().st_size if output_path.exists() else 0,
            )
            return True
        except Exception as e:
            logger.error(
                "Failed schema-preserving zarr filter for %s: %s",
                zarr_zip_path.name,
                e,
            )
            return False
        finally:
            if tmp_output_zip is not None and tmp_output_zip.exists():
                try:
                    tmp_output_zip.unlink()
                except Exception:
                    logger.debug("Failed to clean up temporary output zip %s", tmp_output_zip, exc_info=True)


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
    # Expected MTX layout:
    # - Header: %%MatrixMarket matrix coordinate integer general
    # - Metadata: %metadata_json: {...}
    # - Dimensions: num_features num_barcodes num_values
    # - Data rows: row col value (1-indexed)
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
    second compression pass and is much faster. After rezipping, deduplicates
    any duplicate entries (e.g., .zattrs) that may have accumulated during
    zarr group metadata rewrites.
    """
    with zipfile.ZipFile(dst_path, "w", zipfile.ZIP_STORED) as zf:
        for fpath in src_path.rglob("*"):
            if not fpath.is_file():
                continue

            rel = fpath.relative_to(src_path).as_posix()
            # Keep archive parity with Xenium source: per-level grids/<n>/.zattrs
            # are not present in source transcripts archives and can break viewers.
            if rel.startswith("grids/") and rel.endswith("/.zattrs"):
                parts = rel.split("/")
                if len(parts) == 3 and parts[1].isdigit():
                    continue

            zf.write(fpath, arcname=rel)
    
    # Deduplicate any .zattrs files or other duplicate entries in the zip
    _dedupe_zip_entries_keep_last(dst_path)


def _dedupe_zip_entries_keep_last(zip_path: Path) -> None:
    """Rewrite a ZIP file so each member name appears once (last entry wins).

    ZipStore can append duplicate members (for example repeated `.zattrs` writes).
    Some unzip tools materialize these as many sibling files, which breaks
    downstream consumers. This repack step keeps only the last version per name.
    """
    with zipfile.ZipFile(zip_path, "r") as zin:
        infos = zin.infolist()
        if not infos:
            return

        last_index_by_name: dict[str, int] = {}
        for idx, info in enumerate(infos):
            last_index_by_name[info.filename] = idx

        # Fast path: no duplicates.
        if len(last_index_by_name) == len(infos):
            return

        ordered_unique_infos = [
            info
            for idx, info in enumerate(infos)
            if last_index_by_name.get(info.filename) == idx
        ]

        with tempfile.NamedTemporaryFile(
            dir=zip_path.parent,
            prefix="dedupe_zip_",
            suffix=".zip",
            delete=False,
        ) as tmpf:
            tmp_zip = Path(tmpf.name)

        try:
            with zipfile.ZipFile(tmp_zip, "w") as zout:
                for info in ordered_unique_infos:
                    data = zin.read(info.filename)
                    new_info = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                    new_info.compress_type = info.compress_type
                    new_info.comment = info.comment
                    new_info.create_system = info.create_system
                    new_info.external_attr = info.external_attr
                    new_info.extra = info.extra
                    new_info.flag_bits = info.flag_bits
                    new_info.internal_attr = info.internal_attr
                    new_info.volume = info.volume
                    zout.writestr(new_info, data)

            zip_path.unlink(missing_ok=True)
            shutil.move(str(tmp_zip), str(zip_path))
        finally:
            if tmp_zip.exists():
                try:
                    tmp_zip.unlink()
                except Exception:
                    logger.debug("Failed to clean up temporary dedupe zip %s", tmp_zip, exc_info=True)


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


def _read_cfm_cell_features_array_specs(source_zarr_zip_path: Path | None) -> dict[str, dict]:
    """Read per-array zarr metadata from source cell_feature_matrix.zarr.zip.

    Returns a mapping for ``cell_features`` arrays keyed by array name, with
    optional keys: ``compressor``, ``compressors``, ``chunks``.
    """
    if source_zarr_zip_path is None or not source_zarr_zip_path.exists():
        return {}

    specs: dict[str, dict] = {}
    try:
        with zipfile.ZipFile(source_zarr_zip_path, "r") as zf:
            for key in ["cell_id", "data", "indices", "indptr"]:
                meta_name = f"cell_features/{key}/.zarray"
                if meta_name not in zf.namelist():
                    continue
                meta = json.loads(zf.read(meta_name))
                specs[key] = {
                    "compressor": meta.get("compressor"),
                    "compressors": meta.get("compressors"),
                    "chunks": tuple(meta.get("chunks", [])) if meta.get("chunks") is not None else None,
                    "shape": tuple(meta.get("shape", [])) if meta.get("shape") is not None else None,
                    "dimension_separator": meta.get("dimension_separator", None),
                }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read source CFM zarr specs from %s: %s", source_zarr_zip_path, exc)
        return {}

    return specs


def build_cell_feature_matrix_zarr_from_h5(
    filtered_h5_path: Path,
    output_zarr_zip_path: Path,
    source_zarr_zip_path: Path | None = None,
) -> tuple[int, int] | None:
    """Rebuild cell_feature_matrix.zarr.zip from a filtered cell_feature_matrix.h5.

    Returns:
        (num_features, num_cells) when successful, else None.
    """
    try:
        import h5py
        import numpy as np
        import zarr
    except ImportError:
        logger.warning("h5py/zarr not installed; cannot rebuild cell_feature_matrix.zarr.zip")
        return None

    t0 = time.perf_counter()
    logger.debug(
        "[build_cfm_zarr] START: input_h5=%s output_zarr=%s",
        filtered_h5_path.name,
        output_zarr_zip_path.name,
    )
    try:
        t_read = time.perf_counter()
        with h5py.File(filtered_h5_path, "r") as h5f:
            if "matrix" not in h5f:
                logger.error("[build_cfm_zarr] FAILED: no matrix group in H5")
                return None
            matrix = h5f["matrix"]
            required = {"barcodes", "data", "indices", "indptr", "shape", "features"}
            if not required.issubset(set(matrix.keys())):
                logger.error("[build_cfm_zarr] FAILED: missing required keys in matrix group")
                return None

            barcodes = matrix["barcodes"][:]
            data = matrix["data"][:]
            indices = matrix["indices"][:]
            indptr = matrix["indptr"][:]
            shape = matrix["shape"][:]
            feat_grp = matrix["features"]
            feature_ids = feat_grp["id"][:]
            # feature_keys = gene symbol ("name" in 10X H5); feature_types = assay type
            def _decode_arr(arr):
                return [x.decode("utf-8", errors="replace") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
            feature_keys = _decode_arr(feat_grp["name"][:]) if "name" in feat_grp else _decode_arr(feature_ids)
            feature_types = _decode_arr(feat_grp["feature_type"][:]) if "feature_type" in feat_grp else ["gene"] * len(feature_ids)

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

            # Normalize to Xenium-style fixed-width byte strings for stable .zarray metadata.
            barcodes_text = _decode_arr(barcodes)
            max_bc_len = max((len(x.encode("utf-8")) for x in barcodes_text), default=1)
            barcodes_fixed = np.asarray(barcodes_text, dtype=f"S{max_bc_len}")
            chunks_cell_id = (max(len(barcodes_fixed), 1),)

            # Canonicalize numeric arrays to uint32 and single full-length chunks.
            # This yields stable .zarray metadata like: dtype "<u4", chunks=[N], fill_value=0.
            data = np.asarray(data, dtype=np.uint32)
            indices = np.asarray(indices, dtype=np.uint32)
            indptr = np.asarray(indptr, dtype=np.uint32)
            chunks_data = (max(len(data), 1),)
            chunks_indices = (max(len(indices), 1),)
            chunks_indptr = (max(len(indptr), 1),)

        logger.info(
            "  H5 read: %.2fs  (%d features, %d cells, data chunks=%s)",
            time.perf_counter() - t_read,
            int(shape[0]),
            int(shape[1]),
            chunks_data,
        )

        output_zarr_zip_path.parent.mkdir(parents=True, exist_ok=True)
        t_zarr = time.perf_counter()
        source_specs = _read_cfm_cell_features_array_specs(source_zarr_zip_path)

        def _create_ds(grp, name, arr, chunks):
            """Create a zarr dataset, stripping unknown compression kwargs gracefully."""
            kwargs: dict = {"shape": arr.shape, "dtype": arr.dtype, "data": arr, "chunks": chunks}
            spec = source_specs.get(name, {})
            src_compressors = spec.get("compressors")
            src_compressor = spec.get("compressor")

            try:
                if src_compressors is not None:
                    return grp.create_dataset(name, compressors=src_compressors, **kwargs)
                if src_compressor is not None:
                    return grp.create_dataset(name, compressor=src_compressor, **kwargs)
                return grp.create_dataset(name, compressor=None, **kwargs)
            except TypeError:
                pass
            try:
                if src_compressors is not None:
                    return grp.create_dataset(name, compressors=src_compressors, **kwargs)
                if src_compressor is not None:
                    return grp.create_dataset(name, compressor=src_compressor, **kwargs)
                return grp.create_dataset(name, compressor=None, **kwargs)
            except TypeError:
                pass
            return grp.create_dataset(name, **kwargs)

        # Write to a temp DIRECTORY store then rezip — the same technique used by
        # filter_zarr_zip_by_row_indices_preserve_schema which is known to work.
        with tempfile.TemporaryDirectory(dir=_xenium_temp_root(), prefix="cfm_zarr_build_") as tmpdir_out:
            try:
                try:
                    root = zarr.open_group(tmpdir_out, mode="w", zarr_format=2)
                except TypeError:
                    root = zarr.open_group(tmpdir_out, mode="w")

                g = root.require_group("cell_features")
                _create_ds(g, "cell_id", barcodes_fixed, chunks_cell_id)
                _create_ds(g, "data",    data,     chunks_data)
                _create_ds(g, "indices", indices,  chunks_indices)
                _create_ds(g, "indptr",  indptr,   chunks_indptr)
                decoded_feature_ids = [
                    x.decode("utf-8", errors="replace") if isinstance(x, (bytes, bytearray)) else str(x)
                    for x in feature_ids
                ]
                g.attrs["feature_ids"] = decoded_feature_ids
                g.attrs["feature_keys"] = feature_keys
                g.attrs["feature_types"] = feature_types
                g.attrs["major_version"] = 3
                g.attrs["minor_version"] = 0
                g.attrs["number_cells"] = int(shape[1])
                g.attrs["number_features"] = int(shape[0])

                _rezip_zarr(Path(tmpdir_out), output_zarr_zip_path)
            except Exception as inner_exc:  # noqa: BLE001
                raise RuntimeError(f"zarr write to tmpdir failed: {inner_exc}") from inner_exc
        logger.info("  Zarr write+rezip: %.2fs", time.perf_counter() - t_zarr)

        logger.info(
            "[build_cfm_zarr] SUCCESS: %s in %.2fs total",
            output_zarr_zip_path.name,
            time.perf_counter() - t0,
        )
        return int(shape[0]), int(shape[1])
    except Exception as exc:  # noqa: BLE001
        logger.error("[build_cfm_zarr] FAILED: %s from %s: %s", output_zarr_zip_path.name, filtered_h5_path.name, exc, exc_info=True)
        return None


def build_cell_feature_matrix_zarr_from_sparse_bundle(
    filtered_cfm_dir: Path,
    output_zarr_zip_path: Path,
    source_cfm_dir: Path | None = None,
    source_zarr_zip_path: Path | None = None,
) -> tuple[int, int] | None:
    """Build ``cell_feature_matrix.zarr.zip`` from a filtered sparse bundle.

    Expects ``filtered_cfm_dir`` to contain at least:
    - barcodes file
    - matrix.mtx(.gz)
    - optional features file (for ``feature_ids`` attribute)

    Returns:
        (num_features, num_cells) on success, else None.
    """
    try:
        import numpy as np
        import zarr
    except ImportError:
        logger.warning("numpy/zarr not installed; cannot rebuild cell_feature_matrix.zarr.zip")
        return None

    files = get_cell_feature_matrix_files(filtered_cfm_dir)
    if not files["barcodes"] or not files["matrix"]:
        logger.error(
            "[build_cfm_zarr_sparse] FAILED: missing required files in %s",
            filtered_cfm_dir,
        )
        return None

    t0 = time.perf_counter()
    logger.debug(
        "[build_cfm_zarr_sparse] START: input_dir=%s output_zarr=%s",
        filtered_cfm_dir,
        output_zarr_zip_path,
    )

    try:
        barcodes = read_barcodes_file(files["barcodes"])
        data_rows, (num_features, num_cells, _num_values) = read_mtx_file(files["matrix"])

        if len(barcodes) != num_cells:
            logger.warning(
                "[build_cfm_zarr_sparse] barcodes/matrix mismatch: %d barcodes vs %d matrix cols; using matrix cols",
                len(barcodes),
                num_cells,
            )

        # Build source-compatible cell_id uint32[N,2]. First column is global cell index (1-based).
        # Use source_cfm_dir/barcodes as the global ordering reference when available.
        global_ids = np.arange(1, num_cells + 1, dtype=np.uint32)
        if source_cfm_dir is not None:
            src_files = get_cell_feature_matrix_files(source_cfm_dir)
            src_bc = src_files.get("barcodes")
            if src_bc is not None and src_bc.exists():
                source_barcodes = read_barcodes_file(src_bc)
                bc_to_global = {bc: i + 1 for i, bc in enumerate(source_barcodes)}
                mapped = [bc_to_global.get(bc, i + 1) for i, bc in enumerate(barcodes[:num_cells])]
                global_ids = np.asarray(mapped, dtype=np.uint32)

        cell_id = np.zeros((num_cells, 2), dtype=np.uint32)
        cell_id[:, 0] = global_ids
        cell_id[:, 1] = 1

        source_specs = _read_cfm_cell_features_array_specs(source_zarr_zip_path)
        source_indptr_shape = source_specs.get("indptr", {}).get("shape")
        if source_indptr_shape and len(source_indptr_shape) >= 1 and source_indptr_shape[0] >= 1:
            source_num_features = int(source_indptr_shape[0]) - 1
            if source_num_features > num_features:
                num_features = source_num_features

        # Build source-compatible CSR-like arrays by feature rows:
        # - indices are cell-column indices (0-based, subset-local)
        # - indptr length is num_features + 1
        row_data: list[list[int]] = [[] for _ in range(num_features)]
        row_indices: list[list[int]] = [[] for _ in range(num_features)]
        for row_1based, col_1based, value in data_rows:
            r = row_1based - 1
            c = col_1based - 1
            if 0 <= r < num_features and 0 <= c < num_cells:
                row_indices[r].append(c)
                row_data[r].append(value)

        indptr = np.zeros(num_features + 1, dtype=np.uint32)
        nnz = 0
        for r in range(num_features):
            nnz += len(row_data[r])
            indptr[r + 1] = nnz

        if nnz:
            data = np.empty(nnz, dtype=np.uint32)
            indices = np.empty(nnz, dtype=np.uint32)
            offset = 0
            for r in range(num_features):
                n = len(row_data[r])
                if n:
                    data[offset : offset + n] = row_data[r]
                    indices[offset : offset + n] = row_indices[r]
                    offset += n
        else:
            data = np.array([], dtype=np.uint32)
            indices = np.array([], dtype=np.uint32)

        feature_ids: list[str] = []
        feature_keys: list[str] = []
        feature_types: list[str] = []
        if files["features"] and files["features"].exists():
            open_fn = gzip.open if str(files["features"]).endswith(".gz") else open
            with open_fn(files["features"], "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if not parts or not parts[0]:
                        continue
                    feature_ids.append(parts[0])
                    feature_keys.append(parts[1] if len(parts) > 1 else parts[0])
                    feature_types.append(parts[2] if len(parts) > 2 else "gene")

        def _chunk_1d(n: int) -> tuple[int, ...]:
            return (max(n, 1),)

        def _create_ds_sparse(grp, name, arr, chunks):
            kwargs: dict = {"shape": arr.shape, "dtype": arr.dtype, "data": arr, "chunks": chunks}
            spec = source_specs.get(name, {})
            src_compressors = spec.get("compressors")
            src_compressor = spec.get("compressor")

            try:
                if src_compressors is not None:
                    return grp.create_dataset(name, compressors=src_compressors, **kwargs)
                if src_compressor is not None:
                    return grp.create_dataset(name, compressor=src_compressor, **kwargs)
                return grp.create_dataset(name, compressor=None, **kwargs)
            except TypeError:
                pass
            try:
                if src_compressors is not None:
                    return grp.create_dataset(name, compressors=src_compressors, **kwargs)
                if src_compressor is not None:
                    return grp.create_dataset(name, compressor=src_compressor, **kwargs)
                return grp.create_dataset(name, compressor=None, **kwargs)
            except TypeError:
                pass
            return grp.create_dataset(name, **kwargs)

        output_zarr_zip_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to temp DIRECTORY store then rezip (same approach as filter_zarr_zip_by_row_indices_preserve_schema)
        with tempfile.TemporaryDirectory(dir=_xenium_temp_root(), prefix="cfm_zarr_sparse_") as tmpdir_out:
            try:
                root = zarr.open_group(tmpdir_out, mode="w", zarr_format=2)
            except TypeError:
                root = zarr.open_group(tmpdir_out, mode="w")

            g = root.require_group("cell_features")
            _create_ds_sparse(g, "cell_id", cell_id, (max(num_cells, 1), 2))
            _create_ds_sparse(g, "data",    data,    _chunk_1d(len(data)))
            _create_ds_sparse(g, "indices", indices, _chunk_1d(len(indices)))
            _create_ds_sparse(g, "indptr",  indptr,  _chunk_1d(len(indptr)))
            g.attrs["feature_ids"] = feature_ids
            g.attrs["feature_keys"] = feature_keys
            g.attrs["feature_types"] = feature_types
            g.attrs["major_version"] = 3
            g.attrs["minor_version"] = 0
            g.attrs["number_cells"] = int(num_cells)
            g.attrs["number_features"] = int(num_features)

            _rezip_zarr(Path(tmpdir_out), output_zarr_zip_path)

        logger.info(
            "[build_cfm_zarr_sparse] SUCCESS: %s (%d features, %d cells) in %.2fs",
            output_zarr_zip_path.name,
            int(num_features),
            int(num_cells),
            time.perf_counter() - t0,
        )
        return int(num_features), int(num_cells)
    except Exception as exc:  # noqa: BLE001
        logger.error("[build_cfm_zarr_sparse] FAILED: %s", exc, exc_info=True)
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
        # Deduplicate any .zattrs files at the top level that ZipStore may have created
        _dedupe_zip_entries_keep_last(output_zarr_zip_path)

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
        # Deduplicate any .zattrs files at the top level that ZipStore may have created
        _dedupe_zip_entries_keep_last(output_zarr_zip_path)

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
