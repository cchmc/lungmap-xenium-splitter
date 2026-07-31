"""LASSO region parsing from multiple file formats.

Supported Formats:
1. GeoJSON (.geojson, .json): RFC 7946 FeatureCollection/Feature
   - Geometry: Point, LineString, Polygon, MultiPolygon
   - Properties: region_id, id, name used for region ID

2. Tabular (CSV/TSV):
   - Format A: region_id + polygon_wkt (WKT format)
   - Format B: region_id + x,y (vertices, one per row)
   - Format C: selection + x,y (selection ID instead of region_id)
   - Supports metadata: lines starting with # or quoted "#" are skipped

Coordinate System:
All regions are in coordinate space (typically micrometers) matching input
entity data. Conversion to pixel space happens during image processing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon, shape

from xenium_splitter.models import LassoRegion


def load_lasso_regions(lasso_path: Path) -> list[LassoRegion]:
    """Load LASSO regions from GeoJSON or tabular file.
    
    Detects file format based on extension and delegates to appropriate loader.
    
    Args:
        lasso_path: Path to GeoJSON, JSON, CSV, TSV, or TXT LASSO file
    
    Returns:
        List of LassoRegion objects, one per region
    
    Raises:
        ValueError: If file format unsupported or no valid regions found
    """
    suffixes = [s.lower() for s in lasso_path.suffixes]
    if any(s in {".geojson", ".json"} for s in suffixes):
        return _load_geojson_regions(lasso_path)
    if any(s in {".csv", ".tsv", ".txt"} for s in suffixes):
        return _load_tabular_regions(lasso_path)
    raise ValueError(f"Unsupported LASSO file format: {lasso_path.name}")


def _load_geojson_regions(lasso_path: Path) -> list[LassoRegion]:
    """Parse LASSO regions from a GeoJSON/JSON file.

    Handles FeatureCollection, a single Feature, or a bare geometry object.
    Each feature is converted to a single Polygon (MultiPolygon geometries are
    reduced to their largest member). Features with empty or invalid geometries
    are silently skipped.

    Args:
        lasso_path: Path to a ``.geojson`` or ``.json`` file.

    Returns:
        List of :class:`LassoRegion` objects, one per valid polygon feature.

    Raises:
        ValueError: If no valid polygon regions are found in the file.
    """
    payload = json.loads(lasso_path.read_text(encoding="utf-8"))
    regions: list[LassoRegion] = []

    if payload.get("type") == "FeatureCollection":
        features = payload.get("features", [])
    elif payload.get("type") == "Feature":
        features = [payload]
    else:
        features = [{"type": "Feature", "geometry": payload, "properties": {}}]

    for idx, feature in enumerate(features, start=1):
        geometry = feature.get("geometry")
        if not geometry:
            continue

        geom = shape(geometry)
        polygon = _normalize_to_polygon(geom)
        if polygon is None or polygon.is_empty:
            continue

        props = feature.get("properties") or {}
        region_id = str(
            props.get("region_id")
            or props.get("id")
            or props.get("name")
            or feature.get("id")
            or idx
        )
        regions.append(LassoRegion(region_id=region_id, polygon=polygon))

    if not regions:
        raise ValueError("No valid polygon regions found in LASSO GeoJSON/JSON file.")
    return regions


def _load_tabular_regions(lasso_path: Path) -> list[LassoRegion]:
    """Load LASSO regions from CSV/TSV file.
    
    Supports three tabular formats:
    1. region_id + polygon_wkt: WKT polygon strings
    2. region_id + x,y: Vertex coordinates (one per row, grouped by ID)
    3. selection + x,y: Alternative naming for vertex format
    
    Metadata Handling:
    Lines starting with # or quoted "#" are skipped before header detection.
    This allows for CSV files with metadata comments.
    
    Args:
        lasso_path: Path to CSV/TSV LASSO file
    
    Returns:
        List of LassoRegion objects reconstructed from tabular data
    
    Raises:
        ValueError: If required columns missing or no valid regions found
    """
    # Infer delimiter from extension
    ext = lasso_path.suffix.lower()
    if ext == ".tsv":
        sep = "\t"
    elif ext == ".csv":
        sep = ","
    else:
        sep = None  # let pandas sniff
    
    # Skip metadata lines (starting with # or quoted #)
    skip_rows = 0
    with open(lasso_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and empty lines
            if line.startswith("#") or (line.startswith('"#') and line.endswith('"')):
                skip_rows += 1
            elif not line:  # empty line
                skip_rows += 1
            else:
                # Found non-comment line (header)
                break
    
    df = pd.read_csv(lasso_path, sep=sep, engine="python" if sep is None else "c", skiprows=skip_rows)
    columns = {c.lower(): c for c in df.columns}

    if {"region_id", "polygon_wkt"}.issubset(columns):
        return _load_regions_from_wkt(df, columns)
    if {"region_id", "x", "y"}.issubset(columns):
        return _load_regions_from_points(df, columns)
    if {"selection", "x", "y"}.issubset(columns):
        return _load_regions_from_points(df, columns, "selection")
    raise ValueError(
        "Tabular LASSO file must include either region_id+polygon_wkt or region_id+x+y columns."
    )


def _load_regions_from_wkt(df: pd.DataFrame, columns: dict[str, str]) -> list[LassoRegion]:
    """Build LassoRegion objects from a DataFrame with ``region_id`` and ``polygon_wkt`` columns.

    Each row must contain a valid WKT polygon string.  Rows that produce an empty
    or un-normalizable geometry are skipped.

    Args:
        df: Input DataFrame.
        columns: Case-normalised column name mapping (``{lower_name: actual_name}``).

    Returns:
        List of :class:`LassoRegion` objects.

    Raises:
        ValueError: If no valid polygons can be parsed.
    """
    regions: list[LassoRegion] = []
    for _, row in df.iterrows():
        region_id = str(row[columns["region_id"]])
        polygon_wkt = row[columns["polygon_wkt"]]
        geom = wkt.loads(str(polygon_wkt))
        polygon = _normalize_to_polygon(geom)
        if polygon is None:
            continue
        regions.append(LassoRegion(region_id=region_id, polygon=polygon))

    if not regions:
        raise ValueError("No valid polygons parsed from region_id+polygon_wkt table.")
    return regions


def _load_regions_from_points(df: pd.DataFrame, columns: dict[str, str], region_id_col: str = "region_id") -> list[LassoRegion]:
    """Build LassoRegion objects from a DataFrame with per-vertex x/y coordinates.

    Rows are grouped by the region identifier column and each group is assembled
    into a :class:`shapely.geometry.Polygon`.  Groups with fewer than 3 points
    are skipped.  Invalid polygons are repaired with ``buffer(0)``.

    Args:
        df: Input DataFrame with at least a region ID column and x/y columns.
        columns: Case-normalised column name mapping (``{lower_name: actual_name}``).
        region_id_col: Lowercase name of the region identifier column
            (``"region_id"`` or ``"selection"``).

    Returns:
        List of :class:`LassoRegion` objects, one per valid polygon.

    Raises:
        ValueError: If no valid polygons can be assembled.
    """
    grouped = df.groupby(columns[region_id_col], sort=False)
    regions: list[LassoRegion] = []

    for region_id, group in grouped:
        points = list(zip(group[columns["x"]].astype(float), group[columns["y"]].astype(float)))
        if len(points) < 3:
            continue
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        polygon = _normalize_to_polygon(polygon)
        if polygon is None:
            continue
        regions.append(LassoRegion(region_id=str(region_id), polygon=polygon))

    if not regions:
        raise ValueError("No valid polygons parsed from region_id+x+y table.")
    return regions


def _normalize_to_polygon(geom: object) -> Polygon | None:
    """Return a Polygon from a geometry, or ``None`` if the conversion is not possible.

    For MultiPolygon inputs, the largest member by area is returned.  Any other
    geometry type (Point, LineString, etc.) returns ``None``.
    """
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        if len(geom.geoms) == 0:
            return None
        return max(geom.geoms, key=lambda p: p.area)
    return None
