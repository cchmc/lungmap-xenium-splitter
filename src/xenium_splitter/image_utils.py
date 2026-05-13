"""Image I/O and masking utilities for xenium_splitter.

Coordinate System:
The xenium_splitter uses a unified coordinate system where:
- Coordinate space: Micrometers (um), from experiment.xenium or raw polygon bounds
- Pixel space: Integer pixel indices in the cropped image
- Conversion: pixel = coordinate / pixel_size_um (when pixel_size_um provided)

Image Cropping and Masking:
When splitting regions:
1. Polygon bounds are converted to pixel space via _polygon_to_pixel_space()
2. Crop bounding box is computed with integer pixel bounds using _bbox_int()
3. Cropped region is extracted from the image
4. Polygon is translated to local (crop-relative) coordinates
5. Local mask is applied via _apply_local_mask()

Coordinate Rebasing:
Entity coordinates (cells, transcripts, nucleus) in tables must be rebased
to align with the cropped image:
- Crop origin: floor(polygon_min_bounds / pixel_size) in pixel space
- Entity coordinates -= origin (in coordinate units)
This alignment happens in io_utils.rebase_table_coordinates_to_region_crop().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def read_image(path: Path, squash_layers: bool = True) -> np.ndarray:
    lower_name = path.name.lower()
    if path.suffix.lower() == ".svs":
        return _read_svs(path)

    if lower_name.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        array = tifffile.imread(path)
        return _squash_if_needed(array, path=path) if squash_layers else np.asarray(array)

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def supports_windowed_region_read(path: Path) -> bool:
    lower_name = path.name.lower()
    return path.suffix.lower() == ".svs" or lower_name.endswith(
        (".tif", ".tiff", ".ome.tif", ".ome.tiff")
    )


def read_masked_cropped_region(
    path: Path,
    polygon: Polygon,
    pixel_size_um: float | None = None,
    squash_layers: bool = True,
) -> np.ndarray:
    if path.suffix.lower() == ".svs":
        return _read_masked_cropped_svs_region(path, polygon, pixel_size_um)

    lower_name = path.name.lower()
    if lower_name.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        return _read_masked_cropped_tiff_region(path, polygon, pixel_size_um, squash_layers)

    image = read_image(path, squash_layers=squash_layers)
    return mask_and_crop_region(image, polygon, pixel_size_um=pixel_size_um)


def _read_svs(path: Path) -> np.ndarray:
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError(
            "SVS reading requires openslide-python and OpenSlide shared libraries."
        ) from exc

    with openslide.OpenSlide(str(path)) as slide:
        width, height = slide.dimensions
        region = slide.read_region((0, 0), 0, (width, height))
        return np.asarray(region.convert("RGB"))


def _read_masked_cropped_svs_region(
    path: Path,
    polygon: Polygon,
    pixel_size_um: float | None,
) -> np.ndarray:
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError(
            "SVS reading requires openslide-python and OpenSlide shared libraries."
        ) from exc

    polygon_px = _polygon_to_pixel_space(polygon, pixel_size_um)
    min_x_i, min_y_i, max_x_i, max_y_i = _bbox_int(polygon_px.bounds)
    if min_x_i >= max_x_i or min_y_i >= max_y_i:
        return np.empty((0, 0), dtype=np.uint8)

    with openslide.OpenSlide(str(path)) as slide:
        width, height = slide.dimensions
        min_x_i = max(min_x_i, 0)
        min_y_i = max(min_y_i, 0)
        max_x_i = min(max_x_i, width)
        max_y_i = min(max_y_i, height)
        if min_x_i >= max_x_i or min_y_i >= max_y_i:
            return np.empty((0, 0), dtype=np.uint8)

        crop_w = max_x_i - min_x_i
        crop_h = max_y_i - min_y_i
        region = slide.read_region((min_x_i, min_y_i), 0, (crop_w, crop_h))
        image = np.asarray(region.convert("RGB"))
        local_polygon = affinity.translate(polygon_px, xoff=-min_x_i, yoff=-min_y_i)
        return _apply_local_mask(image, local_polygon)


def _read_masked_cropped_tiff_region(
    path: Path,
    polygon: Polygon,
    pixel_size_um: float | None,
    squash_layers: bool,
) -> np.ndarray:
    """Read TIFF region with masking, using full-read + crop (more reliable than zarr slicing)."""
    polygon_px = _polygon_to_pixel_space(polygon, pixel_size_um)
    min_x_i, min_y_i, max_x_i, max_y_i = _bbox_int(polygon_px.bounds)
    if min_x_i >= max_x_i or min_y_i >= max_y_i:
        return np.empty((0, 0), dtype=np.uint8)

    # Read full image with squashing, then crop to region bbox.
    # This is simpler and more reliable than trying to do windowed zarr reads.
    try:
        full = read_image(path, squash_layers=squash_layers)
    except Exception as e:
        logger.error(f"Failed to read TIFF {path.name}: {e}")
        raise

    h, w = full.shape[:2]
    min_x_i_safe = max(min_x_i, 0)
    min_y_i_safe = max(min_y_i, 0)
    max_x_i_safe = min(max_x_i, w)
    max_y_i_safe = min(max_y_i, h)
    
    if min_x_i_safe >= max_x_i_safe or min_y_i_safe >= max_y_i_safe:
        logger.warning(
            f"Crop bounds out of image range for {path.name}: "
            f"requested ({min_x_i}, {min_y_i}, {max_x_i}, {max_y_i}), "
            f"image shape ({h}, {w})"
        )
        return np.empty((0, 0), dtype=full.dtype)
    
    cropped = full[min_y_i_safe:max_y_i_safe, min_x_i_safe:max_x_i_safe]
    local_polygon = affinity.translate(polygon_px, xoff=-min_x_i_safe, yoff=-min_y_i_safe)
    masked = _apply_local_mask(cropped, local_polygon)
    
    if masked.size == 0 or np.all(masked == 0):
        logger.warning(f"Result is empty or all-black for {path.name} region bbox ({min_x_i_safe}, {min_y_i_safe}, {max_x_i_safe}, {max_y_i_safe})")
    
    return masked


def _squash_if_needed(array: np.ndarray, path: Path | None = None) -> np.ndarray:
    """Flatten multi-page TIFF arrays intelligently.
    
    For OME-TIFF with axes metadata, flatten based on what each axis represents.
    For Z-stacks: max projection. For time series: take first frame.
    Fallback: max projection for unknown multi-dimensional arrays.
    """
    arr = np.asarray(array)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr

    axes = _get_tiff_axes(path) if path else None
    if axes:
        return _squash_ome_by_axes(arr, axes)

    if arr.ndim == 3:
        return np.max(arr, axis=0)
    if arr.ndim == 4:
        return np.max(arr, axis=0)
    return np.squeeze(arr)


def _get_tiff_axes(path: Path) -> str | None:
    """Extract axes string from OME-TIFF metadata."""
    if not path.name.lower().endswith((".ome.tif", ".ome.tiff")):
        return None
    try:
        with tifffile.TiffFile(path) as tif:
            if tif.is_ome:
                series = tif.series[0]
                return series.axes
    except Exception:
        pass
    return None


def _squash_ome_by_axes(arr: np.ndarray, axes: str) -> np.ndarray:
    """Flatten multi-dimensional OME-TIFF using axis knowledge.
    
    Axes: Z=depth, T=time, C=channel, Y=row, X=column.
    Keeps Y, X (spatial) and final C if RGB/RGBA.
    Takes max projection for Z and T, collapses other dimensions.
    """
    if arr.ndim <= 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr

    axes_lower = axes.lower()
    keep_axes: list[int] = []
    for i, ax in enumerate(axes_lower):
        is_spatial = ax in {"y", "x"}
        is_color_channel = ax == "c" and i == len(axes_lower) - 1 and arr.shape[i] in (3, 4)
        if is_spatial or is_color_channel:
            keep_axes.append(i)

    if len(keep_axes) == arr.ndim:
        return arr

    result = arr
    for i in range(arr.ndim - 1, -1, -1):
        if i not in keep_axes:
            result = np.max(result, axis=i)

    return result


def mask_and_crop_region(image: np.ndarray, polygon: Polygon, pixel_size_um: float | None = None) -> np.ndarray:
    """Mask and crop image to polygon bounds.
    
    Workflow:
    1. Convert polygon from coordinate space to pixel space using pixel_size_um
    2. Compute integer bounding box and clamp to image dimensions
    3. Crop image to bounding box
    4. Translate polygon to local (crop-relative) coordinates
    5. Apply binary mask (polygon interior = keep, exterior = 0)
    
    Args:
        image: Input image array (2D or 3D)
        polygon: Shapely Polygon in coordinate space (micrometers)
        pixel_size_um: Pixel size in micrometers (for coordinate conversion)
    
    Returns:
        Masked and cropped image array
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

    polygon_px = _polygon_to_pixel_space(polygon, pixel_size_um)
    height, width = image.shape[:2]
    min_x_i, min_y_i, max_x_i, max_y_i = _bbox_int(polygon_px.bounds)
    min_x_i = max(min_x_i, 0)
    min_y_i = max(min_y_i, 0)
    max_x_i = min(max_x_i, width)
    max_y_i = min(max_y_i, height)
    bbox_width_px = max(0, max_x_i - min_x_i)
    bbox_height_px = max(0, max_y_i - min_y_i)
    bbox_width_um = bbox_width_px * pixel_size_um if pixel_size_um is not None and pixel_size_um > 0 else None
    bbox_height_um = bbox_height_px * pixel_size_um if pixel_size_um is not None and pixel_size_um > 0 else None
    logger.debug(
        "Image crop bbox: image_shape=%s bbox_px=(x:%d-%d,y:%d-%d,w:%d,h:%d) bbox_um=(w:%s,h:%s)",
        image.shape,
        min_x_i,
        max_x_i,
        min_y_i,
        max_y_i,
        bbox_width_px,
        bbox_height_px,
        f"{bbox_width_um:.3f}" if bbox_width_um is not None else "n/a",
        f"{bbox_height_um:.3f}" if bbox_height_um is not None else "n/a",
    )
    if min_x_i >= max_x_i or min_y_i >= max_y_i:
        return np.empty((0, 0), dtype=image.dtype)

    cropped = image[min_y_i:max_y_i, min_x_i:max_x_i]
    local_polygon = affinity.translate(polygon_px, xoff=-min_x_i, yoff=-min_y_i)
    return _apply_local_mask(cropped, local_polygon)


def _polygon_to_pixel_space(polygon: Polygon, pixel_size_um: float | None) -> Polygon:
    """Convert polygon from coordinate space to pixel space.
    
    Scales polygon coordinates by 1/pixel_size_um to transform micrometers to pixels.
    
    Args:
        polygon: Polygon in coordinate units (micrometers)
        pixel_size_um: Pixel size in micrometers; None means polygon is already in pixel space
    
    Returns:
        Polygon in pixel space (or original if pixel_size_um is None)
    """
    if pixel_size_um is not None and pixel_size_um > 0:
        scale_factor = 1.0 / pixel_size_um
        return affinity.scale(polygon, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    return polygon


def _bbox_int(bounds: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Convert polygon bounds to integer pixel coordinates.
    
    Uses floor() for min and ceil() for max to ensure crop fully contains polygon.
    CRITICAL: This behavior MUST match io_utils._region_crop_origin_um() to align
    entity coordinates with the cropped image.
    
    Args:
        bounds: (min_x, min_y, max_x, max_y) as floats
    
    Returns:
        (min_x_int, min_y_int, max_x_int, max_y_int) as integers
    """
    min_x, min_y, max_x, max_y = bounds
    return int(np.floor(min_x)), int(np.floor(min_y)), int(np.ceil(max_x)), int(np.ceil(max_y))


def _apply_local_mask(image: np.ndarray, local_polygon: Polygon) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

    crop_h, crop_w = image.shape[:2]
    if crop_h == 0 or crop_w == 0:
        return image

    # Check polygon validity and bounds
    if local_polygon.is_empty:
        logger.warning("Local polygon is empty after translation")
        return np.zeros_like(image)
    
    polygon_bounds = local_polygon.bounds
    if (polygon_bounds[0] >= crop_w or polygon_bounds[2] <= 0 or 
        polygon_bounds[1] >= crop_h or polygon_bounds[3] <= 0):
        logger.warning(f"Polygon bounds {polygon_bounds} outside crop region ({crop_w}, {crop_h})")
        return np.zeros_like(image)

    mask_img = Image.new("L", (crop_w, crop_h), 0)
    draw = ImageDraw.Draw(mask_img)
    
    # Convert polygon coords to tuples (int conversion for ImageDraw)
    coords = [(float(x), float(y)) for x, y in local_polygon.exterior.coords]
    try:
        draw.polygon(coords, fill=255)
    except Exception as e:
        logger.error(f"Failed to draw polygon with coords {coords}: {e}")
        return np.zeros_like(image)
    
    mask = np.asarray(mask_img, dtype=bool)
    
    if mask.sum() == 0:
        logger.warning(f"Mask is all zeros for polygon bounds {polygon_bounds}")
    
    if image.ndim == 2:
        return np.where(mask, image, 0)
    return np.where(mask[..., None], image, 0)


def _pyramid_levels(image: np.ndarray) -> list[np.ndarray]:
    """Build a list of progressively halved images for a pyramid OME-TIFF.

    Stops when either spatial dimension drops to 512 px or below.
    Each level is a 2×2 block-average of the previous (preserves dtype).
    """
    levels: list[np.ndarray] = [image]
    while True:
        prev = levels[-1]
        h, w = prev.shape[:2]
        if h <= 512 or w <= 512:
            break
        nh, nw = h // 2, w // 2
        trimmed = prev[: nh * 2, : nw * 2]
        if trimmed.ndim == 3:
            down = trimmed.reshape(nh, 2, nw, 2, trimmed.shape[2]).mean(axis=(1, 3))
        else:
            down = trimmed.reshape(nh, 2, nw, 2).mean(axis=(1, 3))
        levels.append(down.astype(prev.dtype))
    return levels


def _write_pyramidal_ome_tiff(
    image: np.ndarray,
    output_path: Path,
    axes: str | None = None,
    pixel_size_um: float | None = None,
) -> None:
    """Write a tiled, pyramidal OME-TIFF compatible with Xenium Explorer.

    Sub-resolution levels are stored as sub-IFDs (the TIFF pyramid standard).
    Tile size is 256×256 as required by Xenium Explorer 4+.
    When pixel_size_um is provided, PhysicalSizeX/Y are written into the OME-XML
    so Xenium Explorer shows coordinates in micrometers.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    levels = _pyramid_levels(image)
    n_subifds = len(levels) - 1
    metadata: dict = {}
    if axes:
        metadata["axes"] = axes
    if pixel_size_um is not None and pixel_size_um > 0:
        metadata["PhysicalSizeX"] = pixel_size_um
        metadata["PhysicalSizeY"] = pixel_size_um
        metadata["PhysicalSizeXUnit"] = "µm"
        metadata["PhysicalSizeYUnit"] = "µm"
    photometric = "rgb" if (image.ndim == 3 and image.shape[2] in (3, 4)) else "minisblack"
    options: dict = {"tile": (256, 256), "photometric": photometric}
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(levels[0], subifds=n_subifds, metadata=metadata, **options)
        for level in levels[1:]:
            tif.write(level, subfiletype=1, **options)


def save_image_like(
    source_path: Path,
    output_path: Path,
    image: np.ndarray,
    pixel_size_um: float | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_name = source_path.name.lower()
    if source_name.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        if image.ndim == 2:
            axes = "YX"
        elif image.ndim == 3:
            axes = "YXC"
        else:
            axes = None
        _write_pyramidal_ome_tiff(image, output_path, axes=axes, pixel_size_um=pixel_size_um)
        return output_path

    if source_path.suffix.lower() == ".svs":
        fallback = output_path.with_suffix(".tiff")
        _write_pyramidal_ome_tiff(image, fallback, axes="YXC", pixel_size_um=pixel_size_um)
        return fallback

    pil_image = Image.fromarray(_coerce_for_pillow(image))
    pil_image.save(output_path)
    return output_path


def convert_svs_to_ome_tiff(svs_path: Path, output_path: Path) -> Path:
    image = _read_svs(svs_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pyramidal_ome_tiff(image, output_path, axes="YXC")
    return output_path


def write_array_as_ome_tiff(
    image: np.ndarray,
    output_path: Path,
    pixel_size_um: float | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        axes = "YX"
    elif image.ndim == 3:
        axes = "YXC"
    else:
        raise ValueError(f"Unsupported array shape for OME-TIFF: {image.shape}")
    _write_pyramidal_ome_tiff(image, output_path, axes=axes, pixel_size_um=pixel_size_um)
    return output_path


def _coerce_for_pillow(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)
