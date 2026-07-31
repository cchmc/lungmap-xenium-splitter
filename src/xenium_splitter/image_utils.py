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
from PIL import Image, ImageDraw, ImageFont
from shapely import affinity
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def read_image(path: Path, squash_layers: bool = True) -> np.ndarray:
    """Read an image file into a NumPy array.

    Supports SVS (via ``openslide``), TIFF/OME-TIFF (via ``tifffile``), and
    common raster formats (PNG, JPEG, etc.) via Pillow.  Multi-layer arrays
    are max-projected when ``squash_layers`` is ``True``.

    Args:
        path: Path to the image file.
        squash_layers: When ``True``, flatten multi-dimensional stacks to 2D/3D.

    Returns:
        Image as a NumPy array (H × W or H × W × C).
    """
    lower_name = path.name.lower()
    if path.suffix.lower() == ".svs":
        return _read_svs(path)

    if lower_name.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        array = tifffile.imread(path)
        return _squash_if_needed(array, path=path) if squash_layers else np.asarray(array)

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def supports_windowed_region_read(path: Path) -> bool:
    """Return ``True`` when the image format supports windowed (partial) region reads.

    Windowed reading avoids loading the full image when only a small region crop
    is needed.  Currently supported for TIFF/OME-TIFF and SVS files.
    """
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
    """Read, crop, and mask a single polygon region from an image file.

    Dispatches to the appropriate windowed reader for TIFF/SVS, or falls back
    to a full-read + crop for other formats.

    Args:
        path: Path to the source image.
        polygon: Region polygon in coordinate space (micrometers).
        pixel_size_um: Pixel size in micrometers; used to convert polygon coordinates
            to pixel indices.  ``None`` treats the polygon as already in pixel space.
        squash_layers: Flatten multi-layer stacks when ``True``.

    Returns:
        Masked and cropped image array.
    """
    if path.suffix.lower() == ".svs":
        return _read_masked_cropped_svs_region(path, polygon, pixel_size_um)

    lower_name = path.name.lower()
    if lower_name.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        return _read_masked_cropped_tiff_region(path, polygon, pixel_size_um, squash_layers)

    image = read_image(path, squash_layers=squash_layers)
    return mask_and_crop_region(image, polygon, pixel_size_um=pixel_size_um)


def _read_svs(path: Path) -> np.ndarray:
    """Read a full SVS slide as an RGB NumPy array (requires ``openslide-python``)."""
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
    """Read a polygon-bounded crop of an SVS slide, applying a binary mask.

    Uses ``openslide`` windowed reads; only the bounding box of the polygon
    is loaded from disk.
    """
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

    spatial_axes = _spatial_axes_for_array(full, path=path)
    y_axis, x_axis = spatial_axes
    h = int(full.shape[y_axis])
    w = int(full.shape[x_axis])
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
    
    crop_slices = [slice(None)] * full.ndim
    crop_slices[y_axis] = slice(min_y_i_safe, max_y_i_safe)
    crop_slices[x_axis] = slice(min_x_i_safe, max_x_i_safe)
    cropped = full[tuple(crop_slices)]
    local_polygon = affinity.translate(polygon_px, xoff=-min_x_i_safe, yoff=-min_y_i_safe)
    masked = _apply_local_mask(cropped, local_polygon, spatial_axes=spatial_axes)
    
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


def _apply_local_mask(
    image: np.ndarray,
    local_polygon: Polygon,
    spatial_axes: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply a binary polygon mask to a cropped image array.

    Pixels outside the polygon interior are zeroed.  The mask is drawn with
    Pillow and broadcast across non-spatial axes.

    Args:
        image: Cropped image array (2-D or higher).
        local_polygon: Polygon whose coordinates are relative to the crop origin.
        spatial_axes: ``(y_axis, x_axis)`` indices within ``image.shape``; inferred
            when ``None``.

    Returns:
        Image array with out-of-polygon pixels set to zero.
    """
    if image.ndim < 2:
        raise ValueError(f"Expected image array with at least 2 dims, got shape {image.shape}")

    if spatial_axes is None:
        spatial_axes = _spatial_axes_for_array(image)
    y_axis, x_axis = spatial_axes
    crop_h = int(image.shape[y_axis])
    crop_w = int(image.shape[x_axis])
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

    broadcast_mask = mask
    for axis in range(image.ndim):
        if axis not in spatial_axes:
            broadcast_mask = np.expand_dims(broadcast_mask, axis=axis)
    return np.where(broadcast_mask, image, 0)


def _pyramid_levels(image: np.ndarray, axes: str | None = None) -> list[np.ndarray]:
    """Build a list of progressively halved images for a pyramid OME-TIFF.

    Stops when either spatial dimension drops to 512 px or below.
    Each level is a 2×2 block-average of the previous (preserves dtype).
    """
    levels: list[np.ndarray] = [image]
    spatial_axes = _spatial_axes_for_array(image, axes=axes)
    while True:
        prev = levels[-1]
        y_axis, x_axis = _spatial_axes_for_array(prev, axes=axes)
        h = int(prev.shape[y_axis])
        w = int(prev.shape[x_axis])
        if h <= 512 or w <= 512:
            break
        nh, nw = h // 2, w // 2
        trimmed = _trim_spatial_axes(prev, spatial_axes, nh * 2, nw * 2)
        down = _downsample_spatial_axes(trimmed, spatial_axes, nh, nw)
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
    Tile size is 1024×1024 per 10x Xenium Explorer guidance.
    When pixel_size_um is provided, PhysicalSizeX/Y are written into the OME-XML
    so Xenium Explorer shows coordinates in micrometers.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    levels = _pyramid_levels(image, axes=axes)
    n_subifds = len(levels) - 1
    metadata: dict = {}
    if axes:
        metadata["axes"] = axes
    if pixel_size_um is not None and pixel_size_um > 0:
        metadata["PhysicalSizeX"] = pixel_size_um
        metadata["PhysicalSizeY"] = pixel_size_um
        metadata["PhysicalSizeXUnit"] = "µm"
        metadata["PhysicalSizeYUnit"] = "µm"
    photometric = "rgb" if _is_rgb_like(image, axes=axes) else "minisblack"
    options: dict = {
        "tile": (1024, 1024),
        "photometric": photometric,
        "compression": "zlib",
        "resolutionunit": "CENTIMETER",
    }

    def _resolution_for_level(scale: float) -> tuple[float, float] | None:
        if pixel_size_um is None or pixel_size_um <= 0:
            return None
        effective_pixel_size_um = float(pixel_size_um) * float(scale)
        pixels_per_centimeter = 1.0e4 / effective_pixel_size_um
        return (pixels_per_centimeter, pixels_per_centimeter)

    with tifffile.TiffWriter(output_path, bigtiff=True, ome=True) as tif:
        base_kwargs = dict(options)
        resolution = _resolution_for_level(1.0)
        if resolution is not None:
            base_kwargs["resolution"] = resolution
        tif.write(levels[0], subifds=n_subifds, metadata=metadata, **base_kwargs)
        for level_index, level in enumerate(levels[1:], start=1):
            level_kwargs = dict(options)
            resolution = _resolution_for_level(2.0 ** level_index)
            if resolution is not None:
                level_kwargs["resolution"] = resolution
            tif.write(level, subfiletype=1, **level_kwargs)


def save_image_like(
    source_path: Path,
    output_path: Path,
    image: np.ndarray,
    pixel_size_um: float | None = None,
) -> Path:
    """Save ``image`` in a format that matches the source file's format.

    - TIFF/OME-TIFF → pyramidal OME-TIFF (``_write_pyramidal_ome_tiff``).
    - SVS → pyramidal OME-TIFF with ``YXC`` axes (SVS output not supported).
    - Other raster formats → Pillow write using the original extension.

    Args:
        source_path: Original image path used to determine the output format.
        output_path: Destination path; parent directories are created if needed.
        image: Array to write.
        pixel_size_um: Physical pixel size in micrometers; written to OME-XML
            so the image registers correctly in Xenium Explorer.

    Returns:
        Actual path written (may differ from ``output_path`` for SVS inputs).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_name = source_path.name.lower()
    if source_name.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        axes = _infer_write_axes(image, source_path)
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
    """Convert an SVS slide to a pyramidal OME-TIFF.

    The full slide is read at the highest resolution level and written as a
    tiled, pyramidal OME-TIFF with ``YXC`` axes.
    """
    image = _read_svs(svs_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pyramidal_ome_tiff(image, output_path, axes="YXC")
    return output_path


def write_array_as_ome_tiff(
    image: np.ndarray,
    output_path: Path,
    pixel_size_um: float | None = None,
) -> Path:
    """Write an image array as a pyramidal OME-TIFF.

    Axes are inferred from the array shape.  Use :func:`save_image_like` when
    you need to match the source format of an existing image file.

    Args:
        image: Array to write.
        output_path: Destination path; parent directories are created if needed.
        pixel_size_um: Physical pixel size in micrometers written to OME-XML.

    Returns:
        The path that was written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axes = _infer_write_axes(image)
    _write_pyramidal_ome_tiff(image, output_path, axes=axes, pixel_size_um=pixel_size_um)
    return output_path


def _infer_write_axes(image: np.ndarray, source_path: Path | None = None) -> str:
    """Infer the OME-TIFF axis string for ``image``.

    Reads the axis string from the source file's OME metadata when available.
    Falls back to shape-based heuristics: ``YX`` (2-D), ``YXC`` (3-D RGB),
    ``ZYX`` (3-D grayscale), ``TZYX`` (4-D), or ``ZYXC`` (4-D RGB).
    """
    arr = np.asarray(image)
    source_axes = _get_tiff_axes(source_path) if source_path is not None else None
    if source_axes and len(source_axes) == arr.ndim:
        return source_axes
    if arr.ndim == 2:
        return "YX"
    if _is_rgb_like(arr):
        if arr.ndim == 3:
            return "YXC"
        if arr.ndim == 4:
            return "ZYXC"
    if arr.ndim == 3:
        return "ZYX"
    if arr.ndim == 4:
        return "TZYX"
    raise ValueError(f"Unsupported array shape for OME-TIFF: {arr.shape}")


def _is_rgb_like(image: np.ndarray, axes: str | None = None) -> bool:
    """Return ``True`` when ``image`` looks like an RGB or RGBA array.

    When ``axes`` is provided the C axis is checked directly; otherwise a last
    dimension of 3 or 4 is used as a heuristic.
    """
    arr = np.asarray(image)
    if axes and len(axes) == arr.ndim:
        axes_lower = axes.lower()
        try:
            c_axis = axes_lower.index("c")
            return int(arr.shape[c_axis]) in (3, 4)
        except ValueError:
            return False
    return arr.ndim >= 3 and int(arr.shape[-1]) in (3, 4)


def _spatial_axes_for_array(image: np.ndarray, path: Path | None = None, axes: str | None = None) -> tuple[int, int]:
    """Return ``(y_axis_index, x_axis_index)`` for a given array and optional axes string.

    Uses OME axes metadata when available (from file or explicit argument),
    then falls back to shape-based heuristics (last two non-color axes).
    """
    arr = np.asarray(image)
    source_axes = axes or (_get_tiff_axes(path) if path is not None else None)
    if source_axes and len(source_axes) == arr.ndim:
        axes_lower = source_axes.lower()
        try:
            return axes_lower.index("y"), axes_lower.index("x")
        except ValueError:
            pass
    if _is_rgb_like(arr, axes=source_axes):
        return arr.ndim - 3, arr.ndim - 2
    return arr.ndim - 2, arr.ndim - 1


def _trim_spatial_axes(
    image: np.ndarray,
    spatial_axes: tuple[int, int],
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Crop ``image`` to ``target_h × target_w`` on the spatial axes (used before downsampling)."""
    slices = [slice(None)] * image.ndim
    y_axis, x_axis = spatial_axes
    slices[y_axis] = slice(0, target_h)
    slices[x_axis] = slice(0, target_w)
    return image[tuple(slices)]


def _downsample_spatial_axes(
    image: np.ndarray,
    spatial_axes: tuple[int, int],
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Halve resolution on spatial axes by 2×2 block averaging (used for pyramid levels)."""
    y_axis, x_axis = spatial_axes
    moved = np.moveaxis(image, (y_axis, x_axis), (-2, -1))
    lead_shape = moved.shape[:-2]
    reshaped = moved.reshape(lead_shape + (out_h, 2, out_w, 2))
    down = reshaped.mean(axis=(-3, -1))
    return np.moveaxis(down, (-2, -1), (y_axis, x_axis))


def _coerce_for_pillow(image: np.ndarray) -> np.ndarray:
    """Convert ``image`` to uint8, clipping float values to the [0, 255] range."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def generate_morphology_mip(image: np.ndarray) -> np.ndarray:
    """Generate a morphology maximum-intensity projection from a cropped morphology stack.

    Delegates to :func:`_squash_if_needed` which applies max projection across
    all non-spatial axes.
    """
    return _squash_if_needed(np.asarray(image))


def generate_morphology_focus(image: np.ndarray, path: Path | None = None) -> np.ndarray:
    """Return the best-focus Z-plane from a morphology stack (discards focus stats).

    See :func:`generate_morphology_focus_with_stats` for details.
    """
    focus_image, _stats = generate_morphology_focus_with_stats(image, path)
    return focus_image


def generate_morphology_focus_with_stats(
    image: np.ndarray,
    path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Generate a morphology focus image from a cropped morphology stack.

    Uses Laplacian-variance focus scoring across Z-like planes. If the image is
    already 2D or RGB, it is returned unchanged.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr, None
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr, None

    stack = _focus_plane_stack(arr, path)
    if stack is None or stack.shape[0] == 0:
        return _squash_if_needed(arr, path=path), None

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Morphology focus generation requires opencv-python-headless."
        ) from exc

    focus_scores: list[float] = []
    for plane in stack:
        plane_gray = _focus_plane_to_grayscale(plane)
        lap = cv2.Laplacian(plane_gray, cv2.CV_32F)
        focus_scores.append(float(lap.var()))

    best_index = int(np.argmax(np.asarray(focus_scores, dtype=np.float64)))
    stats = {
        "stack_count": int(stack.shape[0]),
        "selected_index": int(best_index),
        "focus_scores": [float(score) for score in focus_scores],
    }
    return np.asarray(stack[best_index]).astype(arr.dtype, copy=False), stats


def _focus_plane_stack(image: np.ndarray, path: Path | None = None) -> np.ndarray | None:
    """Reshape a multi-dimensional morphology array into a stack of 2-D (or RGB) planes.

    Uses OME axis metadata to identify non-spatial (Z/T) axes and moves them to
    the leading dimension.  Falls back to simple shape-based heuristics.
    Returns ``None`` when the array is already 2-D or RGB.
    """
    arr = np.asarray(image)
    axes = _get_tiff_axes(path) if path else None

    if axes and len(axes) == arr.ndim:
        axes_lower = axes.lower()
        keep_axes = [i for i, ax in enumerate(axes_lower) if ax in {"y", "x"}]
        color_axis = next(
            (
                i
                for i, ax in enumerate(axes_lower)
                if ax == "c" and arr.shape[i] in (3, 4)
            ),
            None,
        )
        if color_axis is not None:
            keep_axes.append(color_axis)
        keep_axes = sorted(set(keep_axes), key=lambda idx: idx)
        plane_axes = [i for i in range(arr.ndim) if i not in keep_axes]
        if plane_axes:
            moved = np.moveaxis(arr, plane_axes + keep_axes, range(arr.ndim))
            tail_shape = tuple(moved.shape[len(plane_axes):])
            return moved.reshape((-1,) + tail_shape)

    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        return arr
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        return arr.reshape((-1,) + arr.shape[-3:])
    return None


def _focus_plane_to_grayscale(plane: np.ndarray) -> np.ndarray:
    """Convert a single focus plane to a float32 grayscale array for Laplacian scoring.

    RGB/RGBA planes are converted with standard luminance weights (0.299R + 0.587G + 0.114B).
    Multi-channel planes that are neither 2-D nor RGB are collapsed with max-projection.
    """
    arr = np.asarray(plane)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32, copy=False)
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32, copy=False)
    return _squash_if_needed(arr).astype(np.float32, copy=False)


def _resize_for_overlay(
    image: np.ndarray,
    *,
    max_dimension_px: int,
) -> tuple[np.ndarray, float]:
    """Downscale ``image`` so its longest edge is at most ``max_dimension_px``.

    Returns the (possibly unchanged) image and the scale factor applied
    (1.0 when no downscaling was needed, >1.0 when the image was shrunk).
    The scale factor converts overlay pixel coordinates back to the original
    resolution.
    """
    if max_dimension_px <= 0:
        raise ValueError("max_dimension_px must be > 0")

    height_px, width_px = image.shape[:2]
    longest_edge = max(height_px, width_px)
    if longest_edge <= max_dimension_px:
        return image, 1.0

    scale = float(longest_edge) / float(max_dimension_px)
    new_width = max(1, int(round(width_px / scale)))
    new_height = max(1, int(round(height_px / scale)))
    resized = Image.fromarray(image).resize((new_width, new_height), Image.Resampling.BILINEAR)
    return np.asarray(resized), scale


def render_grid_overlay_image(
    image: np.ndarray,
    *,
    level_name: str,
    tile_size_um: float,
    pixel_size_um: float | None,
    fov_stride_um: tuple[float, float] | None = None,
    fov_size_um: tuple[float, float] | None = None,
    max_dimension_px: int = 2048,
) -> tuple[np.ndarray, float]:
    """Render a grid overlay image with line coordinates and tile labels.

    The input image is expected to already be crop-local for a region, so grid
    line coordinates are shown in the same rebased micrometer space used by the
    transcript outputs.
    """
    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be > 0 for grid overlays")
    if tile_size_um <= 0:
        raise ValueError("tile_size_um must be > 0 for grid overlays")

    base = _coerce_for_pillow(image)
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1)
    elif base.ndim == 3 and base.shape[2] == 4:
        rgb = base[:, :, :3]
    elif base.ndim == 3 and base.shape[2] == 3:
        rgb = base
    else:
        raise ValueError(f"Unsupported image shape for grid overlay: {image.shape}")

    rgb, downsample_scale = _resize_for_overlay(rgb, max_dimension_px=max_dimension_px)
    overlay_pixel_size_um = float(pixel_size_um) * downsample_scale

    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    height_px, width_px = rgb.shape[:2]
    width_um = width_px * overlay_pixel_size_um
    height_um = height_px * overlay_pixel_size_um

    # Title block for quick inspection.
    draw.rectangle((6, 6, 320, 34), fill=(0, 0, 0))
    draw.text((10, 10), f"Level {level_name} grid {tile_size_um:.0f} um", fill=(255, 255, 255), font=font)

    if fov_stride_um is not None:
        stride_x_um, stride_y_um = fov_stride_um
        if stride_x_um > 0:
            x_positions_um = np.arange(0.0, width_um + stride_x_um, stride_x_um, dtype=np.float64)
            for x_um in x_positions_um:
                x_px = int(round(x_um / overlay_pixel_size_um))
                if x_px < 0 or x_px >= width_px:
                    continue
                draw.line((x_px, 0, x_px, height_px - 1), fill=(255, 0, 0), width=1)
        if stride_y_um > 0:
            y_positions_um = np.arange(0.0, height_um + stride_y_um, stride_y_um, dtype=np.float64)
            for y_um in y_positions_um:
                y_px = int(round(y_um / overlay_pixel_size_um))
                if y_px < 0 or y_px >= height_px:
                    continue
                draw.line((0, y_px, width_px - 1, y_px), fill=(255, 0, 0), width=1)

    if fov_stride_um is not None and fov_size_um is not None:
        stride_x_um, stride_y_um = fov_stride_um
        size_x_um, size_y_um = fov_size_um
        if stride_x_um > 0 and size_x_um > 0:
            x_end_positions_um = np.arange(size_x_um, width_um + size_x_um, stride_x_um, dtype=np.float64)
            for x_um in x_end_positions_um:
                x_px = int(round(x_um / overlay_pixel_size_um))
                if x_px < 0 or x_px >= width_px:
                    continue
                draw.line((x_px, 0, x_px, height_px - 1), fill=(255, 96, 96), width=1)
        if stride_y_um > 0 and size_y_um > 0:
            y_end_positions_um = np.arange(size_y_um, height_um + size_y_um, stride_y_um, dtype=np.float64)
            for y_um in y_end_positions_um:
                y_px = int(round(y_um / overlay_pixel_size_um))
                if y_px < 0 or y_px >= height_px:
                    continue
                draw.line((0, y_px, width_px - 1, y_px), fill=(255, 96, 96), width=1)

    x_positions_um = np.arange(0.0, width_um + tile_size_um, tile_size_um, dtype=np.float64)
    y_positions_um = np.arange(0.0, height_um + tile_size_um, tile_size_um, dtype=np.float64)

    for x_um in x_positions_um:
        x_px = int(round(x_um / overlay_pixel_size_um))
        if x_px < 0 or x_px >= width_px:
            continue
        draw.line((x_px, 0, x_px, height_px - 1), fill=(0, 255, 0), width=1)
        label = f"x={x_um:.0f}"
        text_y = 40 if (x_px // 40) % 2 == 0 else 56
        text_x = min(max(2, x_px + 2), max(2, width_px - 50))
        draw.rectangle((text_x - 1, text_y - 1, min(width_px - 1, text_x + 42), text_y + 10), fill=(0, 0, 0))
        draw.text((text_x, text_y), label, fill=(180, 255, 180), font=font)

    for y_um in y_positions_um:
        y_px = int(round(y_um / overlay_pixel_size_um))
        if y_px < 0 or y_px >= height_px:
            continue
        draw.line((0, y_px, width_px - 1, y_px), fill=(0, 255, 0), width=1)
        label = f"y={y_um:.0f}"
        text_y = min(max(2, y_px + 2), max(2, height_px - 12))
        draw.rectangle((6, text_y - 1, 52, min(height_px - 1, text_y + 10)), fill=(0, 0, 0))
        draw.text((8, text_y), label, fill=(180, 255, 180), font=font)

    tile_cols = max(1, int(np.ceil(width_um / tile_size_um)))
    tile_rows = max(1, int(np.ceil(height_um / tile_size_um)))
    for ty in range(tile_rows):
        for tx in range(tile_cols):
            cx_um = (tx + 0.5) * tile_size_um
            cy_um = (ty + 0.5) * tile_size_um
            cx_px = int(round(cx_um / overlay_pixel_size_um))
            cy_px = int(round(cy_um / overlay_pixel_size_um))
            if cx_px < 0 or cx_px >= width_px or cy_px < 0 or cy_px >= height_px:
                continue
            label = f"{tx},{ty}"
            text_w = 7 * len(label)
            left = min(max(0, cx_px - text_w // 2), max(0, width_px - text_w - 4))
            top = min(max(0, cy_px - 6), max(0, height_px - 14))
            draw.rectangle((left - 2, top - 2, min(width_px - 1, left + text_w + 2), min(height_px - 1, top + 10)), fill=(0, 0, 0))
            draw.text((left, top), label, fill=(180, 255, 180), font=font)

    return np.asarray(canvas), overlay_pixel_size_um
