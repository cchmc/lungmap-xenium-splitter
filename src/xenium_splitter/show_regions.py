from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from xenium_splitter.image_utils import read_image
from xenium_splitter.io_utils import load_pixel_size_from_experiment
from xenium_splitter.lasso import load_lasso_regions


_IMAGE_SUFFIXES = (
    ".ome.tif",
    ".ome.tiff",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".svs",
)


def _is_image_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _IMAGE_SUFFIXES)


def _is_he_like_name(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() == ".svs":
        return True
    cleaned = re.sub(r"[^a-z0-9]+", " ", name)
    tokens = set(cleaned.split())
    return bool({"he", "hande", "handeimage", "hematoxylin", "eosin"}.intersection(tokens)) or "h&e" in name


def _morphology_rank(path: Path) -> int | None:
    name = path.name.lower()
    if name in {"morphology_mip.ome.tif", "morphology_mip.ome.tiff"}:
        return 0
    if name in {"morphology_focus.ome.tif", "morphology_focus.ome.tiff"}:
        return 1
    if name in {"morphology.ome.tif", "morphology.ome.tiff"}:
        return 2
    if "morphology" in name:
        return 3
    return None


def _discover_source_image(input_dir: Path, he_image: Path | None) -> tuple[Path, str]:
    if he_image is not None:
        if not he_image.is_file():
            raise ValueError(f"H&E image path does not exist or is not a file: {he_image}")
        return he_image, "he"

    image_paths = sorted(path for path in input_dir.rglob("*") if path.is_file() and _is_image_path(path))
    if not image_paths:
        raise ValueError(f"No image files found under input directory: {input_dir}")

    he_candidates = [path for path in image_paths if _is_he_like_name(path)]
    if he_candidates:
        return he_candidates[0], "he"

    ranked_morphology = [
        (rank, path)
        for path in image_paths
        for rank in [_morphology_rank(path)]
        if rank is not None
    ]
    if ranked_morphology:
        ranked_morphology.sort(key=lambda item: (item[0], str(item[1]).lower()))
        return ranked_morphology[0][1], "morphology"

    raise ValueError(
        "Could not find an H&E-like image or morphology image under input directory. "
        "Provide --he-image explicitly or include morphology.ome.tif data."
    )


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        max_value = float(np.nanmax(arr_f)) if arr_f.size else 0.0
        if max_value > 255.0:
            arr_f = (arr_f / max_value) * 255.0
        arr = np.clip(arr_f, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return arr[:, :, :3]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 3:
        projected = np.max(arr, axis=0)
        projected = np.clip(projected.astype(np.float32), 0, 255).astype(np.uint8)
        return np.stack([projected, projected, projected], axis=-1)

    raise ValueError(f"Unsupported image shape for show_regions: {arr.shape}")


def _resize_max_dimension(image_rgb: np.ndarray, max_dimension_px: int) -> tuple[np.ndarray, float]:
    if max_dimension_px <= 0:
        raise ValueError("max_dimension_px must be > 0")

    height_px, width_px = image_rgb.shape[:2]
    longest = max(height_px, width_px)
    if longest <= max_dimension_px:
        return image_rgb, 1.0

    scale = float(longest) / float(max_dimension_px)
    out_w = max(1, int(round(width_px / scale)))
    out_h = max(1, int(round(height_px / scale)))
    resized = Image.fromarray(image_rgb).resize((out_w, out_h), Image.Resampling.BILINEAR)
    return np.asarray(resized), scale


def _draw_dotted_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: tuple[int, int, int],
    width: int,
    segment_len: float = 8.0,
    gap_len: float = 5.0,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 0:
        return

    ux = dx / length
    uy = dy / length
    pos = 0.0
    while pos < length:
        seg_end = min(pos + segment_len, length)
        sx = x1 + ux * pos
        sy = y1 + uy * pos
        ex = x1 + ux * seg_end
        ey = y1 + uy * seg_end
        draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
        pos += segment_len + gap_len


def _draw_dotted_bbox(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[float, float, float, float],
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    min_x, min_y, max_x, max_y = bounds
    corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
    for i in range(4):
        _draw_dotted_line(draw, corners[i], corners[(i + 1) % 4], color=color, width=width)


def _load_scaled_font(image_size: tuple[int, int]) -> ImageFont.ImageFont:
    width, height = image_size
    # Scale label size with output image size while keeping readable bounds.
    font_size = max(14, min(42, int(round(min(width, height) * 0.022))))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def create_regions_preview(
    *,
    input_dir: Path,
    lasso_file: Path,
    output_image: Path,
    he_image: Path | None = None,
    max_dimension_px: int = 2000,
) -> dict[str, str | int]:
    """Create a low-resolution preview with LASSO regions and dotted region bboxes.

    The function prefers an H&E source image and falls back to morphology data
    if H&E is unavailable.
    """
    regions = load_lasso_regions(lasso_file)
    pixel_size_um = load_pixel_size_from_experiment(input_dir)
    source_image, source_kind = _discover_source_image(input_dir, he_image)

    source_array = read_image(source_image, squash_layers=True)
    rgb = _to_rgb_uint8(source_array)
    rgb, resize_scale = _resize_max_dimension(rgb, max_dimension_px=max_dimension_px)

    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font = _load_scaled_font((canvas.width, canvas.height))

    region_color = (128, 0, 255)
    bbox_color = (0, 0, 0) if source_kind == "he" else (0, 255, 0)
    label_color = (0, 0, 0) if source_kind == "he" else (255, 255, 255)

    px_scale_from_um = (1.0 / float(pixel_size_um)) if pixel_size_um and pixel_size_um > 0 else 1.0
    draw_scale = px_scale_from_um / resize_scale

    for region in regions:
        coords = [
            (float(x) * draw_scale, float(y) * draw_scale)
            for x, y in region.polygon.exterior.coords
        ]
        if len(coords) < 2:
            continue

        draw.line(coords, fill=region_color, width=2)

        min_x, min_y, max_x, max_y = region.bounds
        scaled_bounds = (
            float(min_x) * draw_scale,
            float(min_y) * draw_scale,
            float(max_x) * draw_scale,
            float(max_y) * draw_scale,
        )
        _draw_dotted_bbox(draw, scaled_bounds, color=bbox_color, width=3)

        label = str(region.region_id)
        min_x, min_y, max_x, max_y = scaled_bounds
        margin = max(3, int(round(min(canvas.width, canvas.height) * 0.006)))
        label_x = min(max(0, int(round(min_x)) + margin), max(0, canvas.width - margin))
        label_y = min(max(0, int(round(min_y)) + margin), max(0, canvas.height - margin))
        # Draw a subtle shadow to preserve readability on complex tissue background.
        draw.text((label_x + 2, label_y + 2), label, font=font, fill=(0, 0, 0))
        draw.text((label_x, label_y), label, font=font, fill=label_color)

    # title = f"show_regions: source={source_kind} ({source_image.name})"
    # draw.rectangle((6, 6, min(850, canvas.width - 6), 24), fill=(0, 0, 0))
    # draw.text((10, 10), title, font=font, fill=(255, 255, 255))

    output_image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image)
    return {
        "source_kind": source_kind,
        "source_path": str(source_image),
        "region_count": len(regions),
        "output_path": str(output_image),
    }