from pathlib import Path

import numpy as np
import tifffile
from shapely.geometry import Polygon

from xenium_splitter import image_utils
from xenium_splitter.image_utils import (
    mask_and_crop_region,
    read_image,
    read_masked_cropped_region,
)


def test_tiff_region_squashes_after_cropping(tmp_path: Path) -> None:
    image_path = tmp_path / "stack.ome.tif"
    stack = np.arange(4 * 20 * 30, dtype=np.uint16).reshape(4, 20, 30)
    tifffile.imwrite(image_path, stack, ome=True, metadata={"axes": "ZYX"})
    polygon = Polygon([(4, 3), (16, 3), (16, 15), (4, 15)])

    assert isinstance(tifffile.memmap(image_path, mode="r"), np.memmap)
    expected = mask_and_crop_region(read_image(image_path), polygon)
    actual = read_masked_cropped_region(image_path, polygon)

    np.testing.assert_array_equal(actual, expected)

    expected_stack = np.stack(
        [
            mask_and_crop_region(plane, polygon)
            for plane in read_image(image_path, squash_layers=False)
        ]
    )
    actual_stack = read_masked_cropped_region(
        image_path,
        polygon,
        squash_layers=False,
    )
    np.testing.assert_array_equal(actual_stack, expected_stack)


def test_compressed_tiff_region_falls_back_to_decoding(tmp_path: Path) -> None:
    image_path = tmp_path / "compressed.ome.tif"
    stack = np.arange(4 * 20 * 30, dtype=np.uint16).reshape(4, 20, 30)
    tifffile.imwrite(
        image_path,
        stack,
        ome=True,
        metadata={"axes": "ZYX"},
        compression="deflate",
    )
    polygon = Polygon([(4, 3), (16, 3), (16, 15), (4, 15)])

    with np.testing.assert_raises(ValueError):
        tifffile.memmap(image_path, mode="r")
    expected = mask_and_crop_region(read_image(image_path), polygon)
    actual = read_masked_cropped_region(image_path, polygon)

    np.testing.assert_array_equal(actual, expected)


def test_tiled_jpeg2000_region_decodes_only_intersecting_tiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "tiled.ome.tif"
    stack = np.arange(3 * 35 * 41, dtype=np.uint16).reshape(3, 35, 41)
    tifffile.imwrite(
        image_path,
        stack,
        ome=True,
        metadata={"axes": "ZYX"},
        tile=(16, 16),
        compression="jpeg2000",
    )
    polygon = Polygon([(4, 3), (36, 3), (36, 33), (4, 33)])
    expected = mask_and_crop_region(read_image(image_path), polygon)

    def fail_full_image_read(*args, **kwargs):
        raise AssertionError("full-image reader should not be used for tiled JPEG2000")

    monkeypatch.setattr(image_utils, "read_image", fail_full_image_read)
    actual = read_masked_cropped_region(image_path, polygon)

    np.testing.assert_array_equal(actual, expected)