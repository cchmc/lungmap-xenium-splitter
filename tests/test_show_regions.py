from pathlib import Path

import numpy as np
from PIL import Image

from xenium_splitter.show_regions import create_regions_preview


def _write_lasso_csv(path: Path) -> None:
    path.write_text(
        "region_id,x,y\n"
        "R1,20,20\n"
        "R1,120,20\n"
        "R1,120,120\n"
        "R1,20,120\n"
        "R2,140,30\n"
        "R2,220,30\n"
        "R2,220,90\n"
        "R2,140,90\n",
        encoding="utf-8",
    )


def test_show_regions_prefers_he_image(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    lasso_file = tmp_path / "lasso.csv"
    _write_lasso_csv(lasso_file)

    # Large enough to force downscaling to max_dimension_px.
    he_img = np.zeros((3000, 2400, 3), dtype=np.uint8)
    he_img[:, :, 0] = 180
    he_img[:, :, 1] = 140
    he_img[:, :, 2] = 120
    he_path = input_dir / "slide_he.png"
    Image.fromarray(he_img).save(he_path)

    # Also add morphology to verify H&E preference wins.
    morph_path = input_dir / "morphology.ome.tif"
    Image.fromarray(np.full((300, 300), 32, dtype=np.uint8)).save(morph_path)

    out_path = tmp_path / "preview.png"
    result = create_regions_preview(
        input_dir=input_dir,
        lasso_file=lasso_file,
        output_image=out_path,
        max_dimension_px=2000,
    )

    assert out_path.is_file()
    assert result["source_kind"] == "he"
    assert Path(str(result["source_path"])).name == "slide_he.png"

    out_img = np.asarray(Image.open(out_path).convert("RGB"))
    assert max(out_img.shape[0], out_img.shape[1]) <= 2000

    # Region outlines are purple.
    purple_mask = (
        (out_img[:, :, 0] == 128)
        & (out_img[:, :, 1] == 0)
        & (out_img[:, :, 2] == 255)
    )
    assert int(purple_mask.sum()) > 0


def test_show_regions_falls_back_to_morphology(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    lasso_file = tmp_path / "lasso.csv"
    _write_lasso_csv(lasso_file)

    morph = np.full((400, 500), 90, dtype=np.uint8)
    morph_path = input_dir / "morphology_mip.ome.tif"
    Image.fromarray(morph).save(morph_path)

    out_path = tmp_path / "preview_morph.png"
    result = create_regions_preview(
        input_dir=input_dir,
        lasso_file=lasso_file,
        output_image=out_path,
    )

    assert out_path.is_file()
    assert result["source_kind"] == "morphology"

    out_img = np.asarray(Image.open(out_path).convert("RGB"))

    # Dotted morphology bounding boxes are green.
    green_mask = (
        (out_img[:, :, 0] == 0)
        & (out_img[:, :, 1] == 255)
        & (out_img[:, :, 2] == 0)
    )
    assert int(green_mask.sum()) > 0