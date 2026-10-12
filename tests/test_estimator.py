from pathlib import Path

import pytest
from typer.testing import CliRunner

from xenium_splitter.cli import app
from xenium_splitter.estimator import estimate_ram
from typer.main import get_command


def test_estimate_ram_uses_archive_sizes(tmp_path: Path) -> None:
    for name, size in (("transcripts.zarr.zip", 100_000_000), ("cells.zarr.zip", 20_000_000)):
        with (tmp_path / name).open("wb") as archive:
            archive.truncate(size)

    assert estimate_ram(tmp_path) == pytest.approx(3.64)
    lasso_path = tmp_path / "regions.csv"
    lasso_path.write_text("region_id,x,y\n", encoding="utf-8")
    output_path = tmp_path / "not_created"
    result = CliRunner().invoke(
        app,
        [
            "estimate-ram", "--input-dir", str(tmp_path),
            "--lasso-file", str(lasso_path), "--output-dir", str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3.64 GB" in result.output
    assert not output_path.exists()


def test_estimate_ram_rejects_missing_archive(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="transcripts.zarr.zip"):
        estimate_ram(tmp_path)
    (tmp_path / "transcripts.zarr.zip").touch()
    with pytest.raises(FileNotFoundError, match="cells.zarr.zip"):
        estimate_ram(tmp_path)


def test_estimate_ram_accepts_all_split_options(tmp_path: Path) -> None:
    commands = get_command(app).commands
    estimator_params = commands["estimate-ram"].params
    split_params = commands["split"].params
    assert [
        (param.name, param.opts, param.secondary_opts, param.required, param.default, param.multiple)
        for param in estimator_params
    ] == [
        (param.name, param.opts, param.secondary_opts, param.required, param.default, param.multiple)
        for param in split_params
    ]
    (tmp_path / "transcripts.zarr.zip").touch()
    (tmp_path / "cells.zarr.zip").touch()
    lasso_path = tmp_path / "regions.csv"
    lasso_path.touch()
    he_path = tmp_path / "he.tif"
    he_path.touch()
    output_path = tmp_path / "not_created"
    arguments = [
        "estimate-ram", "--input-dir", str(tmp_path),
        "--lasso-file", str(lasso_path), "--output-dir", str(output_path),
        "--he-image", str(he_path), "--convert-svs-to-ome", "--no-squash-layers",
        "--include-glob", "cells*", "--include-glob", "transcripts*",
        "--process-images", "--overlays", "--skip-diffexp-recalc",
        "--skip-cell-feature-matrix-zarr", "--copy-transcripts", "--images-only", "-v",
    ]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "2.50 GB" in result.output
    assert not output_path.exists()
    result = CliRunner().invoke(app, arguments + ["--skip-images"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output