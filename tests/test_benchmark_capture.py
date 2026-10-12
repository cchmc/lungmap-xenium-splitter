import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
from PIL import Image

from scripts.benchmark_capture import _capture
from scripts.benchmark_report import _assemble_run


def test_capture_sidecar_preserves_metrics_without_source_mounts(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "cells.csv").write_text("cell_id\n1\n", encoding="utf-8")
    for archive_name, size in (
        ("transcripts.zarr.zip", 1_500_000),
        ("cells.zarr.zip", 2_250_000),
    ):
        with (input_dir / archive_name).open("wb") as archive:
            archive.truncate(size)
    morphology_path = input_dir / "morphology.ome.tif"
    tifffile.imwrite(
        morphology_path,
        np.zeros((4, 5, 7), dtype=np.uint16),
        ome=True,
        metadata={"axes": "ZYX"},
    )
    lasso_file = tmp_path / "regions.csv"
    lasso_file.write_text("region_id,x,y\nR1,0,0\n", encoding="utf-8")
    he_image = tmp_path / "slide.png"
    Image.new("RGB", (9, 7)).save(he_image)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "run_metadata_README.md").write_text("- Regions: 1\n", encoding="utf-8")
    log_out = tmp_path / "job.out"
    log_out.write_text("Successfully completed.\n", encoding="utf-8")
    log_err = tmp_path / "job.err"
    log_err.write_text(
        "xenium_splitter.splitter - INFO - Updated metadata for region R1 "
        "(cells=5, transcripts=9, area_um2=12.0)\n",
        encoding="utf-8",
    )
    metrics_path = tmp_path / "job.metrics.json"

    sidecar = _capture(
        SimpleNamespace(
            job_name="job",
            mode="with_images",
            input_dir=str(input_dir),
            lasso_file=str(lasso_file),
            he_image=str(he_image),
            output_dir=str(output_dir),
            err_log=str(log_err),
            exit_code=0,
        )
    )
    metrics_path.write_text(json.dumps(sidecar), encoding="utf-8")

    input_size = sidecar["file_sizes_bytes"]["input_dir"]
    he_size = sidecar["file_sizes_bytes"]["he_image"]
    lasso_size = sidecar["file_sizes_bytes"]["lasso_file"]
    assert input_size > 0 and he_size > 0 and lasso_size > 0
    assert "output_dir" not in sidecar["file_sizes_bytes"]
    assert sidecar["run_metrics"]["cells_total"] == 5
    assert sidecar["run_metrics"]["cells_written_total"] == 5
    assert sidecar["run_metrics"]["transcripts_total"] == 9
    assert sidecar["he_image"]["he_width_px"] == 9
    assert sidecar["he_image"]["he_height_px"] == 7
    morphology_record = next(
        item for item in sidecar["input_files"] if item["relative_path"] == "morphology.ome.tif"
    )
    assert morphology_record["image"]["spatial_pixel_count"] == 35
    assert morphology_record["image"]["plane_count"] == 4
    assert morphology_record["image"]["sample_count"] == 140

    for path in (input_dir, output_dir):
        shutil.rmtree(path)
    he_image.unlink()
    lasso_file.unlink()

    record = _assemble_run(
        {
            "name": "sample",
            "mode": "with_images",
            "job_name": "xsplit_sample_with_images",
            "log_out": str(log_out),
            "log_err": str(log_err),
            "metrics_json": str(metrics_path),
        }
    )

    assert record["status"] == "SUCCESS"
    assert record["input_dir"] == str(input_dir)
    assert record["lasso_file"] == str(lasso_file)
    assert record["he_image"] == str(he_image)
    assert record["regions"] == 1
    assert record["cells_total"] == 5
    assert record["cells_written_total"] == 5
    assert record["transcripts_total"] == 9
    assert record["he_width_px"] == 9
    assert record["he_height_px"] == 7
    assert record["he_size_gb"] is not None
    assert record["input_dir_size_gb"] is not None
    assert record["lasso_size_gb"] is not None
    assert record["input_file_count"] == 4
    assert record["transcript_zarr_mb"] == 1.5
    assert record["cell_zarr_mb"] == 2.25
    assert record["morphology_file_size_gb"] is not None
    assert record["morphology_width_px"] == 7
    assert record["morphology_height_px"] == 5
    assert record["morphology_pixel_count"] == 35
    assert record["morphology_plane_count"] == 4
    assert record["morphology_sample_count"] == 140
