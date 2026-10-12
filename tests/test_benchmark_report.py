from pathlib import Path
import csv
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import tifffile
from PIL import Image
from pytest import MonkeyPatch

from scripts.benchmark_report import (
    _archive_size_mb,
    _assemble_run,
    _discover_from_logs,
    _he_dimensions_from_log,
    _he_image_metadata,
    _image_compression,
    _input_file_inventory,
    _input_image_compressions,
    _parse_region_counts_from_log,
    _TABLE_HEADERS,
    _print_summary_table,
    _print_statistics,
    _write_csv,
)


def test_he_image_metadata_reads_raster_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (7, 5)).save(image_path)

    metadata = _he_image_metadata(str(image_path))

    assert metadata == {
        "he_format": "PNG",
        "he_width_px": 7,
        "he_height_px": 5,
        "he_pixel_count": 35,
        "he_compression": "DEFLATE",
    }


def test_he_image_metadata_reads_tiff_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "slide.tiff"
    Image.new("RGB", (7, 5)).save(image_path, format="TIFF")

    metadata = _he_image_metadata(str(image_path))

    assert metadata["he_format"] == "TIFF"
    assert metadata["he_width_px"] == 7
    assert metadata["he_height_px"] == 5
    assert metadata["he_pixel_count"] == 35
    assert metadata["he_compression"] == "NONE"


def test_he_image_metadata_recognizes_ome_tiff(tmp_path: Path) -> None:
    image_path = tmp_path / "slide.ome.tiff"
    tifffile.imwrite(
        image_path,
        np.zeros((5, 7), dtype=np.uint8),
        ome=True,
        metadata={"axes": "YX"},
    )

    metadata = _he_image_metadata(str(image_path))

    assert metadata["he_format"] == "OME-TIFF"
    assert metadata["he_width_px"] == 7
    assert metadata["he_height_px"] == 5
    assert metadata["he_pixel_count"] == 35
    assert metadata["he_compression"] == "NONE"


def test_he_image_metadata_reads_svs_dimensions_from_tiff_header(tmp_path: Path) -> None:
    image_path = tmp_path / "slide.svs"
    Image.new("RGB", (7, 5)).save(image_path, format="TIFF")

    metadata = _he_image_metadata(str(image_path))

    assert metadata["he_format"] == "SVS"
    assert metadata["he_width_px"] == 7
    assert metadata["he_height_px"] == 5
    assert metadata["he_pixel_count"] == 35
    assert metadata["he_compression"] == "NONE"


def test_input_image_compressions_lists_tiff_and_raster_codecs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    morphology_path = input_dir / "morphology.ome.tif"
    tifffile.imwrite(
        morphology_path,
        np.zeros((5, 7), dtype=np.uint8),
        ome=True,
        metadata={"axes": "YX"},
        compression="deflate",
    )
    he_path = input_dir / "he.png"
    Image.new("RGB", (7, 5)).save(he_path)

    assert _image_compression(morphology_path) == "DEFLATE"
    assert _input_image_compressions(str(input_dir), str(he_path)) == [
        "morphology.ome.tif=DEFLATE",
    ]


def test_write_csv_creates_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "reports" / "benchmark.csv"

    _write_csv([], str(output_path))

    assert output_path.is_file()
    with output_path.open(newline="", encoding="utf-8") as csv_file:
        headers = next(csv.reader(csv_file))
    assert "status" in headers
    assert "files_processed" not in headers
    assert "files_failed" not in headers
    assert "morphology_file_size_gb" in headers
    assert "morphology_width_px" in headers
    assert "morphology_height_px" in headers
    assert "morphology_plane_count" in headers
    assert "transcript_zarr_mb" in headers
    assert "cell_zarr_mb" in headers
    assert "he_pixel_count" not in headers
    assert "morphology_pixel_count" not in headers
    assert "morphology_sample_count" not in headers
    assert "output_size_gb" not in headers
    summary_headers = {header for header, _ in _TABLE_HEADERS}
    assert "Status" in summary_headers
    assert "Files ok" not in summary_headers
    assert "Files fail" not in summary_headers

    metrics_output_path = tmp_path / "nested" / "reports" / "metrics-only.csv"
    _write_csv([], str(metrics_output_path), metrics_only=True)
    with metrics_output_path.open(newline="", encoding="utf-8") as csv_file:
        metrics_headers = next(csv.reader(csv_file))
    assert "status" not in metrics_headers
    assert "transcript_zarr_mb" in metrics_headers
    assert "cell_zarr_mb" in metrics_headers


def test_archive_size_mb_uses_root_input_and_decimal_units(tmp_path: Path) -> None:
    archive_path = tmp_path / "cells.zarr.zip"
    with archive_path.open("wb") as archive:
        archive.truncate(2_250_000)
    inventory = [{"relative_path": "nested/cells.zarr.zip", "size_bytes": 9_000_000}]

    assert _archive_size_mb(str(tmp_path), inventory, "cells.zarr.zip") == 2.25
    archive_path.unlink()
    assert _archive_size_mb(str(tmp_path), inventory, "cells.zarr.zip") is None
    inventory.append({"relative_path": "cells.zarr.zip", "size_bytes": 2_250_000})
    assert _archive_size_mb(str(tmp_path), inventory, "cells.zarr.zip") == 2.25


def test_missing_input_directory_does_not_scan_working_directory(
    monkeypatch: MonkeyPatch,
) -> None:
    def reject_scan(self: Path, pattern: str) -> None:
        raise AssertionError("Missing input directory must not trigger a filesystem scan")

    monkeypatch.setattr(Path, "rglob", reject_scan)

    assert _input_file_inventory("") == []
    assert _input_image_compressions("", None) == []


def test_images_only_discovery_and_statistics(tmp_path: Path) -> None:
    log_out = tmp_path / "xsplit_sample_images_only.out"
    log_out.write_text("Successfully completed.\n", encoding="utf-8")
    rows = _discover_from_logs(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["name"] == "sample"
    assert rows[0]["mode"] == "images_only"
    record = _assemble_run(rows[0])
    record["wall_s"] = 60
    record["peak_ram_gb"] = 2
    output = StringIO()
    with redirect_stdout(output):
        _print_statistics([record])
    assert "Mode: images_only (1 successful run(s))" in output.getvalue()


def test_parse_region_counts_from_splitter_log(tmp_path: Path) -> None:
    log_path = tmp_path / "job.err"
    log_path.write_text(
        "Updated metadata for region R1 (cells=5, transcripts=9, area_um2=12.0)\n"
        "Updated metadata for region R2 (cells=7, transcripts=11, area_um2=14.0)\n",
        encoding="utf-8",
    )

    counts = _parse_region_counts_from_log(str(log_path))

    assert counts == [
        {"region_id": "R1", "cells": 5, "transcripts": 9},
        {"region_id": "R2", "cells": 7, "transcripts": 11},
    ]


def test_assemble_run_falls_back_to_log_counts_and_image_shape(tmp_path: Path) -> None:
    log_out = tmp_path / "job.out"
    log_out.write_text("Successfully completed.\n", encoding="utf-8")
    log_err = tmp_path / "job.err"
    log_err.write_text(
        "Read slide.tif crop via memory-mapped TIFF: source_shape=(7, 9, 3), "
        "source_bbox=(0,0)-(5,5)\n"
        "Updated metadata for region R1 (cells=5, transcripts=9, area_um2=12.0)\n",
        encoding="utf-8",
    )

    record = _assemble_run(
        {
            "name": "sample",
            "mode": "with_images",
            "job_name": "xsplit_sample_with_images",
            "input_dir": "/unavailable/input",
            "he_image": "/unavailable/slide.tif",
            "lasso_file": "/unavailable/regions.csv",
            "output_dir": "",
            "log_out": str(log_out),
            "log_err": str(log_err),
        }
    )

    assert record["status"] == "SUCCESS"
    assert record["cells_total"] == 5
    assert record["cells_written_total"] == 5
    assert record["transcripts_total"] == 9
    assert record["he_width_px"] == 9
    assert record["he_height_px"] == 7
    assert record["he_pixel_count"] == 63

    default_output = StringIO()
    with redirect_stdout(default_output):
        _print_summary_table([record], "name")
    assert "Status" in default_output.getvalue()

    metrics_output = StringIO()
    with redirect_stdout(metrics_output):
        _print_summary_table([record], "name", metrics_only=True)
    assert "Status" not in metrics_output.getvalue()