"""Estimate peak RAM from the encoded Xenium Zarr archive sizes."""
from pathlib import Path


def estimate_ram(input_dir: Path) -> float:
    """Return estimated peak RAM in GB using archive sizes in decimal MB.

    Peak RAM = 2.5 + 0.010 * transcript MB + 0.007 * cell MB.
    Both archives must exist directly inside the Xenium output directory.
    No archive contents are read or extracted.
    """
    transcript_path = input_dir / "transcripts.zarr.zip"
    cell_path = input_dir / "cells.zarr.zip"
    for archive_path in (transcript_path, cell_path):
        if not archive_path.is_file():
            raise FileNotFoundError(f"Required Zarr archive not found: {archive_path}")

    transcript_mb = transcript_path.stat().st_size / 1_000_000
    cell_mb = cell_path.stat().st_size / 1_000_000
    return 2.5 + 0.010 * transcript_mb + 0.007 * cell_mb