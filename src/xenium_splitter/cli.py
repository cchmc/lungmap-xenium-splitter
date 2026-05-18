from __future__ import annotations

import logging
from pathlib import Path
import shutil
import tempfile

import typer

from xenium_splitter.models import SplitConfig
from xenium_splitter.splitter import run_split

app = typer.Typer(help="Split Xenium outputs by LASSO regions.")


def _configure_app_logging(verbose: bool) -> None:
    """Configure logging for xenium_splitter only.

    Avoid configuring the root logger so dependency loggers like zarr do not
    emit through this CLI unless the caller configures them separately.
    """
    package_logger = logging.getLogger("xenium_splitter")
    package_logger.handlers.clear()
    package_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    package_logger.propagate = False

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    package_logger.addHandler(handler)


@app.command("clean-temp")
def clean_temp_command() -> None:
    """Delete xenium-splitter temporary working files."""
    temp_root = Path(tempfile.gettempdir()) / "xenium_splitter"
    if not temp_root.exists():
        typer.echo(f"No temp directory found at: {temp_root}")
        return

    shutil.rmtree(temp_root)
    typer.echo(f"Deleted temp directory: {temp_root}")


@app.command("split")
def split_command(
    input_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    lasso_file: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    output_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
    he_image: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
    convert_svs_to_ome: bool = typer.Option(
        False,
        help="If --he-image is .svs, also emit OME-TIFF outputs.",
    ),
    squash_layers: bool = typer.Option(
        True,
        "--squash-layers/--no-squash-layers",
        help="Flatten multi-layer image stacks when possible.",
    ),
    include_glob: list[str] = typer.Option(
        None,
        help="Optional glob patterns (repeatable) to limit files from input-dir.",
    ),
    skip_images: bool = typer.Option(
        False,
        "--skip-images/--process-images",
        help="Skip image extraction to reduce memory usage.",
    ),
    recalculate_diffexp: bool = typer.Option(
        True,
        "--recalculate-diffexp/--skip-diffexp-recalc",
        help="Recompute analysis/diffexp files from region-filtered matrix and clustering outputs.",
    ),
    write_cell_feature_matrix_zarr: bool = typer.Option(
        True,
        "--write-cell-feature-matrix-zarr/--skip-cell-feature-matrix-zarr",
        help="Write cell_feature_matrix.zarr.zip outputs (disable to benchmark runtime impact).",
    ),
    copy_transcripts: bool = typer.Option(
        False,
        "--copy-transcripts",
        help="Copy transcript files verbatim to each region output with no filtering or rebasing.",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable debug logging to see detailed processing info.",
    ),
) -> None:
    """Split Xenium files and optional H&E image into per-region outputs."""
    _configure_app_logging(verbose)
    config = SplitConfig(
        input_dir=input_dir,
        lasso_file=lasso_file,
        output_dir=output_dir,
        he_image=he_image,
        convert_svs_to_ome=convert_svs_to_ome,
        squash_layers=squash_layers,
        include_globs=include_glob or [],
        skip_images=skip_images,
        recalculate_diffexp=recalculate_diffexp,
        write_cell_feature_matrix_zarr=write_cell_feature_matrix_zarr,
        copy_transcripts=copy_transcripts,
    )
    metrics, metadata_path = run_split(config)
    typer.echo("Split complete.")
    typer.echo(f"Regions: {metrics.region_count}")
    typer.echo(f"Files processed: {metrics.files_processed}")
    typer.echo(f"Files skipped: {metrics.files_skipped}")
    typer.echo(f"Files failed: {metrics.files_failed}")
    typer.echo(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    app()
