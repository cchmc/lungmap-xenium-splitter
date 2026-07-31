from __future__ import annotations

from pathlib import Path

import typer

from xenium_splitter.io_utils import validate_transcripts_id_uuid_schema

app = typer.Typer(help="Validate ID/UUID schema invariants for transcripts.zarr.zip files.")


@app.callback(invoke_without_command=True)
def main(
    zarr_zip: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Validate that transcript identity fields follow Xenium schema.

    Required per-row invariant (when id and uuid are present):
    - id[:,0] == uuid[:,0]
    - uuid[:,1] == 65536 + id[:,1]
    """
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr is required for transcript zarr validation") from exc

    store = zarr.ZipStore(zarr_zip, mode="r")
    try:
        root = zarr.open(store, mode="r")
        summary = validate_transcripts_id_uuid_schema(root)
    finally:
        store.close()

    typer.echo(
        "PASS: "
        f"rows={int(summary.get('checked_rows', 0))} "
        f"tiles={int(summary.get('checked_tiles', 0))} "
        f"max_id1={int(summary.get('max_id1', -1))}"
    )


if __name__ == "__main__":
    app()
