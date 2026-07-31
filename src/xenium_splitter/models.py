from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon


@dataclass(slots=True)
class LassoRegion:
    """A named polygonal region parsed from a LASSO selection file."""

    region_id: str
    polygon: Polygon

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) bounding box of the polygon in coordinate units."""
        return self.polygon.bounds


@dataclass(slots=True)
class SplitConfig:
    """Configuration for a single xenium-splitter run.

    Collects all user-supplied options and is passed through the pipeline.
    ``pixel_size_um`` is populated from ``experiment.xenium`` at runtime when not
    provided by the caller.

    When ``images_only`` is ``True``, all data-processing stages (boundary
    extraction, CFM filtering, tabular/zarr/HDF5 splitting, diffexp
    recalculation) are skipped.  Only image cropping/masking, morphology MIP/
    focus generation, and grid overlays are performed.  Mutually exclusive with
    ``skip_images``.
    """

    input_dir: Path
    lasso_file: Path
    output_dir: Path
    he_image: Path | None = None
    convert_svs_to_ome: bool = False
    squash_layers: bool = True
    include_globs: list[str] = field(default_factory=list)
    pixel_size_um: float | None = None
    skip_images: bool = False
    overlays: bool = False
    recalculate_diffexp: bool = True
    write_cell_feature_matrix_zarr: bool = True
    copy_transcripts: bool = False
    images_only: bool = False


@dataclass(slots=True)
class FileMetric:
    """Per-file processing record written to run metadata.

    Tracks the outcome (status), filter method (detail), row counts, and wall-clock
    duration for every file encountered during a split run.
    """

    source_path: str
    file_type: str
    status: str
    detail: str = ""
    duration_s: float | None = None
    rows_input: int | None = None
    rows_written_total: int | None = None
    rows_written_by_region: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class RunMetrics:
    """Aggregate statistics for a completed split run.

    ``extra`` is an open-ended dict used by pipeline stages to attach structured
    data (entity counts, timing breakdowns, FOV layout summaries, etc.) that is
    later rendered into the run metadata README.
    """

    files_total: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    region_count: int = 0
    file_metrics: list[FileMetric] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
