from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon


@dataclass(slots=True)
class LassoRegion:
    region_id: str
    polygon: Polygon

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.polygon.bounds


@dataclass(slots=True)
class SplitConfig:
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


@dataclass(slots=True)
class FileMetric:
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
    files_total: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    region_count: int = 0
    file_metrics: list[FileMetric] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
