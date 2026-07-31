from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from xenium_splitter.io_utils import get_cell_feature_matrix_files, read_mtx_file

app = typer.Typer(help="Recalculate differential_expression.csv files from filtered region outputs.")


def recalculate_diffexp_for_region(
    region_dir: Path,
    cfm_dir: Path | None = None,
    cfm_h5: Path | None = None,
    analysis_dir: Path | None = None,
    pseudocount: float = 1e-9,
) -> dict[str, int]:
    """Recompute diffexp CSV files for all clustering outputs in a region output directory.

    Returns:
        Mapping of source-relative diffexp file path to written row count.
    """
    chosen_cfm = cfm_dir or (region_dir / "cell_feature_matrix")
    chosen_cfm_h5 = cfm_h5 or (region_dir / "cell_feature_matrix.h5")
    chosen_analysis = analysis_dir or (region_dir / "analysis")

    if not chosen_analysis.exists():
        raise ValueError(f"analysis directory not found: {chosen_analysis}")

    if chosen_cfm.exists():
        expr, barcodes, feature_meta = _load_expression_from_cfm(chosen_cfm)
    elif chosen_cfm_h5.exists():
        expr, barcodes, feature_meta = _load_expression_from_cfm_h5(chosen_cfm_h5)
    else:
        raise ValueError(
            f"No matrix input found. Expected either directory {chosen_cfm} or file {chosen_cfm_h5}"
        )

    clustering_dirs = _find_clustering_dirs(chosen_analysis)
    if not clustering_dirs:
        raise ValueError(
            f"No clustering directories with clusters.csv found under: {chosen_analysis / 'clustering'}"
        )

    written: dict[str, int] = {}
    for cluster_dir in clustering_dirs:
        clusters_path = cluster_dir / "clusters.csv"
        clusters_df = pd.read_csv(clusters_path)
        if not {"Barcode", "Cluster"}.issubset(clusters_df.columns):
            continue

        diffexp_df = _compute_diffexp_table(
            expr=expr,
            barcodes=barcodes,
            feature_meta=feature_meta,
            clusters_df=clusters_df,
            pseudocount=pseudocount,
        )

        diffexp_dir = chosen_analysis / "diffexp" / cluster_dir.name
        diffexp_dir.mkdir(parents=True, exist_ok=True)
        output_csv = diffexp_dir / "differential_expression.csv"
        diffexp_df.to_csv(output_csv, index=False)
        rel_path = str(Path("analysis") / "diffexp" / cluster_dir.name / "differential_expression.csv")
        written[rel_path] = int(len(diffexp_df))

    return written


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction to an array of p-values.

    Returns an array of adjusted p-values in the same order as the input.
    Values are clipped to [0, 1].
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)

    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = (ranked[i] * n) / rank
        running = min(running, val)
        adjusted[i] = running

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _read_features(features_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 10X Genomics features file into ``(feature_ids, feature_names)`` arrays.

    Supports plain and gzip-compressed TSV files.  Lines with a single field
    are used as both ID and name.
    """
    open_fn = gzip.open if str(features_path).endswith(".gz") else open
    rows: list[list[str]] = []
    with open_fn(features_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                rows.append([parts[0], parts[0]])
            else:
                rows.append([parts[0], parts[1]])
    arr = np.asarray(rows, dtype=object)
    return arr[:, 0], arr[:, 1]


def _load_expression_from_cfm(cfm_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load expression matrix, barcodes, and feature metadata from a ``cell_feature_matrix`` directory.

    Returns:
        ``(expr, barcodes, feature_meta)`` where ``expr`` has shape
        ``(n_features, n_cells)`` and ``feature_meta`` has shape ``(n_features, 2)``
        with columns ``[feature_id, feature_name]``.

    Raises:
        ValueError: If required files (barcodes, features, matrix) are missing or dimensions mismatch.
    """
    files = get_cell_feature_matrix_files(cfm_dir)
    if not files["barcodes"] or not files["features"] or not files["matrix"]:
        raise ValueError(f"Missing barcodes/features/matrix in {cfm_dir}")

    barcode_path = files["barcodes"]
    features_path = files["features"]
    matrix_path = files["matrix"]

    with (gzip.open(barcode_path, "rt", encoding="utf-8") if str(barcode_path).endswith(".gz") else open(barcode_path, "rt", encoding="utf-8")) as f:
        barcodes = np.asarray([line.strip() for line in f if line.strip()], dtype=object)

    feature_ids, feature_names = _read_features(features_path)
    rows, (n_features, n_cells, _) = read_mtx_file(matrix_path)

    expr = np.zeros((n_features, n_cells), dtype=np.float64)
    for row_idx, col_idx, val in rows:
        expr[row_idx - 1, col_idx - 1] = float(val)

    if feature_ids.shape[0] != n_features:
        raise ValueError("Feature file length does not match matrix feature dimension")
    if barcodes.shape[0] != n_cells:
        raise ValueError("Barcode file length does not match matrix cell dimension")

    return expr, barcodes, np.stack([feature_ids, feature_names], axis=1)


def _decode_bytes_array(arr: np.ndarray) -> np.ndarray:
    """Decode a NumPy array of bytes/bytearray elements to an object array of Python strings."""
    out: list[str] = []
    for value in arr:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def _load_expression_from_cfm_h5(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load expression matrix, barcodes, and feature metadata from a ``cell_feature_matrix.h5`` file.

    The HDF5 schema must contain a ``matrix`` group with ``barcodes``, ``data``,
    ``indices``, ``indptr``, ``shape``, and ``features`` datasets (10X Genomics format).

    Returns:
        ``(expr, barcodes, feature_meta)`` where ``expr`` has shape
        ``(n_features, n_cells)``.

    Raises:
        RuntimeError: If ``h5py`` is not installed.
        ValueError: If the HDF5 schema is missing required datasets.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read cell_feature_matrix.h5") from exc

    with h5py.File(h5_path, "r") as f:
        if "matrix" not in f:
            raise ValueError(f"Invalid cell_feature_matrix.h5 (missing matrix group): {h5_path}")

        m = f["matrix"]
        required = {"barcodes", "data", "indices", "indptr", "shape", "features"}
        if not required.issubset(set(m.keys())):
            raise ValueError(f"Invalid cell_feature_matrix.h5 schema: {h5_path}")

        barcodes = _decode_bytes_array(m["barcodes"][:])
        feature_ids = _decode_bytes_array(m["features"]["id"][:])
        feature_names = _decode_bytes_array(m["features"]["name"][:])

        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = m["shape"][:]
        n_features, n_cells = int(shape[0]), int(shape[1])

    expr = np.zeros((n_features, n_cells), dtype=np.float64)
    for col in range(n_cells):
        start = int(indptr[col])
        end = int(indptr[col + 1])
        if start == end:
            continue
        rows = indices[start:end].astype(int)
        expr[rows, col] = data[start:end].astype(np.float64)

    return expr, barcodes, np.stack([feature_ids, feature_names], axis=1)


def _normal_approx_pvals(group: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Compute two-sided p-values using a normal approximation to Welch's t-test.

    Args:
        group: Expression matrix for cells in the cluster, shape ``(n_features, n_group)``.
        rest: Expression matrix for all other cells, shape ``(n_features, n_rest)``.

    Returns:
        Array of p-values, one per feature.  Returns 1.0 for features where the
        standard error is zero or either group has fewer than 2 cells.
    """
    n1 = group.shape[1]
    n2 = rest.shape[1]
    if n1 < 2 or n2 < 2:
        return np.ones(group.shape[0], dtype=float)

    m1 = group.mean(axis=1)
    m2 = rest.mean(axis=1)
    v1 = group.var(axis=1, ddof=1)
    v2 = rest.var(axis=1, ddof=1)

    se = np.sqrt((v1 / n1) + (v2 / n2))
    se = np.where(se <= 0, np.nan, se)
    t_stat = (m1 - m2) / se

    # Two-sided p-value using normal approximation to Welch's t statistic.
    pvals = np.empty_like(t_stat, dtype=float)
    sqrt2 = math.sqrt(2.0)
    for i, t in enumerate(t_stat):
        if np.isnan(t):
            pvals[i] = 1.0
        else:
            pvals[i] = math.erfc(abs(float(t)) / sqrt2)
    return np.clip(pvals, 0.0, 1.0)


def _compute_diffexp_table(
    expr: np.ndarray,
    barcodes: np.ndarray,
    feature_meta: np.ndarray,
    clusters_df: pd.DataFrame,
    pseudocount: float,
) -> pd.DataFrame:
    """Compute a differential-expression table for all clusters.

    For each unique cluster label, computes:
    - Mean counts in cluster.
    - Log2 fold change (cluster mean vs. all other cells, with ``pseudocount``).
    - BH-adjusted p-value from a normal approximation to Welch's t-test.

    Args:
        expr: Expression matrix, shape ``(n_features, n_cells)``.
        barcodes: Barcode array of length ``n_cells``.
        feature_meta: Array of shape ``(n_features, 2)`` with columns
            ``[feature_id, feature_name]``.
        clusters_df: DataFrame with columns ``Barcode`` and ``Cluster``.
        pseudocount: Small constant added to means before log2 to avoid log(0).

    Returns:
        DataFrame with one row per feature and three columns per cluster
        (mean counts, log2 fold change, adjusted p-value).

    Raises:
        ValueError: If no barcodes overlap between ``clusters_df`` and the matrix.
    """
    clusters_df = clusters_df.copy()
    clusters_df["Barcode"] = clusters_df["Barcode"].astype(str)

    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes.tolist())}
    clusters_df = clusters_df[clusters_df["Barcode"].isin(barcode_to_idx)]
    if clusters_df.empty:
        raise ValueError("No overlapping barcodes between clusters.csv and matrix barcodes")

    clusters_df["idx"] = clusters_df["Barcode"].map(barcode_to_idx)
    labels = clusters_df["Cluster"].astype(str)

    def _sort_key(x: str) -> tuple[int, str]:
        return (0, f"{int(x):09d}") if x.isdigit() else (1, x)

    unique_labels = sorted(labels.unique().tolist(), key=_sort_key)

    out = pd.DataFrame(
        {
            "Feature ID": feature_meta[:, 0],
            "Feature Name": feature_meta[:, 1],
        }
    )

    all_idxs = clusters_df["idx"].to_numpy(dtype=int)

    for label in unique_labels:
        in_idxs = clusters_df.loc[labels == label, "idx"].to_numpy(dtype=int)
        out_idxs = np.setdiff1d(all_idxs, in_idxs, assume_unique=False)
        if in_idxs.size == 0 or out_idxs.size == 0:
            mean_in = np.zeros(expr.shape[0], dtype=float)
            log2fc = np.zeros(expr.shape[0], dtype=float)
            adj = np.ones(expr.shape[0], dtype=float)
        else:
            in_expr = expr[:, in_idxs]
            out_expr = expr[:, out_idxs]
            mean_in = in_expr.mean(axis=1)
            log2fc = np.log2((mean_in + pseudocount) / (out_expr.mean(axis=1) + pseudocount))
            pvals = _normal_approx_pvals(in_expr, out_expr)
            adj = _bh_adjust(pvals)

        out[f"Cluster {label} Mean Counts"] = mean_in
        out[f"Cluster {label} Log2 fold change"] = log2fc
        out[f"Cluster {label} Adjusted p value"] = adj

    return out


def _find_clustering_dirs(analysis_dir: Path) -> list[Path]:
    """Return all clustering subdirectories that contain a ``clusters.csv`` file."""
    clustering_root = analysis_dir / "clustering"
    if not clustering_root.exists():
        return []
    return sorted(p for p in clustering_root.iterdir() if p.is_dir() and (p / "clusters.csv").exists())


@app.command("run")
def run_recalculate_diffexp(
    region_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    cfm_dir: Path | None = typer.Option(None, file_okay=False, dir_okay=True),
    cfm_h5: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
    analysis_dir: Path | None = typer.Option(None, file_okay=False, dir_okay=True),
    pseudocount: float = typer.Option(1e-9, help="Pseudocount for log2 fold change."),
) -> None:
    """Recompute diffexp CSV files for all clustering outputs under a region directory."""
    try:
        written = recalculate_diffexp_for_region(
            region_dir=region_dir,
            cfm_dir=cfm_dir,
            cfm_h5=cfm_h5,
            analysis_dir=analysis_dir,
            pseudocount=pseudocount,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Done. Recalculated diffexp for {len(written)} clustering result(s).")


if __name__ == "__main__":
    app()
