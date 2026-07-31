# xenium-splitter — To-Do / Feature Backlog

This file tracks planned improvements, known gaps, and open questions. Items are grouped
by category and roughly ordered by priority within each section. Not all items are
fully scoped; some are exploratory.

---

## H&E Image Handling

### H&E Warping / Registration Support

Currently xenium-splitter crops and masks an H&E image purely by polygon coordinates,
with no spatial transformation applied. This assumes the H&E is already in the same
pixel space as the Xenium morphology data, which is often not the case.

**What needs to be added:**

- **Alignment file ingestion** — support reading a registration/alignment file that
  describes the transform between H&E and Xenium coordinate spaces. Candidate formats
  to consider:
  - 2D affine matrix (e.g., from QuPath, FIJI/BigWarp, or a custom CSV)
  - Displacement/warp field (e.g., from elastix, ANTs, or SimpleITK)
  - 10x-style JSON alignment file (if one exists for the instrument)
- **Affine warp application** — for linear transforms (translation, rotation, scale,
  shear), apply the matrix to the H&E before cropping so that the output image aligns
  with the Xenium morphology crop.
- **Non-linear warp application** — for deformable registration outputs, apply the
  displacement field to the H&E. This is more complex and likely requires an optional
  heavy dependency (e.g., `SimpleITK` or `itk`).
- **Coordinate transform direction** — clarify and document whether the alignment file
  maps H&E → Xenium or Xenium → H&E, and handle the inversion correctly.
- **Possibly: alignment scripts** — provide a small helper script or documented
  workflow for users who need to compute an alignment from scratch (e.g., using
  control points between DAPI and H&E, or using cross-modal image registration tools).
  This is out of scope for the core splitter but could live in a `scripts/` directory
  or companion repository.

**Current limitation to document:** The tool assumes the H&E uses the same pixel size
(`pixel_size_um`) as the Xenium morphology data. Any H&E at a different resolution or
with a non-trivial spatial offset will produce incorrect crops until this is implemented.

---

## Benchmarking

### Runtime and Memory Benchmarking Across Data Sizes

The RAM estimates in README.md are derived from static code analysis, not empirical
measurement. A systematic benchmark suite would:

- Confirm or revise the RAM estimates for different dataset sizes (number of
  transcripts, cells, image dimensions).
- Identify the slowest pipeline stages across a range of inputs.
- Quantify the runtime cost of individual options:
  - `--skip-images` vs. default (image processing overhead)
  - `--write-cell-feature-matrix-zarr` vs. `--skip-cell-feature-matrix-zarr`
  - `--recalculate-diffexp` vs. `--skip-diffexp-recalc`
  - `--copy-transcripts` vs. filtered zarr rebuild
  - SVS vs. TIFF H&E (windowed read vs. full load)
  - Number of regions (1 vs. 5 vs. 20)
- Measure peak RAM empirically with `tracemalloc` or `memory_profiler` rather than
  relying on theoretical array sizes.
- Run on real Xenium datasets at multiple scales:
  - Small: < 1 mm², ~500 K transcripts
  - Medium: ~10 mm², ~5 M transcripts
  - Large: ~50 mm², ~50 M transcripts
  - Very large: full slide, ≥ 100 M transcripts

**Output:** Update README RAM table with measured values, and flag any pipeline
stages whose complexity scales non-linearly with input size.

### Parameter Sensitivity Benchmarking

- `--include-glob` — measure overhead of glob-restricted runs vs. full directory scan.
- Number of output regions — measure how per-region loop overhead scales.
- Zarr chunk sizes / compression settings — explore whether tuning these improves
  write throughput for `transcripts.zarr.zip`.

---

## Cell Feature Matrix

- [ ] Investigate streaming / chunked MTX processing for very large matrices
      (current implementation loads the full sparse matrix into memory).
- [ ] Add optional output format conversion: MTX → HDF5 or MTX → AnnData `.h5ad`.
- [ ] Support additional sparse matrix formats as input (COO, CSR, CSC) for tools
      that export in non-MTX formats.
- [ ] Validate that `cell_feature_matrix.zarr.zip` output passes Xenium Explorer
      schema checks after region filtering.

---

## Transcript Processing

- [ ] Profile and tune the `filter_zarr_zip_by_row_indices_preserve_schema` function,
      which is the single most expensive step for large datasets. Investigate whether
      a streaming approach (avoiding full in-memory extraction) is feasible.
- [ ] Add a `--no-rebase` flag to allow users to skip coordinate rebasing and retain
      source-space coordinates when downstream tools expect them.
- [ ] Expose the FOV dimension overrides as CLI options so users with non-standard
      instrument configurations can override the version-based lookup table.

---

## Image Processing

- [ ] Add proper windowed-read support for large TIFF/OME-TIFF files (currently the
      full image is always loaded even for TIFF — only SVS benefits from windowed
      reads). Consider using `tifffile` zarr-based access or `rasterio` for tiled TIFFs.
- [ ] Validate OME-TIFF outputs against Xenium Explorer's expected pyramid structure
      (tile size, sub-IFD layout, resolution tags).
- [ ] Add support for multi-channel fluorescence images beyond the Xenium morphology
      stack (e.g., additional IF channels provided as separate TIFF files).

---

## Testing

- [ ] Add integration tests for CFM filtering (`test_cfm.py` was referenced in older
      dev notes but never created — only `test_lasso.py` currently exists).
- [ ] Add tests for `update_experiment_xenium_for_region` and `copy_gene_panel`.
- [ ] Add a test for the multi-format file grouping path (`_split_file_group`).
- [ ] Add round-trip tests: split → verify output zarr schema → re-open in Xenium Explorer
      (manual step, but a schema validation script would help automate the checks).
- [ ] Add RAM/performance regression tests to catch accidental full-load paths being
      introduced during refactoring.

---

## CLI / Usability

- [ ] Add a `--dry-run` flag that lists which files would be processed and estimates
      RAM usage without writing any output.
- [ ] Add progress bars (e.g., via `tqdm` or `typer`'s progress support) for the
      main per-file loop and for zarr/CFM processing.
- [ ] Add a `--resume` or `--skip-existing` mode to avoid re-processing regions that
      already have output directories.
- [ ] Emit a machine-readable JSON summary alongside `run_metadata_README.md` for
      programmatic consumption by downstream pipelines.

---

## Documentation

- [ ] Add empirical RAM measurements once benchmarking is complete (replace estimates
      in README with measured values).
- [ ] Document the expected H&E coordinate space and pixel size requirements more
      prominently — this is the most common misconfiguration.
- [ ] Add a troubleshooting section to README covering common failure modes (e.g.,
      empty region outputs, misaligned H&E crops, zarr schema errors).
- [ ] Write a short tutorial showing an end-to-end run with the GSM7990532 sample data
      already in the `data/` directory.
