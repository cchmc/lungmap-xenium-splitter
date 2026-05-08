# Metadata Update Integration Guide

## New Functions Added to `io_utils.py`

Two new functions have been added to handle metadata file updates per region:

### 1. `copy_gene_panel(input_dir: Path, output_dir: Path) -> bool`

Copies `gene_panel.json` from input to region output directory (unchanged across regions).

### 2. `update_experiment_xenium_for_region(...)`

Updates `experiment.xenium` with region-specific metadata:

- `region_name`: New region ID
- `num_cells`: Filtered cell count for region
- `transcripts_per_cell`: Calculated as `num_transcripts / num_cells`
- `transcripts_per_100um`: Calculated as `(num_transcripts / region_area_um2) * 100`

Also creates `_region_metadata.json` with area and transcript information.

## Integration Example

Add this to the end of `run_split()` in `splitter.py`, after all regions have been processed:

```python
# Update metadata files for each region
for region in regions:
    region_dir = config.output_dir / f"region_{region.region_id}"
    region_area_um2 = region.polygon.area

    # Copy gene_panel.json
    copy_gene_panel(config.input_dir, region_dir)

    # Get region-specific metrics
    # Count cells and transcripts from processed files
    num_cells = metrics.extra.get("entity_counts_by_region", {}).get(region.region_id, {}).get("cells", 0)

    # For transcripts, check if transcripts file was processed
    # If available in output, count from the filtered transcripts file
    transcripts_file = region_dir / "transcripts.parquet"
    num_transcripts = 0
    if transcripts_file.exists():
        try:
            transcripts_df = read_table(transcripts_file)
            num_transcripts = len(transcripts_df)
        except:
            pass

    # Update experiment.xenium with region-specific metadata
    update_experiment_xenium_for_region(
        config.input_dir,
        region_dir,
        region.region_id,
        num_cells,
        num_transcripts,
        region_area_um2,
    )
    logger.info(f"Updated metadata for region {region.region_id}: {num_cells} cells, {region_area_um2:.2f} um²")
```

## Required Imports

Add to the imports in `splitter.py`:

```python
from xenium_splitter.io_utils import (
    # ... existing imports ...
    copy_gene_panel,
    update_experiment_xenium_for_region,
)
```

## Polygon Area Reference

Region polygon area is accessible via:

```python
region_area_um2 = region.polygon.area  # in square micrometers (coordinate space)
```

This is a Shapely `Polygon` object property that automatically calculates area using the polygon's coordinates.
