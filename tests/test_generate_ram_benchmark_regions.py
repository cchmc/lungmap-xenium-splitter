import csv
from pathlib import Path

from xenium_splitter.lasso import load_lasso_regions
from scripts.generate_ram_benchmark_regions import SPLIT_FACTORS, generate


def test_generator_creates_area_preserving_region_variants(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "sample_lasso.csv"
    source_path.write_text(
        "region_id,polygon_wkt\n"
        'A,"POLYGON ((0 0, 100 0, 100 40, 0 40, 0 0))"\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "base.csv"
    config_path.write_text(
        "name,input_dir,he_image,lasso_file,walltime_data_only,ram_gb_data_only,"
        "walltime_with_images,ram_gb_with_images\n"
        "sample,/data/sample,,/data/sample_lasso.csv,01:00,8,02:00,16\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "generated"
    output_config = tmp_path / "ram_config.csv"

    rows = generate(
        benchmark_config=config_path,
        source_dir=source_dir,
        output_dir=output_dir,
        output_config=output_config,
        cluster_region_dir="/cluster/regions/ram_runtime",
    )

    assert len(rows) == len(SPLIT_FACTORS)
    for factor, row in zip(SPLIT_FACTORS, rows):
        generated_path = output_dir / f"sample_regions_x{factor}.csv"
        generated_regions = load_lasso_regions(generated_path)
        assert len(generated_regions) == factor
        assert sum(region.polygon.area for region in generated_regions) == 4000
        assert row["lasso_file"] == f"/cluster/regions/ram_runtime/sample_regions_x{factor}.csv"

    with output_config.open(newline="", encoding="utf-8") as file:
        config_rows = list(csv.DictReader(file))
    assert len(config_rows) == 5
