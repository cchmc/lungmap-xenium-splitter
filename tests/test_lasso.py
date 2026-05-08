from pathlib import Path

from xenium_splitter.lasso import load_lasso_regions


def test_load_regions_from_geojson(tmp_path: Path) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"region_id": "A"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                },
            }
        ],
    }
    lasso_file = tmp_path / "lasso.geojson"
    lasso_file.write_text(__import__("json").dumps(payload), encoding="utf-8")

    regions = load_lasso_regions(lasso_file)
    assert len(regions) == 1
    assert regions[0].region_id == "A"


def test_load_regions_from_csv_points(tmp_path: Path) -> None:
    lasso_file = tmp_path / "lasso.csv"
    lasso_file.write_text(
        "region_id,x,y\nR1,0,0\nR1,5,0\nR1,5,5\nR1,0,5\n",
        encoding="utf-8",
    )

    regions = load_lasso_regions(lasso_file)
    assert len(regions) == 1
    assert regions[0].region_id == "R1"
