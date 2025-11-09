"""Tests for geojson utils."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from storm_water_management.geojson_utils import (
    write_geojson_points_from_tif_to_file,
    write_geojson_polygons_from_tif_to_file,
)


@pytest.fixture
def small_tif(tmp_path):
    """Create a small GeoTIFF file for testing."""
    data = np.array(
        [
            [0.0, 0.15, 0.25],
            [0.6, 0.8, 1.2],
            [0.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    transform = from_origin(0, 3, 1, 1)  # x_min, y_max, x_res, y_res
    tif_path = tmp_path / "test.tif"

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return tif_path


def test_geojson_output_created(small_tif):
    """Test that the GeoJSON file is created and contains correct fields."""
    out_path = small_tif.with_name("test_polygons.geojson")

    write_geojson_polygons_from_tif_to_file(str(small_tif))

    assert out_path.exists(), (
        f"GeoJSON file should have been created. Out path: {out_path}"
    )
    gdf = gpd.read_file(out_path)
    assert not gdf.empty, "GeoDataFrame should not be empty."
    assert "depth_class" in gdf.columns, "Missing depth_class column."
    valid_labels = {"0.1-0.2 m", "0.2-0.5 m", "0.5-1 m", "1 <"}
    assert set(gdf["depth_class"]).issubset(valid_labels)


def test_geojson_points_output_created(small_tif):
    """Test that the GeoJSON file is created and contains correct fields."""
    out_path = small_tif.with_name("test_points.geojson")

    write_geojson_points_from_tif_to_file(str(small_tif))

    assert out_path.exists(), (
        f"GeoJSON file should have been created. Out path: {out_path}"
    )
    gdf = gpd.read_file(out_path)
    assert not gdf.empty, "GeoDataFrame should not be empty."
    assert "depth" in gdf.columns, "Missing depth column."
