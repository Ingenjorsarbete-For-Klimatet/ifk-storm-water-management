"""Tests for configurable geographic workflow coordinates."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from storm_water_management.terrain import create_buffered_mosaic, projected_center


def test_projected_center_converts_supplied_wgs84_coordinates():
    """The projected center responds to longitude/latitude inputs."""
    enkoping = projected_center(17.0778, 59.6356)
    nearby = projected_center(17.0878, 59.6356)

    assert nearby[0] > enkoping[0]


@pytest.mark.parametrize(
    ("longitude", "latitude", "message"),
    [
        (181.0, 59.0, "Longitude"),
        (17.0, -91.0, "Latitude"),
        (float("nan"), 59.0, "Longitude"),
        (17.0, float("inf"), "Latitude"),
    ],
)
def test_projected_center_rejects_invalid_coordinates(longitude, latitude, message):
    """Invalid geographic coordinates fail with a useful validation error."""
    with pytest.raises(ValueError, match=message):
        projected_center(longitude, latitude)


def _write_constant_tile(path, transform, value):
    """Write a 10x10 float32 elevation tile with a constant value."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:3006",
        transform=transform,
        nodata=-9.0,
    ) as destination:
        destination.write(np.full((10, 10), value, dtype=np.float32), 1)


def test_create_buffered_mosaic_leaves_no_nodata_strip_at_fractional_bounds(
    tmp_path,
):
    """Fractional bounds must not create a nodata strip inside covered tiles."""
    west_tile = tmp_path / "west.tif"
    east_tile = tmp_path / "east.tif"
    _write_constant_tile(west_tile, from_origin(100, 200, 1, 1), 1.0)
    _write_constant_tile(east_tile, from_origin(110, 200, 1, 1), 2.0)
    output_path = tmp_path / "mosaic.tif"

    create_buffered_mosaic(
        [west_tile, east_tile], (102.4, 192.5, 116.6, 197.7), output_path
    )

    with rasterio.open(output_path) as source:
        mosaic = source.read(1)
        assert source.bounds == (102.0, 192.0, 117.0, 198.0)
    assert set(np.unique(mosaic)) == {1.0, 2.0}
