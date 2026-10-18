"""Tests for configurable geographic workflow coordinates."""

import pytest

from storm_water_management.utils_workflows import projected_center


def test_projected_center_converts_supplied_wgs84_coordinates():
    """The projected center responds to longitude/latitude inputs."""
    enkoping = projected_center(17.0778, 59.6356)
    nearby = projected_center(17.0878, 59.6356)

    assert nearby[0] > enkoping[0]
    assert nearby != enkoping


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
