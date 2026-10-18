"""Tests for generic elevation-download area and naming helpers."""

import pytest

from examples.download_dem import bbox_for_radius, location_slug


def test_bbox_for_radius_uses_supplied_center():
    """Radius bounds are centered on the requested WGS84 coordinate."""
    bounds = bbox_for_radius(10, longitude=10.0, latitude=50.0)

    assert bounds[0] < 10.0 < bounds[2]
    assert bounds[1] < 50.0 < bounds[3]
    assert bounds[0] == pytest.approx(20.0 - bounds[2])
    assert bounds[1] == pytest.approx(100.0 - bounds[3])


def test_location_slug_transliterates_and_sanitizes_names():
    """Location names produce safe portable output-directory components."""
    assert location_slug("Enköping, Sweden") == "enkoping-sweden"
    assert location_slug("  St. John's / North  ") == "st-john-s-north"


@pytest.mark.parametrize(
    ("radius_km", "longitude", "latitude"),
    [
        (0, 10, 50),
        (float("nan"), 10, 50),
        (10, 181, 50),
        (10, 10, -91),
        (10, 10, 90),
    ],
)
def test_bbox_for_radius_rejects_invalid_inputs(radius_km, longitude, latitude):
    """Invalid extents and polar radius bounds are rejected."""
    with pytest.raises(ValueError):
        bbox_for_radius(radius_km, longitude, latitude)


def test_location_slug_rejects_empty_slug():
    """A name that cannot form a directory slug is rejected."""
    with pytest.raises(ValueError, match="location name"):
        location_slug("東京")
