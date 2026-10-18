"""Tests for the rainfall and runoff screening calculations."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from storm_water_management.utils_workflows import (
    effective_rainfall_m,
    estimate_ponding_depth_m,
    estimate_runoff_volume_m3,
    write_estimated_ponding_depth,
    write_estimated_runoff_volume,
)


def test_effective_rainfall_defaults_to_fifty_percent_infiltration():
    """The default infiltration assumption halves event rainfall."""
    assert effective_rainfall_m(100.0) == pytest.approx(0.05)


def test_effective_rainfall_uses_constant_infiltration_capacity():
    """A constant infiltration capacity is applied over the storm duration."""
    assert effective_rainfall_m(
        80.0,
        infiltration_capacity_mm_h=10.0,
        rainfall_duration_hours=3.0,
    ) == pytest.approx(0.05)


def test_effective_rainfall_cannot_be_negative():
    """Infiltration exceeding rainfall leaves no event runoff."""
    assert effective_rainfall_m(
        10.0,
        infiltration_capacity_mm_h=5.0,
        rainfall_duration_hours=3.0,
    ) == 0.0


def test_runoff_volume_subtracts_upslope_storage():
    """Residual runoff depth is multiplied by contributing area."""
    area = np.array([100.0, 250.0])
    storage = np.array([0.01, 0.06])

    result = estimate_runoff_volume_m3(area, storage, effective_rain_m=0.05)

    np.testing.assert_allclose(result, [4.0, 0.0])


def test_ponding_depth_is_bounded_by_rainfall_and_depression_depth():
    """The rainfall-conditioned depth cannot exceed rain or local depression fill."""
    storage = np.array([0.001, 0.0, 0.006, 0.0])
    depression = np.array([0.02, 0.02, 0.02, 0.003])

    result = estimate_ponding_depth_m(storage, depression, effective_rain_m=0.005)

    np.testing.assert_allclose(result, [0.004, 0.005, 0.0, 0.003])
    assert np.all(result <= 0.005)


def test_runoff_volume_writer_preserves_nodata(tmp_path):
    """The generated volume raster keeps invalid cells as NoData."""
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3006",
        "transform": from_origin(0, 2, 1, 1),
        "nodata": -9999.0,
    }
    area_path = tmp_path / "area.tif"
    storage_path = tmp_path / "storage.tif"
    with rasterio.open(area_path, "w", **profile) as destination:
        destination.write(np.array([[100, 200], [300, -9999]], dtype=np.float32), 1)
    with rasterio.open(storage_path, "w", **profile) as destination:
        destination.write(np.array([[0.01, 0.06], [0.02, -9999]], dtype=np.float32), 1)

    output_path = write_estimated_runoff_volume(
        area_path, storage_path, tmp_path / "volume.tif", effective_rain_m=0.05
    )

    with rasterio.open(output_path) as result:
        np.testing.assert_allclose(
            result.read(1), np.array([[4.0, 0.0], [9.0, -9999]], dtype=np.float32)
        )


def test_ponding_depth_writer_preserves_nodata_and_rainfall_scale(tmp_path):
    """The event-depth raster is in metres and respects the 5 mm input limit."""
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3006",
        "transform": from_origin(0, 2, 1, 1),
        "nodata": -9999.0,
    }
    storage_path = tmp_path / "storage_depth.tif"
    depression_path = tmp_path / "depression_depth.tif"
    with rasterio.open(storage_path, "w", **profile) as destination:
        destination.write(np.array([[0.001, 0.0], [0.006, -9999]], dtype=np.float32), 1)
    with rasterio.open(depression_path, "w", **profile) as destination:
        destination.write(np.array([[0.02, 0.003], [0.02, -9999]], dtype=np.float32), 1)

    output_path = write_estimated_ponding_depth(
        storage_path,
        depression_path,
        tmp_path / "ponding_depth.tif",
        effective_rain_m=0.005,
    )

    with rasterio.open(output_path) as result:
        values = result.read(1)
        np.testing.assert_allclose(
            values,
            np.array([[0.004, 0.003], [0.0, -9999]], dtype=np.float32),
            atol=1e-8,
        )
        assert np.max(values[values != -9999]) <= 0.005
