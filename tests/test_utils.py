"""Tests for plot utilities."""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from storm_water_management.utils_plots import create_terrain_plot


def test_create_terrain_plot_ignores_misaligned_optional_scenario(tmp_path):
    """A stale or mismatched optional scenario raster should not crash the plot."""
    depth_path = tmp_path / "depression_depth.tif"
    elevation_path = tmp_path / "dem.tif"
    scenario_path = tmp_path / "runoff_volume.tif"
    output_path = tmp_path / "plot.png"

    depth = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    elevation = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    scenario = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    transform = from_origin(0, 2, 1, 1)
    for path, data in ((depth_path, depth), (elevation_path, elevation), (scenario_path, scenario)):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

    scenario_path_2 = tmp_path / "misaligned_runoff_volume.tif"
    with rasterio.open(
        scenario_path_2,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=from_origin(0, 3, 1, 1),
    ) as dst:
        dst.write(np.ones((3, 3), dtype=np.float32), 1)

    create_terrain_plot(
        depth_path,
        elevation_path,
        output_path,
        runoff_volume_path=scenario_path_2,
    )

    assert output_path.exists()
