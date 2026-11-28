"""Tests for result plotting."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from storm_water_management.plotting import (
    _any_downsample,
    _read_basin_water_depth_layer,
    create_nodata_plot,
    create_pyswmm_plot,
    create_terrain_plot,
)


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
    for path, data in (
        (depth_path, depth),
        (elevation_path, elevation),
        (scenario_path, scenario),
    ):
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


def _write_pyswmm_result_files(
    tmp_path,
    outlet_xy=((10.0, 10.0), (20.0, 20.0)),
    water_levels=(1.0, 2.0),
):
    """Write minimal PySWMM peak, time-series and summary result files."""
    peaks = gpd.GeoDataFrame(
        {
            "basin_id": [1, 2],
            "max_depth_m": [0.15, 0.3],
            "max_spill_flow_m3s": [0.0, 0.0],
            "downstream_basin_id": [0, 0],
            "outlet_x": [xy[0] for xy in outlet_xy],
            "outlet_y": [xy[1] for xy in outlet_xy],
        },
        geometry=gpd.points_from_xy(
            [xy[0] for xy in outlet_xy], [xy[1] for xy in outlet_xy]
        ),
        crs="EPSG:3006",
    )
    peaks_path = tmp_path / "pyswmm_basin_peaks.gpkg"
    peaks.to_file(peaks_path, layer="basin_peaks", driver="GPKG")

    steps = pd.to_datetime(
        ["2025-01-01 00:05:00", "2025-01-01 00:10:00", "2025-01-01 00:15:00"]
    )
    basin_depth = {1: (0.05, 0.15, 0.05), 2: (0.0, 0.0, 0.30)}
    basin_volume = {1: (10.0, 100.0, 50.0), 2: (0.0, 0.0, 0.0)}
    node_rows = []
    link_rows = []
    for step_index, step in enumerate(steps):
        for basin_id in (1, 2):
            node_rows.append(
                {
                    "datetime": step,
                    "basin_id": basin_id,
                    "node_id": f"B{basin_id:05d}",
                    "water_level_m": water_levels[basin_id - 1],
                    "depth_m": basin_depth[basin_id][step_index],
                    "stored_volume_m3": basin_volume[basin_id][step_index],
                    "lateral_inflow_m3s": 0.0,
                    "flooding_rate_m3s": 0.0,
                }
            )
            link_rows.append(
                {
                    "datetime": step,
                    "basin_id": basin_id,
                    "link_id": f"W{basin_id:05d}",
                    "flow_m3s": 0.0,
                    "depth_m": 0.0,
                }
            )
    node_path = tmp_path / "node_timeseries.csv"
    pd.DataFrame(node_rows).to_csv(node_path, index=False)
    link_path = tmp_path / "spill_link_timeseries.csv"
    pd.DataFrame(link_rows).to_csv(link_path, index=False)
    summary_path = tmp_path / "pyswmm_basin_summary.csv"
    pd.DataFrame(
        {
            "basin_id": [1, 2],
            "modeled_destination": ["boundary_outfall", "boundary_outfall"],
        }
    ).to_csv(summary_path, index=False)
    return peaks_path, node_path, link_path, summary_path


def _run_pyswmm_plot(tmp_path, **kwargs):
    """Create the PySWMM plot from the synthetic result files."""
    peaks_path, node_path, link_path, summary_path = _write_pyswmm_result_files(
        tmp_path
    )
    output_path = tmp_path / "pyswmm_plot.png"
    create_pyswmm_plot(
        peaks_path,
        node_path,
        link_path,
        None,
        summary_path,
        tmp_path / "missing_basin_ids.tif",
        tmp_path / "missing_dem.tif",
        output_path,
        **kwargs,
    )
    return output_path


def test_create_pyswmm_plot_depth_snapshot_at_requested_time(tmp_path, capsys):
    """The depth map uses the reported depths at the requested snapshot time."""
    output_path = _run_pyswmm_plot(tmp_path, depth_time="2025-01-01 00:15:00")

    assert output_path.exists()
    assert "Depth map snapshot time: 2025-01-01 00:15" in capsys.readouterr().out


def test_create_pyswmm_plot_depth_snapshot_defaults_to_peak_storage(tmp_path, capsys):
    """Without a requested time the map shows peak total stored volume."""
    output_path = _run_pyswmm_plot(tmp_path)

    assert output_path.exists()
    assert "Depth map snapshot time: 2025-01-01 00:10" in capsys.readouterr().out


def test_create_pyswmm_plot_depth_snapshot_snaps_to_nearest_step(tmp_path, capsys):
    """A requested time between report steps snaps to the nearest step."""
    output_path = _run_pyswmm_plot(tmp_path, depth_time="2025-01-01 00:07:00")

    assert output_path.exists()
    assert "Depth map snapshot time: 2025-01-01 00:05" in capsys.readouterr().out


def _write_bowl_rasters(tmp_path):
    """Write a one-basin raster pair: flat rim at 2 m around a 1 m centre."""
    transform = from_origin(0, 3, 1, 1)
    labels = np.ones((3, 3), dtype=np.int32)
    dem = np.array(
        [[2.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.float32
    )
    basin_ids_path = tmp_path / "basin_ids.tif"
    dem_path = tmp_path / "dem.tif"
    for path, data in ((basin_ids_path, labels), (dem_path, dem)):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=3,
            width=3,
            count=1,
            dtype=data.dtype,
            crs="EPSG:3006",
            transform=transform,
        ) as dst:
            dst.write(data, 1)
    return basin_ids_path, dem_path


def test_basin_water_depth_layer_deeper_in_basin_middle(tmp_path):
    """Per-cell depth follows the terrain: wet and deepest at the low centre."""
    basin_ids_path, dem_path = _write_bowl_rasters(tmp_path)

    layer, extent = _read_basin_water_depth_layer(
        basin_ids_path, dem_path, "EPSG:3006", (0.0, 0.0, 3.0, 3.0), {1: 1.5}, 100
    )

    assert layer is not None
    assert layer[1, 1] == pytest.approx(0.5)
    assert layer.mask[0, 0]
    assert layer.mask[1, 0]
    assert extent == (0.0, 3.0, 0.0, 3.0)


def test_basin_water_depth_layer_requires_dem_coverage(tmp_path):
    """A DEM that does not cover the map area falls back (returns None)."""
    basin_ids_path, dem_path = _write_bowl_rasters(tmp_path)

    layer = _read_basin_water_depth_layer(
        basin_ids_path, dem_path, "EPSG:3006", (5.0, 5.0, 8.0, 8.0), {1: 1.5}, 100
    )

    assert layer is None


def test_create_pyswmm_plot_water_depth_per_cell(tmp_path):
    """With DEM and basin-ID rasters the depth map varies within each basin."""
    peaks_path, node_path, link_path, summary_path = _write_pyswmm_result_files(
        tmp_path, outlet_xy=((1.5, 1.5), (2.5, 2.5)), water_levels=(1.5, 1.5)
    )
    basin_ids_path, dem_path = _write_bowl_rasters(tmp_path)
    output_path = tmp_path / "plot.png"

    create_pyswmm_plot(
        peaks_path,
        node_path,
        link_path,
        None,
        summary_path,
        basin_ids_path,
        dem_path,
        output_path,
    )

    assert output_path.exists()


def _write_nodata_dem(tmp_path):
    """Write a 3x3 DEM with two declared-nodata cells and one NaN cell."""
    dem_path = tmp_path / "dem.tif"
    dem = np.array(
        [[1.0, 2.0, -9999.0], [4.0, np.nan, 6.0], [-9999.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="float32",
        crs="EPSG:3006",
        transform=from_origin(0, 3, 1, 1),
        nodata=-9999.0,
    ) as dst:
        dst.write(dem, 1)
    return dem_path


def test_create_nodata_plot_counts_declared_and_nan_pixels(tmp_path, capsys):
    """Declared nodata and NaN pixels are both counted and plotted."""
    output_path = tmp_path / "nodata_plot.png"

    create_nodata_plot(_write_nodata_dem(tmp_path), output_path)

    assert output_path.exists()
    out = capsys.readouterr().out
    assert "Nodata value: -9999" in out
    assert "3 of 9 pixels (33.33%) are nodata" in out


def test_create_nodata_plot_downsamples_while_keeping_nodata_visible(tmp_path):
    """Downsampling to fewer display pixels still renders a plot."""
    output_path = tmp_path / "nodata_plot.png"

    create_nodata_plot(_write_nodata_dem(tmp_path), output_path, max_display_pixels=2)

    assert output_path.exists()


def test_create_nodata_plot_requires_existing_raster(tmp_path):
    """A missing DEM raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        create_nodata_plot(tmp_path / "missing.tif", tmp_path / "plot.png")


def test_any_downsample_marks_cell_when_any_source_pixel_is_nodata():
    """A display cell is True when at least one source pixel is True."""
    mask = np.zeros((6, 6), dtype=bool)
    mask[4, 5] = True

    shrunk = _any_downsample(mask, (2, 2))

    assert shrunk.shape == (2, 2)
    assert shrunk.sum() == 1
    assert shrunk[1, 1]
