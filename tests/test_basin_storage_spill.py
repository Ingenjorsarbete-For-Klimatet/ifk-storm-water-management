"""Tests for static DEM depression-basin storage and spill calculations."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from storm_water_management.storm_scenario import (
    simulate_basin_storage_spill,
    write_basin_storage_spill_outputs,
)


def test_single_basin_fills_to_sill_then_spills():
    """Runoff volume above basin capacity is reported as overflow."""
    dem = np.full((7, 7), 5.0, dtype=np.float32)
    dem[3, 2] = 2.0  # One-cell depression's spill saddle and outlet.
    dem[3, 3] = 0.0
    depression = np.zeros_like(dem)
    depression[3, 3] = 2.0
    pointer = np.zeros_like(dem)
    pointer[3, 3] = 2  # Whitebox D8: east.
    accumulation = np.ones_like(dem)
    accumulation[3, 3] = 100.0

    labels, ponding_depth, basins = simulate_basin_storage_spill(
        dem,
        depression,
        pointer,
        accumulation,
        cell_area_m2=1.0,
        effective_rain_m=0.05,
        minimum_depression_depth_m=0.02,
        minimum_basin_area_m2=1.0,
    )

    assert len(basins) == 1
    basin = basins[0]
    assert basin["capacity_m3"] == 2.0
    assert basin["incoming_volume_m3"] == 5.0
    assert basin["retained_volume_m3"] == 2.0
    assert basin["spill_volume_m3"] == 3.0
    assert ponding_depth[3, 3] == 2.0
    assert np.count_nonzero(labels) == 1


def test_basin_stage_storage_curve_uses_cell_elevations():
    """A low inflow raises water level only above the basin's lowest cell."""
    dem = np.full((7, 7), 5.0, dtype=np.float32)
    dem[3, 2] = 2.0
    dem[3, 3] = 0.0
    depression = np.zeros_like(dem)
    depression[3, 2:4] = 1.0
    pointer = np.zeros_like(dem)
    pointer[3, 3] = 32  # From the low cell toward the saddle on its west.
    accumulation = np.ones_like(dem)
    accumulation[3, 2:4] = 22.0

    _, ponding_depth, basins = simulate_basin_storage_spill(
        dem,
        depression,
        pointer,
        accumulation,
        cell_area_m2=1.0,
        effective_rain_m=0.05,
        minimum_depression_depth_m=0.02,
        minimum_basin_area_m2=1.0,
    )

    assert basins[0]["incoming_volume_m3"] == pytest.approx(1.1)
    assert basins[0]["water_level_m"] == pytest.approx(1.1)
    assert ponding_depth[3, 3] == pytest.approx(1.1)
    assert ponding_depth[3, 2] == 0.0


def test_boundary_basin_is_marked_censored():
    """A depression clipped by the DEM edge is not assigned a closed-basin depth."""
    dem = np.full((5, 5), 5.0, dtype=np.float32)
    dem[0, 2] = 0.0
    depression = np.zeros_like(dem)
    depression[0, 2] = 2.0
    pointer = np.zeros_like(dem)
    accumulation = np.ones_like(dem) * 10.0

    _, ponding_depth, basins = simulate_basin_storage_spill(
        dem,
        depression,
        pointer,
        accumulation,
        cell_area_m2=1.0,
        effective_rain_m=0.05,
        minimum_basin_area_m2=1.0,
    )

    assert basins[0]["status"] == "edge_or_nodata_censored"
    assert np.count_nonzero(ponding_depth) == 0


def test_spill_from_first_basin_is_routed_to_downstream_basin():
    """Overflow enters the next depression along the D8 flow path."""
    dem = np.full((5, 11), 5.0, dtype=np.float32)
    dem[2, 2] = 0.0
    dem[2, 3:7] = [1.0, 0.8, 0.6, 0.4]
    dem[2, 7] = 0.0
    dem[2, 8] = 0.4
    depression = np.zeros_like(dem)
    depression[2, 2] = 1.0
    depression[2, 7] = 1.0
    pointer = np.zeros_like(dem)
    pointer[2, 2] = 2
    pointer[2, 3:7] = 2
    pointer[2, 7] = 2
    pointer[2, 8:10] = 2
    accumulation = np.ones_like(dem)
    accumulation[2, 2] = 10.0
    accumulation[2, 7] = 100.0

    _, ponding_depth, basins = simulate_basin_storage_spill(
        dem,
        depression,
        pointer,
        accumulation,
        cell_area_m2=1.0,
        effective_rain_m=0.2,
        minimum_depression_depth_m=0.02,
        minimum_basin_area_m2=1.0,
    )

    by_id = {int(basin["basin_id"]): basin for basin in basins}
    assert len(by_id) == 2
    assert by_id[1]["downstream_basin_id"] == 2
    assert by_id[1]["spill_volume_m3"] == 1.0
    assert by_id[2]["incoming_volume_m3"] == 19.0
    assert by_id[2]["retained_volume_m3"] == pytest.approx(0.4)
    assert by_id[2]["spill_volume_m3"] == pytest.approx(18.6)
    assert ponding_depth[2, 2] == 1.0
    assert ponding_depth[2, 7] == 0.4
    total_local_runoff = sum(float(basin["local_event_runoff_m3"]) for basin in basins)
    final_stored = sum(float(basin["retained_volume_m3"]) for basin in basins)
    final_external_spill = float(by_id[2]["spill_volume_m3"])
    assert total_local_runoff == pytest.approx(final_stored + final_external_spill)


def test_basin_writer_creates_aligned_geotiffs_and_csv(tmp_path):
    """The basin scenario writer persists its depth, labels, and summary."""
    dem = np.full((7, 7), 5.0, dtype=np.float32)
    dem[3, 2] = 2.0
    dem[3, 3] = 0.0
    depression = np.zeros_like(dem)
    depression[3, 3] = 2.0
    pointer = np.zeros_like(dem)
    pointer[3, 3] = 2
    accumulation = np.ones_like(dem)
    accumulation[3, 3] = 100.0
    profile = {
        "driver": "GTiff",
        "height": 7,
        "width": 7,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3006",
        "transform": from_origin(0, 7, 1, 1),
        "nodata": -9999.0,
    }
    paths = {
        "dem": tmp_path / "dem.tif",
        "depression": tmp_path / "depression.tif",
        "pointer": tmp_path / "pointer.tif",
        "area": tmp_path / "area.tif",
    }
    for name, values in (
        ("dem", dem),
        ("depression", depression),
        ("pointer", pointer),
        ("area", accumulation),
    ):
        with rasterio.open(paths[name], "w", **profile) as destination:
            destination.write(values, 1)

    pond_path = tmp_path / "ponding.tif"
    id_path = tmp_path / "basin_ids.tif"
    csv_path = tmp_path / "basins.csv"
    summaries = write_basin_storage_spill_outputs(
        paths["dem"],
        paths["depression"],
        paths["pointer"],
        paths["area"],
        pond_path,
        id_path,
        csv_path,
        effective_rain_m=0.05,
        minimum_basin_area_m2=1.0,
    )

    assert len(summaries) == 1
    with rasterio.open(pond_path) as ponding, rasterio.open(id_path) as basin_ids:
        assert ponding.crs == basin_ids.crs
        assert ponding.transform == basin_ids.transform
        assert ponding.read(1)[3, 3] == pytest.approx(2.0)
        assert basin_ids.read(1)[3, 3] == 1
    assert "capacity_m3" in csv_path.read_text(encoding="utf-8")
