"""Tests for PySWMM inputs from prepared or static basin summaries."""

import pytest

from storm_water_management.utils_pyswmm import (
    Basin,
    load_basins,
    select_storage_basins,
)

NETWORK_COLUMNS = (
    "basin_id,incremental_contributing_area_m2,spill_elevation_m,"
    "downstream_basin_id,status,capacity_m3,depression_area_m2,outlet_x,outlet_y,crs\n"
)
NETWORK_ROW = "1,120.0,10.0,0,spills_out_of_analysis_area,50.0,25.0,1.0,2.0,EPSG:3006\n"


def test_load_prepared_network_with_explicit_rainfall(tmp_path):
    """A prepared network CSV omits event rainfall if the CLI supplies it."""
    path = tmp_path / "basin_network.csv"
    path.write_text(NETWORK_COLUMNS + NETWORK_ROW, encoding="utf-8")

    basins, rainfall_mm = load_basins(path, effective_rainfall_mm=12.5)

    assert len(basins) == 1
    assert basins[0].incremental_area_m2 == 120.0
    assert rainfall_mm == 12.5


def test_load_prepared_network_requires_explicit_rainfall(tmp_path):
    """A network-only file cannot silently choose a rainfall event."""
    path = tmp_path / "basin_network.csv"
    path.write_text(NETWORK_COLUMNS + NETWORK_ROW, encoding="utf-8")

    with pytest.raises(ValueError, match="provide --effective-rainfall-mm"):
        load_basins(path)


def test_load_static_summary_keeps_median_rainfall_default(tmp_path):
    """Legacy static summary files continue to supply their median rain depth."""
    path = tmp_path / "static_summary.csv"
    path.write_text(
        NETWORK_COLUMNS.replace(
            "outlet_x,outlet_y,crs", "effective_rainfall_m,outlet_x,outlet_y,crs"
        )
        + "1,120.0,10.0,0,spills_out_of_analysis_area,50.0,25.0,0.01,1.0,2.0,EPSG:3006\n"
        + "2,80.0,12.0,0,spills_out_of_analysis_area,30.0,20.0,0.03,3.0,4.0,EPSG:3006\n",
        encoding="utf-8",
    )

    basins, rainfall_mm = load_basins(path)

    assert len(basins) == 2
    assert rainfall_mm == pytest.approx(20.0)


def test_explicit_rainfall_overrides_static_summary(tmp_path):
    """The command-line rainfall depth takes precedence over CSV defaults."""
    path = tmp_path / "static_summary.csv"
    path.write_text(
        NETWORK_COLUMNS.replace(
            "outlet_x,outlet_y,crs", "effective_rainfall_m,outlet_x,outlet_y,crs"
        )
        + "1,120.0,10.0,0,spills_out_of_analysis_area,50.0,25.0,0.01,1.0,2.0,EPSG:3006\n",
        encoding="utf-8",
    )

    _, rainfall_mm = load_basins(path, effective_rainfall_mm=8.0)

    assert rainfall_mm == 8.0


def test_censored_basins_route_to_boundary_instead_of_storage():
    """Open edge basins are excluded as storage nodes and their area is conserved."""
    basins = [
        Basin(
            basin_id=1,
            incremental_area_m2=120.0,
            spill_elevation_m=5.0,
            downstream_basin_id=0,
            status="edge_or_nodata_censored",
            outlet_x=None,
            outlet_y=None,
            source_capacity_m3=0.0,
            source_area_m2=5000.0,
            crs="EPSG:3006",
        ),
        Basin(
            basin_id=2,
            incremental_area_m2=80.0,
            spill_elevation_m=10.0,
            downstream_basin_id=0,
            status="spills_out_of_analysis_area",
            outlet_x=None,
            outlet_y=None,
            source_capacity_m3=50.0,
            source_area_m2=2000.0,
            crs="EPSG:3006",
        ),
    ]

    modeled, external_area = select_storage_basins(basins, minimum_area_m2=1000.0)

    assert [basin.basin_id for basin in modeled] == [2]
    assert external_area == 120.0
