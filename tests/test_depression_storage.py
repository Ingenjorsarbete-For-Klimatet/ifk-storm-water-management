"""Synthetic-DEM tests for the depression storage / spill module.

Set DEPRESSION_MODULE to the import path of the module, for example
``export DEPRESSION_MODULE=storm_water_management.depression_storage``.
"""

from __future__ import annotations

import heapq
import importlib
import os
import warnings

import numpy as np
import pytest

mod = importlib.import_module(
    os.environ.get("DEPRESSION_MODULE", "storm_water_management.depression_storage")
)

EPS = 1e-6


# ---------------------------------------------------------------- helpers
def condition_dem(dem: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Priority-flood fill with an epsilon gradient (edges drain)."""
    h, w = dem.shape
    filled = dem.astype(np.float64).copy()
    closed = np.zeros(dem.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                heapq.heappush(heap, (filled[r, c], r, c))
                closed[r, c] = True
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in mod.D8_OFFSETS.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not closed[rr, cc]:
                closed[rr, cc] = True
                filled[rr, cc] = max(filled[rr, cc], z + eps)
                heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


def pointer_and_accumulation(filled: np.ndarray):
    """Steepest-descent D8 pointer and accumulation (in cells) on a drained surface."""
    h, w = filled.shape
    pointer = np.zeros((h, w), dtype=np.float32)
    downstream = {}
    for r in range(h):
        for c in range(w):
            best, best_slope = None, 0.0
            for code, (dr, dc) in mod.D8_OFFSETS.items():
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    slope = (filled[r, c] - filled[rr, cc]) / np.hypot(dr, dc)
                    if slope > best_slope + 1e-15:
                        best, best_slope = (code, rr, cc), slope
            if best:
                pointer[r, c] = best[0]
                downstream[(r, c)] = (best[1], best[2])
    acc = np.ones((h, w), dtype=np.float64)
    for r, c in sorted(downstream, key=lambda rc: -filled[rc]):
        rr, cc = downstream[(r, c)]
        acc[rr, cc] += acc[r, c]
    return pointer, acc.astype(np.float32)


def d8_inputs(dem: np.ndarray):
    """Return depth, D8 pointer and accumulation (in cells) for a DEM."""
    filled = condition_dem(dem)
    depth = (filled - dem).astype(np.float32)
    pointer, acc = pointer_and_accumulation(filled)
    return depth, pointer, acc


def pipeline_like_inputs(dem: np.ndarray, flat_increment: float = 1e-3):
    """Like the real pipeline: exact fill for depth, gradient fill for D8."""
    exact = condition_dem(dem, 0.0)
    gradient = condition_dem(dem, flat_increment)
    pointer, acc = pointer_and_accumulation(gradient)
    return (exact - dem).astype(np.float32), pointer, acc


def noisy_plateau_terrain(seed: int, h: int = 80, w: int = 110, amp: float = 0.5):
    """Smooth random terrain, quantised so that natural flats (plateaus) occur."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    noise = ndimage.gaussian_filter(rng.normal(size=(h, w)), 5)
    noise = noise / noise.std() * amp
    z = 10 - 0.004 * np.arange(w) + noise
    return (np.round(z / 0.02) * 0.02).astype(np.float32)


def tilted_plane(h=30, w=40, slope=0.2, top=10.0):
    cols = np.arange(w, dtype=np.float64)
    return np.tile(top - slope * cols, (h, 1))


def add_pit(dem, rows, cols, depth=1.0):
    """Cut a flat-floored pit whose floor is `depth` below the plane at its east rim."""
    rim_col = cols[-1] + 1
    floor = dem[rows[0], rim_col] - depth
    for r in rows:
        for c in cols:
            dem[r, c] = floor
    # Single pour point: raise the north, south and east rim except one notch
    # cell (the west side stays open so upstream water can still enter).
    notch = (rows[len(rows) // 2], rim_col)
    r0, r1 = rows[0] - 1, rows[-1] + 1
    for r in range(r0, r1 + 1):
        for c in range(cols[0], rim_col + 1):
            if (r in (r0, r1) or c == rim_col) and (r, c) != notch:
                dem[r, c] += 0.3


def cascade_dem():
    dem = tilted_plane()
    add_pit(dem, range(10, 15), range(8, 13))     # upstream pit, 25 cells
    add_pit(dem, range(10, 15), range(25, 30))    # downstream pit, 25 cells
    return dem.astype(np.float32)


# ------------------------------------------------------------------ tests
def test_single_pit_spill_level_and_capacity():
    dem = tilted_plane()
    add_pit(dem, range(10, 15), range(8, 13), depth=1.0)
    dem = dem.astype(np.float32)
    depth, pointer, acc = d8_inputs(dem)
    labels, records = mod.prepare_basin_network(
        dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=1.0
    )
    assert len(records) == 1
    rec = records[0]
    sill = float(dem[12, 13])  # the notch
    assert rec["spill_elevation_m"] == pytest.approx(sill, abs=1e-3)
    assert rec["capacity_m3"] == pytest.approx(25.0, rel=0.02)
    assert rec["max_depression_depth_m"] == pytest.approx(1.0, abs=1e-3)
    assert rec["status"] in {mod.STATUS_OUT, mod.STATUS_TO_BASIN}


def test_cascade_mass_balance_and_depth_raster():
    dem = cascade_dem()
    depth, pointer, acc = d8_inputs(dem)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labels, depths, records = mod.simulate_basin_storage_spill(
            dem, depth, pointer, acc, 1.0, 1.0, minimum_basin_area_m2=1.0
        )
    assert len(records) == 2
    up, down = sorted(records, key=lambda r: r["spill_elevation_m"], reverse=True)
    assert up["downstream_basin_id"] == down["basin_id"]
    # Heavy rain: upstream pit is full and passes overflow on.
    assert up["retained_volume_m3"] == pytest.approx(up["capacity_m3"])
    assert up["spill_volume_m3"] > 0
    assert down["incoming_volume_m3"] == pytest.approx(
        down["local_event_runoff_m3"] + up["spill_volume_m3"]
    )
    # Mass balance: runoff = retained + exported.
    runoff = sum(r["local_event_runoff_m3"] for r in records)
    retained = sum(r["retained_volume_m3"] for r in records)
    exported = sum(r["spill_volume_m3"] for r in records if r["downstream_basin_id"] == 0)
    assert runoff == pytest.approx(retained + exported)
    # Raster volume matches retained volume and stays below the spill level.
    for rec in records:
        mask = labels == rec["basin_id"]
        assert float(depths[mask].sum()) == pytest.approx(rec["retained_volume_m3"], rel=1e-3)
        assert rec["water_level_m"] <= rec["spill_elevation_m"] + 1e-6


def test_light_rain_partial_fill():
    dem = cascade_dem()
    depth, pointer, acc = d8_inputs(dem)
    _, depths, records = mod.simulate_basin_storage_spill(
        dem, depth, pointer, acc, 1.0, 0.001, minimum_basin_area_m2=1.0
    )
    for rec in records:
        assert 0 < rec["retained_volume_m3"] < rec["capacity_m3"]
        assert rec["spill_volume_m3"] == 0


def test_prepare_and_simulate_agree_on_network_fields():
    dem = cascade_dem()
    depth, pointer, acc = d8_inputs(dem)
    _, net = mod.prepare_basin_network(dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=1.0)
    _, _, sim = mod.simulate_basin_storage_spill(
        dem, depth, pointer, acc, 1.0, 0.05, minimum_basin_area_m2=1.0
    )
    for a, b in zip(net, sim):
        for key in mod.NETWORK_FIELDS:
            if key in a and key in b:
                if isinstance(a[key], float) and np.isnan(a[key]):
                    assert np.isnan(b[key])
                else:
                    assert a[key] == b[key], key


def test_contributing_area_in_cells_matches_m2():
    dem = cascade_dem()
    depth, pointer, acc = d8_inputs(dem)
    cell_area = 4.0
    _, m2 = mod.prepare_basin_network(
        dem, depth, pointer, acc * cell_area, cell_area, minimum_basin_area_m2=1.0
    )
    _, cells = mod.prepare_basin_network(
        dem, depth, pointer, acc, cell_area, minimum_basin_area_m2=1.0,
        contributing_area_in_cells=True,
    )
    assert [r["contributing_area_m2"] for r in m2] == [r["contributing_area_m2"] for r in cells]
    with pytest.warns(RuntimeWarning, match="cell count"):
        mod.prepare_basin_network(dem, depth, pointer, acc, cell_area, minimum_basin_area_m2=1.0)


def _manual_pit_arrays(pit_cells, nodata_cells=()):
    dem = np.full((9, 9), 5.0, dtype=np.float32)
    depth = np.zeros_like(dem)
    for rc in pit_cells:
        depth[rc] = 1.0
    for rc in nodata_cells:
        dem[rc] = -9999.0
    pointer = np.zeros_like(dem)
    acc = np.ones_like(dem)
    return dem, depth, pointer, acc


def test_raster_edge_basin_is_censored():
    dem, depth, pointer, acc = _manual_pit_arrays([(0, 3), (0, 4)])
    _, recs = mod.prepare_basin_network(
        dem, depth, pointer, acc, 1.0, minimum_depression_depth_m=0.5, minimum_basin_area_m2=1.0
    )
    assert recs[0]["status"] == mod.STATUS_CENSORED
    assert recs[0]["capacity_m3"] == 0.0
    assert np.isnan(recs[0]["spill_elevation_m"])


def test_nodata_neighbour_censors_basin_only_when_masked():
    cells = [(4, 3), (4, 4)]
    dem, depth, pointer, acc = _manual_pit_arrays(cells, nodata_cells=[(4, 5)])
    kwargs = dict(minimum_depression_depth_m=0.5, minimum_basin_area_m2=1.0)
    _, masked = mod.prepare_basin_network(dem, depth, pointer, acc, 1.0, dem_nodata=-9999.0, **kwargs)
    assert masked[0]["status"] == mod.STATUS_CENSORED
    # Without masking, -9999 is treated as terrain and the basin looks closed.
    _, unmasked = mod.prepare_basin_network(dem, depth, pointer, acc, 1.0, **kwargs)
    assert unmasked[0]["status"] != mod.STATUS_CENSORED


def test_component_threshold_uses_max_depth_and_area():
    dem, depth, pointer, acc = _manual_pit_arrays([(4, 3), (4, 4), (4, 5)])
    depth[4, 3] = depth[4, 5] = 0.01  # shallow fringe around a 0.3 m core
    depth[4, 4] = 0.3
    _, recs = mod.prepare_basin_network(
        dem, depth, pointer, acc, 1.0, minimum_depression_depth_m=0.2, minimum_basin_area_m2=3.0
    )
    assert len(recs) == 1 and recs[0]["cell_count"] == 3
    _, recs = mod.prepare_basin_network(
        dem, depth, pointer, acc, 1.0, minimum_depression_depth_m=0.2, minimum_basin_area_m2=4.0
    )
    assert recs == []


def test_deterministic_output():
    dem = cascade_dem()
    depth, pointer, acc = d8_inputs(dem)
    a = mod.prepare_basin_network(dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=1.0)[1]
    b = mod.prepare_basin_network(dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=1.0)[1]
    assert a == b


def test_downstream_trace_stops_on_cycle_and_border():
    labels = np.zeros((5, 5), dtype=np.int32)
    valid = np.ones((5, 5), dtype=bool)
    pointer = np.zeros((5, 5), dtype=np.float32)
    pointer[2, 2] = 2   # east
    pointer[2, 3] = 32  # west -> two-cell loop
    start = 2 * 5 + 2
    assert mod._downstream_basin(start, 1, labels, pointer, valid, 100) == (0, mod.TRACE_CYCLE)
    pointer[2, 3] = 0   # interior cell without flow direction
    assert mod._downstream_basin(start, 1, labels, pointer, valid, 100) == (0, mod.TRACE_NO_FLOW)
    pointer[2, 3] = 2   # flows on to the border cell, which has no direction
    pointer[2, 4] = 0
    assert mod._downstream_basin(start, 1, labels, pointer, valid, 100) == (0, mod.TRACE_EDGE)


def test_empty_result_when_no_basins():
    dem = tilted_plane().astype(np.float32)
    depth, pointer, acc = d8_inputs(dem)
    labels, depths, recs = mod.simulate_basin_storage_spill(dem, depth, pointer, acc, 1.0, 0.05)
    assert recs == [] and labels.max() == 0 and depths.max() == 0


# ------------------------------------------------------------ file I/O
def _write_tif(path, array, nodata=None):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype="float32", crs="EPSG:3006",
        transform=from_origin(0.0, float(array.shape[0]), 1.0, 1.0), nodata=nodata,
    ) as dst:
        dst.write(array.astype("float32"), 1)


def test_storm_outputs_roundtrip_with_nodata_column(tmp_path):
    import csv

    import rasterio

    dem = cascade_dem()
    depth, pointer, acc = d8_inputs(dem)
    nd = -9999.0
    arrays = {"dem": dem, "depth": depth, "ptr": pointer, "acc": acc}
    for name, arr in arrays.items():
        arr = arr.copy()
        arr[:, -1] = nd
        _write_tif(tmp_path / f"{name}.tif", arr, nodata=nd)
    records = mod.write_basin_storage_spill_outputs(
        tmp_path / "dem.tif", tmp_path / "depth.tif", tmp_path / "ptr.tif", tmp_path / "acc.tif",
        tmp_path / "ponding.tif", tmp_path / "ids.tif", tmp_path / "summary.csv",
        effective_rain_m=0.05, minimum_basin_area_m2=1.0,
        scenario_tags={"rainfall_mm": "100"},
    )
    assert len(records) == 2
    with (tmp_path / "summary.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == mod.STORM_FIELDS
        assert len(list(reader)) == 2
    with rasterio.open(tmp_path / "ponding.tif") as src:
        ponding = src.read(1)
        assert (ponding[:, -1] == nd).all()
        assert src.tags()["rainfall_mm"] == "100"
        assert "label_epsilon_m" in src.tags()
    assert all(r["crs"] for r in records)


def test_network_outputs_header_only_when_no_basins(tmp_path):
    import csv

    dem = tilted_plane().astype(np.float32)
    depth, pointer, acc = d8_inputs(dem)
    for name, arr in {"dem": dem, "depth": depth, "ptr": pointer, "acc": acc}.items():
        _write_tif(tmp_path / f"{name}.tif", arr)
    rows = mod.write_basin_network_outputs(
        tmp_path / "dem.tif", tmp_path / "depth.tif", tmp_path / "ptr.tif", tmp_path / "acc.tif",
        tmp_path / "ids.tif", tmp_path / "summary.csv",
    )
    assert rows == []
    with (tmp_path / "summary.csv").open(newline="") as handle:
        assert csv.DictReader(handle).fieldnames == mod.NETWORK_FIELDS


# ------------------------------------- pipeline-like terrain with plateaus
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_outlets_consistent_on_plateau_terrain(seed):
    dem = noisy_plateau_terrain(seed)
    depth, pointer, acc = pipeline_like_inputs(dem)
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # no dropped links, no unresolved paths
        _, recs = mod.prepare_basin_network(
            dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=4.0
        )
    assert len(recs) >= 5
    for rec in recs:
        # All water of a basin leaves through its real exit.
        assert rec["contributing_area_m2"] >= rec["depression_area_m2"] - 1e-6
        assert rec["status"] in {mod.STATUS_OUT, mod.STATUS_TO_BASIN}


def test_trace_continues_through_source_basin_to_next_basin():
    labels = np.zeros((1, 9), dtype=np.int32)
    labels[0, 1:3] = 1          # source basin
    labels[0, 6:8] = 2          # receiving basin
    valid = np.ones_like(labels, dtype=bool)
    pointer = np.full((1, 9), 2, dtype=np.float32)   # everything flows east
    # Start inside the source basin (as after a mis-chosen outlet).
    assert mod._downstream_basin(1, 1, labels, pointer, valid, 100) == (2, mod.TRACE_BASIN)
