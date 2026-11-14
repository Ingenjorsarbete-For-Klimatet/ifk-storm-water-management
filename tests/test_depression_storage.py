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
    """Return a plane DEM sloping east with the given dimensions."""
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
    """Return a plane DEM with an upstream and a downstream pit."""
    dem = tilted_plane()
    add_pit(dem, range(10, 15), range(8, 13))  # upstream pit, 25 cells
    add_pit(dem, range(10, 15), range(25, 30))  # downstream pit, 25 cells
    return dem.astype(np.float32)


# ------------------------------------------------------------------ tests
def test_single_pit_spill_level_and_capacity():
    """A single pit reports the notch spill level and its fill capacity."""
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
    """Heavy rain fills both cascade pits and keeps the volume balance."""
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
    exported = sum(
        r["spill_volume_m3"] for r in records if r["downstream_basin_id"] == 0
    )
    assert runoff == pytest.approx(retained + exported)
    # Raster volume matches retained volume and stays below the spill level.
    for rec in records:
        mask = labels == rec["basin_id"]
        assert float(depths[mask].sum()) == pytest.approx(
            rec["retained_volume_m3"], rel=1e-3
        )
        assert rec["water_level_m"] <= rec["spill_elevation_m"] + 1e-6


def test_light_rain_partial_fill():
    """Light rain only partially fills the cascade pits without spilling."""
    dem = cascade_dem()
    depth, pointer, acc = d8_inputs(dem)
    _, depths, records = mod.simulate_basin_storage_spill(
        dem, depth, pointer, acc, 1.0, 0.001, minimum_basin_area_m2=1.0
    )
    for rec in records:
        assert 0 < rec["retained_volume_m3"] < rec["capacity_m3"]
        assert rec["spill_volume_m3"] == 0


def test_prepare_and_simulate_agree_on_network_fields():
    """Network preparation and storm simulation report identical network fields."""
    dem = cascade_dem()
    depth, pointer, acc = d8_inputs(dem)
    _, net = mod.prepare_basin_network(
        dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=1.0
    )
    _, _, sim = mod.simulate_basin_storage_spill(
        dem, depth, pointer, acc, 1.0, 0.05, minimum_basin_area_m2=1.0
    )
    for a, b in zip(net, sim, strict=True):
        for key in mod.NETWORK_FIELDS:
            if key in a and key in b:
                if isinstance(a[key], float) and np.isnan(a[key]):
                    assert np.isnan(b[key])
                else:
                    assert a[key] == b[key], key


def test_contributing_area_in_cells_matches_m2():
    """Accumulation in cells and in m2 give the same contributing areas."""
    dem = cascade_dem()
    depth, pointer, acc = d8_inputs(dem)
    cell_area = 4.0
    _, m2 = mod.prepare_basin_network(
        dem, depth, pointer, acc * cell_area, cell_area, minimum_basin_area_m2=1.0
    )
    _, cells = mod.prepare_basin_network(
        dem,
        depth,
        pointer,
        acc,
        cell_area,
        minimum_basin_area_m2=1.0,
        contributing_area_in_cells=True,
    )
    assert [r["contributing_area_m2"] for r in m2] == [
        r["contributing_area_m2"] for r in cells
    ]
    with pytest.warns(RuntimeWarning, match="cell count"):
        mod.prepare_basin_network(
            dem, depth, pointer, acc, cell_area, minimum_basin_area_m2=1.0
        )


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
    """A basin touching the raster edge is censored with no capacity."""
    dem, depth, pointer, acc = _manual_pit_arrays([(0, 3), (0, 4)])
    _, recs = mod.prepare_basin_network(
        dem,
        depth,
        pointer,
        acc,
        1.0,
        minimum_depression_depth_m=0.5,
        minimum_basin_area_m2=1.0,
    )
    assert recs[0]["status"] == mod.STATUS_CENSORED
    assert recs[0]["capacity_m3"] == 0.0
    assert np.isnan(recs[0]["spill_elevation_m"])


def test_nodata_neighbour_censors_basin_only_when_masked():
    """A NoData neighbour censors a basin only when the NoData value is masked."""
    cells = [(4, 3), (4, 4)]
    dem, depth, pointer, acc = _manual_pit_arrays(cells, nodata_cells=[(4, 5)])
    kwargs = dict(minimum_depression_depth_m=0.5, minimum_basin_area_m2=1.0)
    _, masked = mod.prepare_basin_network(
        dem, depth, pointer, acc, 1.0, dem_nodata=-9999.0, **kwargs
    )
    assert masked[0]["status"] == mod.STATUS_CENSORED
    # Without masking, -9999 is treated as terrain and the basin looks closed.
    _, unmasked = mod.prepare_basin_network(dem, depth, pointer, acc, 1.0, **kwargs)
    assert unmasked[0]["status"] != mod.STATUS_CENSORED


def test_component_threshold_uses_max_depth_and_area():
    """Component retention uses the maximum depth and the total area."""
    dem, depth, pointer, acc = _manual_pit_arrays([(4, 3), (4, 4), (4, 5)])
    depth[4, 3] = depth[4, 5] = 0.01  # shallow fringe around a 0.3 m core
    depth[4, 4] = 0.3
    _, recs = mod.prepare_basin_network(
        dem,
        depth,
        pointer,
        acc,
        1.0,
        minimum_depression_depth_m=0.2,
        minimum_basin_area_m2=3.0,
    )
    assert len(recs) == 1 and recs[0]["cell_count"] == 3
    _, recs = mod.prepare_basin_network(
        dem,
        depth,
        pointer,
        acc,
        1.0,
        minimum_depression_depth_m=0.2,
        minimum_basin_area_m2=4.0,
    )
    assert recs == []


def test_deterministic_output():
    """Repeated network preparation returns identical records."""
    dem = cascade_dem()
    depth, pointer, acc = d8_inputs(dem)
    a = mod.prepare_basin_network(
        dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=1.0
    )[1]
    b = mod.prepare_basin_network(
        dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=1.0
    )[1]
    assert a == b


def test_downstream_trace_stops_on_cycle_and_border():
    """The downstream trace reports cycles, missing flow and the raster edge."""
    labels = np.zeros((5, 5), dtype=np.int32)
    valid = np.ones((5, 5), dtype=bool)
    pointer = np.zeros((5, 5), dtype=np.float32)
    pointer[2, 2] = 2  # east
    pointer[2, 3] = 32  # west -> two-cell loop
    start = 2 * 5 + 2
    assert mod._downstream_basin(start, 1, labels, pointer, valid, 100) == (
        0,
        mod.TRACE_CYCLE,
    )
    pointer[2, 3] = 0  # interior cell without flow direction
    assert mod._downstream_basin(start, 1, labels, pointer, valid, 100) == (
        0,
        mod.TRACE_NO_FLOW,
    )
    pointer[2, 3] = 2  # flows on to the border cell, which has no direction
    pointer[2, 4] = 0
    assert mod._downstream_basin(start, 1, labels, pointer, valid, 100) == (
        0,
        mod.TRACE_EDGE,
    )


def test_empty_result_when_no_basins():
    """A DEM without depressions yields no basins and a zero depth raster."""
    dem = tilted_plane().astype(np.float32)
    depth, pointer, acc = d8_inputs(dem)
    labels, depths, recs = mod.simulate_basin_storage_spill(
        dem, depth, pointer, acc, 1.0, 0.05
    )
    assert recs == [] and labels.max() == 0 and depths.max() == 0


# ------------------------------------------------------------ file I/O
def _write_tif(path, array, nodata=None):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3006",
        transform=from_origin(0.0, float(array.shape[0]), 1.0, 1.0),
        nodata=nodata,
    ) as dst:
        dst.write(array.astype("float32"), 1)


def test_storm_outputs_roundtrip_with_nodata_column(tmp_path):
    """Storm outputs round-trip rasters, tags and the summary CSV."""
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
        tmp_path / "dem.tif",
        tmp_path / "depth.tif",
        tmp_path / "ptr.tif",
        tmp_path / "acc.tif",
        tmp_path / "ponding.tif",
        tmp_path / "ids.tif",
        tmp_path / "summary.csv",
        effective_rain_m=0.05,
        minimum_basin_area_m2=1.0,
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
    """Network outputs write a header-only CSV when no basins remain."""
    import csv

    dem = tilted_plane().astype(np.float32)
    depth, pointer, acc = d8_inputs(dem)
    for name, arr in {"dem": dem, "depth": depth, "ptr": pointer, "acc": acc}.items():
        _write_tif(tmp_path / f"{name}.tif", arr)
    rows = mod.write_basin_network_outputs(
        tmp_path / "dem.tif",
        tmp_path / "depth.tif",
        tmp_path / "ptr.tif",
        tmp_path / "acc.tif",
        tmp_path / "ids.tif",
        tmp_path / "summary.csv",
    )
    assert rows == []
    with (tmp_path / "summary.csv").open(newline="") as handle:
        assert csv.DictReader(handle).fieldnames == mod.NETWORK_FIELDS


# ------------------------------------- pipeline-like terrain with plateaus
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_outlets_consistent_on_plateau_terrain(seed):
    """Plateau terrain keeps every outlet on its basin's real exit."""
    dem = noisy_plateau_terrain(seed)
    depth, pointer, acc = pipeline_like_inputs(dem)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no dropped links, no unresolved paths
        _, recs = mod.prepare_basin_network(
            dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=4.0
        )
    assert len(recs) >= 5
    for rec in recs:
        # All water of a basin leaves through its real exit.
        assert rec["contributing_area_m2"] >= rec["depression_area_m2"] - 1e-6
        assert rec["status"] in {mod.STATUS_OUT, mod.STATUS_TO_BASIN}


def test_trace_continues_through_source_basin_to_next_basin():
    """A trace starting inside the source basin still reaches the next basin."""
    labels = np.zeros((1, 9), dtype=np.int32)
    labels[0, 1:3] = 1  # source basin
    labels[0, 6:8] = 2  # receiving basin
    valid = np.ones_like(labels, dtype=bool)
    pointer = np.full((1, 9), 2, dtype=np.float32)  # everything flows east
    # Start inside the source basin (as after a mis-chosen outlet).
    assert mod._downstream_basin(1, 1, labels, pointer, valid, 100) == (
        2,
        mod.TRACE_BASIN,
    )


# ------------------------------------------------------- outlet tie-breaking
def flat_tie_puddle_arrays(main_channel_acc=100.0, side_arm_acc=2.0, upstream_acc=50.0):
    """A puddle in a flat channel whose conditioned exit crosses a higher rim.

    The exact-fill spill level ties between the main channel (large
    accumulation) and a dead side arm (small accumulation), and the pointer
    exits over a rim above the spill level, so no crossing at the spill is
    pointer-followed. This mirrors the enkoping rasters, where the old
    lowest-index tie-break put the outlet on the side arm.
    """
    dem = np.full((7, 9), 12.0, dtype=np.float32)
    depth = np.zeros_like(dem)
    pointer = np.zeros_like(dem)
    acc = np.ones_like(dem)

    # Flat channel along row 3 flowing east; the puddle sits in it.
    dem[3, 1:6] = 10.0
    dem[3, 6] = 10.5  # rim the conditioned flow crosses
    dem[3, 7] = 10.0
    dem[3, 4], dem[3, 5] = 9.0, 9.0
    depth[3, 4], depth[3, 5] = 1.0, 1.0
    # Shallow arm of the same depression, ending at an unfilled flat.
    dem[2, 5], dem[1, 5] = 9.9, 9.9
    depth[2, 5], depth[1, 5] = 0.1, 0.1
    dem[0, 5] = 10.0
    # Upstream basin in the same channel.
    dem[3, 1], dem[4, 1] = 9.0, 9.0
    depth[3, 1], depth[4, 1] = 1.0, 1.0

    # Conditioned flow: west to east along row 3, over the rim at (3, 6).
    pointer[3, 1:8] = 2
    pointer[4, 1] = 128  # north, into the upstream pit
    pointer[1, 5] = 8  # south, into the arm
    pointer[2, 5] = 8  # south, into the puddle

    acc[3, 1] = upstream_acc  # upstream outlet
    acc[3, 4] = main_channel_acc  # puddle's main-channel crossing
    acc[1, 5] = side_arm_acc  # puddle's side-arm crossing
    return dem, depth, pointer, acc


def test_flat_tie_outlet_prefers_the_main_flow_crossing():
    """A flat-tie spill picks the crossing that carries the main flow."""
    dem, depth, pointer, acc = flat_tie_puddle_arrays()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a mis-picked outlet drops the link
        _, recs = mod.prepare_basin_network(
            dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=1.0
        )
    assert len(recs) == 2
    puddle = next(r for r in recs if r["cell_count"] == 4)
    upstream = next(r for r in recs if r["cell_count"] == 2)
    # The outlet is the main-channel crossing, not the side arm.
    assert (puddle["outlet_row"], puddle["outlet_col"]) == (3, 4)
    assert puddle["contributing_area_m2"] == pytest.approx(100.0)
    # The upstream link survives the outlet-area check.
    assert upstream["status"] == mod.STATUS_TO_BASIN
    assert upstream["downstream_basin_id"] == puddle["basin_id"]


def test_receiver_with_clearly_less_area_is_still_dropped():
    """A receiver with clearly less outlet area still drops the link."""
    dem, depth, pointer, acc = flat_tie_puddle_arrays(
        main_channel_acc=50.0, upstream_acc=100.0
    )
    with pytest.warns(RuntimeWarning, match="dropped"):
        _, recs = mod.prepare_basin_network(
            dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=1.0
        )
    upstream = next(r for r in recs if r["cell_count"] == 2)
    assert upstream["status"] == mod.STATUS_DROPPED
    assert upstream["downstream_basin_id"] == 0
    assert upstream["dropped_receiver_id"] != 0


def test_equal_outlet_area_link_is_kept_and_routed():
    """float32 accumulation ties must not drop or mis-order a genuine link."""
    dem = cascade_dem()[:, ::-1].copy()  # downstream basin gets the lower id
    depth, pointer, acc = d8_inputs(dem)
    _, net = mod.prepare_basin_network(
        dem, depth, pointer, acc, 1.0, minimum_basin_area_m2=1.0
    )
    linked = [r for r in net if r["downstream_basin_id"] != 0]
    assert len(linked) == 1
    up = linked[0]
    down = next(r for r in net if r["basin_id"] == up["downstream_basin_id"])
    # float32 accumulation identical at both outlets (quantisation tie).
    acc[up["outlet_row"], up["outlet_col"]] = 2e7
    acc[down["outlet_row"], down["outlet_col"]] = 2e7
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, sim = mod.simulate_basin_storage_spill(
            dem, depth, pointer, acc, 1.0, 0.05, minimum_basin_area_m2=1.0
        )
    up_s = next(r for r in sim if r["basin_id"] == up["basin_id"])
    down_s = next(r for r in sim if r["basin_id"] == down["basin_id"])
    assert up_s["status"] == mod.STATUS_TO_BASIN
    assert up_s["downstream_basin_id"] == down["basin_id"]
    assert down_s["incoming_volume_m3"] == pytest.approx(
        down_s["local_event_runoff_m3"] + up_s["spill_volume_m3"]
    )
    runoff = sum(r["local_event_runoff_m3"] for r in sim)
    retained = sum(r["retained_volume_m3"] for r in sim)
    exported = sum(r["spill_volume_m3"] for r in sim if r["downstream_basin_id"] == 0)
    assert runoff == pytest.approx(retained + exported)


def test_break_parent_cycles_drops_weakest_link():
    """Cycle breaking drops the link of the smallest-area member."""
    parent = np.array([0, 2, 3, 1, 0], dtype=np.int32)  # 1 -> 2 -> 3 -> 1
    outlet_area = np.array([0.0, 5.0, 3.0, 4.0, 1.0])
    dropped = np.zeros(5, dtype=bool)
    dropped_receiver = np.zeros(5, dtype=np.int32)
    broken = mod._break_parent_cycles(parent, outlet_area, dropped, dropped_receiver)
    assert broken == 1
    assert parent.tolist() == [0, 2, 0, 1, 0]  # weakest member (area 3) cut
    assert dropped[2] and dropped_receiver[2] == 3
    assert not dropped[[1, 3, 4]].any()


def test_topological_order_places_sender_before_receiver():
    """The topological order processes every sender before its receiver."""
    parent = np.array([0, 0, 1, 1, 2], dtype=np.int32)  # 2 -> 1, 3 -> 1, 4 -> 2
    order = mod._topological_order(parent).tolist()
    assert sorted(order) == [1, 2, 3, 4]
    position = {basin: i for i, basin in enumerate(order)}
    assert position[2] < position[1]
    assert position[3] < position[1]
    assert position[4] < position[2]
