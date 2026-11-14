"""DEM-derived static depression storage and spill screening.

Behaviour notes (changes relative to the first version):

* Depression components are labelled from every cell with fill depth above a
  tiny epsilon, and the user thresholds (``minimum_depression_depth_m`` and
  ``minimum_basin_area_m2``) are applied to whole components (maximum depth and
  area). Spill saddles are evaluated on the filled surface (``dem + depth``),
  so shallow wall/fringe cells no longer bias the spill level low.
* NoData cells are masked explicitly when rasters are read, so a finite NoData
  value (for example -9999) is never treated as terrain.
* D8 contributing area may be given in m^2 (default) or in cells.
* Network preparation and the storm simulation share one network builder, so
  capacity, spill level and status are identical in both modes.
* When several boundary crossings tie at the lowest saddle (a puddle inside a
  flat channel), the outlet is the tied crossing that carries the most D8
  flow, not the first one in raster order.
* A downstream link is kept when the receiver's outlet area is larger within a
  small relative tolerance (the accumulation raster is float32, so outlets of
  large basins can tie or swap by a few ulps); remaining links are dropped and
  any parent cycle is broken explicitly, so the spill graph stays acyclic.
"""

from __future__ import annotations

import csv
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import rasterio
import rasterio.transform
from rasterio.windows import Window
from scipy import ndimage

from storm_water_management.config import AnalysisConfig
from storm_water_management.utils_workflows import (
    clip_to_area,
    create_buffered_mosaic,
    effective_rainfall_m,
    projected_center,
    run_depression_analysis,
    select_tiles,
)

D8_OFFSETS = {
    64: (-1, -1),
    128: (-1, 0),
    1: (-1, 1),
    32: (0, -1),
    2: (0, 1),
    16: (1, -1),
    8: (1, 0),
    4: (1, 1),
}
D8_CODES = {offset: code for code, offset in D8_OFFSETS.items()}

# Fill depth above which a cell belongs to a depression component. This must
# exceed any epsilon gradient added by the DEM conditioning (fix-flats) step.
LABEL_EPSILON_M = 1e-4
# Bumped when behaviour changes; diagnose_basin_network.py refuses older copies.
MODULE_VERSION = 4
# Outlet selection prefers the crossing the D8 pointer follows if its saddle is
# within this distance of the lowest saddle (covers fix-flats epsilon gradients).
OUTLET_FOLLOW_TOLERANCE_M = 5e-3
# Relative tolerance when a downstream receiver must have more outlet
# contributing area than its sender; covers float32 quantisation of the
# accumulation raster (adjacent outlets of large basins can tie or swap ulps).
OUTLET_AREA_TOLERANCE = 1e-6
# Used when the configuration does not give an infiltration fraction.
# TODO: move into AnalysisConfig.
DEFAULT_INFILTRATION_FRACTION = 0.5

_BLOCK_CELLS = 1_000_000

# Reasons a downstream trace from a spill saddle can end.
TRACE_BASIN = "basin"
TRACE_EDGE = "raster_edge"
TRACE_NODATA = "nodata"
TRACE_NO_FLOW = "no_flow_direction"
TRACE_CYCLE = "cycle"
TRACE_MAX_STEPS = "max_steps"
_OPEN_END_REASONS = {TRACE_EDGE, TRACE_NODATA}

STATUS_CENSORED = "edge_or_nodata_censored"
STATUS_NO_AREA = "no_contributing_area"
STATUS_TO_BASIN = "spills_to_basin"
STATUS_DROPPED = "spill_link_dropped"
STATUS_OUT = "spills_out_of_analysis_area"
STATUS_UNRESOLVED = "spill_path_unresolved"


class BasinRecord(TypedDict, total=False):
    """One row of the basin summary CSV (event fields only in storm mode)."""

    basin_id: int
    cell_count: int
    depression_area_m2: float
    max_depression_depth_m: float
    min_elevation_m: float
    spill_elevation_m: float
    capacity_m3: float
    contributing_area_m2: float
    incremental_contributing_area_m2: float
    local_event_runoff_m3: float
    incoming_volume_m3: float
    retained_volume_m3: float
    spill_volume_m3: float
    water_level_m: float
    effective_rainfall_m: float
    downstream_basin_id: int
    downstream_link_dropped: int
    dropped_receiver_id: int
    spill_trace_end: str
    outlet_row: int
    outlet_col: int
    status: str
    outlet_x: float
    outlet_y: float
    crs: str


_FIELDS_HEAD = [
    "basin_id",
    "cell_count",
    "depression_area_m2",
    "max_depression_depth_m",
    "min_elevation_m",
    "spill_elevation_m",
    "capacity_m3",
    "contributing_area_m2",
    "incremental_contributing_area_m2",
]
_FIELDS_EVENT = [
    "local_event_runoff_m3",
    "incoming_volume_m3",
    "retained_volume_m3",
    "spill_volume_m3",
    "water_level_m",
    "effective_rainfall_m",
]
_FIELDS_TAIL = [
    "downstream_basin_id",
    "downstream_link_dropped",
    "dropped_receiver_id",
    "spill_trace_end",
    "outlet_row",
    "outlet_col",
    "status",
    "outlet_x",
    "outlet_y",
    "crs",
]
NETWORK_FIELDS = _FIELDS_HEAD + _FIELDS_TAIL
STORM_FIELDS = _FIELDS_HEAD + _FIELDS_EVENT + _FIELDS_TAIL


# --------------------------------------------------------------------------
# Labelling and spill detection
# --------------------------------------------------------------------------
def _row_block(width: int) -> int:
    return max(1, _BLOCK_CELLS // max(1, width))


def _valid_mask(
    dem: np.ndarray,
    depression_depth: np.ndarray,
    d8_pointer: np.ndarray,
    contributing_area: np.ndarray,
    dem_nodata: float | None = None,
) -> np.ndarray:
    """Cells where every input is usable; optionally masks a finite NoData value."""
    valid = (
        np.isfinite(dem)
        & np.isfinite(depression_depth)
        & np.isfinite(d8_pointer)
        & np.isfinite(contributing_area)
        & (contributing_area >= 0)
    )
    if dem_nodata is not None and np.isfinite(dem_nodata):
        valid &= dem != dem.dtype.type(dem_nodata)
    return valid


def label_depression_components(
    dem: np.ndarray,
    depression_depth: np.ndarray,
    valid: np.ndarray,
    cell_area_m2: float,
    minimum_depth_m: float,
    minimum_area_m2: float,
    label_epsilon_m: float = LABEL_EPSILON_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Label retained 8-connected depression-fill components.

    Cells with fill depth above ``label_epsilon_m`` form components. A component
    is retained when its maximum depth exceeds ``minimum_depth_m`` and its area
    reaches ``minimum_area_m2``. Returns an int32 label grid (zero is
    background) and a per-label cell count. ``dem`` is unused and kept for API
    compatibility.
    """
    if minimum_depth_m < 0 or minimum_area_m2 < 0 or label_epsilon_m < 0:
        raise ValueError(
            "Basin depth, area and epsilon thresholds must be non-negative."
        )
    depression_mask = valid & (depression_depth > label_epsilon_m)
    labels, component_count = ndimage.label(
        depression_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if component_count == 0:
        return labels.astype(np.int32), np.zeros(1, dtype=np.int64)
    counts = np.bincount(labels.ravel(), minlength=component_count + 1)
    max_depth = np.zeros(component_count + 1, dtype=np.float64)
    max_depth[1:] = ndimage.maximum(
        depression_depth, labels, index=np.arange(1, component_count + 1)
    )
    minimum_cells = max(1, math.ceil(minimum_area_m2 / cell_area_m2))
    keep = (counts >= minimum_cells) & (max_depth > minimum_depth_m)
    keep[0] = False
    kept_ids = np.flatnonzero(keep)
    remap = np.zeros(component_count + 1, dtype=np.int32)
    remap[kept_ids] = np.arange(1, kept_ids.size + 1, dtype=np.int32)
    labels = remap[labels]
    counts = np.bincount(labels.ravel(), minlength=kept_ids.size + 1)
    return labels, counts


def _neighbor_slices(
    height: int, width: int, row_offset: int, col_offset: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Return aligned source/neighbor slices for one D8 offset."""
    source_rows = slice(max(0, -row_offset), min(height, height - row_offset))
    source_cols = slice(max(0, -col_offset), min(width, width - col_offset))
    target_rows = slice(max(0, row_offset), min(height, height + row_offset))
    target_cols = slice(max(0, col_offset), min(width, width + col_offset))
    return (source_rows, source_cols), (target_rows, target_cols)


def _find_spill_outlets(
    surface: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    pointer: np.ndarray,
    counts: np.ndarray,
    accumulation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find each component's lowest valid boundary saddle and D8 outlet pair.

    ``surface`` is the filled surface (``dem + depression_depth``). The saddle of
    a boundary pair is the higher of its two filled elevations, and the spill
    level is the lowest saddle.

    The outlet pair is the crossing the D8 pointer follows (saddle within
    ``OUTLET_FOLLOW_TOLERANCE_M`` of the spill level) that carries the largest
    ``accumulation``: after conditioning, all water of a basin leaves through its
    real exit, which therefore has the largest accumulation, whereas side
    crossings that merely tie on saddle elevation (plateau rims) carry little.
    Without ``accumulation`` the lowest-saddle crossing is used. If no crossing is
    followed by the pointer, the tied lowest-saddle crossing with the largest
    ``accumulation`` is used: on a flat (for example a puddle inside a flat
    channel) the real spill is the crossing that carries the main flow, not an
    arbitrary one in raster order. Remaining ties go to
    the lowest linear-index pair, so the result is deterministic.
    """
    basin_count = counts.size - 1
    height, width = labels.shape
    total_cells = height * width
    if total_cells * total_cells >= np.iinfo(np.int64).max:
        raise ValueError(
            "Raster too large for the int64 outlet-pair key (about 3e9 cells)."
        )

    touches_invalid = np.zeros(basin_count + 1, dtype=bool)
    # Raster-edge and NoData-adjacent components are open/censored; their true
    # spill level may lie outside the available DEM and is not inferred here.
    edge_labels = np.concatenate(
        (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
    )
    touches_invalid[np.unique(edge_labels[edge_labels > 0])] = True

    # Boundary pairs are few compared with the raster, so collect them once.
    id_parts, saddle_parts, key_parts, follow_parts, area_parts = [], [], [], [], []
    for (row_offset, col_offset), code in D8_CODES.items():
        source_slice, target_slice = _neighbor_slices(
            height, width, row_offset, col_offset
        )
        source_labels = labels[source_slice]
        target_labels = labels[target_slice]
        invalid_boundary = (source_labels > 0) & ~valid[target_slice]
        if invalid_boundary.any():
            touches_invalid[np.unique(source_labels[invalid_boundary])] = True
        boundary = (
            (source_labels > 0)
            & valid[source_slice]
            & valid[target_slice]
            & (target_labels != source_labels)
        )
        if not boundary.any():
            continue
        boundary_rows, boundary_cols = np.nonzero(boundary)
        source_flat = (
            (boundary_rows + source_slice[0].start).astype(np.int64) * width
            + boundary_cols
            + source_slice[1].start
        )
        target_flat = source_flat + row_offset * width + col_offset
        id_parts.append(source_labels[boundary])
        saddle_parts.append(
            np.maximum(surface[source_slice][boundary], surface[target_slice][boundary])
        )
        key_parts.append(source_flat * total_cells + target_flat)
        follow_parts.append(pointer[source_slice][boundary] == code)
        if accumulation is None:
            area_parts.append(np.zeros(int(boundary.sum()), dtype=np.float64))
        else:
            area_parts.append(accumulation[source_slice][boundary].astype(np.float64))

    spill_levels = np.full(basin_count + 1, np.inf, dtype=np.float64)
    outlet_inside = np.full(basin_count + 1, -1, dtype=np.int64)
    outlet_outside = np.full(basin_count + 1, -1, dtype=np.int64)
    if id_parts:
        ids = np.concatenate(id_parts)
        saddle = np.concatenate(saddle_parts).astype(np.float64)
        keys = np.concatenate(key_parts)
        follows = np.concatenate(follow_parts)
        areas = np.concatenate(area_parts)
        np.minimum.at(spill_levels, ids, saddle)

        # Ties are detected within a few float32 ulps of the elevation scale.
        scale = float(np.max(np.abs(surface[valid]))) if valid.any() else 1.0
        tie_atol = max(1e-6, 4.0 * float(np.finfo(np.float32).eps) * max(scale, 1.0))
        int_max = np.iinfo(np.int64).max
        follow_ok = follows & (saddle <= spill_levels[ids] + OUTLET_FOLLOW_TOLERANCE_M)
        best_area = np.full(basin_count + 1, -np.inf, dtype=np.float64)
        np.maximum.at(best_area, ids[follow_ok], areas[follow_ok])
        chosen_follow = follow_ok & (areas >= best_area[ids])
        best_follow = np.full(basin_count + 1, int_max, dtype=np.int64)
        np.minimum.at(best_follow, ids[chosen_follow], keys[chosen_follow])
        at_lowest = saddle <= spill_levels[ids] + tie_atol
        # Among equally low saddles, the crossing carrying the most flow is the
        # real spill; the remaining tie goes to the lowest key.
        best_any_area = np.full(basin_count + 1, -np.inf, dtype=np.float64)
        np.maximum.at(best_any_area, ids[at_lowest], areas[at_lowest])
        chosen_any = at_lowest & (areas >= best_any_area[ids])
        best_any = np.full(basin_count + 1, int_max, dtype=np.int64)
        np.minimum.at(best_any, ids[chosen_any], keys[chosen_any])
        selected_pair = np.where(best_follow != int_max, best_follow, best_any)
        has_outlet = selected_pair != int_max
        outlet_inside[has_outlet] = selected_pair[has_outlet] // total_cells
        outlet_outside[has_outlet] = selected_pair[has_outlet] % total_cells
    else:
        has_outlet = np.zeros(basin_count + 1, dtype=bool)
    touches_invalid[~has_outlet] = True
    return spill_levels, outlet_inside, outlet_outside, touches_invalid


def _trace_downstream(
    start: int,
    source_basin: int,
    labels: np.ndarray,
    pointer: np.ndarray,
    valid: np.ndarray,
    max_steps: int,
) -> tuple[int, str, int, int]:
    """Trace D8 from a spill saddle to the next retained basin.

    Returns ``(basin_id, reason, last_cell, steps)``; ``basin_id`` is 0 unless
    the trace reaches another retained basin. ``last_cell`` is the flat index of
    the last in-raster cell examined. The trace stops at another retained basin,
    at a cycle, at the raster edge, at NoData, or where no flow direction exists.
    """
    height, width = labels.shape
    row, col = divmod(int(start), width)
    visited: set[int] = set()
    last = int(start)
    for step in range(max_steps):
        if row < 0 or row >= height or col < 0 or col >= width:
            return 0, TRACE_EDGE, last, step
        last = row * width + col
        if not valid[row, col]:
            return 0, TRACE_NODATA, last, step
        basin = int(labels[row, col])
        if basin > 0 and basin != source_basin:
            return basin, TRACE_BASIN, last, step
        # Cells of the source basin are passed through: the path may cross its
        # own basin before leaving it for good. D8 is acyclic, so a revisited
        # cell means corrupt flow directions.
        if last in visited:
            return 0, TRACE_CYCLE, last, step
        visited.add(last)
        offset = D8_OFFSETS.get(int(pointer[row, col]))
        if offset is None:
            # A border cell without a flow direction drains out of the raster.
            on_border = row in (0, height - 1) or col in (0, width - 1)
            reason = TRACE_EDGE if on_border else TRACE_NO_FLOW
            return 0, reason, last, step
        row += offset[0]
        col += offset[1]
    return 0, TRACE_MAX_STEPS, last, max_steps


def _downstream_basin(
    start: int,
    source_basin: int,
    labels: np.ndarray,
    pointer: np.ndarray,
    valid: np.ndarray,
    max_steps: int,
) -> tuple[int, str]:
    """Return ``(basin_id, reason)`` of :func:`_trace_downstream`."""
    basin, reason, _last, _steps = _trace_downstream(
        start, source_basin, labels, pointer, valid, max_steps
    )
    return basin, reason


def _break_parent_cycles(
    parent: np.ndarray,
    outlet_area: np.ndarray,
    dropped: np.ndarray,
    dropped_receiver: np.ndarray,
) -> int:
    """Drop one link per parent cycle so the spill graph stays acyclic.

    Area-monotone links cannot cycle, but links kept within the outlet-area
    tolerance could in principle. Each basin has at most one parent, so cycles
    are found by following parent pointers with path colouring. In each cycle
    the link of the member with the smallest outlet area (then smallest id) is
    dropped, so its overflow leaves the system. Returns the number of cycles
    broken.
    """
    broken = 0
    color = np.zeros(parent.size, dtype=np.uint8)  # 0 unvisited, 1 on path, 2 done
    for start in range(1, parent.size):
        if color[start]:
            continue
        path: list[int] = []
        node = start
        while node > 0 and color[node] == 0:
            color[node] = 1
            path.append(node)
            node = int(parent[node])
        if node > 0 and color[node] == 1:
            cycle = path[path.index(node) :]
            weakest = min(cycle, key=lambda b: (outlet_area[b], b))
            dropped_receiver[weakest] = parent[weakest]
            dropped[weakest] = True
            parent[weakest] = 0
            broken += 1
        for member in path:
            color[member] = 2
    return broken


def _topological_order(parent: np.ndarray) -> np.ndarray:
    """Order basin ids so every basin precedes its downstream receiver.

    Kahn's algorithm on the spill graph. Outlet area alone is not a reliable
    order once ties are kept within ``OUTLET_AREA_TOLERANCE``.
    """
    n = parent.size - 1
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    incoming = np.zeros(n + 1, dtype=np.int32)
    senders = np.flatnonzero(parent > 0)
    np.add.at(incoming, parent[senders], 1)
    ready = [b for b in range(1, n + 1) if incoming[b] == 0]
    order = np.empty(n, dtype=np.int64)
    written = 0
    while ready:
        basin = ready.pop()
        order[written] = basin
        written += 1
        receiver = int(parent[basin])
        if receiver > 0:
            incoming[receiver] -= 1
            if incoming[receiver] == 0:
                ready.append(receiver)
    if written != n:  # defensive: cycles are broken when the network is built
        remaining = np.setdiff1d(np.arange(1, n + 1), order[:written])
        order[written:] = np.sort(remaining)
    return order


def _water_level_for_volume(
    sorted_elevations: np.ndarray,
    volume_m3: float,
    cell_area_m2: float,
    spill_level_m: float,
) -> float:
    """Invert a basin's elevation-derived stage-storage curve."""
    if volume_m3 <= 0 or sorted_elevations.size == 0:
        return float(sorted_elevations[0]) if sorted_elevations.size else spill_level_m
    if not np.isfinite(spill_level_m):
        return float(sorted_elevations[0])

    cumulative_elevation = np.cumsum(sorted_elevations, dtype=np.float64)
    target_depth_area = volume_m3 / cell_area_m2
    lower = 1
    upper = sorted_elevations.size
    while lower < upper:
        active_cells = (lower + upper) // 2
        level = (
            target_depth_area + cumulative_elevation[active_cells - 1]
        ) / active_cells
        if level <= sorted_elevations[active_cells]:
            upper = active_cells
        else:
            lower = active_cells + 1
    water_level = (target_depth_area + cumulative_elevation[lower - 1]) / lower
    return min(float(water_level), spill_level_m)


# --------------------------------------------------------------------------
# Shared network builder
# --------------------------------------------------------------------------
@dataclass
class _BasinNetwork:
    labels: np.ndarray
    counts: np.ndarray
    spill: np.ndarray
    inside_outlet: np.ndarray
    censored: np.ndarray
    trace_reason: list[str]
    trace_end: np.ndarray
    trace_steps: np.ndarray
    parent: np.ndarray
    link_dropped: np.ndarray
    dropped_receiver: np.ndarray
    outlet_area: np.ndarray
    local_area: np.ndarray
    capacity: np.ndarray
    min_elevation: np.ndarray
    max_depth: np.ndarray
    cell_area_m2: float
    width: int

    @property
    def basin_count(self) -> int:
        return self.counts.size - 1

    @property
    def closed(self) -> np.ndarray:
        """Basins with a finite spill level that are not censored."""
        return ~self.censored & np.isfinite(self.spill)


def _validate_inputs(
    dem: np.ndarray,
    depression_depth: np.ndarray,
    d8_pointer: np.ndarray,
    contributing_area: np.ndarray,
    cell_area_m2: float,
) -> None:
    if cell_area_m2 <= 0 or not np.isfinite(cell_area_m2):
        raise ValueError("Cell area must be finite and positive.")
    if not (
        dem.shape
        == depression_depth.shape
        == d8_pointer.shape
        == contributing_area.shape
    ):
        raise ValueError(
            "DEM, depression, pointer, and contributing-area arrays must align."
        )


def _warn_if_accumulation_in_cells(
    contributing_area: np.ndarray,
    valid: np.ndarray,
    cell_area_m2: float,
    in_cells: bool,
) -> None:
    """Warn when the smallest positive accumulation looks like a cell count."""
    if in_cells or cell_area_m2 <= 1.5:
        return
    positive = contributing_area[valid & (contributing_area > 0)]
    if positive.size and float(positive.min()) < 0.5 * cell_area_m2:
        warnings.warn(
            "Contributing area looks like a cell count (smallest positive value "
            f"{float(positive.min()):g} < cell area {cell_area_m2:g} m2). Pass "
            "contributing_area_in_cells=True if it is not in m2.",
            RuntimeWarning,
            stacklevel=3,
        )


def _build_network(
    dem: np.ndarray,
    depression_depth: np.ndarray,
    d8_pointer: np.ndarray,
    contributing_area: np.ndarray,
    cell_area_m2: float,
    minimum_depression_depth_m: float,
    minimum_basin_area_m2: float,
    label_epsilon_m: float,
    contributing_area_in_cells: bool,
    dem_nodata: float | None,
) -> _BasinNetwork:
    """Label basins, find spill outlets and topology, and compute capacities."""
    _validate_inputs(dem, depression_depth, d8_pointer, contributing_area, cell_area_m2)
    height, width = dem.shape
    valid = _valid_mask(
        dem, depression_depth, d8_pointer, contributing_area, dem_nodata
    )
    _warn_if_accumulation_in_cells(
        contributing_area, valid, cell_area_m2, contributing_area_in_cells
    )
    area_scale = cell_area_m2 if contributing_area_in_cells else 1.0

    labels, counts = label_depression_components(
        dem,
        depression_depth,
        valid,
        cell_area_m2,
        minimum_depression_depth_m,
        minimum_basin_area_m2,
        label_epsilon_m,
    )
    n = counts.size - 1
    empty = np.zeros(n + 1)
    if n == 0:
        return _BasinNetwork(
            labels,
            counts,
            np.full(1, np.inf),
            np.full(1, -1, dtype=np.int64),
            np.zeros(1, dtype=bool),
            [""],
            np.full(1, -1, dtype=np.int64),
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=bool),
            np.zeros(1, dtype=np.int32),
            empty,
            empty,
            empty,
            empty,
            empty,
            cell_area_m2,
            width,
        )

    surface = dem + depression_depth
    spill, inside_outlet, outside_outlet, censored = _find_spill_outlets(
        surface, labels, valid, d8_pointer, counts, contributing_area
    )
    del surface

    outlet_area = np.zeros(n + 1, dtype=np.float64)
    has_inside = inside_outlet >= 0
    flat_area = np.ravel(contributing_area)
    outlet_area[has_inside] = (
        flat_area[inside_outlet[has_inside]].astype(np.float64) * area_scale
    )

    parent = np.zeros(n + 1, dtype=np.int32)
    reasons = [""] * (n + 1)
    trace_end = np.full(n + 1, -1, dtype=np.int64)
    trace_steps = np.zeros(n + 1, dtype=np.int32)
    max_steps = min(dem.size, 1_000_000)
    for basin_id in range(1, n + 1):
        if censored[basin_id] or not np.isfinite(spill[basin_id]):
            reasons[basin_id] = "censored"
            continue
        outside = int(outside_outlet[basin_id])
        if outside < 0:
            reasons[basin_id] = "no_outlet"
            continue
        (
            parent[basin_id],
            reasons[basin_id],
            trace_end[basin_id],
            trace_steps[basin_id],
        ) = _trace_downstream(outside, basin_id, labels, d8_pointer, valid, max_steps)

    # A downstream node must have more D8 contributing area at its outlet. The
    # comparison tolerates float32 quantisation of the accumulation raster
    # (outlets of large basins can tie or swap by a few ulps). Links that
    # violate it beyond that indicate disagreeing fill surfaces and are
    # dropped and counted; any cycle left among the kept ties is broken too.
    has_parent = parent > 0
    dropped = np.zeros(n + 1, dtype=bool)
    senders = np.flatnonzero(has_parent)
    if senders.size:
        sender_area = outlet_area[senders]
        tolerance = OUTLET_AREA_TOLERANCE * np.maximum(sender_area, 1.0)
        dropped[senders] = outlet_area[parent[senders]] < sender_area - tolerance
    dropped_receiver = np.where(dropped, parent, 0).astype(np.int32)
    parent[dropped] = 0
    _break_parent_cycles(parent, outlet_area, dropped, dropped_receiver)
    has_parent = parent > 0
    child_area = np.bincount(
        parent[has_parent], weights=outlet_area[has_parent], minlength=n + 1
    )
    local_area = np.maximum(outlet_area - child_area, 0.0)

    if dropped.any():
        warnings.warn(
            f"{int(dropped.sum())} downstream basin link(s) were dropped because the "
            "receiver has no larger contributing area or the link would close a "
            "cycle; their overflow leaves the "
            "system (status 'spill_link_dropped').",
            RuntimeWarning,
            stacklevel=3,
        )
    unresolved_counts = Counter(
        r for r in reasons if r in (TRACE_NO_FLOW, TRACE_CYCLE, TRACE_MAX_STEPS)
    )
    if unresolved_counts:
        breakdown = ", ".join(f"{k}: {v}" for k, v in unresolved_counts.most_common())
        warnings.warn(
            f"{sum(unresolved_counts.values())} spill path(s) ended without reaching "
            f"a basin, the edge or NoData ({breakdown}); status 'spill_path_unresolved'.",
            RuntimeWarning,
            stacklevel=3,
        )

    closed = ~censored & np.isfinite(spill)
    capacity = np.zeros(n + 1, dtype=np.float64)
    min_elevation = np.full(n + 1, np.inf, dtype=np.float64)
    block = _row_block(width)
    for r0 in range(0, height, block):
        r1 = min(r0 + block, height)
        block_labels = labels[r0:r1]
        selected = block_labels > 0
        if not selected.any():
            continue
        ids = block_labels[selected]
        elevations = dem[r0:r1][selected]
        np.minimum.at(min_elevation, ids, elevations)
        in_closed = closed[ids]
        if not in_closed.any():
            continue
        ids_c = ids[in_closed]
        depth_to_spill = np.maximum(spill[ids_c] - elevations[in_closed], 0.0)
        capacity += np.bincount(
            ids_c, weights=depth_to_spill * cell_area_m2, minlength=n + 1
        )
    min_elevation[~np.isfinite(min_elevation)] = np.nan

    max_depth = np.zeros(n + 1, dtype=np.float64)
    max_depth[1:] = ndimage.maximum(depression_depth, labels, index=np.arange(1, n + 1))

    return _BasinNetwork(
        labels=labels,
        counts=counts,
        spill=spill,
        inside_outlet=inside_outlet,
        censored=censored,
        trace_reason=reasons,
        trace_end=trace_end,
        trace_steps=trace_steps,
        parent=parent,
        link_dropped=dropped,
        dropped_receiver=dropped_receiver,
        outlet_area=outlet_area,
        local_area=local_area,
        capacity=capacity,
        min_elevation=min_elevation,
        max_depth=max_depth,
        cell_area_m2=cell_area_m2,
        width=width,
    )


def _status(net: _BasinNetwork, basin_id: int) -> str:
    if net.censored[basin_id] or not np.isfinite(net.spill[basin_id]):
        return STATUS_CENSORED
    if net.outlet_area[basin_id] <= 0:
        return STATUS_NO_AREA
    if net.parent[basin_id] > 0:
        return STATUS_TO_BASIN
    if net.link_dropped[basin_id]:
        return STATUS_DROPPED
    if net.trace_reason[basin_id] in _OPEN_END_REASONS:
        return STATUS_OUT
    return STATUS_UNRESOLVED


def _network_record(net: _BasinNetwork, basin_id: int) -> BasinRecord:
    inside = int(net.inside_outlet[basin_id])
    outlet_row, outlet_col = divmod(inside, net.width) if inside >= 0 else (-1, -1)
    spill = float(net.spill[basin_id])
    if net.censored[basin_id]:
        spill = math.nan
    return BasinRecord(
        basin_id=basin_id,
        cell_count=int(net.counts[basin_id]),
        depression_area_m2=float(net.counts[basin_id] * net.cell_area_m2),
        max_depression_depth_m=float(net.max_depth[basin_id]),
        min_elevation_m=float(net.min_elevation[basin_id]),
        spill_elevation_m=spill if np.isfinite(spill) else math.nan,
        capacity_m3=float(net.capacity[basin_id]),
        contributing_area_m2=float(net.outlet_area[basin_id]),
        incremental_contributing_area_m2=float(net.local_area[basin_id]),
        downstream_basin_id=int(net.parent[basin_id]),
        downstream_link_dropped=int(net.link_dropped[basin_id]),
        dropped_receiver_id=int(net.dropped_receiver[basin_id]),
        spill_trace_end=net.trace_reason[basin_id],
        outlet_row=outlet_row,
        outlet_col=outlet_col,
        status=_status(net, basin_id),
    )


# --------------------------------------------------------------------------
# Public array-level API
# --------------------------------------------------------------------------
def prepare_basin_network(
    dem: np.ndarray,
    depression_depth: np.ndarray,
    d8_pointer: np.ndarray,
    contributing_area: np.ndarray,
    cell_area_m2: float,
    minimum_depression_depth_m: float = 0.02,
    minimum_basin_area_m2: float = 25.0,
    label_epsilon_m: float = LABEL_EPSILON_M,
    contributing_area_in_cells: bool = False,
    dem_nodata: float | None = None,
) -> tuple[np.ndarray, list[BasinRecord]]:
    """Prepare basin labels, capacities, and downstream topology without rainfall.

    The returned records contain terrain/network properties suitable for
    building a dynamic model. No static event volume or ponding estimate is
    computed here. ``spill_elevation_m`` is NaN for censored basins; their
    lowest cell elevation is in ``min_elevation_m``.
    """
    net = _build_network(
        dem,
        depression_depth,
        d8_pointer,
        contributing_area,
        cell_area_m2,
        minimum_depression_depth_m,
        minimum_basin_area_m2,
        label_epsilon_m,
        contributing_area_in_cells,
        dem_nodata,
    )
    records = [_network_record(net, b) for b in range(1, net.basin_count + 1)]
    return net.labels, records


def simulate_basin_storage_spill(
    dem: np.ndarray,
    depression_depth: np.ndarray,
    d8_pointer: np.ndarray,
    contributing_area: np.ndarray,
    cell_area_m2: float,
    effective_rain_m: float,
    minimum_depression_depth_m: float = 0.02,
    minimum_basin_area_m2: float = 25.0,
    label_epsilon_m: float = LABEL_EPSILON_M,
    contributing_area_in_cells: bool = False,
    dem_nodata: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[BasinRecord]]:
    """Simulate static event volume filling DEM depressions and spilling onward.

    Depression fill zones are treated as 8-connected basin components; this is
    not a nested depression hierarchy, so adjacent/nested fill zones can merge.
    D8 contributing area at each component's lowest saddle supplies uniform
    event runoff; downstream saddle paths carry only overflow to the next
    retained component. Edge/NoData-censored components are treated as open
    outlets.

    This is a static DEM-only screening balance: it has no storm timing,
    infiltration variability, culverts, or surface-water hydraulics.
    """
    if effective_rain_m < 0 or not np.isfinite(effective_rain_m):
        raise ValueError("Effective rainfall must be finite and non-negative.")
    net = _build_network(
        dem,
        depression_depth,
        d8_pointer,
        contributing_area,
        cell_area_m2,
        minimum_depression_depth_m,
        minimum_basin_area_m2,
        label_epsilon_m,
        contributing_area_in_cells,
        dem_nodata,
    )
    n = net.basin_count
    depths = np.zeros(dem.shape, dtype=np.float32)
    if n == 0:
        return net.labels, depths, []

    local_volume = net.local_area * effective_rain_m
    incoming = local_volume.copy()
    retained = np.zeros(n + 1, dtype=np.float64)
    spill_volume = np.zeros(n + 1, dtype=np.float64)
    closed = net.closed
    # A topological order of the spill graph: every basin is processed after
    # all its upstream senders, also when outlet areas tie within tolerance.
    for basin_id in _topological_order(net.parent):
        if not closed[basin_id]:
            spill_volume[basin_id] = incoming[basin_id]
            continue
        retained[basin_id] = min(incoming[basin_id], net.capacity[basin_id])
        spill_volume[basin_id] = max(incoming[basin_id] - retained[basin_id], 0.0)
        receiver = int(net.parent[basin_id])
        if receiver > 0:
            incoming[receiver] += spill_volume[basin_id]

    exported = float(spill_volume[1:][net.parent[1:] == 0].sum())
    balance = float(local_volume.sum()) - float(retained.sum()) - exported
    if abs(balance) > 1e-6 * max(1.0, float(local_volume.sum())):
        warnings.warn(
            f"Volume balance residual {balance:.6g} m3 (runoff - retained - exported).",
            RuntimeWarning,
            stacklevel=2,
        )

    # Per-basin stage-storage inversion on cell elevations sorted by (label, z).
    water_levels = np.full(n + 1, np.nan, dtype=np.float64)
    cell_mask = net.labels > 0
    ids = net.labels[cell_mask]
    elevations = dem[cell_mask].astype(np.float64)
    order = np.lexsort((elevations, ids))
    sorted_elevations = elevations[order]
    del cell_mask, ids, elevations, order
    offsets = np.concatenate(([0], np.cumsum(net.counts[1:], dtype=np.int64)))
    for basin_id in range(1, n + 1):
        if not closed[basin_id]:
            continue
        cells = sorted_elevations[offsets[basin_id - 1] : offsets[basin_id]]
        water_levels[basin_id] = _water_level_for_volume(
            cells, retained[basin_id], cell_area_m2, net.spill[basin_id]
        )

    # Rasterized water depth is zero outside retained, closed depression basins.
    height, width = dem.shape
    block = _row_block(width)
    for r0 in range(0, height, block):
        r1 = min(r0 + block, height)
        block_labels = net.labels[r0:r1]
        selected = block_labels > 0
        if not selected.any():
            continue
        levels = water_levels[block_labels[selected]]
        cell_elev = dem[r0:r1][selected]
        wet = np.isfinite(levels)
        result = np.zeros(levels.shape, dtype=np.float32)
        result[wet] = np.maximum(levels[wet] - cell_elev[wet], 0.0).astype(np.float32)
        depths[r0:r1][selected] = result

    summaries: list[BasinRecord] = []
    for basin_id in range(1, n + 1):
        record = _network_record(net, basin_id)
        record.update(
            local_event_runoff_m3=float(local_volume[basin_id]),
            incoming_volume_m3=float(incoming[basin_id]),
            retained_volume_m3=float(retained[basin_id]),
            spill_volume_m3=float(spill_volume[basin_id]),
            water_level_m=float(water_levels[basin_id]),
            effective_rainfall_m=float(effective_rain_m),
        )
        summaries.append(record)
    return net.labels, depths, summaries


# --------------------------------------------------------------------------
# Raster / CSV I/O
# --------------------------------------------------------------------------
def _masked_float32(array: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return float32 data with a finite NoData value replaced by NaN."""
    data = array.astype(np.float32, copy=False)
    if nodata is not None and np.isfinite(nodata):
        data = np.where(data == np.float32(nodata), np.float32(np.nan), data)
    return data


@dataclass
class _Inputs:
    dem: np.ndarray
    depression: np.ndarray
    pointer: np.ndarray
    accumulation: np.ndarray
    nodata: float | None
    cell_area: float
    transform: object
    crs: object
    profile: dict


def _read_inputs(
    dem_path: Path,
    depression_depth_path: Path,
    d8_pointer_path: Path,
    contributing_area_path: Path,
    description: str,
) -> _Inputs:
    with rasterio.open(dem_path) as dem_source:
        arrays = []
        for path in (depression_depth_path, d8_pointer_path, contributing_area_path):
            with rasterio.open(path) as source:
                if (
                    source.shape != dem_source.shape
                    or source.transform != dem_source.transform
                    or source.crs != dem_source.crs
                ):
                    raise ValueError(f"{description} raster is not aligned: {path}")
                arrays.append(_masked_float32(source.read(1), source.nodata))
        dem = _masked_float32(dem_source.read(1), dem_source.nodata)
        transform = dem_source.transform
        return _Inputs(
            dem=dem,
            depression=arrays[0],
            pointer=arrays[1],
            accumulation=arrays[2],
            nodata=dem_source.nodata,
            cell_area=abs(transform.a * transform.e - transform.b * transform.d),
            transform=transform,
            crs=dem_source.crs,
            profile=dem_source.profile.copy(),
        )


def _add_outlet_coordinates(
    summaries: list[BasinRecord], transform: object, crs: object
) -> None:
    for summary in summaries:
        row, col = int(summary["outlet_row"]), int(summary["outlet_col"])
        if row >= 0 and col >= 0:
            outlet_x, outlet_y = rasterio.transform.xy(
                transform, row, col, offset="center"
            )
        else:
            outlet_x, outlet_y = math.nan, math.nan
        summary["outlet_x"] = float(outlet_x)
        summary["outlet_y"] = float(outlet_y)
        summary["crs"] = str(crs)


def _write_csv(path: Path, fields: list[str], rows: list[BasinRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tiled_profile(profile: dict, dtype: str, nodata: float | int) -> dict:
    out = profile.copy()
    out.update(
        driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="lzw", tiled=True
    )
    out.pop("blockxsize", None)
    out.pop("blockysize", None)
    return out


def write_basin_network_outputs(
    dem_path: Path,
    depression_depth_path: Path,
    d8_pointer_path: Path,
    contributing_area_path: Path,
    basin_id_path: Path,
    basin_summary_path: Path,
    minimum_depression_depth_m: float = 0.02,
    minimum_basin_area_m2: float = 25.0,
    label_epsilon_m: float = LABEL_EPSILON_M,
    contributing_area_in_cells: bool = False,
) -> list[BasinRecord]:
    """Write basin labels and network properties without a rainfall scenario."""
    data = _read_inputs(
        dem_path,
        depression_depth_path,
        d8_pointer_path,
        contributing_area_path,
        "Basin preparation",
    )
    labels, summaries = prepare_basin_network(
        data.dem,
        data.depression,
        data.pointer,
        data.accumulation,
        data.cell_area,
        minimum_depression_depth_m,
        minimum_basin_area_m2,
        label_epsilon_m,
        contributing_area_in_cells,
    )
    _add_outlet_coordinates(summaries, data.transform, data.crs)

    basin_id_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        basin_id_path, "w", **_tiled_profile(data.profile, "int32", 0)
    ) as basin_dest:
        basin_dest.update_tags(
            model="dem_derived_basin_network_preparation",
            minimum_depression_depth_m=f"{minimum_depression_depth_m:.9g}",
            minimum_basin_area_m2=f"{minimum_basin_area_m2:.9g}",
            label_epsilon_m=f"{label_epsilon_m:.9g}",
            limitations=(
                "8-connected depression-fill components and D8-derived spill graph; "
                "not a nested depression hierarchy. No rainfall or event simulation."
            ),
        )
        basin_dest.write(labels, 1)
    _write_csv(basin_summary_path, NETWORK_FIELDS, summaries)
    return summaries


def write_basin_storage_spill_outputs(
    dem_path: Path,
    depression_depth_path: Path,
    d8_pointer_path: Path,
    contributing_area_path: Path,
    ponding_depth_path: Path,
    basin_id_path: Path,
    basin_summary_path: Path,
    effective_rain_m: float,
    minimum_depression_depth_m: float = 0.02,
    minimum_basin_area_m2: float = 25.0,
    label_epsilon_m: float = LABEL_EPSILON_M,
    contributing_area_in_cells: bool = False,
    scenario_tags: dict[str, str] | None = None,
) -> list[BasinRecord]:
    """Create basin-ID, ponding-depth, and per-basin CSV outputs."""
    data = _read_inputs(
        dem_path,
        depression_depth_path,
        d8_pointer_path,
        contributing_area_path,
        "Scenario input",
    )
    labels, depths, summaries = simulate_basin_storage_spill(
        data.dem,
        data.depression,
        data.pointer,
        data.accumulation,
        data.cell_area,
        effective_rain_m,
        minimum_depression_depth_m,
        minimum_basin_area_m2,
        label_epsilon_m,
        contributing_area_in_cells,
    )
    _add_outlet_coordinates(summaries, data.transform, data.crs)

    ponding_depth_path.parent.mkdir(parents=True, exist_ok=True)
    basin_id_path.parent.mkdir(parents=True, exist_ok=True)
    depth_nodata = data.nodata if data.nodata is not None else -9999.0
    depth_profile = _tiled_profile(data.profile, "float32", depth_nodata)
    label_profile = _tiled_profile(data.profile, "int32", 0)
    valid_dem = np.isfinite(data.dem)
    tags = {
        "model": "static_dem_d8_depression_storage_spill_screening",
        "effective_rainfall_m": f"{effective_rain_m:.9g}",
        "minimum_depression_depth_m": f"{minimum_depression_depth_m:.9g}",
        "minimum_basin_area_m2": f"{minimum_basin_area_m2:.9g}",
        "label_epsilon_m": f"{label_epsilon_m:.9g}",
        "depth_units": "m",
        "limitations": (
            "Static D8 screening using connected fill-zone components, not "
            "a nested depression hierarchy; no storm timing, spatial "
            "infiltration, culverts, sewers, or hydraulic routing."
        ),
    }
    tags.update(scenario_tags or {})
    height, width = data.dem.shape
    block = _row_block(width)
    with (
        rasterio.open(ponding_depth_path, "w", **depth_profile) as depth_dest,
        rasterio.open(basin_id_path, "w", **label_profile) as basin_dest,
    ):
        depth_dest.update_tags(**tags)
        basin_dest.update_tags(**tags)
        for row in range(0, height, block):
            stop = min(row + block, height)
            depth_block = depths[row:stop].copy()
            depth_block[~valid_dem[row:stop]] = depth_nodata
            window = Window(0, row, width, stop - row)
            depth_dest.write(depth_block, 1, window=window)
            basin_dest.write(labels[row:stop], 1, window=window)

    _write_csv(basin_summary_path, STORM_FIELDS, summaries)
    return summaries


# --------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------
def run_terrain_analysis(config: AnalysisConfig) -> None:
    """Build and analyze a buffered elevation mosaic for one configured area."""
    center_x, center_y = projected_center(config.center_lon, config.center_lat)
    result_bounds = (
        center_x - config.west_km * 1000,
        center_y - config.south_km * 1000,
        center_x + config.east_km * 1000,
        center_y + config.north_km * 1000,
    )
    analysis_bounds = (
        result_bounds[0] - config.buffer_west_km * 1000,
        result_bounds[1] - config.buffer_south_km * 1000,
        result_bounds[2] + config.buffer_east_km * 1000,
        result_bounds[3] + config.buffer_north_km * 1000,
    )
    tile_paths = select_tiles(config.input_dir, analysis_bounds)
    print(f"Analysis center in SWEREF99 TM: ({center_x:.2f}, {center_y:.2f})")
    print(
        "Result extents from center (west/east/north/south): "
        f"{config.west_km:g}/{config.east_km:g}/{config.north_km:g}/"
        f"{config.south_km:g} km."
    )
    print(
        "Buffers beyond result area (west/east/north/south): "
        f"{config.buffer_west_km:g}/{config.buffer_east_km:g}/"
        f"{config.buffer_north_km:g}/{config.buffer_south_km:g} km."
    )
    print(f"Selected {len(tile_paths)} elevation tiles from {config.input_dir}.")
    for tile_path in tile_paths:
        print(f"  {tile_path.name}")
    if config.dry_run:
        print(f"Output rectangle bounds: {result_bounds}")
        print(f"Single-pass analysis bounds: {analysis_bounds}")
        return

    output_dir = config.output_dir
    prefix = config.output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    storm = config.rainfall_mm is not None
    prepare_basins = storm or config.prepare_basins_only
    in_cells = bool(getattr(config, "contributing_area_in_cells", False))

    buffered_dem = output_dir / f"{prefix}_buffered_dem.tif"
    buffered_depth = output_dir / f"{prefix}_buffered_depression_depth.tif"
    result_depth = output_dir / f"{prefix}_depression_depth.tif"
    contributing_area = output_dir / f"{prefix}_buffered_contributing_area.tif"
    upslope_storage = output_dir / f"{prefix}_buffered_upslope_storage.tif"
    d8_pointer = output_dir / f"{prefix}_buffered_d8_pointer.tif"
    d8_contributing_area = output_dir / f"{prefix}_buffered_d8_contributing_area.tif"
    basin_ids = output_dir / f"{prefix}_buffered_depression_basin_id.tif"
    basin_summary = output_dir / f"{prefix}_buffered_basin_summary.csv"
    # When both a network-only run and a storm run are requested, keep their
    # outputs apart so the storm results do not overwrite the network files.
    if config.prepare_basins_only and storm:
        network_ids = output_dir / f"{prefix}_buffered_basin_network_id.tif"
        network_summary = output_dir / f"{prefix}_buffered_basin_network_summary.csv"
        print(
            "Both network preparation and a storm scenario requested; network "
            f"outputs use separate files ({network_summary.name})."
        )
    else:
        network_ids, network_summary = basin_ids, basin_summary

    effective_rain = 0.0
    scenario_tags: dict[str, str] = {}
    if storm:
        result_contributing_area = output_dir / f"{prefix}_contributing_area.tif"
        result_upslope_storage = output_dir / f"{prefix}_upslope_depression_storage.tif"
        buffered_ponding = output_dir / f"{prefix}_buffered_estimated_ponding_depth.tif"
        result_ponding = output_dir / f"{prefix}_estimated_ponding_depth.tif"
        result_basin_ids = output_dir / f"{prefix}_depression_basin_id.tif"
        fraction = (
            DEFAULT_INFILTRATION_FRACTION
            if config.infiltration_fraction is None
            else config.infiltration_fraction
        )
        effective_rain = effective_rainfall_m(
            config.rainfall_mm,
            infiltration_fraction=fraction,
            infiltration_capacity_mm_h=config.infiltration_capacity_mm_h,
            rainfall_duration_hours=config.rainfall_duration_hours,
        )
        scenario_tags = {
            "rainfall_mm": f"{config.rainfall_mm:.9g}",
            "infiltration_fraction": (
                "n/a"
                if config.infiltration_capacity_mm_h is not None
                else f"{fraction:.9g}"
            ),
            "infiltration_capacity_mm_h": (
                "n/a"
                if config.infiltration_capacity_mm_h is None
                else f"{config.infiltration_capacity_mm_h:.9g}"
            ),
            "rainfall_duration_hours": (
                "n/a"
                if config.infiltration_capacity_mm_h is None
                else f"{config.rainfall_duration_hours:.9g}"
            ),
        }
        if config.infiltration_capacity_mm_h is None:
            print(
                f"Storm rainfall: {config.rainfall_mm:g} mm; assuming "
                f"{fraction:.0%} uniform infiltration, leaving "
                f"{effective_rain * 1000:g} mm effective rainfall."
            )
        else:
            print(
                f"Storm rainfall: {config.rainfall_mm:g} mm over "
                f"{config.rainfall_duration_hours:g} h; constant infiltration "
                f"capacity {config.infiltration_capacity_mm_h:g} mm/h leaves "
                f"{effective_rain * 1000:g} mm effective rainfall."
            )

    print("Creating buffered elevation mosaic ...")
    create_buffered_mosaic(tile_paths, analysis_bounds, buffered_dem)
    print("Running one depression-fill calculation over the full buffered mosaic ...")
    if prepare_basins:
        print("Calculating D8 flow and basin-network preparation rasters ...")
    run_depression_analysis(
        buffered_dem,
        buffered_depth,
        contributing_area_path=contributing_area if prepare_basins else None,
        upslope_storage_path=upslope_storage if prepare_basins else None,
        d8_pointer_path=d8_pointer if prepare_basins else None,
        d8_contributing_area_path=d8_contributing_area if prepare_basins else None,
    )
    if config.prepare_basins_only:
        summaries = write_basin_network_outputs(
            buffered_dem,
            buffered_depth,
            d8_pointer,
            d8_contributing_area,
            network_ids,
            network_summary,
            minimum_depression_depth_m=config.minimum_depression_depth_m,
            minimum_basin_area_m2=config.minimum_basin_area_m2,
            contributing_area_in_cells=in_cells,
        )
        print(f"Prepared {len(summaries):,} DEM-derived basins.")
        print(f"Basin IDs: {network_ids}")
        print(f"Basin network summary: {network_summary}")
    print("Clipping to the requested result rectangle ...")
    clip_to_area(buffered_depth, result_depth, result_bounds)

    if storm:
        clip_to_area(contributing_area, result_contributing_area, result_bounds)
        clip_to_area(upslope_storage, result_upslope_storage, result_bounds)
        basin_summaries = write_basin_storage_spill_outputs(
            buffered_dem,
            buffered_depth,
            d8_pointer,
            d8_contributing_area,
            buffered_ponding,
            basin_ids,
            basin_summary,
            effective_rain,
            minimum_depression_depth_m=config.minimum_depression_depth_m,
            minimum_basin_area_m2=config.minimum_basin_area_m2,
            contributing_area_in_cells=in_cells,
            scenario_tags=scenario_tags,
        )
        clip_to_area(buffered_ponding, result_ponding, result_bounds)
        clip_to_area(basin_ids, result_basin_ids, result_bounds)
        retained = sum(float(r["retained_volume_m3"]) for r in basin_summaries)
        exported = sum(
            float(r["spill_volume_m3"])
            for r in basin_summaries
            if int(r["downstream_basin_id"]) == 0
        )
        print(f"Contributing area raster (m²): {result_contributing_area}")
        print(
            f"Average upslope depression storage raster (m): {result_upslope_storage}"
        )
        print(f"Basin-based estimated ponding-depth raster (m): {result_ponding}")
        print(f"Depression basin ID raster: {result_basin_ids}")
        print(f"Basin storage/spill summary: {basin_summary}")
        print(
            f"Processed {len(basin_summaries):,} depression components; "
            f"retained {retained:,.1f} m³ in depressions, "
            f"{exported:,.1f} m³ left the analysis (edge, NoData or unresolved paths)."
        )

    print(f"Finished. Rectangular depression-depth raster: {result_depth}")
