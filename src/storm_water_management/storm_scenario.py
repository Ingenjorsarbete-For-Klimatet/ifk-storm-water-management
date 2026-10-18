"""DEM-derived static depression storage and spill screening."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

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


def label_depression_components(
    dem: np.ndarray,
    depression_depth: np.ndarray,
    valid: np.ndarray,
    cell_area_m2: float,
    minimum_depth_m: float,
    minimum_area_m2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Label retained 8-connected depression-fill components.

    Returns an int32 label grid (zero is background) and a per-label cell count.
    Small components are omitted to reduce sensitivity to tiny DEM artefacts.
    """
    if minimum_depth_m < 0 or minimum_area_m2 < 0:
        raise ValueError("Basin depth and area thresholds must be non-negative.")
    depression_mask = valid & (depression_depth > minimum_depth_m)
    labels, component_count = ndimage.label(
        depression_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    counts = np.bincount(labels.ravel(), minlength=component_count + 1)
    minimum_cells = max(1, math.ceil(minimum_area_m2 / cell_area_m2))
    keep = counts >= minimum_cells
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
    dem: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    pointer: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find each component's lowest valid boundary saddle and D8 outlet pair."""
    basin_count = counts.size - 1
    height, width = labels.shape
    spill_levels = np.full(basin_count + 1, np.inf, dtype=np.float64)
    touches_invalid = np.zeros(basin_count + 1, dtype=bool)
    # Raster-edge and NoData-adjacent components are open/censored; their true
    # spill level may lie outside the available DEM and is not inferred here.
    edge_labels = np.concatenate(
        (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
    )
    touches_invalid[np.unique(edge_labels[edge_labels > 0])] = True

    for (row_offset, col_offset), _code in D8_CODES.items():
        source_slice, target_slice = _neighbor_slices(
            height, width, row_offset, col_offset
        )
        source_labels = labels[source_slice]
        target_labels = labels[target_slice]
        source_valid = valid[source_slice]
        target_valid = valid[target_slice]
        source_dem = dem[source_slice]
        target_dem = dem[target_slice]
        invalid_boundary = (source_labels > 0) & ~target_valid
        if invalid_boundary.any():
            touches_invalid[np.unique(source_labels[invalid_boundary])] = True

        boundary = (
            (source_labels > 0)
            & source_valid
            & target_valid
            & (target_labels != source_labels)
        )
        if not boundary.any():
            continue
        basin_ids = source_labels[boundary]
        saddle = np.maximum(source_dem[boundary], target_dem[boundary])
        np.minimum.at(spill_levels, basin_ids, saddle)

    # On a tied lowest saddle, prefer a crossing that the conditioned D8 pointer
    # actually follows, then choose the lowest linear-index pair deterministically.
    best_pair = np.full(basin_count + 1, np.iinfo(np.int64).max, dtype=np.int64)
    best_any_pair = np.full_like(best_pair, np.iinfo(np.int64).max)
    total_cells = height * width
    for (row_offset, col_offset), code in D8_CODES.items():
        source_slice, target_slice = _neighbor_slices(
            height, width, row_offset, col_offset
        )
        source_labels = labels[source_slice]
        target_labels = labels[target_slice]
        boundary = (
            (source_labels > 0)
            & valid[source_slice]
            & valid[target_slice]
            & (target_labels != source_labels)
        )
        if not boundary.any():
            continue
        source_dem = dem[source_slice]
        target_dem = dem[target_slice]
        basin_ids = source_labels[boundary]
        saddle = np.maximum(source_dem[boundary], target_dem[boundary])
        at_lowest_saddle = np.isclose(
            saddle, spill_levels[basin_ids], rtol=0.0, atol=1e-6
        )
        if not at_lowest_saddle.any():
            continue

        boundary_rows, boundary_cols = np.nonzero(boundary)
        source_rows = boundary_rows + source_slice[0].start
        source_cols = boundary_cols + source_slice[1].start
        source_flat = source_rows * width + source_cols
        target_flat = source_flat + row_offset * width + col_offset
        pair_key = source_flat.astype(np.int64) * total_cells + target_flat
        ids = basin_ids[at_lowest_saddle]
        keys = pair_key[at_lowest_saddle]
        np.minimum.at(best_any_pair, ids, keys)
        pointer_values = pointer[source_slice][boundary]
        follows_saddle = at_lowest_saddle & (pointer_values == code)
        if follows_saddle.any():
            np.minimum.at(
                best_pair,
                basin_ids[follows_saddle],
                pair_key[follows_saddle],
            )

    selected_pair = np.where(
        best_pair != np.iinfo(np.int64).max, best_pair, best_any_pair
    )
    outlet_inside = np.full(basin_count + 1, -1, dtype=np.int64)
    outlet_outside = np.full(basin_count + 1, -1, dtype=np.int64)
    has_outlet = selected_pair != np.iinfo(np.int64).max
    outlet_inside[has_outlet] = selected_pair[has_outlet] // total_cells
    outlet_outside[has_outlet] = selected_pair[has_outlet] % total_cells
    touches_invalid[~has_outlet] = True
    return spill_levels, outlet_inside, outlet_outside, touches_invalid


def _downstream_basin(
    start: int,
    source_basin: int,
    labels: np.ndarray,
    pointer: np.ndarray,
    valid: np.ndarray,
    max_steps: int,
) -> int:
    """Trace D8 from a spill saddle to the next retained basin or raster edge."""
    height, width = labels.shape
    row, col = divmod(int(start), width)
    for _ in range(max_steps):
        if row < 0 or row >= height or col < 0 or col >= width:
            return 0
        if not valid[row, col]:
            return 0
        basin = int(labels[row, col])
        if basin > 0 and basin != source_basin:
            return basin
        code = int(pointer[row, col])
        offset = D8_OFFSETS.get(code)
        if offset is None:
            return 0
        row += offset[0]
        col += offset[1]
    return 0


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


def prepare_basin_network(
    dem: np.ndarray,
    depression_depth: np.ndarray,
    d8_pointer: np.ndarray,
    contributing_area: np.ndarray,
    cell_area_m2: float,
    minimum_depression_depth_m: float = 0.02,
    minimum_basin_area_m2: float = 25.0,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    """Prepare basin labels, capacities, and downstream topology without rainfall.

    The returned records contain terrain/network properties suitable for
    building a dynamic model. No static event volume or ponding estimate is
    computed here.
    """
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

    valid = (
        np.isfinite(dem)
        & np.isfinite(depression_depth)
        & np.isfinite(d8_pointer)
        & np.isfinite(contributing_area)
        & (contributing_area >= 0)
    )
    labels, counts = label_depression_components(
        dem,
        depression_depth,
        valid,
        cell_area_m2,
        minimum_depression_depth_m,
        minimum_basin_area_m2,
    )
    basin_count = counts.size - 1
    if basin_count == 0:
        return labels, []

    spill, inside_outlet, outside_outlet, censored = _find_spill_outlets(
        dem, labels, valid, d8_pointer, counts
    )
    outlet_area = np.zeros(basin_count + 1, dtype=np.float64)
    for basin_id in range(1, basin_count + 1):
        inside = inside_outlet[basin_id]
        if inside >= 0:
            outlet_area[basin_id] = float(contributing_area.flat[inside])

    parent = np.zeros(basin_count + 1, dtype=np.int32)
    max_steps = min(dem.size, 1_000_000)
    for basin_id in range(1, basin_count + 1):
        outside = outside_outlet[basin_id]
        if outside >= 0 and not censored[basin_id]:
            parent[basin_id] = _downstream_basin(
                int(outside), basin_id, labels, d8_pointer, valid, max_steps
            )
        receiver = int(parent[basin_id])
        if receiver > 0 and outlet_area[receiver] <= outlet_area[basin_id]:
            parent[basin_id] = 0

    child_area = np.zeros(basin_count + 1, dtype=np.float64)
    for basin_id in range(1, basin_count + 1):
        receiver = int(parent[basin_id])
        if receiver > 0:
            child_area[receiver] += outlet_area[basin_id]
    local_area = np.maximum(outlet_area - child_area, 0.0)

    capacity = np.zeros(basin_count + 1, dtype=np.float64)
    minimum_elevation = np.full(basin_count + 1, np.inf, dtype=np.float64)
    flat_labels = labels.ravel()
    flat_dem = dem.ravel()
    for start in range(0, flat_labels.size, 1_000_000):
        stop = min(start + 1_000_000, flat_labels.size)
        ids = flat_labels[start:stop]
        selected = ids > 0
        if not selected.any():
            continue
        selected_ids = ids[selected]
        elevations = flat_dem[start:stop][selected]
        np.minimum.at(minimum_elevation, selected_ids, elevations)
        closed = ~censored[selected_ids] & np.isfinite(spill[selected_ids])
        if not closed.any():
            continue
        selected_ids = selected_ids[closed]
        elevations = elevations[closed]
        depths_to_spill = np.maximum(spill[selected_ids] - elevations, 0.0)
        capacity += np.bincount(
            selected_ids,
            weights=depths_to_spill * cell_area_m2,
            minlength=basin_count + 1,
        )

    records: list[dict[str, float | int | str]] = []
    for basin_id in range(1, basin_count + 1):
        inside = int(inside_outlet[basin_id])
        outlet_row, outlet_col = (
            divmod(inside, dem.shape[1]) if inside >= 0 else (-1, -1)
        )
        receiver = int(parent[basin_id])
        if censored[basin_id] or not np.isfinite(spill[basin_id]):
            status = "edge_or_nodata_censored"
        elif outlet_area[basin_id] <= 0:
            status = "no_contributing_area"
        elif receiver > 0:
            status = "spills_to_basin"
        else:
            status = "spills_out_of_analysis_area"
        records.append(
            {
                "basin_id": basin_id,
                "cell_count": int(counts[basin_id]),
                "depression_area_m2": float(counts[basin_id] * cell_area_m2),
                # Censored basins are routed to the model boundary, not treated
                # as storage nodes; use a finite placeholder for CSV consumers.
                "spill_elevation_m": float(spill[basin_id])
                if np.isfinite(spill[basin_id])
                else float(minimum_elevation[basin_id]),
                "capacity_m3": float(capacity[basin_id]),
                "contributing_area_m2": float(outlet_area[basin_id]),
                "incremental_contributing_area_m2": float(local_area[basin_id]),
                "downstream_basin_id": receiver,
                "outlet_row": outlet_row,
                "outlet_col": outlet_col,
                "status": status,
            }
        )
    return labels, records


def simulate_basin_storage_spill(
    dem: np.ndarray,
    depression_depth: np.ndarray,
    d8_pointer: np.ndarray,
    contributing_area: np.ndarray,
    cell_area_m2: float,
    effective_rain_m: float,
    minimum_depression_depth_m: float = 0.02,
    minimum_basin_area_m2: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
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
    if cell_area_m2 <= 0 or not np.isfinite(cell_area_m2):
        raise ValueError("Cell area must be finite and positive.")
    if effective_rain_m < 0 or not np.isfinite(effective_rain_m):
        raise ValueError("Effective rainfall must be finite and non-negative.")
    if not (
        dem.shape
        == depression_depth.shape
        == d8_pointer.shape
        == contributing_area.shape
    ):
        raise ValueError(
            "DEM, depression, pointer, and contributing-area arrays must align."
        )

    valid = (
        np.isfinite(dem)
        & np.isfinite(depression_depth)
        & np.isfinite(d8_pointer)
        & np.isfinite(contributing_area)
        & (contributing_area >= 0)
    )
    labels, counts = label_depression_components(
        dem,
        depression_depth,
        valid,
        cell_area_m2,
        minimum_depression_depth_m,
        minimum_basin_area_m2,
    )
    basin_count = counts.size - 1
    depths = np.zeros(dem.shape, dtype=np.float32)
    if basin_count == 0:
        return labels, depths, []

    spill, inside_outlet, outside_outlet, censored = _find_spill_outlets(
        dem, labels, valid, d8_pointer, counts
    )
    outlet_area = np.zeros(basin_count + 1, dtype=np.float64)
    for basin_id in range(1, basin_count + 1):
        inside = inside_outlet[basin_id]
        if inside >= 0:
            outlet_area[basin_id] = float(contributing_area.flat[inside])

    parent = np.zeros(basin_count + 1, dtype=np.int32)
    max_steps = min(dem.size, 1_000_000)
    for basin_id in range(1, basin_count + 1):
        outside = outside_outlet[basin_id]
        if outside >= 0 and not censored[basin_id]:
            parent[basin_id] = _downstream_basin(
                int(outside),
                basin_id,
                labels,
                d8_pointer,
                valid,
                max_steps,
            )
        receiver = int(parent[basin_id])
        if receiver > 0 and outlet_area[receiver] <= outlet_area[basin_id]:
            # A downstream node must have at least as much D8 contributing area.
            parent[basin_id] = 0

    child_area = np.zeros(basin_count + 1, dtype=np.float64)
    for basin_id in range(1, basin_count + 1):
        receiver = int(parent[basin_id])
        if receiver > 0:
            child_area[receiver] += outlet_area[basin_id]

    local_area = np.maximum(outlet_area - child_area, 0.0)
    local_volume = local_area * effective_rain_m
    capacity = np.zeros(basin_count + 1, dtype=np.float64)
    flat_labels = labels.ravel()
    flat_dem = dem.ravel()
    for start in range(0, flat_labels.size, 1_000_000):
        stop = min(start + 1_000_000, flat_labels.size)
        ids = flat_labels[start:stop]
        selected = ids > 0
        if not selected.any():
            continue
        selected_ids = ids[selected]
        elevations = flat_dem[start:stop][selected]
        depths_to_spill = np.maximum(spill[selected_ids] - elevations, 0.0)
        capacity += np.bincount(
            selected_ids,
            weights=depths_to_spill * cell_area_m2,
            minlength=basin_count + 1,
        )
    incoming = local_volume.copy()
    retained = np.zeros(basin_count + 1, dtype=np.float64)
    spill_volume = np.zeros(basin_count + 1, dtype=np.float64)
    water_levels = np.full(basin_count + 1, np.nan, dtype=np.float64)
    # Flow accumulation normally increases downstream, providing a topological
    # order for this acyclic D8-derived basin graph.
    processing_order = np.argsort(outlet_area, kind="stable")
    for basin_id in processing_order:
        if basin_id == 0:
            continue
        if censored[basin_id] or not np.isfinite(spill[basin_id]):
            spill_volume[basin_id] = incoming[basin_id]
        else:
            retained[basin_id] = min(incoming[basin_id], capacity[basin_id])
            spill_volume[basin_id] = max(incoming[basin_id] - retained[basin_id], 0.0)
            receiver = int(parent[basin_id])
            if receiver > 0:
                incoming[receiver] += spill_volume[basin_id]

    # Sort basin-cell elevations once by label; solve each basin's curve using
    # its sorted cell heights, then rasterize the resulting level per basin.
    cell_indices = np.flatnonzero(flat_labels)
    cell_ids = flat_labels[cell_indices]
    sort_order = np.argsort(cell_ids, kind="stable")
    sorted_elevations = flat_dem[cell_indices[sort_order]].astype(np.float64)
    del cell_indices, cell_ids, sort_order
    offsets = np.cumsum(counts[1:], dtype=np.int64)
    for basin_id in range(1, basin_count + 1):
        start = 0 if basin_id == 1 else int(offsets[basin_id - 2])
        stop = int(offsets[basin_id - 1])
        basin_elevations = np.sort(sorted_elevations[start:stop])
        if not censored[basin_id] and np.isfinite(spill[basin_id]):
            water_levels[basin_id] = _water_level_for_volume(
                basin_elevations,
                retained[basin_id],
                cell_area_m2,
                spill[basin_id],
            )

    # Rasterized water depth is zero outside retained, closed depression basins.
    flat_depths = depths.ravel()
    for start in range(0, flat_labels.size, 1_000_000):
        stop = min(start + 1_000_000, flat_labels.size)
        ids = flat_labels[start:stop]
        selected = ids > 0
        if not selected.any():
            continue
        selected_ids = ids[selected]
        elevations = flat_dem[start:stop][selected]
        levels = water_levels[selected_ids]
        wet = np.isfinite(levels)
        result = np.zeros(selected_ids.shape, dtype=np.float32)
        result[wet] = np.maximum(levels[wet] - elevations[wet], 0.0).astype(np.float32)
        flat_depths[start:stop][selected] = result

    summaries: list[dict[str, float | int | str]] = []
    for basin_id in range(1, basin_count + 1):
        inside = int(inside_outlet[basin_id])
        if inside >= 0:
            outlet_row, outlet_col = divmod(inside, dem.shape[1])
        else:
            outlet_row, outlet_col = -1, -1
        receiver = int(parent[basin_id])
        if censored[basin_id] or not np.isfinite(spill[basin_id]):
            status = "edge_or_nodata_censored"
        elif outlet_area[basin_id] <= 0:
            status = "no_contributing_area"
        elif receiver > 0:
            status = "spills_to_basin"
        else:
            status = "spills_out_of_analysis_area"
        summaries.append(
            {
                "basin_id": basin_id,
                "cell_count": int(counts[basin_id]),
                "depression_area_m2": float(counts[basin_id] * cell_area_m2),
                "spill_elevation_m": float(spill[basin_id])
                if np.isfinite(spill[basin_id])
                else math.nan,
                "capacity_m3": float(capacity[basin_id]),
                "contributing_area_m2": float(outlet_area[basin_id]),
                "incremental_contributing_area_m2": float(local_area[basin_id]),
                "local_event_runoff_m3": float(local_volume[basin_id]),
                "incoming_volume_m3": float(incoming[basin_id]),
                "retained_volume_m3": float(retained[basin_id]),
                "spill_volume_m3": float(spill_volume[basin_id]),
                "water_level_m": float(water_levels[basin_id]),
                "effective_rainfall_m": float(effective_rain_m),
                "downstream_basin_id": receiver,
                "outlet_row": outlet_row,
                "outlet_col": outlet_col,
                "status": status,
            }
        )
    return labels, depths, summaries


def write_basin_network_outputs(
    dem_path: Path,
    depression_depth_path: Path,
    d8_pointer_path: Path,
    contributing_area_path: Path,
    basin_id_path: Path,
    basin_summary_path: Path,
    minimum_depression_depth_m: float = 0.02,
    minimum_basin_area_m2: float = 25.0,
) -> list[dict[str, float | int | str]]:
    """Write basin labels and network properties without a rainfall scenario."""
    paths = (depression_depth_path, d8_pointer_path, contributing_area_path)
    with rasterio.open(dem_path) as dem_source:
        arrays = []
        for path in paths:
            with rasterio.open(path) as source:
                if (
                    source.shape != dem_source.shape
                    or source.transform != dem_source.transform
                    or source.crs != dem_source.crs
                ):
                    raise ValueError(f"Basin preparation raster is not aligned: {path}")
                arrays.append(source.read(1).astype(np.float32, copy=False))
        dem = dem_source.read(1).astype(np.float32, copy=False)
        cell_area = abs(
            dem_source.transform.a * dem_source.transform.e
            - dem_source.transform.b * dem_source.transform.d
        )
        transform = dem_source.transform
        crs = dem_source.crs
        profile = dem_source.profile.copy()

    labels, summaries = prepare_basin_network(
        dem,
        arrays[0],
        arrays[1],
        arrays[2],
        cell_area,
        minimum_depression_depth_m,
        minimum_basin_area_m2,
    )
    for summary in summaries:
        row, col = int(summary["outlet_row"]), int(summary["outlet_col"])
        if row >= 0 and col >= 0:
            outlet_x, outlet_y = transform * (col + 0.5, row + 0.5)
        else:
            outlet_x, outlet_y = math.nan, math.nan
        summary["outlet_x"] = float(outlet_x)
        summary["outlet_y"] = float(outlet_y)
        summary["crs"] = str(crs)

    basin_id_path.parent.mkdir(parents=True, exist_ok=True)
    label_profile = profile.copy()
    label_profile.update(
        driver="GTiff", count=1, dtype="int32", nodata=0, compress="lzw", tiled=True
    )
    label_profile.pop("blockxsize", None)
    label_profile.pop("blockysize", None)
    with rasterio.open(basin_id_path, "w", **label_profile) as basin_dest:
        basin_dest.update_tags(
            model="dem_derived_basin_network_preparation",
            minimum_depression_depth_m=f"{minimum_depression_depth_m:.9g}",
            minimum_basin_area_m2=f"{minimum_basin_area_m2:.9g}",
            limitations=(
                "8-connected depression-fill components and D8-derived spill graph; "
                "not a nested depression hierarchy. No rainfall or event simulation."
            ),
        )
        basin_dest.write(labels, 1)

    basin_summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        list(summaries[0])
        if summaries
        else [
            "basin_id",
            "cell_count",
            "depression_area_m2",
            "spill_elevation_m",
            "capacity_m3",
            "contributing_area_m2",
            "incremental_contributing_area_m2",
            "downstream_basin_id",
            "outlet_row",
            "outlet_col",
            "status",
            "outlet_x",
            "outlet_y",
            "crs",
        ]
    )
    with basin_summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
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
) -> list[dict[str, float | int | str]]:
    """Create basin-ID, ponding-depth, and per-basin CSV outputs."""
    paths = (depression_depth_path, d8_pointer_path, contributing_area_path)
    with rasterio.open(dem_path) as dem_source:
        arrays = []
        for path in paths:
            with rasterio.open(path) as source:
                if (
                    source.shape != dem_source.shape
                    or source.transform != dem_source.transform
                    or source.crs != dem_source.crs
                ):
                    raise ValueError(f"Scenario input raster is not aligned: {path}")
                arrays.append(source.read(1).astype(np.float32, copy=False))
        dem = dem_source.read(1).astype(np.float32, copy=False)
        nodata = dem_source.nodata
        cell_area = abs(
            dem_source.transform.a * dem_source.transform.e
            - dem_source.transform.b * dem_source.transform.d
        )
        transform = dem_source.transform
        crs = dem_source.crs
        profile = dem_source.profile.copy()

    depression, pointer, accumulation = arrays
    if nodata is None:
        valid_dem = np.isfinite(dem)
    else:
        valid_dem = np.isfinite(dem) & (dem != nodata)
    labels, depths, summaries = simulate_basin_storage_spill(
        dem,
        depression,
        pointer,
        accumulation,
        cell_area,
        effective_rain_m,
        minimum_depression_depth_m,
        minimum_basin_area_m2,
    )
    for summary in summaries:
        outlet_row = int(summary["outlet_row"])
        outlet_col = int(summary["outlet_col"])
        if outlet_row >= 0 and outlet_col >= 0:
            outlet_x, outlet_y = transform * (outlet_col + 0.5, outlet_row + 0.5)
        else:
            outlet_x, outlet_y = math.nan, math.nan
        summary["outlet_x"] = float(outlet_x)
        summary["outlet_y"] = float(outlet_y)
        summary["crs"] = str(crs)

    ponding_depth_path.parent.mkdir(parents=True, exist_ok=True)
    basin_id_path.parent.mkdir(parents=True, exist_ok=True)
    depth_profile = profile.copy()
    depth_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=nodata if nodata is not None else -9999.0,
        compress="lzw",
        tiled=True,
    )
    label_profile = profile.copy()
    label_profile.update(
        driver="GTiff",
        count=1,
        dtype="int32",
        nodata=0,
        compress="lzw",
        tiled=True,
    )
    depth_profile.pop("blockxsize", None)
    depth_profile.pop("blockysize", None)
    label_profile.pop("blockxsize", None)
    label_profile.pop("blockysize", None)
    with (
        rasterio.open(ponding_depth_path, "w", **depth_profile) as depth_dest,
        rasterio.open(basin_id_path, "w", **label_profile) as basin_dest,
    ):
        scenario_tags = {
            "model": "static_dem_d8_depression_storage_spill_screening",
            "effective_rainfall_m": f"{effective_rain_m:.9g}",
            "minimum_depression_depth_m": f"{minimum_depression_depth_m:.9g}",
            "minimum_basin_area_m2": f"{minimum_basin_area_m2:.9g}",
            "depth_units": "m",
            "limitations": (
                "Static D8 screening using connected fill-zone components, not "
                "a nested depression hierarchy; no storm timing, spatial "
                "infiltration, culverts, sewers, or hydraulic routing."
            ),
        }
        depth_dest.update_tags(**scenario_tags)
        basin_dest.update_tags(**scenario_tags)
        row_step = 512
        for row in range(0, dem.shape[0], row_step):
            stop = min(row + row_step, dem.shape[0])
            depth_block = depths[row:stop].copy()
            depth_block[~valid_dem[row:stop]] = depth_profile["nodata"]
            window = rasterio.windows.Window(0, row, dem.shape[1], stop - row)
            depth_dest.write(depth_block, 1, window=window)
            basin_dest.write(labels[row:stop], 1, window=window)

    basin_summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summaries:
        with basin_summary_path.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    else:
        basin_summary_path.write_text("basin_id,status\n", encoding="utf-8")
    return summaries
