"""Utilities for building and running conceptual PySWMM storage models."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

RAINFALL_INTERVAL_MINUTES = 5
CURVE_POINTS = 31
MINIMUM_STORAGE_DEPTH_M = 0.001
SIMULATION_START = datetime(2025, 1, 1, 0, 0)


@dataclass(frozen=True)
class Basin:
    """DEM-derived data needed to construct one conceptual storage basin."""

    basin_id: int
    incremental_area_m2: float
    spill_elevation_m: float
    downstream_basin_id: int
    status: str
    outlet_x: float | None
    outlet_y: float | None
    source_capacity_m3: float
    source_area_m2: float
    crs: str

    @property
    def node_id(self) -> str:
        """Return a stable SWMM identifier for the basin."""
        return f"B{self.basin_id:05d}"

    @property
    def subcatchment_id(self) -> str:
        """Return a stable SWMM identifier for the contributing area."""
        return f"C{self.basin_id:05d}"

    @property
    def curve_id(self) -> str:
        """Return a stable SWMM identifier for the storage curve."""
        return f"S{self.basin_id:05d}"


@dataclass(frozen=True)
class BasinGeometry:
    """Raster-derived elevation and stage-area data for one basin."""

    floor_elevation_m: float
    maximum_depth_m: float
    curve: tuple[tuple[float, float], ...]
    raster_cell_count: int
    curve_capacity_m3: float


def load_basins(summary_path: Path) -> tuple[list[Basin], float]:
    """Load basins and the existing effective-rainfall depth from the CSV."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"Basin summary does not exist: {summary_path}")

    basins: list[Basin] = []
    rainfall_depths: list[float] = []
    with summary_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {
            "basin_id",
            "incremental_contributing_area_m2",
            "spill_elevation_m",
            "downstream_basin_id",
            "status",
            "capacity_m3",
            "depression_area_m2",
            "effective_rainfall_m",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Basin summary is missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            basin_id = int(row["basin_id"])
            rainfall_depths.append(float(row["effective_rainfall_m"]))
            basins.append(
                Basin(
                    basin_id=basin_id,
                    incremental_area_m2=float(
                        row["incremental_contributing_area_m2"]
                    ),
                    spill_elevation_m=float(row["spill_elevation_m"]),
                    downstream_basin_id=int(row["downstream_basin_id"]),
                    status=row["status"],
                    outlet_x=_optional_float(row.get("outlet_x")),
                    outlet_y=_optional_float(row.get("outlet_y")),
                    source_capacity_m3=float(row["capacity_m3"]),
                    source_area_m2=float(row["depression_area_m2"]),
                    crs=row.get("crs", "") or "",
                )
            )

    if not basins:
        raise ValueError(f"No basin rows found in {summary_path}")
    if len({basin.basin_id for basin in basins}) != len(basins):
        raise ValueError("Basin IDs in the summary must be unique.")
    crs_values = {basin.crs for basin in basins if basin.crs}
    if len(crs_values) > 1:
        raise ValueError("Basin summary rows must use a single coordinate reference system.")
    if any(not math.isfinite(depth) or depth < 0 for depth in rainfall_depths):
        raise ValueError("Effective rainfall values in the summary must be finite and non-negative.")
    rainfall_mm = float(np.median(rainfall_depths)) * 1000.0
    basin_ids = {basin.basin_id for basin in basins}
    for basin in basins:
        if basin.incremental_area_m2 < 0 or not math.isfinite(basin.incremental_area_m2):
            raise ValueError(f"Invalid incremental area for basin {basin.basin_id}.")
        if not math.isfinite(basin.spill_elevation_m):
            raise ValueError(f"Invalid spill elevation for basin {basin.basin_id}.")
        if basin.downstream_basin_id and basin.downstream_basin_id not in basin_ids:
            raise ValueError(
                f"Basin {basin.basin_id} points to missing downstream basin "
                f"{basin.downstream_basin_id}."
            )
    return basins, rainfall_mm


def _optional_float(value: str | None) -> float | None:
    """Convert a nonempty optional CSV cell to a finite float."""
    if value is None or not value.strip():
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def select_storage_basins(
    basins: list[Basin], minimum_area_m2: float
) -> tuple[list[Basin], float]:
    """Filter minor storage nodes while conserving all incremental runoff area.

    Incremental area from each source basin is transferred to the first
    retained basin downstream. If no retained basin occurs before the analysis
    boundary, it is sent to a separate external-runoff catchment. Retained
    basin links are collapsed across omitted nodes to the next retained basin.
    """
    if minimum_area_m2 < 0 or not math.isfinite(minimum_area_m2):
        raise ValueError("Minimum modeled basin area must be finite and non-negative.")
    by_id = {basin.basin_id: basin for basin in basins}
    retained = {
        basin.basin_id
        for basin in basins
        if basin.source_area_m2 >= minimum_area_m2
    }
    if not retained:
        raise ValueError(
            "No basin meets the storage-area cutoff; lower "
            "--minimum-storage-basin-area-m2."
        )

    routed_area = {basin_id: 0.0 for basin_id in retained}
    external_area_m2 = 0.0
    for basin in basins:
        current_id = basin.basin_id
        visited: set[int] = set()
        while current_id and current_id not in retained:
            if current_id in visited:
                raise ValueError(f"Cycle found in basin spill graph at basin {current_id}.")
            visited.add(current_id)
            current_id = by_id[current_id].downstream_basin_id
        if current_id:
            routed_area[current_id] += basin.incremental_area_m2
        else:
            external_area_m2 += basin.incremental_area_m2

    selected: list[Basin] = []
    for basin in basins:
        if basin.basin_id not in retained:
            continue
        downstream_id = basin.downstream_basin_id
        visited = {basin.basin_id}
        while downstream_id and downstream_id not in retained:
            if downstream_id in visited:
                raise ValueError(f"Cycle found in basin spill graph at basin {downstream_id}.")
            visited.add(downstream_id)
            downstream_id = by_id[downstream_id].downstream_basin_id
        selected.append(
            replace(
                basin,
                incremental_area_m2=routed_area[basin.basin_id],
                downstream_basin_id=downstream_id,
            )
        )

    expected_area = sum(basin.incremental_area_m2 for basin in basins)
    represented_area = sum(basin.incremental_area_m2 for basin in selected)
    if not math.isclose(
        expected_area, represented_area + external_area_m2, rel_tol=1e-10, abs_tol=1e-6
    ):
        raise RuntimeError("Storage-basin filtering failed to conserve incremental runoff area.")
    return selected, external_area_m2


def extract_basin_elevations(
    dem_path: Path, basin_ids_path: Path, basin_ids: set[int]
) -> tuple[dict[int, np.ndarray], float]:
    """Read basin-cell elevations blockwise and verify the rasters align."""
    if not dem_path.is_file() or not basin_ids_path.is_file():
        raise FileNotFoundError("The configured DEM and basin-ID rasters must both exist.")

    elevations: dict[int, list[np.ndarray]] = {basin_id: [] for basin_id in basin_ids}
    with rasterio.open(dem_path) as dem, rasterio.open(basin_ids_path) as labels:
        if (
            dem.width != labels.width
            or dem.height != labels.height
            or dem.crs != labels.crs
            or dem.transform != labels.transform
        ):
            raise ValueError("The DEM and basin-ID rasters must have identical grids and CRS.")
        cell_area_m2 = abs(dem.transform.a * dem.transform.e)
        if not math.isfinite(cell_area_m2) or cell_area_m2 <= 0:
            raise ValueError("Could not determine a positive raster cell area.")

        for _, window in dem.block_windows(1):
            heights = dem.read(1, window=window, masked=True).filled(np.nan)
            labels_block = labels.read(1, window=window)
            relevant = np.isin(labels_block, tuple(basin_ids))
            if not relevant.any():
                continue
            ids_in_block = np.unique(labels_block[relevant])
            for basin_id_value in ids_in_block:
                basin_id = int(basin_id_value)
                selected = (labels_block == basin_id) & np.isfinite(heights)
                if selected.any():
                    elevations[basin_id].append(heights[selected].astype(np.float64))

    combined: dict[int, np.ndarray] = {}
    missing_ids: list[int] = []
    for basin_id, blocks in elevations.items():
        if not blocks:
            missing_ids.append(basin_id)
        else:
            combined[basin_id] = np.concatenate(blocks)
    if missing_ids:
        raise ValueError(
            "No DEM cells were found for basin IDs: "
            + ", ".join(str(value) for value in missing_ids[:20])
        )
    return combined, cell_area_m2


def build_storage_geometries(
    basins: list[Basin], elevations: dict[int, np.ndarray], cell_area_m2: float
) -> dict[int, BasinGeometry]:
    """Derive sampled SWMM storage curves from basin-cell DEM elevations."""
    geometries: dict[int, BasinGeometry] = {}
    for basin in basins:
        cell_heights = np.sort(elevations[basin.basin_id])
        floor = float(cell_heights[0])
        max_depth = basin.spill_elevation_m - floor
        if not math.isfinite(max_depth):
            raise ValueError(f"Basin {basin.basin_id} has an invalid DEM-derived depth.")
        # SWMM requires positive storage depth; preserve zero-capacity basins as 1 mm buffers.
        max_depth = max(max_depth, MINIMUM_STORAGE_DEPTH_M)
        depths = np.linspace(0.0, max_depth, CURVE_POINTS)
        areas = np.searchsorted(cell_heights, floor + depths, side="right") * cell_area_m2
        curve = tuple(
            (float(depth), float(area))
            for depth, area in zip(depths, areas, strict=True)
        )
        capacity = float(np.sum((areas[:-1] + areas[1:]) * np.diff(depths) * 0.5))
        geometries[basin.basin_id] = BasinGeometry(
            floor_elevation_m=floor,
            maximum_depth_m=max_depth,
            curve=curve,
            raster_cell_count=int(cell_heights.size),
            curve_capacity_m3=capacity,
        )
    return geometries


def _format_date_time(value: datetime) -> tuple[str, str]:
    """Format date and time as expected by SWMM input sections."""
    return value.strftime("%m/%d/%Y"), value.strftime("%H:%M")


def render_swmm_input(
    basins: list[Basin],
    geometries: dict[int, BasinGeometry],
    rainfall_mm: float,
    duration_hours: float,
    drain_hours: float,
    spill_channel_width_m: float,
    spill_channel_roughness: float,
    report_interval_minutes: int,
    external_runoff_area_m2: float,
) -> str:
    """Build a reproducible, self-contained conceptual SWMM model input."""
    interval = RAINFALL_INTERVAL_MINUTES
    interval_count = round(duration_hours * 60 / interval)
    if not math.isclose(interval_count * interval, duration_hours * 60, abs_tol=1e-8):
        raise ValueError(f"Storm duration must be a multiple of {interval} minutes.")
    end = SIMULATION_START + timedelta(hours=duration_hours + drain_hours)
    end_date, end_time = _format_date_time(end)
    start_date, start_time = _format_date_time(SIMULATION_START)
    sections: list[str] = [
        "[TITLE]",
        "Enkoping DEM-derived conceptual storage network - screening only",
        "Uniform effective rainfall; no sewer network or design hyetograph supplied.",
        "",
        "[OPTIONS]",
        "FLOW_UNITS CMS",
        "INFILTRATION HORTON",
        "FLOW_ROUTING DYNWAVE",
        "LINK_OFFSETS DEPTH",
        "ALLOW_PONDING YES",
        f"START_DATE {start_date}",
        f"START_TIME {start_time}:00",
        f"REPORT_START_DATE {start_date}",
        f"REPORT_START_TIME {start_time}:00",
        f"END_DATE {end_date}",
        f"END_TIME {end_time}:00",
        f"WET_STEP 00:{interval:02d}:00",
        "DRY_STEP 01:00:00",
        "ROUTING_STEP 00:00:15",
        f"REPORT_STEP 00:{report_interval_minutes:02d}:00",
        "",
        "[EVAPORATION]",
        "CONSTANT 0",
        "DRY_ONLY NO",
        "",
        "[RAINGAGES]",
        ";;Name  Format     Interval  SCF  Source",
        f"RG1     VOLUME     0:{interval:02d}      1.0  TIMESERIES TS1",
        "",
        "[TIMESERIES]",
        ";;Name  Date  Time  Value (mm per interval)",
    ]
    rainfall_per_interval_mm = rainfall_mm / interval_count
    for step in range(1, interval_count + 1):
        timestamp = SIMULATION_START + timedelta(minutes=step * interval)
        date_value, time_value = _format_date_time(timestamp)
        sections.append(f"TS1 {date_value} {time_value} {rainfall_per_interval_mm:.10g}")
    after_storm = SIMULATION_START + timedelta(minutes=(interval_count + 1) * interval)
    date_value, time_value = _format_date_time(after_storm)
    sections.append(f"TS1 {date_value} {time_value} 0")

    sections.extend(["", "[SUBCATCHMENTS]", ";;Name  RainGage  Outlet  Area(ha)  %Imperv  Width  %Slope  CurbLen  SnowPack"])
    for basin in basins:
        if basin.incremental_area_m2 <= 0:
            continue
        area_ha = basin.incremental_area_m2 / 10_000.0
        width_m = max(math.sqrt(basin.incremental_area_m2), 1.0)
        sections.append(
            f"{basin.subcatchment_id} RG1 {basin.node_id} {area_ha:.10g} "
            f"100 {width_m:.6g} 0.5 0"
        )
    if external_runoff_area_m2 > 0:
        external_width_m = max(math.sqrt(external_runoff_area_m2), 1.0)
        sections.append(
            f"C00000 RG1 OF1 {external_runoff_area_m2 / 10_000.0:.10g} "
            f"100 {external_width_m:.6g} 0.5 0"
        )
    sections.extend(
        ["", "[SUBAREAS]", ";;Name  N-Imperv  N-Perv  S-Imperv  S-Perv  PctZero  RouteTo  PctRouted"]
    )
    for basin in basins:
        if basin.incremental_area_m2 > 0:
            sections.append(f"{basin.subcatchment_id} 0.015 0.1 0 0 100 OUTLET 0")
    if external_runoff_area_m2 > 0:
        sections.append("C00000 0.015 0.1 0 0 100 OUTLET 0")
    sections.extend(["", "[STORAGE]", ";;Name  Elev  MaxDepth  InitDepth  Shape  Curve  A1  A2  A0  Fevap  Psi  Ksat  IMD"])
    for basin in basins:
        geometry = geometries[basin.basin_id]
        sections.append(
            f"{basin.node_id} {geometry.floor_elevation_m:.6f} "
            f"{geometry.maximum_depth_m + 0.1:.6f} 0 TABULAR {basin.curve_id} 0 0 0 0 0 0 0"
        )
    sections.extend(["", "[CURVES]", ";;Name  Type  Depth(m)  Area(m2)"])
    for basin in basins:
        sections.append(f"{basin.curve_id} STORAGE")
        for depth, area in geometries[basin.basin_id].curve:
            sections.append(f"{basin.curve_id} {depth:.8g} {area:.8g}")

    sections.extend(
        ["", "[CONDUITS]", ";;Name  FromNode  ToNode  Length  Roughness  InOffset  OutOffset  InitFlow  MaxFlow"]
    )
    basin_by_id = {basin.basin_id: basin for basin in basins}
    external_basins = [basin for basin in basins if basin.downstream_basin_id == 0]
    for basin in basins:
        downstream = (
            basin_by_id[basin.downstream_basin_id].node_id
            if basin.downstream_basin_id
            else f"O{basin.basin_id:05d}"
        )
        if basin.downstream_basin_id:
            receiver = basin_by_id[basin.downstream_basin_id]
            if all(value is not None for value in (basin.outlet_x, basin.outlet_y, receiver.outlet_x, receiver.outlet_y)):
                channel_length_m = math.hypot(
                    receiver.outlet_x - basin.outlet_x,
                    receiver.outlet_y - basin.outlet_y,
                )
            else:
                channel_length_m = 10.0
        else:
            channel_length_m = 10.0
        channel_length_m = max(channel_length_m, 1.0)
        crest_depth = geometries[basin.basin_id].maximum_depth_m
        sections.append(
            f"W{basin.basin_id:05d} {basin.node_id} {downstream} "
            f"{channel_length_m:.6g} {spill_channel_roughness:.6g} "
            f"{crest_depth:.6f} 0 0 0"
        )
    sections.extend(["", "[XSECTIONS]", ";;Link  Shape  Geom1  Geom2  Geom3  Geom4  Barrels  Culvert"])
    for basin in basins:
        channel_height_m = max(geometries[basin.basin_id].maximum_depth_m + 0.1, 0.2)
        sections.append(
            f"W{basin.basin_id:05d} RECT_OPEN {channel_height_m:.6f} "
            f"{spill_channel_width_m:.6g} 0 0 1"
        )
    sections.extend(["", "[OUTFALLS]", ";;Name  Elevation  Type  StageData  Gated"])
    if external_runoff_area_m2 > 0:
        outfall_elevation = min(geometry.floor_elevation_m for geometry in geometries.values())
        sections.append(f"OF1 {outfall_elevation:.6f} FREE NO")
    for basin in external_basins:
        sections.append(f"O{basin.basin_id:05d} {basin.spill_elevation_m - 0.01:.6f} FREE NO")
    sections.extend(["", "[COORDINATES]", ";;Node  X  Y"])
    for basin in basins:
        if basin.outlet_x is not None and basin.outlet_y is not None:
            sections.append(f"{basin.node_id} {basin.outlet_x:.3f} {basin.outlet_y:.3f}")
    sections.extend(["", "[REPORT]", "INPUT NO", "CONTROLS NO", "SUBCATCHMENTS NONE", "NODES NONE", "LINKS NONE", ""])
    return "\n".join(sections)


def run_model(
    input_path: Path,
    report_path: Path,
    output_path: Path,
    basins: list[Basin],
    report_interval_minutes: int,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Run SWMM through PySWMM and export node, link and basin summaries."""
    try:
        from pyswmm import Links, Nodes, Simulation
    except ImportError as exc:
        raise RuntimeError(
            "PySWMM is required. Install the optional dependency with "
            "pip install -e '.[swmm]'."
        ) from exc

    node_series_path = output_dir / "node_timeseries.csv"
    link_series_path = output_dir / "spill_link_timeseries.csv"
    basin_summary_path = output_dir / "pyswmm_basin_summary.csv"
    maxima: dict[str, dict[str, float]] = {
        basin.node_id: {
            "max_depth_m": 0.0,
            "max_water_level_m": -math.inf,
            "max_volume_m3": 0.0,
            "max_flooding_m3s": 0.0,
            "max_outflow_m3s": 0.0,
        }
        for basin in basins
    }

    with (
        node_series_path.open("w", newline="", encoding="utf-8") as node_file,
        link_series_path.open("w", newline="", encoding="utf-8") as link_file,
    ):
        node_writer = csv.DictWriter(
            node_file,
            fieldnames=["datetime", "basin_id", "node_id", "water_level_m", "depth_m", "stored_volume_m3", "lateral_inflow_m3s", "flooding_rate_m3s"],
        )
        link_writer = csv.DictWriter(
            link_file,
            fieldnames=["datetime", "basin_id", "link_id", "flow_m3s", "depth_m"],
        )
        node_writer.writeheader()
        link_writer.writeheader()
        with Simulation(str(input_path), str(report_path), str(output_path)) as sim:
            sim.step_advance(report_interval_minutes * 60)
            nodes = Nodes(sim)
            links = Links(sim)
            for _ in sim:
                timestamp = sim.current_time.isoformat(sep=" ")
                for basin in basins:
                    node = nodes[basin.node_id]
                    depth = float(node.depth)
                    volume = float(node.volume)
                    flooding = float(node.flooding)
                    inflow = float(node.lateral_inflow)
                    stats = maxima[basin.node_id]
                    stats["max_depth_m"] = max(stats["max_depth_m"], depth)
                    stats["max_water_level_m"] = max(stats["max_water_level_m"], float(node.invert_elevation) + depth)
                    stats["max_volume_m3"] = max(stats["max_volume_m3"], volume)
                    stats["max_flooding_m3s"] = max(stats["max_flooding_m3s"], flooding)
                    node_writer.writerow({
                        "datetime": timestamp,
                        "basin_id": basin.basin_id,
                        "node_id": basin.node_id,
                        "water_level_m": float(node.invert_elevation) + depth,
                        "depth_m": depth,
                        "stored_volume_m3": volume,
                        "lateral_inflow_m3s": inflow,
                        "flooding_rate_m3s": flooding,
                    })
                for basin in basins:
                    link = links[f"W{basin.basin_id:05d}"]
                    flow = float(link.flow)
                    maxima[basin.node_id]["max_outflow_m3s"] = max(maxima[basin.node_id]["max_outflow_m3s"], flow)
                    link_writer.writerow({
                        "datetime": timestamp,
                        "basin_id": basin.basin_id,
                        "link_id": f"W{basin.basin_id:05d}",
                        "flow_m3s": flow,
                        "depth_m": float(link.depth),
                    })
    with basin_summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["basin_id", "source_status", "modeled_destination", "downstream_basin_id", "outlet_x", "outlet_y", "max_water_level_m", "max_depth_m", "max_stored_volume_m3", "max_spill_flow_m3s", "max_flooding_rate_m3s"],
        )
        writer.writeheader()
        for basin in basins:
            stats = maxima[basin.node_id]
            writer.writerow({
                "basin_id": basin.basin_id,
                "source_status": basin.status,
                "modeled_destination": "modeled_basin" if basin.downstream_basin_id else "boundary_outfall",
                "downstream_basin_id": basin.downstream_basin_id,
                "outlet_x": basin.outlet_x,
                "outlet_y": basin.outlet_y,
                "max_water_level_m": f"{stats['max_water_level_m']:.6g}",
                "max_depth_m": f"{stats['max_depth_m']:.6g}",
                "max_stored_volume_m3": f"{stats['max_volume_m3']:.6g}",
                "max_spill_flow_m3s": f"{stats['max_outflow_m3s']:.6g}",
                "max_flooding_rate_m3s": f"{stats['max_flooding_m3s']:.6g}",
            })
    _write_run_metadata(output_dir, report_path)
    _write_peak_geopackage(output_dir, basin_summary_path, basins)
    return node_series_path, link_series_path, basin_summary_path


def _write_run_metadata(output_dir: Path, report_path: Path) -> None:
    """Extract SWMM report continuity percentages for quality checks."""
    report_lines = report_path.read_text(encoding="utf-8", errors="replace").splitlines()

    def continuity_error(section_heading: str) -> float:
        try:
            start = next(index for index, line in enumerate(report_lines) if section_heading in line)
        except StopIteration as exc:
            raise ValueError(f"Could not find {section_heading!r} in SWMM report.") from exc
        for line in report_lines[start : start + 30]:
            if "Continuity Error (%)" in line:
                return float(line.split()[-1])
        raise ValueError(f"Could not parse continuity error after {section_heading!r}.")

    runoff_error = continuity_error("Runoff Quantity Continuity")
    routing_error = continuity_error("Flow Routing Continuity")
    path = output_dir / "swmm_continuity.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["metric", "percent"])
        writer.writerow(["runoff_continuity_error", runoff_error])
        writer.writerow(["flow_routing_continuity_error", routing_error])


def _write_peak_geopackage(
    output_dir: Path, basin_summary_path: Path, basins: list[Basin]
) -> Path | None:
    """Write CRS-aware basin outlet points with modeled peak metrics."""
    georeferenced = [basin for basin in basins if basin.outlet_x is not None and basin.outlet_y is not None]
    crs_values = {basin.crs for basin in georeferenced if basin.crs}
    if not georeferenced or len(crs_values) != 1:
        return None
    with basin_summary_path.open(newline="", encoding="utf-8") as source:
        summary_rows = {int(row["basin_id"]): row for row in csv.DictReader(source)}
    rows = [summary_rows[basin.basin_id] for basin in georeferenced]
    frame = gpd.GeoDataFrame(
        rows,
        geometry=gpd.points_from_xy(
            [basin.outlet_x for basin in georeferenced],
            [basin.outlet_y for basin in georeferenced],
        ),
        crs=next(iter(crs_values)),
    )
    path = output_dir / "pyswmm_basin_peaks.gpkg"
    frame.to_file(path, layer="basin_peaks", driver="GPKG", index=False)
    return path


def write_model_notes(
    output_dir: Path,
    basins: list[Basin],
    geometries: dict[int, BasinGeometry],
    rainfall_mm: float,
    duration_hours: float,
    report_interval_minutes: int,
    cell_area_m2: float,
    source_basin_count: int,
    minimum_storage_basin_area_m2: float,
    external_runoff_area_m2: float,
) -> None:
    """Record assumptions, provenance and raster-to-curve capacity checks."""
    path = output_dir / "model_notes.txt"
    curve_capacity = sum(item.curve_capacity_m3 for item in geometries.values())
    source_capacity = sum(basin.source_capacity_m3 for basin in basins)
    capacity_ratio = curve_capacity / source_capacity if source_capacity else math.nan
    with path.open("w", encoding="utf-8") as output:
        output.write("Enköping DEM-informed conceptual PySWMM scenario\n")
        output.write("=================================================\n\n")
        output.write("Interpretation: screening scenario; not a calibrated flood prediction.\n")
        output.write("Network: DEM-derived depressions and D8 basin spill graph only; no surveyed sewer assets.\n")
        output.write("Rainfall: uniform net/effective depth; no additional infiltration loss applied.\n")
        output.write(f"Effective rainfall: {rainfall_mm:.6g} mm over {duration_hours:g} h.\n")
        output.write("Temporal pattern: constant intensity; source data has no hyetograph.\n")
        output.write("Runoff: incremental D8 area assigned to its corresponding basin as 100% impervious.\n")
        output.write("Overland routing: SWMM width=sqrt(area), slope=0.5%, impervious n=0.015.\n")
        output.write("Storage curves: 31 DEM-sampled stage-area ordinates per basin, truncated at spill elevation.\n")
        output.write("Zero-capacity basins use a 1 mm minimum storage depth for SWMM numerical validity.\n")
        output.write("Spillways: conceptual rectangular open channels at basin sills; channel dimensions and roughness are assumptions.\n")
        output.write(f"Sample interval: {report_interval_minutes} minutes.\n")
        output.write(f"DEM cell area: {cell_area_m2:g} m². Basin count: {len(basins)}.\n")
        output.write(f"Source basin count: {source_basin_count}; modeled storage nodes: {len(basins)}.\n")
        output.write(f"Storage-area cutoff: {minimum_storage_basin_area_m2:g} m².\n")
        output.write(f"Incremental runoff area routed directly to boundary outfall: {external_runoff_area_m2:.6g} m².\n")
        output.write(f"Sum of source basin capacities: {source_capacity:.6g} m³.\n")
        output.write(f"Sum of sampled-curve capacities: {curve_capacity:.6g} m³.\n")
        output.write(f"Sampled/source capacity ratio: {capacity_ratio:.6g}.\n")
        output.write("Check the ratio above: sampled curves are discretized from the DEM and may differ from the CSV capacity.\n")
        output.write("All basin coordinates are inherited from the source CSV CRS (EPSG:5845); outputs are tabular.\n")
