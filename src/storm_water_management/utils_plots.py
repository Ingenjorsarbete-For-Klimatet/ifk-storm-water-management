"""Plot Enköping terrain/rainfall rasters and conceptual PySWMM results."""

# ruff: noqa: W191

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from rasterio.enums import Resampling

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPOSITORY_ROOT / "results" / "enkoping"
DEPTH_LIMITS_M = (0.1, 0.2, 0.5, 1.0, float("inf"))
DEPTH_LABELS = ("0.1–0.2 m", "0.2–0.5 m", "0.5–1 m", ">1 m")
DEPTH_COLORS = ("#440154", "#31688e", "#35b779", "#fde725")
DEPTH_CLASS_CMAP = ListedColormap(DEPTH_COLORS)
DEPTH_CLASS_CMAP.set_bad((0, 0, 0, 0))
DEPTH_CLASS_NORM = BoundaryNorm(
	np.arange(len(DEPTH_LABELS) + 1) - 0.5, DEPTH_CLASS_CMAP.N
)


def _depth_classes(values: np.ndarray | np.ma.MaskedArray) -> np.ma.MaskedArray:
	masked = np.ma.asarray(values)
	data = np.asarray(masked.data)
	mask = np.ma.getmaskarray(masked) | ~np.isfinite(data) | (data < 0.1)
	return np.ma.array(np.digitize(data, (0.2, 0.5, 1.0)), mask=mask)


def _display_raster(
	source: rasterio.io.DatasetReader,
	max_pixels: int,
	window: rasterio.windows.Window | None = None,
	resampling: Resampling = Resampling.average,
) -> np.ma.MaskedArray:
	if window is None:
		window = rasterio.windows.Window(0, 0, source.width, source.height)
	scale = min(1.0, max_pixels / max(window.width, window.height))
	shape = (max(1, round(window.height * scale)), max(1, round(window.width * scale)))
	return np.ma.masked_invalid(
		source.read(1, window=window, out_shape=shape, resampling=resampling, masked=True)
	)


def create_terrain_plot(
	raster_path: Path,
	elevation_path: Path,
	output_path: Path,
	max_display_pixels: int = 1600,
	runoff_volume_path: Path | None = None,
	ponding_depth_path: Path | None = None,
	flow_accumulation_path: Path | None = None,
) -> None:
	"""Plot depression depth, elevation, and optional runoff/routing rasters."""
	if not raster_path.is_file():
		raise FileNotFoundError(f"Enköping result raster not found: {raster_path}")
	if not elevation_path.is_file():
		raise FileNotFoundError(f"Enköping elevation raster not found: {elevation_path}")
	with rasterio.open(raster_path) as depth_src:
		if depth_src.crs is None:
			raise ValueError("The result raster has no CRS.")
		depth = _display_raster(depth_src, max_display_pixels)
		bounds, crs, transform = depth_src.bounds, depth_src.crs, depth_src.transform
		raster_width, raster_height = depth_src.width, depth_src.height
		valid = int(depth.count())
		extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
		valid_values = depth.compressed()

	if not valid:
		raise ValueError("The result raster contains no valid depth pixels.")
	with rasterio.open(elevation_path) as elevation_src:
		if elevation_src.crs != crs:
			raise ValueError("Elevation and result rasters must use the same CRS.")
		window = rasterio.windows.from_bounds(
			bounds.left, bounds.bottom, bounds.right, bounds.top,
			transform=elevation_src.transform,
		)
		elevation = _display_raster(elevation_src, max_display_pixels, window)
		eb = rasterio.windows.bounds(window, elevation_src.transform)
		elevation_extent = (eb[0], eb[2], eb[1], eb[3])

	if ponding_depth_path is None and runoff_volume_path is None:
		candidate = raster_path.with_name(
			raster_path.name.replace("depression_depth", "estimated_ponding_depth", 1)
		)
		if candidate.is_file():
			ponding_depth_path = candidate
		else:
			candidate = raster_path.with_name(
				raster_path.name.replace("depression_depth", "runoff_volume", 1)
			)
			if candidate.is_file():
				runoff_volume_path = candidate
	if flow_accumulation_path is None:
		candidate = raster_path.with_name(
			raster_path.name.replace("depression_depth", "contributing_area", 1)
		)
		if candidate.is_file():
			flow_accumulation_path = candidate

	scenario_path = ponding_depth_path or runoff_volume_path
	scenario_depth = ponding_depth_path is not None
	figure, axes = plt.subplots(1, 3, figsize=(17, 7), constrained_layout=True)
	image = axes[0].imshow(
		_depth_classes(depth), extent=extent, origin="upper", cmap=DEPTH_CLASS_CMAP,
		norm=DEPTH_CLASS_NORM, interpolation="nearest",
	)
	axes[0].set_title("Terrain depression-fill depth (≥ 0.1 m)")
	colorbar = figure.colorbar(image, ax=axes[0], ticks=np.arange(4), label="Depth class (m)")
	colorbar.ax.set_yticklabels(DEPTH_LABELS)

	elevations = elevation.compressed()
	if not elevations.size:
		raise ValueError("No valid elevation pixels overlap the result raster.")
	low, high = np.percentile(elevations, (2, 98))
	elevation_image = axes[1].imshow(
		elevation, extent=elevation_extent, origin="upper", cmap="terrain",
		norm=Normalize(float(low), float(high) if high > low else float(low) + 1),
	)
	axes[1].set_title("Ground elevation (2nd–98th percentile scale)")
	figure.colorbar(elevation_image, ax=axes[1], label="Elevation (m)")

	scenario_max = 0.0
	positive_cells = 0
	if scenario_path is not None:
		if not scenario_path.is_file():
			raise FileNotFoundError(f"Scenario raster not found: {scenario_path}")
		with rasterio.open(scenario_path) as scenario_src:
			if (
				scenario_src.crs != crs or scenario_src.transform != transform
				or scenario_src.width != raster_width
				or scenario_src.height != raster_height
			):
				raise ValueError("Scenario raster must align with the depression raster.")
			scenario = _display_raster(scenario_src, max_display_pixels)
			scenario = np.ma.masked_less_equal(scenario, 0)
			scenario_max = float(scenario.max()) if scenario.count() else 0.0
			for _, block_window in scenario_src.block_windows(1):
				block = scenario_src.read(1, window=block_window, masked=True)
				block_values = np.asarray(block.data)
				positive_cells += int(
					((~np.ma.getmaskarray(block)) & np.isfinite(block_values)
					 & (block_values > 0)).sum()
				)
		if scenario_depth:
			scenario_image = axes[2].imshow(
				_depth_classes(scenario), extent=extent, origin="upper",
				cmap=DEPTH_CLASS_CMAP, norm=DEPTH_CLASS_NORM,
			)
			scenario_colorbar = figure.colorbar(
				scenario_image, ax=axes[2], ticks=np.arange(4)
			)
			scenario_colorbar.ax.set_yticklabels(DEPTH_LABELS)
			axes[2].set_title("Estimated rainfall ponding depth")
		else:
			values = scenario.compressed()
			if values.size:
				minimum = max(float(values.min()), np.finfo(float).tiny)
				maximum = max(float(np.percentile(values, 99.5)), minimum * 1.01)
				scenario_image = axes[2].imshow(
					scenario, extent=extent, origin="upper", cmap="magma",
					norm=LogNorm(minimum, maximum, clip=True),
				)
				figure.colorbar(scenario_image, ax=axes[2], label="Runoff volume (m³, log)")
			axes[2].set_title("Accumulated runoff volume")
	elif flow_accumulation_path is not None:
		with rasterio.open(flow_accumulation_path) as flow_src:
			if (
				flow_src.crs != crs or flow_src.transform != transform
				or flow_src.width != raster_width or flow_src.height != raster_height
			):
				raise ValueError("Flow-accumulation raster must align with the depression raster.")
			flow = np.ma.masked_less_equal(_display_raster(flow_src, max_display_pixels), 0)
		values = flow.compressed()
		if values.size:
			flow_image = axes[2].imshow(
				flow, extent=extent, origin="upper", cmap="viridis",
				norm=LogNorm(max(float(values.min()), np.finfo(float).tiny),
							 max(float(values.max()), float(values.min()) * 1.01)),
			)
			figure.colorbar(flow_image, ax=axes[2], label="Contributing area (m², log)")
		axes[2].set_title("Flow accumulation / contributing area")
	else:
		axes[2].set_visible(False)

	for axis in axes[:2] if scenario_path is None and flow_accumulation_path is None else axes:
		if axis.get_visible():
			axis.set_xlabel("Easting (m)")
			axis.set_ylabel("Northing (m)")
			axis.set_aspect("equal", adjustable="box")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.suptitle(f"Enköping terrain results — {crs}", fontsize=15, fontweight="bold")
	figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
	plt.close(figure)
	threshold_cells = int((valid_values >= DEPTH_LIMITS_M[0]).sum())
	print(f"Saved plot: {output_path}")
	print(
		f"Analyzed {valid:,} displayed pixels; {threshold_cells:,} pixels are "
		f"at least {DEPTH_LIMITS_M[0]:.1f} m deep; maximum depth "
		f"{float(valid_values.max()):.3f} m."
	)
	if scenario_path is not None:
		print(f"{positive_cells:,} cells have a positive scenario value; maximum {scenario_max:.3f}.")


def _read_dem_backdrop(
	path: Path,
	crs: object,
	bounds: tuple[float, float, float, float],
	max_pixels: int,
) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float]] | None:
	"""Read a downsampled DEM map subset when the optional raster exists."""
	if not path.is_file():
		return None
	with rasterio.open(path) as source:
		if source.crs != crs:
			raise ValueError(f"DEM CRS ({source.crs}) does not match basin CRS ({crs}).")
		left, bottom = max(bounds[0], source.bounds.left), max(bounds[1], source.bounds.bottom)
		right, top = min(bounds[2], source.bounds.right), min(bounds[3], source.bounds.top)
		if left >= right or bottom >= top:
			return None
		window = rasterio.windows.from_bounds(
			left, bottom, right, top, transform=source.transform
		).round_offsets().round_lengths()
		image = _display_raster(source, max_pixels, window)
		transform = source.window_transform(window)
		extent = (
			transform.c,
			transform.c + transform.a * window.width,
			transform.f + transform.e * window.height,
			transform.f,
		)
	return image, extent


def _read_basin_depth_layer(
	path: Path,
	crs: object,
	bounds: tuple[float, float, float, float],
	basin_depths: dict[int, float],
	max_pixels: int,
) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float]] | None:
	"""Rasterize modeled basin depths into the corresponding basin-ID cells."""
	if not path.is_file():
		return None
	with rasterio.open(path) as source:
		if source.crs != crs:
			raise ValueError(f"Basin-ID CRS ({source.crs}) does not match basin CRS ({crs}).")
		left, bottom = max(bounds[0], source.bounds.left), max(bounds[1], source.bounds.bottom)
		right, top = min(bounds[2], source.bounds.right), min(bounds[3], source.bounds.top)
		if left >= right or bottom >= top:
			return None
		window = rasterio.windows.from_bounds(
			left, bottom, right, top, transform=source.transform
		).round_offsets().round_lengths()
		window = window.intersection(rasterio.windows.Window(0, 0, source.width, source.height))
		labels = source.read(
			1,
			window=window,
			out_shape=(
				max(1, round(window.height * min(1, max_pixels / max(window.width, window.height)))),
				max(1, round(window.width * min(1, max_pixels / max(window.width, window.height)))),
			),
			resampling=Resampling.mode,
		)
		transform = source.window_transform(window)
		extent = (
			transform.c,
			transform.c + transform.a * window.width,
			transform.f + transform.e * window.height,
			transform.f,
		)
	lookup = np.full(max(max(basin_depths, default=0), int(labels.max())) + 1, np.nan)
	for basin_id, depth in basin_depths.items():
		if 0 <= basin_id < lookup.size:
			lookup[basin_id] = depth
	return np.ma.masked_invalid(lookup[labels]), extent


def create_pyswmm_plot(
	peaks_path: Path,
	node_path: Path,
	link_path: Path,
	summary_path: Path,
	basin_ids_path: Path,
	dem_path: Path,
	output_path: Path,
	max_display_pixels: int = 1400,
	minimum_flow_m3s: float = 0.005,
) -> None:
	"""Plot PySWMM basin peaks, spill links, aggregate hydrographs and storage."""
	if not peaks_path.is_file():
		raise FileNotFoundError(f"PySWMM peak GeoPackage not found: {peaks_path}")
	for path in (node_path, link_path, summary_path):
		if not path.is_file():
			raise FileNotFoundError(f"PySWMM result file not found: {path}")
	peaks = gpd.read_file(peaks_path)
	nodes = pd.read_csv(node_path, parse_dates=["datetime"])
	links = pd.read_csv(link_path, parse_dates=["datetime"])
	summary = pd.read_csv(summary_path)
	if peaks.empty or peaks.crs is None:
		raise ValueError("Peak GeoPackage must have basin records and a CRS.")
	required = {"basin_id", "max_depth_m", "max_spill_flow_m3s", "geometry"}
	if missing := required.difference(peaks.columns):
		raise ValueError(f"Peak data lacks columns: {', '.join(sorted(missing))}")
	if not {"datetime", "stored_volume_m3", "flooding_rate_m3s"}.issubset(nodes):
		raise ValueError("Node time series is missing required columns.")
	if not {"datetime", "basin_id", "flow_m3s"}.issubset(links):
		raise ValueError("Link time series is missing required columns.")
	if not {"basin_id", "modeled_destination"}.issubset(summary):
		raise ValueError("Basin summary is missing required columns.")
	totals = nodes.groupby("datetime", sort=True).agg(
		stored=("stored_volume_m3", "sum"), flooding=("flooding_rate_m3s", "sum")
	)
	boundary = summary.loc[summary.modeled_destination == "boundary_outfall", "basin_id"]
	outflow = links.loc[links.basin_id.isin(boundary)].groupby("datetime").flow_m3s.sum()
	totals["outflow"] = outflow.reindex(totals.index, fill_value=0)

	bounds = peaks.total_bounds
	margin_x = max((bounds[2] - bounds[0]) * 0.04, 1.0)
	margin_y = max((bounds[3] - bounds[1]) * 0.04, 1.0)
	map_bounds = (bounds[0] - margin_x, bounds[1] - margin_y,
				  bounds[2] + margin_x, bounds[3] + margin_y)
	dem_layer = _read_dem_backdrop(dem_path, peaks.crs, map_bounds, max_display_pixels)
	basin_depths = dict(zip(
		peaks.basin_id.astype(int), peaks.max_depth_m.astype(float), strict=True
	))
	depth_layer = _read_basin_depth_layer(
		basin_ids_path, peaks.crs, map_bounds, basin_depths, max_display_pixels
	)
	fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
	depth_ax, flow_ax, hydro_ax, storage_ax = axes.ravel()
	depths = pd.to_numeric(peaks.max_depth_m, errors="coerce").to_numpy()
	if depth_layer is not None:
		basin_image, basin_extent = depth_layer
		points = depth_ax.imshow(
			_depth_classes(basin_image), extent=basin_extent, origin="upper",
			cmap=DEPTH_CLASS_CMAP, norm=DEPTH_CLASS_NORM, interpolation="nearest",
		)
	else:
		points = depth_ax.scatter(
			peaks.geometry.x, peaks.geometry.y, c=_depth_classes(depths), s=18,
			cmap=DEPTH_CLASS_CMAP, norm=DEPTH_CLASS_NORM, edgecolors="none",
		)
	depth_ax.set_title("Maximum modeled basin storage depth")
	bar = fig.colorbar(points, ax=depth_ax, ticks=np.arange(4))
	bar.ax.set_yticklabels(DEPTH_LABELS)
	bar.set_label("Peak water depth (m)")

	flows = pd.to_numeric(peaks.max_spill_flow_m3s, errors="coerce").fillna(0)
	active = peaks.loc[(flows >= minimum_flow_m3s) & (flows > 0)].copy()
	downstream = pd.to_numeric(peaks.get("downstream_basin_id", 0), errors="coerce")
	if not isinstance(downstream, pd.Series):
		downstream = pd.Series(0, index=peaks.index)
	basin_positions = {
		int(row.basin_id): (float(row.outlet_x), float(row.outlet_y))
		for row in peaks.itertuples()
		if hasattr(row, "outlet_x") and hasattr(row, "outlet_y")
		and pd.notna(row.outlet_x) and pd.notna(row.outlet_y)
	}
	if dem_layer is not None:
		dem_image, dem_extent = dem_layer
		flow_ax.imshow(dem_image, extent=dem_extent, origin="upper", cmap="Greys", alpha=.45)
	flow_ax.set_xlim(map_bounds[0], map_bounds[2])
	flow_ax.set_ylim(map_bounds[1], map_bounds[3])
	flow_ax.scatter(peaks.geometry.x, peaks.geometry.y, s=8, color="#777777", alpha=.5)
	if not active.empty:
		flow_min = max(float(flows.loc[active.index].min()), np.finfo(float).tiny)
		flow_max = max(float(flows.loc[active.index].max()), flow_min * 1.001)
		flow_norm = LogNorm(flow_min, flow_max)
		flow_cmap = plt.get_cmap("plasma")
		for row in active.itertuples():
			source_id = int(row.basin_id)
			destination_id = int(downstream.loc[row.Index]) if pd.notna(downstream.loc[row.Index]) else 0
			if source_id not in basin_positions:
				continue
			start = basin_positions[source_id]
			flow = float(flows.loc[row.Index])
			color = flow_cmap(flow_norm(flow))
			if destination_id in basin_positions:
				end = basin_positions[destination_id]
				if start != end:
					flow_ax.annotate(
						"", xy=end, xytext=start,
						arrowprops={"arrowstyle": "-|>", "color": color,
									"lw": 1.2, "connectionstyle": "arc3,rad=0.04"},
					)
			else:
				flow_ax.scatter([start[0]], [start[1]], marker="s", s=35,
								color=[color], edgecolors="white", linewidths=.4)
		fig.colorbar(
			matplotlib.cm.ScalarMappable(norm=flow_norm, cmap=flow_cmap),
			ax=flow_ax, label="Peak spill-link flow (m³/s, log scale)",
		)
	flow_ax.set_title(f"Basins with peak spill flow ≥ {minimum_flow_m3s:g} m³/s")
	for axis in (depth_ax, flow_ax):
		axis.set_xlabel(f"Easting (m; {peaks.crs.to_string()})")
		axis.set_ylabel("Northing (m)")
		axis.set_aspect("equal", adjustable="box")
		axis.grid(alpha=.2)

	hydro_ax.plot(totals.index, totals.outflow, label="Boundary outflow")
	hydro_ax.plot(totals.index, totals.flooding, label="Node flooding")
	hydro_ax.set_title("Aggregate discharge rates")
	hydro_ax.set_ylabel("Rate (m³/s)")
	hydro_ax.legend()
	storage_ax.plot(totals.index, totals.stored / 1_000_000, color="#238b57")
	storage_ax.fill_between(totals.index, totals.stored / 1_000_000,
							color="#238b57", alpha=.15)
	storage_ax.set_title("Total modeled basin storage")
	storage_ax.set_ylabel("Storage (million m³)")
	for axis in (hydro_ax, storage_ax):
		axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
		axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
		axis.tick_params(axis="x", labelrotation=25)
		axis.grid(alpha=.25)
	fig.suptitle(
		f"Enköping conceptual PySWMM scenario — {len(peaks)} basins; "
		f"peak depth {np.nanmax(depths):.2f} m",
		fontsize=15, fontweight="bold",
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
	plt.close(fig)
	print(f"PySWMM visualization written to {output_path}")


def main(plot_type: str, argv: Sequence[str] | None = None) -> None:
	"""Run one of the package plotting workflows from its example CLI."""
	parser = argparse.ArgumentParser(description=__doc__)
	if plot_type == "terrain":
		parser.add_argument("--raster", type=Path,
							default=RESULTS_DIR / "enkoping_depression_depth.tif")
		parser.add_argument("--elevation", type=Path,
							default=RESULTS_DIR / "enkoping_buffered_dem.tif")
		parser.add_argument("--output", type=Path,
							default=RESULTS_DIR / "enkoping_depression_depth_plot.png")
		parser.add_argument("--flow-accumulation", type=Path)
		parser.add_argument("--ponding-depth", type=Path)
		parser.add_argument("--runoff-volume", type=Path)
		parser.add_argument("--max-display-pixels", type=int, default=1600)
		args = parser.parse_args(argv)
		if args.max_display_pixels < 100:
			parser.error("--max-display-pixels must be at least 100")
		create_terrain_plot(args.raster, args.elevation, args.output,
							args.max_display_pixels, args.runoff_volume,
							args.ponding_depth, args.flow_accumulation)
	elif plot_type == "pyswmm":
		directory = RESULTS_DIR / "pyswmm"
		parser.add_argument("--peaks", type=Path, default=directory / "pyswmm_basin_peaks.gpkg")
		parser.add_argument("--nodes", type=Path, default=directory / "node_timeseries.csv")
		parser.add_argument("--links", type=Path, default=directory / "spill_link_timeseries.csv")
		parser.add_argument("--summary", type=Path, default=directory / "pyswmm_basin_summary.csv")
		parser.add_argument("--basin-ids", type=Path,
							default=RESULTS_DIR / "enkoping_buffered_depression_basin_id.tif")
		parser.add_argument("--dem", type=Path,
							default=RESULTS_DIR / "enkoping_buffered_dem.tif")
		parser.add_argument("--output", type=Path,
							default=directory / "enkoping_pyswmm_results.png")
		parser.add_argument("--max-display-pixels", type=int, default=1400)
		parser.add_argument("--minimum-flow-m3s", type=float, default=.005)
		args = parser.parse_args(argv)
		if args.max_display_pixels < 100 or args.minimum_flow_m3s < 0:
			parser.error("display pixels must be ≥100 and flow threshold non-negative")
		create_pyswmm_plot(args.peaks, args.nodes, args.links, args.summary,
						   args.basin_ids, args.dem, args.output,
						   args.max_display_pixels, args.minimum_flow_m3s)
	else:
		raise ValueError(f"Unknown plot type: {plot_type}")
