"""Calculate depression depth in a rectangle centered on Enköping.

The requested result extent can reach different distances west, east, north,
and south of the center. The DEM for the entire result extent plus directional
buffers is mosaicked and analyzed once, then clipped to the requested result
rectangle to reduce edge effects.

Run from the repository root with ``python examples/analyze_enkoping.py``.
Elevation tiles are read from ``data/elevation/enkoping`` and outputs are
written to ``results/enkoping``. Distances are independently configurable;
for example, ``--west-km 2 --east-km 2.5 --north-km 2 --south-km 5`` sets the
result rectangle, while ``--buffer-west-km`` and the other directional buffer
options extend the terrain used for the single calculation beyond that area.

Optionally, ``--rainfall-mm`` adds screening outputs using D8 flow accumulation
and connected DEM depression basins. By default, 50% of event rainfall is
treated as infiltrated. Runoff volume is estimated from incremental contributing
area, stored in each basin to its DEM-derived spill level, and excess is routed
to the next mapped basin along D8 flow. The depth raster and per-basin CSV are
static screening estimates, not hydraulic flood predictions. Select the
100-year event depth and duration from a local design-rain source. By default,
basins shallower than 2 cm or smaller than 25 m² are excluded from the basin
graph.
"""

import argparse
from pathlib import Path

from storm_water_management import postprocess
from storm_water_management.storm_scenario import write_basin_storage_spill_outputs
from storm_water_management.utils_workflows import (
    add_south_edge_drain,
    clip_to_area,
    create_buffered_mosaic,
    effective_rainfall_m,
    projected_center,
    run_depression_analysis,
    select_tiles,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "data" / "elevation" / "enkoping"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "enkoping"
DEFAULT_RESULT_EXTENT_KM = 4.0
DEFAULT_BUFFER_KM = 2.5
DEFAULT_SOUTH_DRAIN_ELEVATION_M = 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing downloaded elevation tiles (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for analysis outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--rainfall-mm",
        type=float,
        help=(
            "Optional event rainfall depth in mm (use a locally sourced "
            "100-year design depth to represent that return period). Creates a "
            "basin-based ponding-depth raster and per-basin inflow/storage/spill summary."
        ),
    )
    infiltration = parser.add_mutually_exclusive_group()
    infiltration.add_argument(
        "--infiltration-fraction",
        type=float,
        default=None,
        help=(
            "Uniform fraction of event rainfall assumed to infiltrate "
            "(default: 0.5; valid range: 0-1)."
        ),
    )
    infiltration.add_argument(
        "--infiltration-capacity-mm-h",
        type=float,
        help=(
            "Alternative constant, spatially uniform infiltration capacity "
            "in mm/hour; requires --rainfall-duration-hours."
        ),
    )
    parser.add_argument(
        "--rainfall-duration-hours",
        type=float,
        help="Storm duration in hours, required with --infiltration-capacity-mm-h.",
    )
    parser.add_argument(
        "--minimum-depression-depth-m",
        type=float,
        default=0.02,
        help=(
            "Ignore filled depressions shallower than this threshold when "
            "building the basin graph (default: 0.02 m)."
        ),
    )
    parser.add_argument(
        "--minimum-basin-area-m2",
        type=float,
        default=25.0,
        help=(
            "Ignore connected depression components smaller than this area "
            "to reduce DEM-noise sinks (default: 25 m²)."
        ),
    )
    for direction in ("west", "east", "north", "south"):
        parser.add_argument(
            f"--{direction}-km",
            type=float,
            default=DEFAULT_RESULT_EXTENT_KM,
            help=(
                f"Result extent from the center toward {direction} in km "
                f"(default: {DEFAULT_RESULT_EXTENT_KM:g})."
            ),
        )
        parser.add_argument(
            f"--buffer-{direction}-km",
            type=float,
            default=DEFAULT_BUFFER_KM,
            help=(
                f"Additional analysis buffer beyond the result area to the "
                f"{direction} in km (default: {DEFAULT_BUFFER_KM:g})."
            ),
        )
    parser.add_argument(
        "--south-drain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Lower the southern edge of the buffered DEM to create an artificial "
            "open outlet. Disabled by default; --no-south-drain disables it."
        ),
    )
    parser.add_argument(
        "--south-drain-elevation-m",
        type=float,
        default=DEFAULT_SOUTH_DRAIN_ELEVATION_M,
        help=(
            "Elevation assigned to valid cells along the southern edge when "
            f"--south-drain is enabled (default: {DEFAULT_SOUTH_DRAIN_ELEVATION_M:g} m)."
        ),
    )
    parser.add_argument(
        "--south-drain-rows",
        type=int,
        default=1,
        help=(
            "Number of southernmost DEM rows lowered to the drain elevation "
            "when --south-drain is enabled (default: 1; at 1 m resolution, "
            "100 rows makes a 100 m-wide artificial strip)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the selected tiles and projected bounds without calculating.",
    )
    args = parser.parse_args()
    if args.rainfall_mm is not None and args.rainfall_mm <= 0:
        parser.error("--rainfall-mm must be greater than zero.")
    if args.rainfall_mm is None and (
        args.infiltration_fraction is not None
        or args.infiltration_capacity_mm_h is not None
        or args.rainfall_duration_hours is not None
    ):
        parser.error("Infiltration and duration options require --rainfall-mm.")
    if args.infiltration_fraction is not None and not 0 <= args.infiltration_fraction <= 1:
        parser.error("--infiltration-fraction must be between 0 and 1.")
    if args.minimum_depression_depth_m < 0:
        parser.error("--minimum-depression-depth-m cannot be negative.")
    if args.minimum_basin_area_m2 < 0:
        parser.error("--minimum-basin-area-m2 cannot be negative.")
    if args.infiltration_capacity_mm_h is not None:
        if args.infiltration_capacity_mm_h < 0:
            parser.error("--infiltration-capacity-mm-h cannot be negative.")
        if args.rainfall_duration_hours is None or args.rainfall_duration_hours <= 0:
            parser.error(
                "--infiltration-capacity-mm-h requires a positive "
                "--rainfall-duration-hours."
            )
    elif args.rainfall_duration_hours is not None:
        parser.error(
            "--rainfall-duration-hours can only be used with "
            "--infiltration-capacity-mm-h."
        )
    for direction in ("west", "east", "north", "south"):
        if getattr(args, f"{direction}_km") <= 0:
            parser.error(f"--{direction}-km must be greater than zero.")
        if getattr(args, f"buffer_{direction}_km") < 0:
            parser.error(f"--buffer-{direction}-km cannot be negative.")
    if args.south_drain_rows < 1:
        parser.error("--south-drain-rows must be at least 1.")
    return args


def main() -> None:
    """Build one buffered mosaic, analyze it once, and crop to the result area."""
    args = parse_args()
    center_x, center_y = projected_center()
    result_bounds = (
        center_x - args.west_km * 1000,
        center_y - args.south_km * 1000,
        center_x + args.east_km * 1000,
        center_y + args.north_km * 1000,
    )
    analysis_bounds = (
        result_bounds[0] - args.buffer_west_km * 1000,
        result_bounds[1] - args.buffer_south_km * 1000,
        result_bounds[2] + args.buffer_east_km * 1000,
        result_bounds[3] + args.buffer_north_km * 1000,
    )

    tile_paths = select_tiles(args.input_dir, analysis_bounds)
    print(f"Enköping center in SWEREF99 TM: ({center_x:.2f}, {center_y:.2f})")
    print(
        "Result extents from center (west/east/north/south): "
        f"{args.west_km:g}/{args.east_km:g}/{args.north_km:g}/"
        f"{args.south_km:g} km."
    )
    print(
        "Buffers beyond result area (west/east/north/south): "
        f"{args.buffer_west_km:g}/{args.buffer_east_km:g}/"
        f"{args.buffer_north_km:g}/{args.buffer_south_km:g} km."
    )
    if args.south_drain:
        if args.buffer_south_km <= 0:
            raise ValueError(
                "--south-drain requires --buffer-south-km greater than zero so "
                "the artificial outlet stays outside the result area."
            )
        print(
            f"South drain enabled: lowering the southernmost {args.south_drain_rows} "
            f"raster row(s) to {args.south_drain_elevation_m:g} m "
            "(artificial boundary condition)."
        )
    else:
        print("South drain disabled.")
    print(f"Selected {len(tile_paths)} elevation tiles from {args.input_dir}.")
    for tile_path in tile_paths:
        print(f"  {tile_path.name}")

    if args.dry_run:
        print(f"Output rectangle bounds: {result_bounds}")
        print(f"Single-pass analysis bounds: {analysis_bounds}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_south_drain" if args.south_drain else ""
    buffered_dem = args.output_dir / f"enkoping_buffered_dem{suffix}.tif"
    analysis_dem = buffered_dem
    if args.south_drain:
        analysis_dem = args.output_dir / "enkoping_buffered_dem_south_drain_conditioned.tif"
    buffered_depth = (
        args.output_dir / f"enkoping_buffered_depression_depth{suffix}.tif"
    )
    result_raster = args.output_dir / f"enkoping_depression_depth{suffix}.tif"
    contributing_area_path = None
    upslope_storage_path = None
    if args.rainfall_mm is not None:
        contributing_area_path = (
            args.output_dir / f"enkoping_buffered_contributing_area{suffix}.tif"
        )
        upslope_storage_path = (
            args.output_dir / f"enkoping_buffered_upslope_storage{suffix}.tif"
        )
        result_contributing_area = (
            args.output_dir / f"enkoping_contributing_area{suffix}.tif"
        )
        result_upslope_storage = (
            args.output_dir / f"enkoping_upslope_depression_storage{suffix}.tif"
        )
        buffered_ponding_depth = (
            args.output_dir / f"enkoping_buffered_estimated_ponding_depth{suffix}.tif"
        )
        result_ponding_depth = (
            args.output_dir / f"enkoping_estimated_ponding_depth{suffix}.tif"
        )
        d8_pointer_path = (
            args.output_dir / f"enkoping_buffered_d8_pointer{suffix}.tif"
        )
        d8_contributing_area_path = (
            args.output_dir / f"enkoping_buffered_d8_contributing_area{suffix}.tif"
        )
        buffered_basin_ids = (
            args.output_dir / f"enkoping_buffered_depression_basin_id{suffix}.tif"
        )
        result_basin_ids = (
            args.output_dir / f"enkoping_depression_basin_id{suffix}.tif"
        )
        basin_summary_path = (
            args.output_dir / f"enkoping_buffered_basin_summary{suffix}.csv"
        )
        fraction = (
            0.5
            if args.infiltration_fraction is None
            else args.infiltration_fraction
        )
        effective_rain = effective_rainfall_m(
            args.rainfall_mm,
            infiltration_fraction=fraction,
            infiltration_capacity_mm_h=args.infiltration_capacity_mm_h,
            rainfall_duration_hours=args.rainfall_duration_hours,
        )
        if args.infiltration_capacity_mm_h is None:
            print(
                f"Storm rainfall: {args.rainfall_mm:g} mm; assuming "
                f"{fraction:.0%} uniform infiltration, leaving "
                f"{effective_rain * 1000:g} mm effective rainfall."
            )
        else:
            print(
                f"Storm rainfall: {args.rainfall_mm:g} mm over "
                f"{args.rainfall_duration_hours:g} h; constant infiltration "
                f"capacity {args.infiltration_capacity_mm_h:g} mm/h leaves "
                f"{effective_rain * 1000:g} mm effective rainfall."
            )
    else:
        effective_rain = None

    print("Creating buffered elevation mosaic ...")
    create_buffered_mosaic(tile_paths, analysis_bounds, buffered_dem)
    if args.south_drain:
        print("Applying the optional southern-edge drain condition ...")
        add_south_edge_drain(
            buffered_dem,
            analysis_dem,
            args.south_drain_elevation_m,
            rows=args.south_drain_rows,
        )
    print("Running one depression-fill calculation over the full buffered mosaic ...")
    if effective_rain is not None:
        print("Also calculating contributing area and upslope depression storage ...")
    run_depression_analysis(
        analysis_dem,
        buffered_depth,
        contributing_area_path=contributing_area_path,
        upslope_storage_path=upslope_storage_path,
        d8_pointer_path=d8_pointer_path if effective_rain is not None else None,
        d8_contributing_area_path=(
            d8_contributing_area_path if effective_rain is not None else None
        ),
    )
    print("Clipping to the requested result rectangle ...")
    clip_to_area(buffered_depth, result_raster, result_bounds)
    if effective_rain is not None:
        clip_to_area(contributing_area_path, result_contributing_area, result_bounds)
        clip_to_area(upslope_storage_path, result_upslope_storage, result_bounds)
        basin_summaries = write_basin_storage_spill_outputs(
            analysis_dem,
            buffered_depth,
            d8_pointer_path,
            d8_contributing_area_path,
            buffered_ponding_depth,
            buffered_basin_ids,
            basin_summary_path,
            effective_rain,
            minimum_depression_depth_m=args.minimum_depression_depth_m,
            minimum_basin_area_m2=args.minimum_basin_area_m2,
        )
        clip_to_area(buffered_ponding_depth, result_ponding_depth, result_bounds)
        clip_to_area(buffered_basin_ids, result_basin_ids, result_bounds)
        print(f"Contributing area raster (m²): {result_contributing_area}")
        print(f"Average upslope depression storage raster (m): {result_upslope_storage}")
        total_spill = sum(float(row["spill_volume_m3"]) for row in basin_summaries)
        print(f"Basin-based estimated ponding-depth raster (m): {result_ponding_depth}")
        print(f"Depression basin ID raster: {result_basin_ids}")
        print(f"Basin storage/spill summary: {basin_summary_path}")
        print(
            f"Processed {len(basin_summaries):,} depression components; "
            f"sum of basin spill transfers {total_spill:,.1f} m³ "
            "(a routed volume may spill at multiple basins)."
        )

    postprocess.write_geojson_polygons_from_tif_to_file(
        str(result_raster), output_format="FlatGeobuf"
    )
    print(f"Finished. Rectangular depression-depth raster: {result_raster}")
    polygons_path = result_raster.with_name(result_raster.stem + "_polygons.fgb")
    print(f"FlatGeobuf polygons: {polygons_path}")


if __name__ == "__main__":
    main()
