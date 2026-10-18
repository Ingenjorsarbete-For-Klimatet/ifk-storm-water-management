"""Run a DEM-informed, conceptual PySWMM event analysis for Enköping.

This standalone analysis turns a DEM-derived basin-network summary and
aligned elevation/basin rasters into a simplified SWMM storage network. Major
basins are represented by storage nodes whose area-versus-depth curves are
sampled from DEM hypsometry; incremental D8 contributing area is represented
as subcatchments; and basin connections follow a simplified spill graph. Minor
basins are collapsed into the next modeled basin or bypassed to an external
outfall so the 12,000-plus DEM components remain computationally tractable.

The repository has no surveyed storm-drain network or design hyetograph. This
model therefore represents surface depressions only, assumes a uniform storm
of effective rainfall, and uses assumed overland-flow parameters. Treat output
as a dynamic screening scenario, not a calibrated or infrastructure-grade
flood prediction. Change the event depth and duration to test alternatives.

Install the optional dependency with ``pip install -e '.[swmm]'`` then run
``python examples/analyze_enkoping_pyswmm.py`` from the repository root.
Create the network inputs first with ``python examples/analyze_enkoping.py
--prepare-basins-only``. Pass ``--effective-rainfall-mm`` when using that
rainfall-independent network summary.
"""

import argparse
import math
import sys
from pathlib import Path

from storm_water_management.utils_pyswmm import (
    build_storage_geometries,
    extract_basin_elevations,
    load_basins,
    render_swmm_input,
    run_model,
    select_storage_basins,
    write_model_notes,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results" / "enkoping"
DEFAULT_BASIN_SUMMARY = DEFAULT_RESULTS_DIR / "enkoping_buffered_basin_summary.csv"
DEFAULT_DEM = DEFAULT_RESULTS_DIR / "enkoping_buffered_dem.tif"
DEFAULT_BASIN_IDS = DEFAULT_RESULTS_DIR / "enkoping_buffered_depression_basin_id.tif"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "pyswmm"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basin-summary", type=Path, default=DEFAULT_BASIN_SUMMARY)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--basin-ids", type=Path, default=DEFAULT_BASIN_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--effective-rainfall-mm",
        type=float,
        default=100,
        help=(
            "Net rainfall depth in mm. Defaults to the median effective rainfall "
            "recorded in a static basin summary; required for a prepared network "
            "summary without rainfall data. No further infiltration is applied."
        ),
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=6.0,
        help="Uniform storm duration in hours (default: 6; no local hyetograph is available).",
    )
    parser.add_argument(
        "--drain-hours",
        type=float,
        default=12.0,
        help="Additional simulation time after rainfall ends (default: 12).",
    )
    parser.add_argument(
        "--report-interval-minutes",
        type=int,
        default=5,
        help="Sampling interval for exported time series in minutes (default: 5).",
    )
    parser.add_argument(
        "--minimum-storage-basin-area-m2",
        type=float,
        default=1000.0,
        help=(
            "Model only DEM depressions at least this large; smaller basins are "
            "collapsed into the next modeled basin or bypassed to an external "
            "outfall (default: 1000 m²)."
        ),
    )
    parser.add_argument(
        "--spill-channel-width-m",
        type=float,
        default=1.0,
        help="Assumed width of conceptual spill channels in metres (default: 1).",
    )
    parser.add_argument(
        "--spill-channel-roughness",
        type=float,
        default=0.03,
        help="Assumed Manning roughness of spill channels (default: 0.03).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory's generated files if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Create a conceptual SWMM model, run it, and export time-series results."""
    args = parse_args()
    if args.duration_hours <= 0 or not math.isfinite(args.duration_hours):
        raise ValueError("Storm duration must be finite and positive.")
    if args.drain_hours < 0 or not math.isfinite(args.drain_hours):
        raise ValueError("Drain time must be finite and non-negative.")
    if args.report_interval_minutes <= 0 or 60 % args.report_interval_minutes:
        raise ValueError("Report interval must be a positive divisor of 60 minutes.")
    if args.spill_channel_width_m <= 0 or not math.isfinite(args.spill_channel_width_m):
        raise ValueError("Spill-channel width must be finite and positive.")
    if args.spill_channel_roughness <= 0 or not math.isfinite(
        args.spill_channel_roughness
    ):
        raise ValueError("Spill-channel roughness must be finite and positive.")

    source_basins, rainfall_mm = load_basins(
        args.basin_summary, args.effective_rainfall_mm
    )
    basins, external_runoff_area_m2 = select_storage_basins(
        source_basins, args.minimum_storage_basin_area_m2
    )
    if rainfall_mm < 0 or not math.isfinite(rainfall_mm):
        raise ValueError("Effective rainfall must be finite and non-negative.")
    basin_elevations, cell_area_m2 = extract_basin_elevations(
        args.dem, args.basin_ids, {basin.basin_id for basin in basins}
    )
    geometries = build_storage_geometries(basins, basin_elevations, cell_area_m2)

    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite to replace files."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "enkoping_dem_conceptual.inp"
    report_path = args.output_dir / "enkoping_dem_conceptual.rpt"
    output_path = args.output_dir / "enkoping_dem_conceptual.out"
    input_path.write_text(
        render_swmm_input(
            basins,
            geometries,
            rainfall_mm,
            args.duration_hours,
            args.drain_hours,
            args.spill_channel_width_m,
            args.spill_channel_roughness,
            args.report_interval_minutes,
            external_runoff_area_m2,
        ),
        encoding="utf-8",
    )
    write_model_notes(
        args.output_dir,
        basins,
        geometries,
        rainfall_mm,
        args.duration_hours,
        args.report_interval_minutes,
        cell_area_m2,
        len(source_basins),
        args.minimum_storage_basin_area_m2,
        external_runoff_area_m2,
    )
    node_csv, link_csv, basin_csv = run_model(
        input_path,
        report_path,
        output_path,
        basins,
        args.report_interval_minutes,
        args.output_dir,
    )
    print(
        f"Ran {len(basins)} of {len(source_basins)} DEM-derived basins with "
        f"{rainfall_mm:g} mm effective rainfall; {external_runoff_area_m2:,.0f} m² "
        "of incremental area drains directly to the boundary outfall."
    )
    print(f"SWMM input: {input_path}")
    print(f"Node time series: {node_csv}")
    print(f"Spill-link time series: {link_csv}")
    print(f"Basin summary: {basin_csv}")
    print(f"Assumptions and capacity check: {args.output_dir / 'model_notes.txt'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
