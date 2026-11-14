"""Run the Enköping elevation, terrain, and optional PySWMM workflow."""

from __future__ import annotations

# Package imports intentionally follow the repository path bootstrap.
# ruff: noqa: E402, I001

import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_ROOT / "src"
for import_path in (REPOSITORY_ROOT, SOURCE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from storm_water_management.config import (  # noqa: E402
    AnalysisConfig,
    DownloadConfig,
    PySwmmConfig,
)
from storm_water_management.elevation_download import download_elevation_tiles  # noqa: E402
from storm_water_management import postprocess  # noqa: E402
from storm_water_management.storm_scenario import (  # noqa: E402
    run_terrain_analysis,
)
from storm_water_management.utils_plots import (  # noqa: E402
    create_pyswmm_plot,
    create_terrain_plot,
)
from storm_water_management.utils_pyswmm import (  # noqa: E402
    run_pyswmm_analysis,
)

TILE_DIR = REPOSITORY_ROOT / "data" / "elevation" / "enkoping"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "enkoping"
LOCATION_NAME = "Enköping"
CENTER_LONGITUDE = 17.0778
CENTER_LATITUDE = 59.6356
OUTPUT_PREFIX = "enkoping"
DEFAULT_DOWNLOAD = DownloadConfig(
    name=LOCATION_NAME,
    center_lon=CENTER_LONGITUDE,
    center_lat=CENTER_LATITUDE,
    output_dir=TILE_DIR,
)
DEFAULT_ANALYSIS = AnalysisConfig(
    input_dir=TILE_DIR,
    output_dir=RESULTS_DIR,
    center_lon=CENTER_LONGITUDE,
    center_lat=CENTER_LATITUDE,
    output_prefix=OUTPUT_PREFIX,
)
DEFAULT_PYSWMM = PySwmmConfig()


def run_pipeline(
    *,
    scenario: Literal["terrain", "pyswmm"] = "pyswmm",
    download: bool = False,
    plot: bool = True,
    download_config: DownloadConfig = DEFAULT_DOWNLOAD,
    analysis: AnalysisConfig = DEFAULT_ANALYSIS,
    pyswmm: PySwmmConfig = DEFAULT_PYSWMM,
    max_display_pixels: int | None = None,
) -> None:
    """Run download, terrain preparation, optional PySWMM, and plotting.

    Location defaults and output naming are intentionally configured in this
    example module. The package-level analysis functions accept coordinates and
    paths explicitly and do not assume Enköping.
    """
    if scenario not in ("terrain", "pyswmm"):
        raise ValueError(f"Unknown scenario: {scenario}")
    output_prefix = analysis.output_prefix
    if analysis.dry_run and (scenario != "terrain" or download or plot):
        raise ValueError("dry_run is only available for a terrain-only pipeline with plot=False.")

    if download:
        print("[1] Download elevation tiles")
        download_elevation_tiles(
            name=download_config.name,
            center_lon=download_config.center_lon,
            center_lat=download_config.center_lat,
            radius_km=download_config.radius_km,
            bbox=download_config.bbox,
            output_dir=download_config.output_dir or analysis.input_dir,
            collection=download_config.collection,
        )

    terrain_config = analysis
    if scenario == "pyswmm":
        terrain_config = replace(
            analysis,
            rainfall_mm=None,
            prepare_basins_only=True,
            infiltration_fraction=None,
            infiltration_capacity_mm_h=None,
            rainfall_duration_hours=None,
        )
    else:
        terrain_config = replace(analysis, prepare_basins_only=False)

    stage_number = 2 if download else 1
    print(f"[{stage_number}] Prepare terrain and basin outputs")
    run_terrain_analysis(terrain_config)
    if analysis.dry_run:
        return

    output_dir = analysis.output_dir
    depth_rasters = [
        output_dir / f"{output_prefix}_buffered_depression_depth.tif",
        output_dir / f"{output_prefix}_depression_depth.tif",
    ]
    if terrain_config.rainfall_mm is not None:
        depth_rasters.extend(
            [
                output_dir / f"{output_prefix}_buffered_estimated_ponding_depth.tif",
                output_dir / f"{output_prefix}_estimated_ponding_depth.tif",
            ]
        )
    for depth_raster in depth_rasters:
        postprocess.write_geojson_polygons_from_tif_to_file(
            str(depth_raster), output_format="FlatGeobuf"
        )
        polygon_path = depth_raster.with_name(f"{depth_raster.stem}_polygons.fgb")
        print(f"FlatGeobuf polygons: {polygon_path}")

    if scenario == "pyswmm":
        stage_number += 1
        print(f"[{stage_number}] Run conceptual PySWMM scenario")
        run_pyswmm_analysis(
            pyswmm,
            basin_summary_path=output_dir / f"{output_prefix}_buffered_basin_summary.csv",
            dem_path=output_dir / f"{output_prefix}_buffered_dem.tif",
            basin_ids_path=output_dir / f"{output_prefix}_buffered_depression_basin_id.tif",
            output_dir=output_dir / "pyswmm",
        )
    if not plot:
        return

    stage_number += 1
    print(f"[{stage_number}] Create results visualization")
    output_dir = analysis.output_dir
    if scenario == "terrain":
        create_terrain_plot(
            output_dir / f"{output_prefix}_depression_depth.tif",
            output_dir / f"{output_prefix}_buffered_dem.tif",
            output_dir / f"{output_prefix}_depression_depth_plot.png",
            max_display_pixels=max_display_pixels or 1600,
            ponding_depth_path=(
                output_dir / f"{output_prefix}_estimated_ponding_depth.tif"
                if terrain_config.rainfall_mm is not None
                else None
            ),
        )
        return

    swmm_dir = output_dir / "pyswmm"
    create_pyswmm_plot(
        swmm_dir / "pyswmm_basin_peaks.gpkg",
        swmm_dir / "node_timeseries.csv",
        swmm_dir / "spill_link_timeseries.csv",
        swmm_dir / "pyswmm_basin_summary.csv",
        output_dir / f"{output_prefix}_buffered_depression_basin_id.tif",
        output_dir / f"{output_prefix}_buffered_dem.tif",
        swmm_dir / f"{output_prefix}_pyswmm_results.png",
        max_display_pixels=max_display_pixels or 1400,
    )


def main() -> None:
    """Run the default Enköping PySWMM pipeline."""
    run_pipeline()


if __name__ == "__main__":
    main()
