"""End-to-end terrain analysis workflow for one configured study area.

Builds a buffered elevation mosaic from downloaded tiles, runs the
depression-fill and D8 flow calculations, prepares the DEM-derived basin
network, optionally adds a static rainfall storage/spill scenario, and clips
every result to the requested rectangle. All heavy lifting lives in
``terrain`` (processing steps) and ``basin_network`` (basin science); this
module only orchestrates them for an ``AnalysisConfig``.
"""

from __future__ import annotations

from storm_water_management.basin_network import (
    write_basin_network_outputs,
    write_basin_storage_spill_outputs,
)
from storm_water_management.config import (
    DEFAULT_INFILTRATION_FRACTION,
    AnalysisConfig,
)
from storm_water_management.terrain import (
    clip_to_area,
    create_buffered_mosaic,
    effective_rainfall_m,
    projected_center,
    run_depression_analysis,
    select_tiles,
)


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
