"""Main function for analysis."""

import os
import time

import matplotlib.pyplot as plt
from whitebox_workflows import WbEnvironment, show

from storm_water_management.utils import (
    get_control_values,
    get_tif_as_np_array,
    get_tif_from_np_array,
    info,
    saturated_upper_limit,
    transform_epsg,
    write_to_png,
)


def do_analysis(
    filename: str, do_calculation_of_control_values=False, rewrite_tif=True
) -> None:
    """Main function.

    Args:
        filename: path to file
        do_calculation_of_control_values: whether to calculate control values (number of filled cells and total volume)
        rewrite_tif: whether to rewrite tif to array and back to raster before analysis
    """
    start = time.time()
    filename_path = os.path.dirname(filename)
    tif_filename = os.path.basename(filename)
    wbe = WbEnvironment()
    wbe.verbose = False
    wbe.working_directory = filename_path
    dem = wbe.read_raster(tif_filename)
    dem.configs.epsg_code = 3006

    if rewrite_tif:
        raster_as_array = get_tif_as_np_array(filename)
        dem_from_array = get_tif_from_np_array(dem, raster_as_array)
        dem_from_array.configs.epsg_code = 3006
        dem_smoothed = dem_from_array
    else:
        dem_smoothed = dem

    # info(dem_from_array)

    # Smooth DEM. Parameters need to be set to proper values.
    # dem_smoothed = wbe.feature_preserving_smoothing(dem_from_array, filter_size=11, normal_diff_threshold=10.0, iterations=3)
    # dem_smoothed = dem

    # Fill depressions
    # dem_no_deps = wbe.fill_depressions_planchon_and_darboux(
    #    dem_smoothed, flat_increment=0.001
    # )
    # dem_no_deps = wbe.fill_depressions(dem_smoothed, flat_increment=0.001)
    dem_no_deps = wbe.fill_depressions_wang_and_liu(dem_smoothed, flat_increment=0.001)
    depression_depth = wbe.raster_calculator(
        "('dem_no_deps'-'dem')", [dem_no_deps, dem_smoothed]
    )

    # Flow accumulation analysis
    # channel_threshold = 50000.0
    # flow_accum = wbe.qin_flow_accumulation(dem_no_deps, out_type='cells', convergence_threshold=channel_threshold, log_transform=True)
    # wbe.write_raster(flow_accum, filename + 'flow_accum.tif'))

    depression_depth = transform_epsg(depression_depth)

    # Plot depression filling
    plot_depression = False
    if plot_depression:
        fig, ax = plt.subplots()
        ax = show(
            depression_depth,
            ax=ax,
            title="Depression Filling",
            figsize=(10, 7),
            colorbar_kwargs={
                "label": "Elevation (m)",
                "location": "right",
                "shrink": 0.5,
            },
            zorder=1,
            vmin=0,
            vmax=1,
        )
        ax.legend()
        plt.show()

    output_filename = tif_filename[:-4] + "_depression_depth.tif"
    wbe.write_raster(depression_depth, output_filename)

    write_to_png_bool = False
    if write_to_png_bool:
        depression_depth_saturated = saturated_upper_limit(depression_depth)
        wbe.write_raster(depression_depth_saturated, "depression_depth_saturated.tif")
        write_to_png(
            filename_path + "/depression_depth_saturated.tif", "output_colormap.png"
        )
        info(depression_depth_saturated)

    end = time.time()
    print(f"Analysis total time: {end - start:.2f} s")

    if do_calculation_of_control_values:
        number_of_filled_cells, total_volume = get_control_values(depression_depth)
        print("number_of_filled_cells: ", number_of_filled_cells)
        print("total water volume: ", total_volume)
    return os.path.join(filename_path, output_filename)
