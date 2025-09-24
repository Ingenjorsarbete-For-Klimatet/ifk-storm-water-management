"""Main function."""

import matplotlib.pyplot as plt
from geojson_utils import write_geojson_polynomials_from_tif_to_file
from utils import (
    get_tif_as_np_array,
    get_tif_from_np_array,
    info,
    saturated_upper_limit,
    transform_epsg,
    write_to_png,
)
from whitebox_workflows import WbEnvironment, show

filename_path = "/home/chris/repos/storm_temp/data/1m"
filename = "63950_3150_25.tif"


def main():
    """Main function."""
    wbe = WbEnvironment()
    wbe.verbose = True
    wbe.working_directory = filename_path
    dem = wbe.read_raster(filename)

    raster_as_array = get_tif_as_np_array(filename_path, filename)
    dem_from_array = get_tif_from_np_array(dem, raster_as_array)
    dem_from_array.configs.epsg_code = 3006
    info(dem_from_array)

    # Smooth DEM. Parameters need to be set to proper values.
    # dem_smoothed = wbe.feature_preserving_smoothing(dem_from_array, filter_size=11, normal_diff_threshold=10.0, iterations=3)
    dem_smoothed = dem_from_array
    # Fill depressions
    # dem_no_deps = wbe.fill_depressions_planchon_and_darboux(
    #    dem_smoothed, flat_increment=0.001
    # )
    # dem_no_deps = wbe.fill_depressions(dem_smoothed, flat_increment=0.001)
    dem_no_deps = wbe.fill_depressions_wang_and_liu(dem_smoothed, flat_increment=0.001)
    depression_depth = wbe.raster_calculator(
        "('dem_no_deps'-'dem')", [dem_no_deps, dem_from_array]
    )
    # wbe.write_raster(depression_depth, filename + 'depression_depth.tif')

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

    wbe.write_raster(depression_depth, filename[:-4] + "_depression_depth.tif")
    write_geojson_polynomials_from_tif_to_file(
        filename_path + "/" + filename[:-4] + "_depression_depth.tif"
    )

    write_to_png_bool = False
    if write_to_png_bool:
        # prepare and save as png
        depression_depth_saturated = saturated_upper_limit(depression_depth)
        wbe.write_raster(depression_depth_saturated, "depression_depth_saturated.tif")
        write_to_png(
            filename_path + "/depression_depth_saturated.tif", "output_colormap.png"
        )
        info(depression_depth_saturated)


if __name__ == "__main__":
    main()
