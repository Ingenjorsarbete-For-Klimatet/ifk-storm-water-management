filename_path = '/home/chris/repos/storm_temp/data/1m'
filename = '63950_3150_25.tif'

from whitebox_workflows import WbEnvironment, show
import matplotlib.pyplot as plt

from utils import info, get_tif_as_np_array, get_tif_from_np_array

def main():

    # create environment
    wbe = WbEnvironment()
    wbe.verbose = True
    wbe.working_directory = filename_path
    dem = wbe.read_raster(filename)

    raster_as_array = get_tif_as_np_array(filename_path, filename)
    dem_from_array = get_tif_from_np_array(dem, raster_as_array)
    info(dem_from_array)

    # Smooth DEM. Parameters need to be set to proper values. 
    #dem_smoothed = wbe.feature_preserving_smoothing(dem_from_array, filter_size=11, normal_diff_threshold=10.0, iterations=3)
    dem_smoothed = dem_from_array
    # Fill depressions
    dem_no_deps = wbe.fill_depressions_planchon_and_darboux(dem_smoothed, flat_increment=0.001)
    #dem_no_deps = wbe.fill_depressions(dem_smoothed, flat_increment=0.001)
    #dem_no_deps = wbe.fill_depressions_wang_and_liu(dem_smoothed, flat_increment=0.001)
    depression_depth = wbe.raster_calculator("('dem_no_deps'-'dem')",[dem_no_deps, dem_from_array])
    #wbe.write_raster(depression_depth, filename + 'depression_depth.tif')

    # Flow accumulation analysis
    #channel_threshold = 50000.0
    #flow_accum = wbe.qin_flow_accumulation(dem_no_deps, out_type='cells', convergence_threshold=channel_threshold, log_transform=True)

    #wbe.write_raster(flow_accum, filename + 'flow_accum.tif'))

    # Plot depression filling
    fig, ax = plt.subplots()
    ax = show(depression_depth, ax=ax, title='Depression Filling', figsize=(10,7), colorbar_kwargs={'label': 'Elevation (m)', 'location': "right", 'shrink': 0.5}, zorder=1, vmin=0, vmax=1)
    ax.legend() 
    plt.show()

    return depression_depth

main()

