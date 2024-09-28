'''From elevation data, utilize flow and depression analysis, and save as geojson
'''
import richdem as rd
from storm_water_management import utils

def main(tif_filename):
    # input data
    tfw_filename = tif_filename.split(".")[0] + r".tfw"

    # read data
    elevation_data = utils.get_elevation_data_from_tif(tif_filename)
    tfw = utils.get_coordinates_from_tfw(tfw_filename)

    # Flow analysis
    dem = rd.LoadGDAL(tif_filename, no_data=-9999)

    #Fill depressions with epsilon gradient to ensure drainage
    fd_epsilon = rd.FillDepressions(dem, epsilon=True, in_place=False)

    #Get flow accumulation with no explicit weighting. The default will be 1.
    accum_d8 = rd.FlowAccumulation(fd_epsilon, method='D8')

    # fill depression analysis
    rda_fill = rd.FillDepressions(dem, epsilon=False, in_place=False)
    rda_fill_diff = rda_fill - dem
    rd.rdShow(rda_fill_diff, zxmin=750, zxmax=850, zymin=750, zymax=550, figsize=(8,5.5), cmap='jet')

    # save geojson
    export_sample = True
    if export_sample:
        n_x = min(200, elevation_data.shape[0])
        n_y = min(200, elevation_data.shape[1])
    else:
        n_x = elevation_data.shape[0]
        n_y = elevation_data.shape[1]
    
    geojson_data = utils.get_coordinate_center_points_from_tfw(tfw, elevation_data, accum_d8, rda_fill_diff, n_x, n_y)
    utils.save_to_geojson(geojson_data, "64_3_2023")

if __name__ == "__main__":
    main(r"/home/chris/projects/ifk-storm-water-management/data/64_3_2023.tif")