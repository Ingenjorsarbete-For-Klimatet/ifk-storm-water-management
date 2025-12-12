import os
from storm_water_management import tif_preprocessing_utils, analysis, postprocess
from dataclasses import dataclass

@dataclass
class City:
    name: str
    longitudinal_position: float
    lateral_position: float
    distance_limit_calculation: float = 5000
    distance_limit_results: float = 3000

"""cities = [
    City("lilla_edet", 58.13230982643911, 12.124035673595857),
    City("henån", 58.23579871150094, 11.675051043473774),
    City("trollhättan", 58.28852773933211, 12.296952550648106),
    City("lerum", 57.77086736070489, 12.267423222012443),
    City("kungälv", 57.87207410821738, 11.971864246035913),
    City("nossebro", 58.18883716555037, 12.718009440512605),
    City("vara", 58.26239241191423, 12.959222384878474),
    City("grästorp", 58.33175977452058, 12.680429620162242),
    City("vårgårda", 58.03449974409668, 12.808469423654577),
    City("vänersborg", 58.368002509503505, 12.327689164411954),
    City("herrljunga", 58.078478603473194, 13.028066378329038),
    City("lidköping", 58.51017437523059, 13.136038799241717),
    City("alingsås", 57.93048034773409, 12.540678237239264),
    City("uddevalla", 58.3514080080046, 11.933445688117272),
    City("munkedal", 58.47044880677707, 11.678226890362856),
    City("stenungsund", 58.07150531530186, 11.850248960467628)
]"""

cities = [
    City("gbg_north", 57.78186270404611, 11.959186965623235),
    City("gbg_center", 57.70550004878299, 11.939731567136613),
    City("gbg_south", 57.61938252109898, 11.889806099289737)
]

folder_to_search_for_tif_files = "/home/chris/repos/data/elevation_data_sweden/lilla_edet"
output_folder = "/home/chris/repos/data/results/"

for city in cities:
    do_preprocess = False
    do_analysis = False
    do_postprocessing = False

    x, y = tif_preprocessing_utils.get_sweref99_coordinate_from_wgs84(city.longitudinal_position, city.lateral_position)

    # Merge all tif files that are within distance_limit_calculation (= half width of rectangle).
    if do_preprocess:
        filename = city.name + ".tif"
        output_filename = os.path.join(output_folder, filename)
        files_in_folder = tif_preprocessing_utils.get_all_tif_files_recursively(folder_to_search_for_tif_files)
        file_list = tif_preprocessing_utils.filter_files_by_distance(files_in_folder, x,y, city.distance_limit_calculation)
        #tif_preprocessing_utils.save_and_plot_area_of_all_files_in_folder(file_list, x, y)
        tif_preprocessing_utils.concat_tif_in_folder(file_list, folder_to_search_for_tif_files, output_filename)
        print("Preprocess done.")

    # Analyze
    if do_analysis:
        analysis_tif_filename = analysis.do_analysis(output_filename)
        print("Analysis done.")

    # Post processing
    if do_postprocessing:
        cropped_output_tif_filename = postprocess.crop_tif(analysis_tif_filename, x, y, city.distance_limit_results)
        postprocess.write_geojson_polygons_from_tif_to_file(cropped_output_tif_filename)
        print("Postprocess done.")

    if True:
        import glob
        import geopandas as gpd
        # can be done faster by pyogrio or GeoParquet
        
        files = glob.glob(os.path.join(output_folder, "*.geojson"))
        gdfs = [gpd.read_file(f) for f in files]
        merged = gpd.pd.concat(gdfs, ignore_index=True)
        merged.to_file("merged.geojson", driver="GeoJSON")
