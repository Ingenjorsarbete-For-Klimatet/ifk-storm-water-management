"""Example on how to run from files."""

import os

import rasterio

from storm_water_management import analysis, postprocess, tif_preprocessing_utils

# Distance from mid point of bbox for inclusion of other bboxs.
include_tile_distance = 2000

# Files for depression depth analysis.
tifs_to_explore = [
    "/home/chris/repos/data/elevation_data_sweden/goteborg/639_31_2550/63925_3150_25.tif",
    "/home/chris/repos/data/elevation_data_sweden/goteborg/639_31_2525/63925_3125_25.tif",
    "/home/chris/repos/data/elevation_data_sweden/goteborg/639_31_5075/63950_3175_25.tif",
    "/home/chris/repos/data/elevation_data_sweden/goteborg/639_31_5050/63950_3150_25.tif",
    "/home/chris/repos/data/elevation_data_sweden/goteborg/639_31_5025/63950_3125_25.tif",
    "/home/chris/repos/data/elevation_data_sweden/goteborg/639_31_7525/63975_3125_25.tif",
    "/home/chris/repos/data/elevation_data_sweden/goteborg/639_31_2575/63925_3175_25.tif",
    "/home/chris/repos/data/elevation_data_sweden/goteborg/639_31_7550/63975_3150_25.tif",
    "/home/chris/repos/data/elevation_data_sweden/goteborg/639_31_7575/63975_3175_25.tif",
]

folder_to_search_for_tif_files = "/home/chris/repos/data/elevation_data_sweden/goteborg"
output_folder = "/home/chris/repos/data/res/hogsbo"

# utils.write_to_png("/home/chris/repos/data/res/hogsbo/63975_3175_25_depression_depth.tif", "t.png")
do_analysis = True
do_postprocessing = True
for filename in tifs_to_explore:
    # Get midpoint of tile
    with rasterio.open(filename) as ds:
        b = ds.bounds
        center_x = (b.left + b.right) / 2
        center_y = (b.bottom + b.top) / 2

    output_filename = os.path.join(
        output_folder,
        f"incl_distance_{include_tile_distance}_" + os.path.basename(filename),
    )
    print(output_filename)
    # copy files to dir
    files_in_folder = tif_preprocessing_utils.get_all_tif_files_recursively(
        folder_to_search_for_tif_files
    )
    file_list = tif_preprocessing_utils.filter_files_by_distance(
        files_in_folder, center_x, center_y, include_tile_distance
    )
    # tif_preprocessing_utils.save_and_plot_area_of_all_files_in_folder(file_list, x, y)
    tif_preprocessing_utils.concat_tif_in_folder(
        file_list, folder_to_search_for_tif_files, output_filename
    )

    # Analyze
    if do_analysis:
        analysis_tif_filename = analysis.do_analysis(output_filename, do_calculation_of_control_values=False, rewrite_tif=False)
        print("Analysis done.")

    # Post processing
    if do_postprocessing:
        cropped_output_tif_filename = postprocess.crop_tif(
            analysis_tif_filename, center_x, center_y, 2500 / 2
        )
        postprocess.write_geojson_polygons_from_tif_to_file(cropped_output_tif_filename, output_format="FlatGeobuf")
        print("Postprocess done.")
