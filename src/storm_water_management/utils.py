"""Utils."""

import json

import numpy as np
from PIL import Image
from pyproj import Transformer
from whitebox_workflows import WbEnvironment


def get_coordinates_from_tfw(filename: str) -> list:
    """Get coordinates from tfw.

    Args:
        filename: name of tif file

    Retrun:
        elevation data
    """
    with open(filename, "r") as f:
        tfw = [float(x) for x in f.readlines()]

    return tfw


def save_to_geojson(data: list, filename: str) -> None:
    """Save to geojson.

    Args:
        data: data to save
        filename: name of file to save
    """
    with open(f"{filename}.json", "w") as f:
        json.dump(data, f)


def transform_geojson_points_to_polygons(
    filename: str, output_filename: str = "kvadrater.geojson"
) -> None:
    """Transform points to polygon.

    Args:
        filename: name of geojson file
        output_filename: name of output file
    Retrun:
        None
    """
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.read_file(filename)

    # Robust beräkning av cellstorlek
    x_coords = sorted([p.x for p in gdf.geometry])
    x_diffs = np.diff(x_coords)
    cell_size = np.median(x_diffs[x_diffs > 0])  # eller sätt värde manuellt

    print(f"Beräknad cellstorlek: {cell_size}")

    def create_square_around_point(point, size):
        half = size / 2
        return box(point.x - half, point.y - half, point.x + half, point.y + half)

    gdf["geometry"] = gdf.geometry.apply(
        lambda p: create_square_around_point(p, cell_size)
    )

    gdf.to_file(output_filename, driver="GeoJSON")


def info(dem) -> None:
    """Print meta data of tif file.

    Args:
        dem: tif file
    Retrun:
        None
    """
    print(f"Rows: {dem.configs.rows}")
    print(f"Columns: {dem.configs.columns}")
    print(f"Resolution (x direction): {dem.configs.resolution_x}")
    print(f"Resolution (y direction): {dem.configs.resolution_y}")
    print(f"North: {dem.configs.north}")
    print(f"South: {dem.configs.south}")
    print(f"East: {dem.configs.east}")
    print(f"West: {dem.configs.west}")
    print(f"Min value: {dem.configs.minimum}")
    print(f"Max value: {dem.configs.maximum}")
    print(f"EPSG code: {dem.configs.epsg_code}")  # 0 if not set
    print(f"Nodata value: {dem.configs.nodata}")
    print(f"Data type: {dem.configs.data_type}")
    print(f"Photometric interpretation: {dem.configs.photometric_interp}")


# transform_geojson_points_to_polygons("/home/chris/repos/ifk-storm-water-management/notebooks/FillDepressionsBestCoast.geojson")


def get_tif_as_np_array(filename_path: str, filename: str) -> np.array:
    """Transform tif raster to numpy array.

    Args:
        filename_path: name of directory including file
        filename: name of tif file

    Retrun:
        raster as numpy array
    """
    im = Image.open(filename_path + "/" + filename)
    return np.array(im)


def get_tif_from_np_array(dem, tif_as_array: np.array):
    """Get tif raster from numpy array.

    Args:
        dem: raster
        tif_as_array: name of tif file

    Retrun:
        raster
    """
    out_configs = dem.configs

    # verkar vara EPSG:5845 i.e.,  Horizontal CRS: EPSG:3006 Vertical CRS: EPSG:5613
    # a little bit unclear if 3006 or 5845
    transformer = Transformer.from_crs("EPSG:5845", "EPSG:4326")
    lower_lat, lower_lon = transformer.transform(dem.configs.west, dem.configs.south)
    upper_lat, upper_lon = transformer.transform(dem.configs.east, dem.configs.north)

    out_configs.east = upper_lat
    out_configs.north = upper_lon
    out_configs.west = lower_lat
    out_configs.south = lower_lon
    out_configs.epsg_code = 4326
    out_configs.resolution_x = (upper_lat - lower_lat) / (
        dem.configs.east - dem.configs.west
    )
    out_configs.resolution_y = (upper_lon - lower_lon) / (
        dem.configs.north - dem.configs.south
    )
    # min/max will be set when dem saved?
    # out_configs.minimum = -10
    # out_configs.maximum = 200

    wbe = WbEnvironment()
    # wbe.verbose = True
    # wbe.working_directory = filename_path
    topo = wbe.new_raster(out_configs)

    num_rows, num_cols = tif_as_array.shape

    for row in range(num_rows):
        for col in range(num_cols):
            topo[row, col] = float(tif_as_array[row, col])
            # topo[row, col] = float(tif_as_array[row, col])

    return topo
