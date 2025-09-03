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

def transform_epsg(dem, epsg_in: int = 5845, epsg_out: int = 4326):
    """Transform raster

    Args:
        dem: raster with epsg_in
        epsg_in: epsg code in
        epsg_out: epsg code out

    Retrun:
        raster with new epsg code
    """
    transformer = Transformer.from_crs("EPSG:" + str(epsg_in), "EPSG:" + str(epsg_out), always_xy=True)
    lower_lon, lower_lat  = transformer.transform(dem.configs.west, dem.configs.south)
    upper_lon, upper_lat  = transformer.transform(dem.configs.east, dem.configs.north)

    # write out config
    out_configs = dem.configs
    out_configs.east = upper_lon
    out_configs.north =  upper_lat
    out_configs.west = lower_lon
    out_configs.south = lower_lat
    out_configs.epsg_code = epsg_in
    out_configs.resolution_x = (upper_lat - lower_lat) / (
        dem.configs.east - dem.configs.west
    )
    out_configs.resolution_y = (upper_lon - lower_lon) / (
        dem.configs.north - dem.configs.south
    )

    wbe = WbEnvironment()
    dem_transformed = wbe.new_raster(out_configs)
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            dem_transformed[row, col] = dem[row, col]

    return dem_transformed


def get_tif_from_np_array(dem, tif_as_array: np.array):
    """Write over dem with np array values

    Args:
        dem: raster
        tif_as_array: name of tif file

    Retrun:
        raster overwritten
    """
    num_rows, num_cols = tif_as_array.shape

    for row in range(num_rows):
        for col in range(num_cols):
            dem[row, col] = float(tif_as_array[row, col])

    return dem

def saturated_upper_limit(dem, upper_limit: float = 1.):
    """Set upper limit of dem

    Args:
        dem: raster
        tif_as_array: name of tif file

    Retrun:
        raster
    """
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            dem[row, col] = min(upper_limit, dem[row, col])

    return dem


def write_to_png(filename: str, output_filename: str, lower_limit: float = 0.1) -> None:
    """Write dem file to png.

    Args:
        filename: input tif file
        output_filename: output png file
        lower_limit: Limit for transparancy
    """

    import rasterio
    from PIL import Image
    import matplotlib.pyplot as plt

    with rasterio.open(filename) as src:
        data = src.read(1)

    # make a mask for transparent pixels
    mask = data < lower_limit

    # Normalize values
    normed = (data - data.min()) / (data.max() - data.min())
    normed[mask] = np.nan

    # Välj colormap från matplotlib
    cmap = plt.get_cmap("viridis")  # t.ex. "viridis", "terrain", "plasma"

    # apply color map colormap -> RGBA (0–1 floats)
    rgba = cmap(normed)
    rgba = (rgba * 255).astype(np.uint8)
    rgba[..., 3] = np.where(mask, 0, 255)

    # Save file
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(output_filename)
