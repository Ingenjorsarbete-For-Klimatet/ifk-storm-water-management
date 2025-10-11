"""Utils."""

import numpy as np
from PIL import Image
from pyproj import Transformer
from whitebox_workflows import WbEnvironment


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


def get_tif_as_np_array(filename: str) -> np.array:
    """Transform tif raster to numpy array.

    Args:
        filename: name of tif file

    Retrun:
        raster as numpy array
    """
    im = Image.open(filename)
    return np.array(im)


def transform_epsg(dem, epsg_in: int = 5845, epsg_out: int = 4326):
    """Transform raster.

    Args:
        dem: raster with epsg_in
        epsg_in: epsg code in
        epsg_out: epsg code out

    Retrun:
        raster with new epsg code
    """
    transformer = Transformer.from_crs(
        "EPSG:" + str(epsg_in), "EPSG:" + str(epsg_out), always_xy=True
    )
    lower_lon, lower_lat = transformer.transform(dem.configs.west, dem.configs.south)
    upper_lon, upper_lat = transformer.transform(dem.configs.east, dem.configs.north)

    # write out config
    out_configs = dem.configs
    out_configs.east = upper_lon
    out_configs.north = upper_lat
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

    print(f"bounds: [{lower_lon}, {lower_lat}, {upper_lon}, {upper_lat}]")

    return dem_transformed


def get_tif_from_np_array(dem, tif_as_array: np.array):
    """Write over dem with np array values.

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


def saturated_upper_limit(dem, upper_limit: float = 1.0):
    """Set upper limit of dem.

    Args:
        dem: raster
        upper_limit: saturation limit.

    Retrun:
        raster
    """
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            dem[row, col] = min(upper_limit, dem[row, col])

    return dem


def write_to_png(filename: str, output_filename: str, lower_limit: float = 0.1) -> None:
    """Write dem file to png. Values lower than lower_limit is set to nan.

    Args:
        filename: input tif file
        output_filename: output png file
        lower_limit: Limit for transparancy
    """
    import matplotlib.pyplot as plt
    import rasterio
    from PIL import Image

    with rasterio.open(filename) as src:
        data = src.read(1)

    # make a mask for transparent pixels
    mask = data < lower_limit

    # Normalize values and apply color map
    normed = (data - data.min()) / (data.max() - data.min())
    normed[mask] = np.nan
    cmap = plt.get_cmap("viridis")  # t.ex. "viridis", "terrain", "plasma"
    rgba = cmap(normed)
    rgba = (rgba * 255).astype(np.uint8)
    rgba[..., 3] = np.where(mask, 0, 255)

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(output_filename)


def write_to_png_alpha(
    filename: str, output_filename: str, alpha_val: float = 0.75
) -> None:
    """Write dem file to png.

    Args:
        filename: input tif file
        output_filename: output png file
        alpha_val: alpha color
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    img = Image.open(filename).convert("L")
    arr = np.array(img)

    # Normalize  to color map
    norm = mcolors.Normalize(vmin=arr.min(), vmax=arr.max())
    cmap = plt.get_cmap("viridis")
    colored = cmap(norm(arr))  # ger RGBA float i [0,1]
    colored[..., 3] = alpha_val
    colored_img = (colored * 255).astype(np.uint8)
    out = Image.fromarray(colored_img, mode="RGBA")
    out.save(output_filename, "PNG")


if __name__ == "__main__":
    # write_to_png_alpha("/home/chris/repos/storm_temp/data/1m/63950_3150_25.tif", "elevation.png")

    filename_path = "/home/chris/repos/storm_temp/data/1m"
    filename = "63950_3150_25.tif"
    wbe = WbEnvironment()
    wbe.verbose = True
    wbe.working_directory = filename_path
    dem = wbe.read_raster(filename)

    transform_epsg(dem, 3006, 4326)
