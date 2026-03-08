"""Utils for geojson."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from shapely.geometry import Point, shape


def write_geojson_polygons_from_tif_to_file(
    tif_filename,
    plot_polynomials: bool = False,
    output_format: str = "GeoJSON",
    apply_dissolve: bool = False,
) -> None:
    """Write tif file to geojson polygons file.

    Args:
        tif_filename: input tif file
        plot_polynomials: plot polynomials
        output_format: output format, either "GeoJSON" or "FlatGeobuf"
        apply_dissolve: apply disolve on output
    """
    with rasterio.open(tif_filename) as src:
        img = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs

    # Define bins for depth
    bins = [0.1, 0.2, 0.5, 1.0, 1000.0]
    labels = ["0.1-0.2 m", "0.2-0.5 m", "0.5-1 m", "1 <"]
    classified = np.digitize(img, bins, right=False).astype("int16")

    # Mask to ignore 0
    mask = img > 0

    # Extract polygons and build geodataframe directly
    geoms = []
    vals = []

    for s, val in features.shapes(classified, mask=mask, transform=transform):
        val = int(val)
        if 0 < val <= len(labels):
            geoms.append(shape(s))
            vals.append(labels[val - 1])

    gdf = gpd.GeoDataFrame({"depth_class": vals}, geometry=geoms, crs=crs)

    # Write file with specified format
    file_ext = "fgb" if output_format == "FlatGeobuf" else "geojson"

    remove_small_polygons = False
    if remove_small_polygons:
        min_area = 1
        referens = gdf.union_all()
        mask_large_areas = gdf.geometry.area > min_area
        mask_within = gdf.geometry.within(referens)
        gdf = gdf[mask_large_areas | mask_within]

    if apply_dissolve:
        gdf = gdf.dissolve(by="depth_class")
        gdf = gdf[gdf.geometry.notna()]

    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(tif_filename[:-4] + f"_polygons.{file_ext}", driver=output_format)

    if plot_polynomials:
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(column="depth_class", cmap="viridis", legend=True, ax=ax)
        plt.title("Vattendjup")
        plt.show()


def write_geojson_points_from_tif_to_file(
    tif_filename, plot_points: bool = False
) -> None:
    """Write tif file to geojson points file.

    Args:
        tif_filename: input tif file
        plot_points: plot polynomials
    """
    with rasterio.open(tif_filename) as src:
        data = src.read(1)
        transform = src.transform

    geoms = []
    values = []

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if value > 0.1 and value != src.nodata:
                x, y = rasterio.transform.xy(transform, row, col)
                geoms.append(Point(x, y))
                values.append(float(value))

    gdf = gpd.GeoDataFrame({"depth": values}, geometry=geoms, crs=src.crs)
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(tif_filename[:-4] + "_points.geojson", driver="GeoJSON")
    if plot_points:
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(column="depth", cmap="viridis", legend=True, ax=ax)
        plt.title("Vattendjup")
        plt.show()


def crop_tif(tif_path, x, y, distance_limit, output_filename: str = "") -> str:
    """Crop tif.

    Args:
        tif_path: tif file
        x: x coord
        y: y coord
        distance_limit: distance to crop from (x,y)
        output_filename: output filename

    Returns:
        New file path
    """
    with rasterio.open(tif_path) as src:
        left = x - distance_limit
        right = x + distance_limit
        bottom = y - distance_limit
        top = y + distance_limit

        window = from_bounds(left, bottom, right, top, src.transform)
        data = src.read(window=window)

        transform = src.window_transform(window)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": data.shape[1],
                "width": data.shape[2],
                "count": data.shape[0],
                "transform": transform,
                "compress": "lzw",
                "tiled": False,
            }
        )

        out_meta.pop("blockxsize", None)
        out_meta.pop("blockysize", None)

        if output_filename:
            output_tif_filename = output_filename
        else:
            output_tif_filename = tif_path[:-4] + "_cropped.tif"

        with rasterio.open(output_tif_filename, "w", **out_meta) as dst:
            dst.write(data)

    return output_tif_filename
