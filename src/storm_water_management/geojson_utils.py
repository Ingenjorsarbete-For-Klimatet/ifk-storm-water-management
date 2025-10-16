"""Utils for geojson."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import Point, shape


def write_geojson_polygons_from_tif_to_file(
    tif_filename, plot_polynomials: bool = False
) -> None:
    """Write tif file to geojson polygons file.

    Args:
        tif_filename: input tif file
        plot_polynomials: plot polynomials
    """
    with rasterio.open(tif_filename) as src:
        img = src.read(1).astype(float)
        transform = src.transform

    # Define bins for depth
    bins = [0.1, 0.2, 0.5, 1.0, 1000.0]
    labels = ["0.1-0.2 m", "0.2-0.5 m", "0.5-1 m", "1.0 <"]
    classified = np.digitize(img, bins, right=False).astype("int16")

    # Mask to ignore 0
    mask = img > 0

    # Extract polynomials
    results = (
        {"properties": {"class": labels[int(val) - 1]}, "geometry": s}
        for s, val in features.shapes(classified, mask=mask, transform=transform)
        if int(val) > 0 and int(val) <= len(labels)
    )

    geoms = []
    vals = []
    for feat in results:
        geoms.append(shape(feat["geometry"]))
        vals.append(feat["properties"]["class"])
    gdf = gpd.GeoDataFrame({"depth_class": vals}, geometry=geoms, crs=src.crs)
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(tif_filename[:-4] + "_polygons" + ".geojson", driver="GeoJSON")

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
    gdf.to_file("output.geojson", driver="GeoJSON")
    gdf.to_file(tif_filename[:-4] + "_points.geojson", driver="GeoJSON")
    if plot_points:
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(column="depth", cmap="viridis", legend=True, ax=ax)
        plt.title("Vattendjup")
        plt.show()


if __name__ == "__main__":
    filename_path = "/home/chris/repos/storm_temp/data/1m"
    filename = filename_path + "/depression_depth_saturated.tif"
    write_geojson_polygons_from_tif_to_file(filename, plot_polynomials=False)
    write_geojson_points_from_tif_to_file(filename, plot_points=False)
