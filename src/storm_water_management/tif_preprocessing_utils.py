"""Utils for preprocessing of tif files."""

import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.plot import show
from shapely.geometry import Polygon


def concat_tif_in_folder(
    files: list,
    folder: str,
    output_filename: str,
    plot_merge: bool = False,
) -> None:
    """Concat tif files in folder.

    Args:
        files: list of tif files to merge
        folder: folder
        output_filename: Filename of merged file
        plot_merge: plot merge (True/False - default True)
    """
    # files = get_all_tif_files_recursively(folder)

    print(f"Number of files to merge: {len(files)}")

    if not files:
        raise ValueError("No .tif-files in list")

    src_files = [rasterio.open(os.path.join(folder, f)) for f in files]

    mosaic, out_trans = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        }
    )

    # folder_name = os.path.basename(os.path.normpath(folder))
    # if len(output_filename) == 0:
    #    output_filename = os.path.join(folder, f"{folder_name}.tif")

    with rasterio.open(output_filename, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files:
        src.close()

    if plot_merge:
        with rasterio.open(output_filename) as src:
            fig, ax = plt.subplots(figsize=(8, 6))
            show(src, ax=ax, cmap="viridis")
            plt.show()

        src.close()

    return output_filename


def get_all_tif_files_recursively(folder: str) -> list:
    """Get all tif files in folder recursively.

    Args:
        folder: folder to search in.
    Retrun:
        list of tif files
    """
    files = [
        os.path.join(root, f)
        for root, _, filenames in os.walk(folder)
        for f in filenames
        if f.lower().endswith(".tif")
    ]
    return files


def is_tif_coordinates_closer_than_limit(
    x: float, y: float, bounds, limit: float
) -> bool:
    """Check if tif file is limit close to (x,y).

    Args:
        x: x coord
        y: y coord
        bounds: bounds of tif
        limit: disance from mid point
    Returns:
        True if tif limit close to (x,y)
    """
    dx = max(bounds.left - x, 0, x - bounds.right)
    dy = max(bounds.bottom - y, 0, y - bounds.top)
    return (dx < limit) & (dy < limit)


def get_sweref99_coordinate_from_wgs84(lon: float, lat: float) -> tuple:
    """Get sweref99 coordinate from wgs84 coordinate.

    Args:
        lon: longitudinal position
        lat: lateral position
    Returns:
        sweref99 coordinate
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5845", always_xy=True)
    x, y = transformer.transform(lat, lon)
    return x, y


def filter_files_by_distance(files: list, x: float, y: float, limit: float) -> list:
    """Filter files by distance.

    Args:
        files: list of tif files
        x: x coord
        y: y coord
        limit: disance from mid point
    Returns:
        list of files
    """
    filtered_files = []
    for f in files:
        with rasterio.open(f) as src:
            bounds = src.bounds
            if is_tif_coordinates_closer_than_limit(x, y, bounds, limit):
                filtered_files.append(f)

    return filtered_files


def save_and_plot_area_of_all_files_in_folder(files, x, y) -> None:
    """Save and plot area off files.

    Args:
        files: list of tif files
        x: x coord to plot
        y: y coord to plot
    """
    polygons = []

    # create polygons of file bounds
    for f in files:
        with rasterio.open(f) as src:
            bounds = src.bounds
            poly = Polygon(
                [
                    (bounds.left, bounds.bottom),
                    (bounds.left, bounds.top),
                    (bounds.right, bounds.top),
                    (bounds.right, bounds.bottom),
                    (bounds.left, bounds.bottom),
                ]
            )
            polygons.append({"geometry": poly, "file": f})

    print(f"Number of tif-files: {len(files)}")
    gdf = gpd.GeoDataFrame(polygons, crs=src.crs)  # use sweref99
    gdf.to_file("tiff_bounds.geojson", driver="GeoJSON")

    ax = gdf.plot(edgecolor="red", facecolor="none", figsize=(8, 8))
    ax.set_title("TIFF File Extents")
    ax.plot([x], [y], "o")
    plt.show()


def parse_and_run() -> None:
    """Parse and run."""
    parser = argparse.ArgumentParser(description="Concat all files in folder to 1 tif")
    parser.add_argument("folder", help="Folder including tif files")
    args = parser.parse_args()
    concat_tif_in_folder(args.folder)


if __name__ == "__main__":
    parse_and_run()
