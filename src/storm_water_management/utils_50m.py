"""Utils for lantmäteriets 50 m elevation data."""

import json

import numpy as np


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
