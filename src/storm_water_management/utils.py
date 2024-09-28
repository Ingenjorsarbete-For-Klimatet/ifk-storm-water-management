import numpy as np
import rasterio
import richdem
import json

def get_coordinate_center_points_from_tfw(
    tfw: list,
    elev_data: richdem.rdarray,
    flow_data: richdem.rdarray,
    depression_filling_data: richdem.rdarray,
    n_points_x: int,
    n_points_y: int,
) -> list:
    """Get coordinates for centerpoints

    Args:
        tfw:

    Retrun:
        elevation data, flow data, and depression fill
    """
    size_x = tfw[0]
    size_y = tfw[3]
    upper_left_x = tfw[4] + size_x / 2
    upper_left_y = tfw[5] - size_y / 2

    points = []
    for row in range(n_points_y):
        for col in range(n_points_x):
            center_x = upper_left_x + (col * size_x)
            center_y = upper_left_y - (row * size_y)
            points.append(
                {
                    "type": "Feature", 
                    "geometry": {"type": "Point", "coordinates": [center_x, center_y]},
                    "properties":{
                        "elevation_data": float(elev_data[row][col]),
                        "water_flow": float(flow_data[row][col]),
                        "depression_filling": float(depression_filling_data[row][col])
                        },
                    }
            )

    return points


def get_elevation_data_from_tif(filename: str) -> np.array:
    """Get elevation data

    Args:
        filename: name of tif file

    Retrun:
        elevation data
    """
    with rasterio.open(filename) as f:
        elevation_data = f.read()

    return elevation_data[0]

def get_coordinates_from_tfw(filename: str) -> np.array:
    """Get elevation data

    Args:
        filename: name of tif file

    Retrun:
        elevation data
    """
    with open(filename, "r") as f:

        tfw = [float(x) for x in f.readlines()]

    return tfw

def save_to_geojson(data: list, filname: str):
    """Get elevation data

    Args:
        data: data to save
        filename: name of file to save

    Retrun:
        elevation data
    """
    with open(f"{filname}.json", "w") as f:
        json.dump(data, f)
