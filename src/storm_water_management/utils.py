import geojson
import numpy as np
import rasterio
import richdem


def get_coordinate_center_points_from_tfw(
    tfw: list,
    elev_data: np.array,
    flow_data: richdem.rdarray,
    n_points_x: int,
    n_points_y: int,
) -> list:
    """Get coordinates for centerpoints

    Args:
        tfw:

    Retrun:
        elevation data
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
                geojson.Feature(
                    geometry=geojson.Point((center_x, center_y)),
                    properties={
                        "elevation_data": float(elev_data[row][col]),
                        "water_flow": flow_data[row][col],
                    },
                )
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


if __name__ == "__main__":
    import numpy as np

    tif_filename = r"/home/chris/projects/ifk-storm-water-management/data/64_3_2023.tif"
    tfw_filename = tif_filename.split(".")[0] + r".tfw"

    elevation_data = get_elevation_data_from_tif(tif_filename)
    tfw = get_coordinates_from_tfw(tfw_filename)

    elev_max = np.amax(elevation_data)
    # 102.80466 , 103.67129 , 105.40455 , 106.492455, 106.93501
    print(type(float(elevation_data[0][0])))
    list(elevation_data[0][0:5])

    coordinates = get_coordinate_center_points_from_tfw(tfw, elevation_data, 20, 20)
    print(coordinates[0])

    with open("temp_file.json", "w") as f:
        geojson.dump(coordinates, f)
