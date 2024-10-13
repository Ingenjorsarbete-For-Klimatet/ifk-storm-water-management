"""Unit test."""

import richdem as rd

from storm_water_management import utils


def test_compile_data_to_geojson():
    """Test compile_data_to_geojson"""
    n_x = 2
    n_y = 2
    rda_fill_diff = rd.rdarray([[1.0, 1.0], [2.0, 2.0]], no_data=-9999)
    elevation_data = rd.rdarray([[100.0, 200], [100.0, 200]], no_data=-9999)
    accum_d8 = rd.rdarray([[3.0, 4.0], [5.0, 6.0]], no_data=-9999)
    tfw = [50.0, 0.0, 0.0, -50.0, 100.0, 200.0]

    print(rda_fill_diff[1])

    geojson_data = utils.compile_data_to_geojson(
        tfw, elevation_data, accum_d8, rda_fill_diff, n_x, n_y
    )

    geojson_data_expected_1 = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [125, 225]},
        "properties": {
            "elevation_data": 100.0,
            "water_flow": 3.0,
            "depression_filling": 1.0,
        },
    }

    assert geojson_data[0] == geojson_data_expected_1
