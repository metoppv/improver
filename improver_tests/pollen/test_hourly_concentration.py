# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.


import numpy as np
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from improver.pollen.hourly_concentration import PollenHourlyConcentration

weed_pollen_data = np.array(
    [[0.0, 2.191227039816113e-07, 0.00000011], [0.000000073, 5.6e-08, 4.4e-08]]
)
WEED_EXPECTED = np.array(
    [[0.0, 190.484086385, 95.623361348], [63.459139804, 48.680983959, 38.24934454]]
)


def get_input_cubes(pollen_name: str) -> Cube:
    """Get a test Cube of "raw" hourly concentration for weed Pollen.

    The input cube has sufficient structure to test all paths through
    code being tested.
    Args:
        pollen_name:
            The name of the pollen species to be used in the cube metadata.
    Returns:
        Cube:
            An iris Cube with the structure and metadata expected by the plugin.
    """

    # Latitude, Longitude coordinates for the test cube to cover a 2x3 grid
    # around Exeter Airport Latitude: 50° 44' 2.39" N Longitude: -3° 24' 29.99" W
    cube_latitudes = np.array([50.732, 50.733])
    cube_longitudes = np.array([-3.416, -3.415, -3.414])

    # create iris DimCoords for latitude and longitude
    latitude_coord = DimCoord(
        cube_latitudes,
        standard_name="latitude",
        long_name="latitude",
        var_name="latitude",
        units="degrees",
    )
    longitude_coord = DimCoord(
        cube_longitudes,
        standard_name="longitude",
        long_name="longitude",
        var_name="longitude",
        units="degrees",
    )

    # create iris AuxCoord for "height" that spans all points in the cube
    height_coord = AuxCoord(
        np.array([10]),
        standard_name="height",
        long_name="height",
        var_name="height",
        units="m",
    )

    # create data cube including the latitude and longitude coordinates
    cube = Cube(
        weed_pollen_data,
        units="g / m3",
        dim_coords_and_dims=[(latitude_coord, 0), (longitude_coord, 1)],
    )
    cube.attributes.update({"species": pollen_name, "quantity": "Concentration"})
    cube.add_aux_coord(height_coord)
    return cube


def test_process():
    cube = get_input_cubes("weed_pollen")
    plugin = PollenHourlyConcentration()
    output_cube = plugin.process(cube)
    np.testing.assert_array_almost_equal(output_cube.data, WEED_EXPECTED)


def test_invalid_species():
    cube = get_input_cubes("weed_pollen")
    cube.attributes["species"] = "invalid_pollen"
    plugin = PollenHourlyConcentration()
    try:
        plugin.process(cube)
    except ValueError as err:
        assert str(err) == "Pollen species invalid_pollen not handled"


def test_scaling_factor():
    cube = get_input_cubes("weed_pollen")
    plugin = PollenHourlyConcentration()
    scaling_factors_dict = {
        "weed_pollen": [1.0, 213.0]
    }  # Use a scaling factor of 213 for weed pollen
    output_cube = plugin.process(cube, scaling_factors_dict)
    expected_scaled_data = WEED_EXPECTED * 213
    np.testing.assert_array_almost_equal(output_cube.data, expected_scaled_data)
