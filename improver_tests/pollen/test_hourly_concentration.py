# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.


import numpy as np
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from improver.metadata.constants import FLOAT_DTYPE
from improver.pollen.hourly_concentration import PollenHourlyConcentration

pollen_data = {
    "weed": np.array(
        [[0.0, 2.191227039816113e-07, 0.00000011], [0.000000073, 5.6e-08, 4.4e-08]]
    ),
    "alder": np.array(
        [
            [0.0, 9.3936263e-13, 4.766694816638051e-10],
            [1.5695971e-29, 1.2977890e-12, 4.6269901e-22],
        ]
    ),
}
EXPECTED = {
    "weed": np.array(
        [[0.0, 190.484086385, 95.623361348], [63.459139804, 48.680983959, 38.24934454]]
    ).astype(FLOAT_DTYPE),
    "alder": np.array(
        [[0.0, 0.000143524, 0.072829732], [0.0, 0.000198288, 0.0]]
    ).astype(FLOAT_DTYPE),
}


def get_input_cubes(pollen_name: str) -> Cube:
    """Get a test Cube of "raw" hourly concentration for weed Pollen.

    The input cube has sufficient structure to test all paths through
    code being tested.
    Args:
        pollen_name:
            The name of the pollen taxa to be used in the cube metadata.
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
        pollen_data[pollen_name],
        units="g / m3",
        dim_coords_and_dims=[(latitude_coord, 0), (longitude_coord, 1)],
    )
    cube.attributes.update({"taxa": pollen_name, "quantity": "Concentration"})
    cube.add_aux_coord(height_coord)
    return cube


def test_process():
    for pollen_name in pollen_data.keys():
        cube = get_input_cubes(pollen_name)
        plugin = PollenHourlyConcentration()
        output_cube = plugin.process(cube)
        np.testing.assert_array_almost_equal(output_cube.data, EXPECTED[pollen_name])


def test_invalid_taxa():
    cube = get_input_cubes("weed")
    cube.attributes["taxa"] = "invalid_pollen"
    plugin = PollenHourlyConcentration()
    try:
        plugin.process(cube)
    except ValueError as err:
        assert str(err) == "Pollen taxa invalid not handled"


def test_scaling_factor():
    scaling_factors_dict = {
        "weed": 213.0,
        "alder": 14.6,
    }
    for taxa, scaling_factor in scaling_factors_dict.items():
        cube = get_input_cubes(taxa)
        plugin = PollenHourlyConcentration(scaling_factors_dict)
        output_cube = plugin.process(cube)
        expected_scaled_data = (EXPECTED[taxa] * scaling_factor).astype(FLOAT_DTYPE)
        np.testing.assert_array_almost_equal(output_cube.data, expected_scaled_data)
