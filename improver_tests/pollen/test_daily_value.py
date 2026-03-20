# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.


import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.pollen.daily_value import PollenDailyValue

INPUT_DATA = {
    "grass_pollen": np.array([[0.011, 0.001, 240.1], [30.0, 51.0, 150.56]]),
    "birch_pollen": np.array([[23.4, 40.0, 0.001], [237.7, 200.0, 65.3]]),
    "oak_pollen": np.array([[0.0, 0.01, 30.0], [50.0, 200.0, 200.0000001]]),
    "hazel_pollen": np.array([[200.0, 79.2, 49.99999999], [30.000001, 0.5, 0.00000]]),
}

EXPECTED_DATA = {
    "grass_pollen": np.array([[1, 0, 4], [1, 3, 4]]),
    "birch_pollen": np.array([[1, 2, 0], [4, 4, 3]]),
    "oak_pollen": np.array([[0, 1, 2], [3, 4, 4]]),
    "hazel_pollen": np.array([[4, 3, 2], [2, 1, 0]]),
}


def get_test_cube(pollen_cube_data, species) -> CubeList:
    """Get a simple Cube for Pollen Daily Value tests.

    Returns:
        Cube for Pollen types given in INPUT_DATA or EXPECTED_DATA dicts above.
    """

    # create data cube
    cube = Cube(
        pollen_cube_data,
        units=1,
    )
    cube.attributes.update({"species": species, "quantity": "Concentration"})
    return cube


def test_process():
    for species in INPUT_DATA:
        input_cube = get_test_cube(INPUT_DATA[species], species)
        expected_cube = get_test_cube(EXPECTED_DATA[species], species)
        plugin = PollenDailyValue()
        cube = plugin.process(input_cube)
        assert cube.data.all() == expected_cube.data.all()


def test_invalid_pollen_name():
    species = "foo_pollen"
    input_pollen_data = np.array(
        [[200.0, 79.2, 49.999999999], [30.00000001, 0.5, 0.00000]]
    )
    input_cube = get_test_cube(input_pollen_data, species)
    plugin = PollenDailyValue()

    msg = f"Pollen species {species} not handled"
    with pytest.raises(ValueError, match=msg):
        plugin.process(input_cube)
