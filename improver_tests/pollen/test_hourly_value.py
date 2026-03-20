# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.


import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.pollen.hourly_value import PollenHourlyValue

INPUT_DATA = {
    "2026-03-22 01:00:00+00:00": {
        "grass_pollen": np.array([[0.0, 0.0999999, 29.0], [50.0, 131.2, 409.0]]),
        "alder_pollen": np.array([[0.01, 0.010000000001, 31.0], [100.0, 50.0, 80.0]]),
        "plane_pollen": np.array([[345.0, 42.0, 25.01], [200.0, 0.0002, 100.3]]),
        "weed_pollen": np.array([[2, 3, 3], [2, 2, 4]]),
    },
}

EXPECTED_DATA = {
    "2026-03-22 01:00:00+00:00": {
        "grass_pollen": np.array([[0, 0, 1], [2, 3, 4]]),
        "alder_pollen": np.array([[1, 1, 2], [4, 2, 3]]),
        "plane_pollen": np.array([[4, 2, 1], [3, 0, 3]]),
        "weed_pollen": np.array([[5.4, 103.0, 80.0], [45.82, 0.01, 200.00000000001]]),
    },
}

BAD_POLLEN_NAME_DATA = {
    "2026-03-22 01:00:00+00:00": {
        "foo_pollen": np.array([[0.0, 0.0999999, 29.0], [50.0, 131.2, 409.0]]),
    },
}


def get_input_cubes(pollen_values_dict: dict) -> CubeList:
    """Create a CubeList of simple input cubes for Pollen Hourly Index tests.

    All cubes have 2-D arrays of integers, with values from 0 to 5.

    Returns:
        CubeList for Pollen types given in INPUT_DATA above.
    """
    cubes = CubeList()
    # for each item in INPUT_DATA create a Cube that has the given data
    for pollen_name in pollen_values_dict:
        # create data cube
        cube = Cube(
            pollen_values_dict[pollen_name],
            units=1,
        )
        cube.attributes.update({"species": pollen_name, "quantity": "Concentration"})
        cubes.append(cube)

    return cubes


def test_process():
    for datetime_key, pollen_values_dict in INPUT_DATA.items():
        cubes = get_input_cubes(pollen_values_dict)
        for cube in cubes:
            plugin = PollenHourlyValue()
            output_cube = plugin.process(cube)
            pollen_name = cube.attributes.get("species")
            expected_data = EXPECTED_DATA[datetime_key][pollen_name]
            assert output_cube.data.all() == expected_data.all()


def test_unknown_pollen_name():
    for datetime_key, pollen_values_dict in BAD_POLLEN_NAME_DATA.items():
        cubes = get_input_cubes(pollen_values_dict)
        for cube in cubes:
            plugin = PollenHourlyValue()

            msg = "Pollen species foo_pollen not handled"
            with pytest.raises(ValueError, match=msg):
                plugin.process(cube)
