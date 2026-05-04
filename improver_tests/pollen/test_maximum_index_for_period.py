# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.


import numpy as np
from iris.cube import Cube, CubeList

from improver.pollen.pollen_maximum_index_for_period import PollenMaximumIndexForPeriod

INPUT_DATA = {
    "2026-03-22 01:00:00+00:00": {
        "grass_pollen": np.array([[0, 0, 1], [1, 2, 1]]),
        "birch_pollen": np.array([[1, 2, 3], [0, 5, 4]]),
        "oak_pollen": np.array([[0, 2, 4], [4, 5, 2]]),
        "hazel_pollen": np.array([[1, 1, 5], [2, 4, 1]]),
        "alder_pollen": np.array([[0, 2, 3], [5, 2, 3]]),
        "ash_pollen": np.array([[0, 0, 1], [1, 2, 1]]),
        "plane_pollen": np.array([[4, 2, 1], [3, 2, 3]]),
        "weed_pollen": np.array([[2, 3, 3], [2, 2, 4]]),
    },
    "2026-03-22 02:00:00+00:00": {
        "grass_pollen": np.array([[0, 5, 1], [1, 2, 1]]),
        "birch_pollen": np.array([[1, 2, 3], [0, 5, 5]]),
        "oak_pollen": np.array([[0, 2, 4], [4, 5, 2]]),
        "hazel_pollen": np.array([[1, 1, 2], [2, 4, 1]]),
        "alder_pollen": np.array([[0, 2, 3], [1, 2, 3]]),
        "ash_pollen": np.array([[0, 0, 1], [1, 2, 1]]),
        "plane_pollen": np.array([[3, 2, 1], [3, 1, 3]]),
        "weed_pollen": np.array([[2, 3, 3], [2, 5, 4]]),
    },
    "2026-03-22 03:00:00+00:00": {
        "grass_pollen": np.array([[0, 1, 1], [1, 2, 1]]),
        "birch_pollen": np.array([[1, 2, 3], [0, 3, 4]]),
        "oak_pollen": np.array([[0, 2, 4], [2, 0, 2]]),
        "hazel_pollen": np.array([[1, 1, 4], [2, 4, 1]]),
        "alder_pollen": np.array([[0, 2, 3], [1, 2, 3]]),
        "ash_pollen": np.array([[0, 0, 1], [1, 2, 1]]),
        "plane_pollen": np.array([[4, 2, 1], [3, 2, 3]]),
        "weed_pollen": np.array([[5, 4, 4], [2, 2, 5]]),
    },
}
EXPECTED_DATA = {
    "2026-03-22 01:00:00+00:00": np.array([[4, 3, 5], [5, 5, 4]]),
    "2026-03-22 02:00:00+00:00": np.array([[3, 5, 4], [4, 5, 5]]),
    "2026-03-22 03:00:00+00:00": np.array([[5, 4, 4], [3, 4, 5]]),
}


def get_input_cubes(pollen_values_dict: dict) -> CubeList:
    """Create a CubeList of simple input cubes for Pollen Index tests.

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
        cubes.append(cube)

    return cubes


def test_process():
    for datetime_key, pollen_values_dict in INPUT_DATA.items():
        cubes = get_input_cubes(pollen_values_dict)
        plugin = PollenMaximumIndexForPeriod()
        cube = plugin.process(cubes)
        assert cube.data.all() == EXPECTED_DATA[datetime_key].all()
