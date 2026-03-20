# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.


import numpy as np
from iris.cube import Cube, CubeList

from improver.pollen.daily_index import PollenDailyIndex

POLLEN_DATA = {
    "grass_pollen": np.array([[0, 0, 1], [1, 2, 1]]),
    "birch_pollen": np.array([[1, 2, 3], [0, 5, 4]]),
    "oak_pollen": np.array([[0, 2, 4], [4, 5, 2]]),
    "hazel_pollen": np.array([[1, 1, 5], [2, 4, 1]]),
    "alder_pollen": np.array([[0, 2, 3], [5, 2, 3]]),
    "ash_pollen": np.array([[0, 0, 1], [1, 2, 1]]),
    "plane_pollen": np.array([[4, 2, 1], [3, 2, 3]]),
    "weed_pollen": np.array([[2, 3, 3], [2, 2, 4]]),
}
EXPECTED_DATA = np.array([[4, 3, 5], [5, 5, 4]])


def get_input_cubes() -> CubeList:
    """Create a CubeList of simple input cubes for Pollen Daily Index tests.

    All cubes have 2-D arrays of integers, with values from 0 to 5.

    Returns:
        CubeList for Pollen types given in POLLEN_NAMES above.
    """

    cubes = CubeList()
    # for each item in POLLEN_DATA create a Cube that has the given data
    for pollen_name, data in POLLEN_DATA.items():
        # create data cube
        cube = Cube(
            data,
            units=1,
        )
        cubes.append(cube)

    return cubes


def test_process():
    cubes = get_input_cubes()
    plugin = PollenDailyIndex()
    cube = plugin.process(cubes)
    assert cube.data.all() == EXPECTED_DATA.all()
