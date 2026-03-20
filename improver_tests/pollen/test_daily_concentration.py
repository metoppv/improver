# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.


import numpy as np
from iris.cube import Cube, CubeList

from improver.pollen.daily_concentration import PollenDailyConcentration

pollen_hourly_concentrations_dict = {
    "weed_pollen": {
        1: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        2: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        3: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        4: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        5: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        6: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        7: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        8: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        9: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        10: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        11: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        12: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        13: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        14: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        15: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        16: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        17: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        18: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        19: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        20: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        21: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        22: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        23: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        24: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
    },
    "insufficient_pollen": {
        1: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        2: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        3: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        4: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        5: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        6: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        7: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        8: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        9: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        10: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        11: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
        12: np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]]),
    },
}

WEED_AVERAGE = np.array([[0.0, 0.01, 29.0], [50.0, 131.2, 409.0]])
INSUFFICIENT_AVERAGE = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])


def get_input_cubes(pollen_name: str) -> CubeList:
    """Get a CubeList of hourly concentration Cubes for a Pollen species."""

    cubes = CubeList()
    pollen_values_dict = pollen_hourly_concentrations_dict[pollen_name]
    # for each item in INPUT_DATA create a Cube that has the given data
    for hour, pollen_values in pollen_values_dict.items():
        # create data cube
        cube = Cube(
            pollen_values,
            units=1,
        )
        cube.attributes.update({"species": pollen_name, "quantity": "Concentration"})
        cubes.append(cube)

    return cubes


def test_process():
    cubes = get_input_cubes("weed_pollen")
    plugin = PollenDailyConcentration()
    output_cube = plugin.process(cubes)
    print(output_cube)
    print("------------")
    print(output_cube.data)
    assert output_cube.data.all() == WEED_AVERAGE.all()


def test_insufficient_data():
    cubes = get_input_cubes("insufficient_pollen")
    plugin = PollenDailyConcentration()
    output_cube = plugin.process(cubes)
    print(output_cube)
    print("------------")
    print(output_cube.data)
    assert output_cube.data.all() == INSUFFICIENT_AVERAGE.all()
