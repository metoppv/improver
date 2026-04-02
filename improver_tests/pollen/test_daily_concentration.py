# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.


import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.metadata.constants import FLOAT_DTYPE
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
        24: np.array([[0.0, 0.01, 29.0], [50.0, 100.0, 200.0]]),
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

WEED_AVERAGE = np.array([[0.0, 0.01, 29.0], [50.0, 129.9, 400.29166667]]).astype(
    FLOAT_DTYPE
)
INSUFFICIENT_AVERAGE = np.array(
    [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
).astype(FLOAT_DTYPE)


def get_input_cubes(pollen_name: str) -> CubeList:
    """Get a CubeList of hourly concentration Cubes for a Pollen taxa."""

    cubes = CubeList()
    pollen_values_dict = pollen_hourly_concentrations_dict[pollen_name]
    # for each item in INPUT_DATA create a Cube that has the given data
    for hour, pollen_values in pollen_values_dict.items():
        # create data cube
        cube = Cube(
            pollen_values,
            units=1,
        )
        cube.attributes.update({"taxa": pollen_name, "quantity": "Concentration"})
        cubes.append(cube)

    return cubes


def test_process():
    cubes = get_input_cubes("weed_pollen")
    plugin = PollenDailyConcentration()
    output_cube = plugin.process(cubes)
    np.testing.assert_array_almost_equal(output_cube.data, WEED_AVERAGE)


def test_insufficient_data():
    cubes = get_input_cubes("insufficient_pollen")
    plugin = PollenDailyConcentration()
    with pytest.warns(
        UserWarning,
        match="Expected at least 23 cubes for hourly data, but got 12. Output values set to NaN.",
    ):
        output_cube = plugin.process(cubes)
    np.testing.assert_array_almost_equal(output_cube.data, INSUFFICIENT_AVERAGE)
