# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.


import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.pollen.pollen_index_for_period import PollenIndexForPeriod

INPUT_DATA = {
    "grass_pollen": np.array([[0.0, 0.0999999, 29.0], [50.0, 131.2, 409.0]]),
    "alder_pollen": np.array([[0.01, 0.010000000001, 31.0], [100.0, 50.0, 80.0]]),
    "plane_pollen": np.array([[345.0, 42.0, 25.01], [200.0, 0.0002, 100.3]]),
    "weed_pollen": np.array([[2, 3, 3], [2, 2, 4]]),
}

EXPECTED_DATA = {
    "grass_pollen": np.array([[0, 0, 1], [2, 3, 4]]),
    "alder_pollen": np.array([[1, 1, 2], [4, 2, 3]]),
    "plane_pollen": np.array([[4, 2, 1], [3, 0, 3]]),
    "weed_pollen": np.array([[5.4, 103.0, 80.0], [45.82, 0.01, 200.00000000001]]),
}

BAD_POLLEN_NAME_HOUR_DATA = {
    "2026-03-22 01:00:00+00:00": {
        "foo_pollen": np.array([[0.0, 0.0999999, 29.0], [50.0, 131.2, 409.0]]),
    },
}


def get_test_cube(pollen_cube_data, taxa) -> CubeList:
    """Get a simple Cube for Pollen Daily Value tests.

    Returns:
        Cube for Pollen types given in INPUT_DATA or EXPECTED_DATA dicts above.
    """

    # create data cube
    cube = Cube(
        pollen_cube_data,
        units=1,
    )
    cube.attributes.update({"taxa": taxa, "quantity": "Concentration"})
    return cube


def test_process():
    for taxa in INPUT_DATA:
        input_cube = get_test_cube(INPUT_DATA[taxa], taxa)
        expected_cube = get_test_cube(EXPECTED_DATA[taxa], taxa)
        plugin = PollenIndexForPeriod()
        cube = plugin.process(input_cube)
        assert cube.data.all() == expected_cube.data.all()


def test_invalid_pollen_name():
    taxa = "foo_pollen"
    input_pollen_data = np.array(
        [[200.0, 79.2, 49.999999999], [30.00000001, 0.5, 0.00000]]
    )
    input_cube = get_test_cube(input_pollen_data, taxa)
    plugin = PollenIndexForPeriod()

    msg = f"Pollen taxa {taxa} not handled"
    with pytest.raises(ValueError, match=msg):
        plugin.process(input_cube)
