# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.


import numpy as np
from iris.cube import Cube

from improver.pollen.hourly_concentration import PollenHourlyConcentration

weed_pollen_data = np.array(
    [[0.0, 2.191227039816113e-07, 0.00000011], [0.000000073, 5.6e-08, 4.4e-08]]
)
WEED_EXPECTED = np.array(
    [[0.0, 190.484086385, 95.623361348], [63.459139804, 48.680983959, 38.24934454]]
)


def get_input_cubes(pollen_name: str) -> Cube:
    """Get a Cube of "raw" hourly concentration for weed Pollen."""
    # create data cube
    cube = Cube(
        weed_pollen_data,
        units=1,
    )
    cube.attributes.update({"species": pollen_name, "quantity": "Concentration"})
    return cube


def test_process():
    cube = get_input_cubes("weed_pollen")
    plugin = PollenHourlyConcentration()
    output_cube = plugin.process(cube)
    np.testing.assert_array_almost_equal(output_cube.data, WEED_EXPECTED)
