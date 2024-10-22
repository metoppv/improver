# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from unittest.mock import patch, sentinel

from improver.psychrometric_calculations.wet_bulb_temperature import (
    MetaWetBulbFreezingLevel,
)


class HaltExecution(Exception):
    pass


@patch("improver.psychrometric_calculations.wet_bulb_temperature.as_cube")
def test_as_cube_called(mock_as_cube):
    mock_as_cube.side_effect = HaltExecution
    try:
        MetaWetBulbFreezingLevel()(sentinel.cube)
    except HaltExecution:
        pass
    mock_as_cube.assert_called_once_with(sentinel.cube)
