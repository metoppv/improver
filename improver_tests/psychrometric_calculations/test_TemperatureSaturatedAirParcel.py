# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the TemperatureSaturatedAirParcel plugin"""

from unittest.mock import patch

import numpy as np
import pytest
from iris.cube import Cube

from improver.psychrometric_calculations.temperature_saturated_air_parcel import (
    TemperatureSaturatedAirParcel,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


@pytest.fixture(name="temperature")
def temperature_cube_fixture() -> Cube:
    """Set up a r, y, x cube of temperature data"""
    data = np.full((2, 2, 2), fill_value=300, dtype=np.float32)
    temperature_cube = set_up_variable_cube(
        data, name="air_temperature", units="K", attributes=LOCAL_MANDATORY_ATTRIBUTES
    )
    return temperature_cube


@pytest.fixture(name="pressure")
def pressure_cube_fixture() -> Cube:
    """Set up a r, y, x cube of pressure data"""
    data = np.full((2, 2, 2), fill_value=1e5, dtype=np.float32)
    pressure_cube = set_up_variable_cube(
        data,
        name="surface_air_pressure",
        units="Pa",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return pressure_cube


class HaltExecution(Exception):
    pass


def test_initialisation():
    result = TemperatureSaturatedAirParcel()
    assert result.pressure_level == 50000.0


@patch(
    "improver.psychrometric_calculations.temperature_saturated_air_parcel.as_cubelist"
)
def test_plugin_calls(mock_as_cubelist, temperature, pressure):
    mock_as_cubelist.side_effect = HaltExecution
    try:
        TemperatureSaturatedAirParcel()(
            temperature,
            pressure,
        )
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(
        (
            temperature,
            pressure,
        )
    )
