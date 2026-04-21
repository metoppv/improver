# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the TemperatureSaturatedAirParcel plugin"""

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


@pytest.fixture
def temperature_cube_fixture() -> Cube:
    """Set up a r, y, x cube of temperature data"""
    data = np.full((2, 2, 2), fill_value=300, dtype=np.float32)
    temperature_cube = set_up_variable_cube(
        data,
        name="air_temperature_at_condensation_level",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return temperature_cube


@pytest.fixture
def pressure_cube_fixture() -> Cube:
    """Set up a r, y, x cube of pressure data"""
    data = np.full((2, 2, 2), fill_value=1e5, dtype=np.float32)
    pressure_cube = set_up_variable_cube(
        data,
        name="air_pressure_at_condensation_level",
        units="Pa",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return pressure_cube


def test_initialisation():
    # Test init requires no params
    result = TemperatureSaturatedAirParcel()
    assert type(result) is TemperatureSaturatedAirParcel


def test_process(temperature_cube_fixture, pressure_cube_fixture):
    # Test init will process cubes when provided as standard improver-type plugin
    test_class = TemperatureSaturatedAirParcel()(
        temperature_cube_fixture, pressure_cube_fixture
    )
    assert (
        test_class.name()
        == "parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level"
    )
