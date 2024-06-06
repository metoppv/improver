# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the HumidityMixingRatio plugin"""
from unittest.mock import patch, sentinel

import numpy as np
import pytest
from iris.cube import Cube

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTES
from improver.psychrometric_calculations.psychrometric_calculations import (
    HumidityMixingRatio,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


class HaltExecution(Exception):
    pass


@patch("improver.psychrometric_calculations.psychrometric_calculations.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    mock_as_cubelist.side_effect = HaltExecution
    try:
        HumidityMixingRatio()(
            sentinel.air_temperature,
            sentinel.surface_air_pressure,
            sentinel.relative_humidity,
        )
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(
        sentinel.air_temperature,
        sentinel.surface_air_pressure,
        sentinel.relative_humidity,
    )


@pytest.fixture(name="temperature")
def temperature_cube_fixture() -> Cube:
    """Set up a r, y, x cube of temperature data"""
    data = np.full((2, 2, 2), fill_value=300, dtype=np.float32)
    temperature_cube = set_up_variable_cube(
        data, name="air_temperature", units="K", attributes=LOCAL_MANDATORY_ATTRIBUTES,
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


@pytest.fixture(name="rel_humidity")
def humidity_cube_fixture() -> Cube:
    """Set up a r, y, x cube of relative humidity data"""
    data = np.full((2, 2, 2), fill_value=1e-1, dtype=np.float32)
    humidity_cube = set_up_variable_cube(
        data,
        name="relative_humidity",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return humidity_cube


def metadata_ok(mixing_ratio: Cube, baseline: Cube, model_id_attr=None) -> None:
    """
    Checks mixing_ratio Cube long_name, units and dtype are as expected.
    Compares mixing_ratio Cube with baseline to make sure everything else matches.

    Args:
        mixing_ratio: Result of HumidityMixingRatio plugin
        baseline: A temperature or similar cube with the same coordinates and attributes.

    Raises:
        AssertionError: If anything doesn't match
    """
    assert mixing_ratio.standard_name == "humidity_mixing_ratio"
    assert mixing_ratio.units == "kg kg-1"
    assert mixing_ratio.dtype == np.float32
    for coord in mixing_ratio.coords():
        base_coord = baseline.coord(coord.name())
        assert mixing_ratio.coord_dims(coord) == baseline.coord_dims(base_coord)
        assert coord == base_coord
    for attr in MANDATORY_ATTRIBUTES:
        assert mixing_ratio.attributes[attr] == baseline.attributes[attr]
    all_attr_keys = list(mixing_ratio.attributes.keys())
    if model_id_attr:
        assert (
            mixing_ratio.attributes[model_id_attr] == baseline.attributes[model_id_attr]
        )
    mandatory_attr_keys = [k for k in all_attr_keys if k != model_id_attr]
    assert sorted(mandatory_attr_keys) == sorted(MANDATORY_ATTRIBUTES)


@pytest.mark.parametrize(
    "temperature_value, pressure_value, rel_humidity_value, expected",
    (
        (293, 100000, 1.0, 1.459832e-2),
        (293, 100000, 0.5, 7.29916e-3),
        (293, 100000, 0.1, 1.459832e-3),
        (300, 100000, 0.1, 2.23855e-3),
    ),
)
def test_basic(
    temperature,
    pressure,
    rel_humidity,
    temperature_value,
    pressure_value,
    rel_humidity_value,
    expected,
):
    """Check that for each pair of values, we get the expected result
    and that the metadata are as expected."""
    temperature.data = np.full_like(temperature.data, temperature_value)
    pressure.data = np.full_like(pressure.data, pressure_value)
    rel_humidity.data = np.full_like(rel_humidity.data, rel_humidity_value)
    result = HumidityMixingRatio()([temperature, pressure, rel_humidity])
    metadata_ok(result, temperature)
    assert np.isclose(result.data, expected, atol=1e-7).all()


@pytest.mark.parametrize("model_id_attr", ("mosg__model_configuration", None))
def test_model_id_attr(temperature, pressure, rel_humidity, model_id_attr):
    """Check that tests pass if model_id_attr is set on inputs and is applied or not"""
    temperature.attributes["mosg__model_configuration"] = "gl_ens"
    pressure.attributes["mosg__model_configuration"] = "gl_ens"
    rel_humidity.attributes["mosg__model_configuration"] = "gl_ens"
    result = HumidityMixingRatio(model_id_attr=model_id_attr)(
        [temperature, pressure, rel_humidity]
    )
    metadata_ok(result, temperature, model_id_attr=model_id_attr)
