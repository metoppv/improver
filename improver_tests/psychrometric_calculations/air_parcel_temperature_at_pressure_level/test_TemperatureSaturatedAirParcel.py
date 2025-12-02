# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the TemperatureSaturatedAirParcel plugin"""

import iris.util
import numpy as np
import pytest
from iris.cube import Cube

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTES
from improver.psychrometric_calculations.temperature_saturated_air_parcel import (
    TemperatureSaturatedAirParcel,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "Post-Processed Global Model Forecast on Global 10 km Standard Grid",
    "source": "Met Office Unified Model",
    "institution": "Met Office",
}


@pytest.fixture(name="temperature")
def temperature_fixture() -> Cube:
    """Set up a cube of temperature data"""
    data = np.full((2, 2), fill_value=293.0, dtype=np.float32)
    data[0, 1] = 295.0
    temperature = set_up_variable_cube(
        data,
        name="air_temperature_at_condensation_level",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return temperature


@pytest.fixture(name="pressure")
def pressure_fixture() -> Cube:
    """Set up a cube of pressure data"""
    data = np.full((2, 2), fill_value=100000.0, dtype=np.float32)
    data[0, 0] = 100200.0
    pressure = set_up_variable_cube(
        data,
        name="air_pressure_at_condensation_level",
        units="Pa",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return pressure


@pytest.fixture(name="pressure_bad_units")
def pressure_bad_units_fixture() -> Cube:
    """Set up a cube of something that could be pressure data
    according to its name but has 'm' as units"""
    data = np.full((2, 2), fill_value=100000.0, dtype=np.float32)
    data[0, 0] = 100200.0
    pressure = set_up_variable_cube(
        data,
        name="air_pressure_at_condensation_level",
        units="m",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return pressure


@pytest.fixture(name="temperature_bad_units")
def temperature_bad_units_fixture() -> Cube:
    """Set up a cube of something that could be temperature data
    according to its name but has 'm' as units"""
    data = np.full((2, 2), fill_value=100000.0, dtype=np.float32)
    data[0, 0] = 100200.0
    pressure = set_up_variable_cube(
        data,
        name="air_temperature_at_condensation_level",
        units="m",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return pressure


@pytest.fixture(name="air_parcel")
def air_parcel_fixture() -> Cube:
    """Set up a result cube"""
    data = np.array([[[266.47, 269.69], [266.58, 266.58]]], np.float32)
    air_parcel = set_up_variable_cube(
        data,
        name="parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[50000.0],
        pressure=True,
    )
    return iris.util.squeeze(air_parcel)


@pytest.fixture(name="air_parcel_diff_pressure")
def air_parcel_diff_pressure_fixture() -> Cube:
    """Set up a result cube using a different pressure level than the default"""
    data = np.array([[[273.48, 276.41], [273.57, 273.57]]], np.float32)
    air_parcel_diff_pressure = set_up_variable_cube(
        data,
        name="parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[60000.0],
        pressure=True,
    )
    return iris.util.squeeze(air_parcel_diff_pressure)


def metadata_ok(air_parcel: Cube, baseline: Cube, model_id_attr=None) -> None:
    """
    Checks parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level
    Cube long_name, units and dtype are as expected. Then compares with baseline
    to make sure everything else matches.

    Args:
        air_parcel: Result of TemperatureSaturatedAirParcel plugin
        baseline: A cube with the expected coordinates and attributes.

    Raises:
        AssertionError: If anything doesn't match
    """
    assert (
        air_parcel.long_name
        == "parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level"
    )
    assert air_parcel.units == "K"
    assert air_parcel.dtype == np.float32
    for coord in baseline.coords():
        result_coord = air_parcel.coord(coord.name())
        assert air_parcel.coord_dims(coord) == baseline.coord_dims(result_coord)
        assert result_coord == coord
    for attr in MANDATORY_ATTRIBUTES:
        assert air_parcel.attributes[attr] == baseline.attributes[attr]
    all_attr_keys = list(air_parcel.attributes.keys())
    if model_id_attr:
        assert (
            air_parcel.attributes[model_id_attr] == baseline.attributes[model_id_attr]
        )
        mandatory_attr_keys = [k for k in all_attr_keys if k != model_id_attr]
    else:
        mandatory_attr_keys = all_attr_keys
    assert sorted(mandatory_attr_keys) == sorted(MANDATORY_ATTRIBUTES)


def test_basic(temperature, pressure, air_parcel):
    """Check that for each pair of values, we get the expected result
    and that the metadata are as expected."""
    result = TemperatureSaturatedAirParcel()([temperature, pressure])
    metadata_ok(result, air_parcel)
    assert np.isclose(result.data, air_parcel.data, atol=1e-2).all()


def test_masked(temperature, pressure, air_parcel):
    """Check that masked inputs result in masked outputs."""
    temperature_mask = np.full_like(temperature.data, False)
    pressure_mask = np.full_like(pressure.data, False)
    temperature_mask[0, 0] = True
    pressure_mask[1, 1] = True
    air_parcel_mask = np.full_like(air_parcel.data, False)
    air_parcel_mask[0, 0] = True
    air_parcel_mask[1, 1] = True
    temperature.data = np.ma.masked_array(temperature.data, mask=temperature_mask)
    pressure.data = np.ma.masked_array(pressure.data, mask=pressure_mask)
    air_parcel.data = np.ma.masked_array(air_parcel.data, mask=air_parcel_mask)
    result = TemperatureSaturatedAirParcel()([temperature, pressure])
    metadata_ok(result, air_parcel)
    assert np.isclose(result.data, air_parcel.data, atol=1e-2).all()


def test_different_pressure(temperature, pressure, air_parcel_diff_pressure):
    """Check that we get the expected result from the plugin when we use
    a different pressure (600hPa)."""
    result = TemperatureSaturatedAirParcel(pressure_level=60000.0)(
        [temperature, pressure]
    )
    metadata_ok(result, air_parcel_diff_pressure)
    assert np.isclose(result.data, air_parcel_diff_pressure.data, atol=1e-2).all()


def test_bad_temperature_units(temperature_bad_units, pressure):
    """Check that if the temperature cube doesn't have the correct units
    then it cannot be used."""
    with pytest.raises(ValueError):
        TemperatureSaturatedAirParcel()([temperature_bad_units, pressure])


def test_bad_pressure_units(temperature, pressure_bad_units):
    """Check that if the pressure cube doesn't have the correct units
    then it cannot be used."""
    with pytest.raises(ValueError):
        TemperatureSaturatedAirParcel()([temperature, pressure_bad_units])
