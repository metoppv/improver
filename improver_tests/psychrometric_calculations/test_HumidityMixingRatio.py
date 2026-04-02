# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the HumidityMixingRatio plugin"""

from typing import List, Tuple
from unittest.mock import patch, sentinel

import iris
import iris.cube as icube
import numpy as np
import pytest
from iris.coords import AncillaryVariable
from iris.cube import Cube

from improver.constants import EARTH_SURFACE_GRAVITY_ACCELERATION, WATER_DENSITY
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTES
from improver.psychrometric_calculations.precipitable_water import (
    PrecipitableWater,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    HumidityMixingRatio,
    get_pressure_points,
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


@pytest.fixture(name="rel_humidity")
def humidity_cube_fixture() -> Cube:
    """Set up a r, y, x cube of relative humidity data"""
    data = np.full((2, 2, 2), fill_value=1e-1, dtype=np.float32)
    humidity_cube = set_up_variable_cube(
        data,
        name="relative_humidity",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES | {"least_significant_digit": 3},
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


@pytest.mark.parametrize(
    "temperature_value, pressure_value, rel_humidity_value, expected",
    (
        (293, 100000, 0.1, 1.459832e-3),
        (300, 100000, 0.0, 1.11927e-5),
        (300, 100000, 0.00001, 1.11927e-5),
    ),
)
def test_zero_humidity(
    temperature,
    pressure,
    rel_humidity,
    temperature_value,
    pressure_value,
    rel_humidity_value,
    expected,
):
    """Check that zero and tiny humidity values are handled correctly."""
    temperature.data = np.full_like(temperature.data, temperature_value)
    pressure.data = np.full_like(pressure.data, pressure_value)
    rel_humidity.data = np.full_like(rel_humidity.data, rel_humidity_value)
    result = HumidityMixingRatio()([temperature, pressure, rel_humidity])
    metadata_ok(result, temperature)
    assert np.isclose(result.data, expected, atol=1e-7).all()


def make_pressure_cube(temp_cube: Cube) -> Cube:
    """
    Create a 3D pressure cube from a temperature_on_pressure_levels cube.
    The resulting cube has shape (levels, y, x).

    :param temp_cube: input temperature on pressure cube
    :return: a 3D pressure cube

    """

    # ----------------------------------------
    # 1. Extract the 1D pressure coordinate
    # ----------------------------------------
    p_coord = temp_cube.coord("pressure")  # DimCoord
    p_vals = p_coord.points  # 1D array (nz,)

    # ----------------------------------------
    # 2. Broadcast pressure to match the cube grid
    # ----------------------------------------

    p_3d = p_vals[:, None, None]  # (nz, 1, 1)
    p_3d = np.broadcast_to(p_3d, temp_cube.shape)  # (nz, ny, nx)

    # ----------------------------------------
    # 3. Build a new pressure cube
    # ----------------------------------------
    pressure_cube = icube.Cube(
        p_3d,
        standard_name="air_pressure",
        long_name="air_pressure unit test",
        units=p_coord.units,
        dim_coords_and_dims=[
            (p_coord, 0),  # vertical dimension
            (temp_cube.coord(axis="y"), 1),  # y coordinate
            (temp_cube.coord(axis="x"), 2),  # x coordinate
        ],
        attributes=temp_cube.attributes,
    )

    return pressure_cube


def set_up_temperature_cube(
    shape: Tuple[int], temperature_value: float, vertical_levels: List[float]
) -> Cube:
    """
    Create a temperature on pressure cube.
    :param: shape: Shape of the temperature cube
    :param: temperature_value: temperature value
    :param: vertical_levels: List of vertical levels
    :return: a temperature cube

    """
    temperature = set_up_variable_cube(
        np.full(shape, temperature_value, dtype=np.float32),
        "latlon",
        name="air_temperature",
        x_grid_spacing=1.0,
        y_grid_spacing=1.0,
        vertical_levels=vertical_levels,
        pressure=True,
    )
    add_attribute_dictionary(temperature)
    return temperature


def set_up_rel_humidity_cube(
    shape: Tuple[int], rel_humidity_value: float, vertical_levels: List[float]
) -> Cube:
    """
    Create a relative humidity on pressure cube.
    :param: shape: Shape of the relative humidity cube
    :param: rel_humidity_value: relative humidity value
    :param: vertical_levels: List of vertical levels
    :return: a relative humidity cube
    """
    rel_humidity = set_up_variable_cube(
        np.full(shape, rel_humidity_value, dtype=np.float32),
        "latlon",
        name="relative_humidity",
        units="1",
        x_grid_spacing=1.0,
        y_grid_spacing=1.0,
        vertical_levels=vertical_levels,
        pressure=True,
    )
    add_attribute_dictionary(rel_humidity)
    return rel_humidity


def test_get_pressure_points() -> None:
    """
    tests function "get_pressure_points" which is a support function
    written to check if a pressure cube has been inadvertantly flipped
    within the Improver implementation of PrecipitableWater

    :return: None
    """
    temperature_value, rel_humidity_value = (
        293,
        0.1,
    )

    # set up cubes
    vertical_levels = [100000.0, 50000.0, 100.0]
    shape = (len(vertical_levels), 3, 3)

    temperature = set_up_temperature_cube(shape, temperature_value, vertical_levels)
    pressure = make_pressure_cube(temperature)
    rel_humidity = set_up_rel_humidity_cube(shape, rel_humidity_value, vertical_levels)

    assert np.allclose(get_pressure_points(temperature), np.array(vertical_levels))
    assert np.allclose(get_pressure_points(pressure), np.array(vertical_levels))
    assert np.allclose(get_pressure_points(rel_humidity), np.array(vertical_levels))

    # check captialisation has no affect
    rel_humidity.coord("pressure").rename("Pressure")
    assert np.allclose(get_pressure_points(rel_humidity), np.array(vertical_levels))

    # check null result when no "pressure" dimension
    rel_humidity.coord("Pressure").rename("Pr3ssure")
    assert np.allclose(get_pressure_points(rel_humidity), np.array([]))


def add_attribute_dictionary(cube: Cube) -> None:
    """
    Adds attributes dictionary to cube attributes
    to allow pre-existing checking function "metadata_ok"
    to be used.

    :param cube: Cube to add attributes to
    :return: None
    """
    # set up meta-data required by testing
    attributes_dictionary = {
        "title": "unit test data",
        "source": "unit test",
        "institution": "somewhere",
        "least_significant_digit": 4,
    }
    for k, v in attributes_dictionary.items():
        cube.attributes[k] = v


def test_mixing_ratio_without_pressure_parameter() -> None:
    """
    the HumidityMixingRatio calculation will generate its own pressure cube
    if one is not supplied. This unit tests verifies that the results are the
    same with/without an explicit pressure parameter.

    This ticket reports values for the total precipitable water (TPW) being far too high.

    https://metoffice.atlassian.net/browse/EPPT-3209

    The reason was that HumidityMixingRadio generated a pressure cube that was wrongly
    flipped veritically. This unit test re-creates the failing scenario to test and
    exercise the bug fix

    The unit test then does a very simple total precipitable water calculation
    ensuring the output from HumidityMixingRadio is suitable.

    The improver calculation is then compared against a DIY calculation as a sanity check.

    """
    iris.FUTURE.save_split_attrs = True  # to stop Iris warning
    temperature_value, rel_humidity_value, expected = (
        293,
        0.1,
        1.459832e-3,
    )

    # set up input cubes
    vertical_levels = [100000.0, 50000.0, 100.0]
    shape = (len(vertical_levels), 3, 3)

    temperature = set_up_temperature_cube(shape, temperature_value, vertical_levels)
    pressure = make_pressure_cube(temperature)
    rel_humidity = set_up_rel_humidity_cube(shape, rel_humidity_value, vertical_levels)

    # mixing ratio calculation with 3 parameters
    w3 = HumidityMixingRatio()([temperature, pressure, rel_humidity])
    metadata_ok(w3, temperature)  # asserts in function call
    # check results on single layer are as expected where pressure is 100000 Pa
    assert np.isclose(w3.data[0], expected, atol=1e-7).all()

    # mixing ratio calculation with 2 parameters
    w2 = HumidityMixingRatio()([temperature, rel_humidity])
    metadata_ok(w2, temperature)  # asserts in function call

    # check 2 parameter calculation gives same results as 3 parameter calculation
    assert np.isclose(w3.data, w2.data).all()

    # use w3 to calculate precipitable water
    pw = PrecipitableWater().process(w3)

    # perform integration step (summing water in vertical atmosphere column) for Improver
    improver_tpw = np.sum(pw.data, axis=0)

    # perform DIY TPW calculation
    delta = pressure.data[1:, :, :] - pressure.data[:-1, :, :]
    mid_w = (w3.data[1:, :, :] + w3.data[:-1, :, :]) / 2.0
    integral_terms = delta * mid_w
    unit_test_tpw_1 = -np.sum(integral_terms, axis=0) / (
        EARTH_SURFACE_GRAVITY_ACCELERATION * WATER_DENSITY
    )
    # numpy's trapezium rule integration
    unit_test_tpw_2 = -np.trapezoid(w3.data, x=pressure.data, axis=0) / (
        EARTH_SURFACE_GRAVITY_ACCELERATION * WATER_DENSITY
    )

    # verify DIY TPW integrations produce same results
    # N.B. Improver uses np.trapezoid
    assert np.isclose(unit_test_tpw_1.data, unit_test_tpw_2.data).all()

    # note for such a small cube the DIY and Improver calculations
    # for TPW are somewhat different.
    # one presumes they will converge for cubes with more cells
    # give a 200% latitude for testing

    assert np.isclose(improver_tpw, unit_test_tpw_1, rtol=2).all()


def test_height_levels():
    """Check that the plugin works with height level data"""

    temperature = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=293, dtype=np.float32),
        name="air_temperature",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[100, 400],
        height=True,
    )
    pressure_cube = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=100000, dtype=np.float32),
        name="surface_air_pressure",
        units="Pa",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[100, 400],
        height=True,
    )
    rel_humidity = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=1.0, dtype=np.float32),
        name="relative_humidity",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[100, 400],
        height=True,
    )
    result = HumidityMixingRatio()([temperature, pressure_cube, rel_humidity])
    metadata_ok(result, temperature)
    assert np.isclose(result.data, 1.459832e-2, atol=1e-7).all()


def test_height_levels_above_surface():
    """Check that the plugin works with height level data"""

    temperature = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=293, dtype=np.float32),
        name="air_temperature",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[100, 400],
        height=True,
    )
    pressure_cube = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=100000, dtype=np.float32),
        name="some_random_pressure",
        units="Pa",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[100, 400],
        height=True,
    )
    rel_humidity = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=1.0, dtype=np.float32),
        name="relative_humidity",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[100, 400],
        height=True,
    )
    result = HumidityMixingRatio()([temperature, pressure_cube, rel_humidity])
    metadata_ok(result, temperature)
    assert np.isclose(result.data, 1.459832e-2, atol=1e-7).all()


def test_pressure_levels():
    """Check that the plugin works with pressure level data when pressure cube is not provided"""
    temperature = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=293, dtype=np.float32),
        name="air_temperature",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[95000, 100000],
        pressure=True,
    )
    rel_humidity = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=1.0, dtype=np.float32),
        name="relative_humidity",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[95000, 100000],
        pressure=True,
    )
    result = HumidityMixingRatio()([temperature, rel_humidity])
    metadata_ok(result, temperature)
    assert np.isclose(result.data[:, 0], 1.537017e-2, atol=1e-7).all()
    assert np.isclose(result.data[:, 1], 1.459832e-2, atol=1e-7).all()


def test_pressure_levels_with_status_flag():
    """Check that the plugin works when the pressure cube is not provided and the input cube has
    a status flag of type ancillary variable."""
    status_flag_values = np.array(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )
    ancillary_var = AncillaryVariable(
        status_flag_values,
        standard_name="status_flag",
        units="1",
    )
    temperature_cube = set_up_variable_cube(
        np.full((3, 3, 3), fill_value=282, dtype=np.float32),
        name="air_temperature",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        pressure=True,
        vertical_levels=[100000.0, 97500.0, 95000.0],
    )
    temperature_cube.add_ancillary_variable(ancillary_var, data_dims=(0, 1))
    rel_humidity_cube = set_up_variable_cube(
        np.full((3, 3, 3), fill_value=282, dtype=np.float32),
        name="relative_humidity",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        pressure=True,
        vertical_levels=[100000.0, 97500.0, 95000.0],
    )
    rel_humidity_cube.add_ancillary_variable(ancillary_var, data_dims=(0, 1))
    result = HumidityMixingRatio()([temperature_cube, rel_humidity_cube])
    metadata_ok(result, temperature_cube)
    assert result.coords()[0].name() == "pressure"


def test_error_raised_no_pressure_coordinate_or_pressure_cube(
    temperature, rel_humidity
):
    """Check that the plugin raises an error if there is no pressure coordinate and no pressure cube"""
    with pytest.raises(
        ValueError,
        match="No pressure cube with name 'pressure' found and no pressure coordinate "
        "found in temperature or relative humidity cubes",
    ):
        HumidityMixingRatio()([temperature, rel_humidity])


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


def test_correct_value_error_returned_when_more_than_one_named_pressure():
    temperature = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=293, dtype=np.float32),
        name="air_temperature",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[95000, 100000],
        pressure=True,
    )
    rel_humidity = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=1.0, dtype=np.float32),
        name="relative_humidity",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[95000, 100000],
        pressure=True,
    )
    some_pressure = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=293, dtype=np.float32),
        name="some_random_pressure",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[95000, 100000],
        pressure=True,
    )
    some_more_pressure = set_up_variable_cube(
        np.full((1, 2, 2, 2), fill_value=1.0, dtype=np.float32),
        name="another_random_pressure",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
        vertical_levels=[95000, 100000],
        pressure=True,
    )
    with pytest.raises(
        ValueError,
        match="Got 2 cubes with 'pressure' in name.",
    ):
        HumidityMixingRatio()(
            [temperature, rel_humidity, some_pressure, some_more_pressure]
        )
