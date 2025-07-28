# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the CloudCondensationLevel plugin"""

from typing import Tuple

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTES
from improver.psychrometric_calculations.cloud_condensation_level import (
    CloudCondensationLevel,
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


@pytest.fixture(name="humidity")
def humidity_cube_fixture() -> Cube:
    """Set up a r, y, x cube of humidity data"""
    data = np.full((2, 2, 2), fill_value=1e-2, dtype=np.float32)
    humidity_cube = set_up_variable_cube(
        data,
        name="humidity_mixing_ratio",
        units="kg kg-1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return humidity_cube


def metadata_ok(ccl: Tuple[Cube, Cube], baseline: Cube, model_id_attr=None) -> None:
    """
    Checks cloud_condensation_level Cube long_name, units and dtype are as expected.
    Ensures that there is no height coord associated with it.
    Compares cloud_condensation_level Cube with baseline to make sure everything else matches.

    Args:
        ccl: Result of CloudCondensationLevel plugin
        baseline: A temperature or similar cube with the same coordinates and attributes.

    Raises:
        AssertionError: If anything doesn't match
    """
    t_at_ccl, p_at_ccl = ccl
    assert t_at_ccl.long_name == "air_temperature_at_condensation_level"
    assert t_at_ccl.units == "K"
    assert p_at_ccl.long_name == "air_pressure_at_condensation_level"
    assert p_at_ccl.units == "Pa"
    for cube in ccl:
        assert "height" not in [c.name() for c in cube.coords()]
        assert cube.dtype == np.float32
        for coord in cube.coords():
            base_coord = baseline.coord(coord.name())
            assert cube.coord_dims(coord) == baseline.coord_dims(base_coord)
            assert coord == base_coord
        for attr in MANDATORY_ATTRIBUTES:
            if attr == "title":
                assert (
                    cube.attributes[attr]
                    == f"Post-Processed {baseline.attributes[attr]}"
                )
            else:
                assert cube.attributes[attr] == baseline.attributes[attr]
        all_attr_keys = list(cube.attributes.keys())
        if model_id_attr:
            assert cube.attributes[model_id_attr] == baseline.attributes[model_id_attr]
        mandatory_attr_keys = [k for k in all_attr_keys if k != model_id_attr]
        assert sorted(mandatory_attr_keys) == sorted(MANDATORY_ATTRIBUTES)


@pytest.mark.parametrize(
    "temperature_value, pressure_value, humidity_value, expected_t, expected_p",
    (
        (293, 100000, 2.7e-3, 264.92, 70278),
        (293, 100000, 1.0e-2, 285.75, 91596),
        (300, 100000, 1.0e-2, 284.22, 82757),
    ),
)
def test_basic(
    temperature,
    pressure,
    humidity,
    temperature_value,
    pressure_value,
    humidity_value,
    expected_t,
    expected_p,
):
    """Check that for each pair of values, we get the expected result
    and that the metadata are as expected."""
    temperature.data = np.full_like(temperature.data, temperature_value)
    pressure.data = np.full_like(pressure.data, pressure_value)
    humidity.data = np.full_like(humidity.data, humidity_value)
    result = CloudCondensationLevel()([temperature, pressure, humidity])
    metadata_ok(result, temperature)
    assert np.isclose(result[0].data, expected_t, atol=1e-2).all()
    assert np.isclose(result[1].data, expected_p, atol=1e-0).all()


@pytest.mark.parametrize(
    "temperature_value, pressure_value, humidity_value",
    (
        (293, 100000, 2.7e-1),
        (300, 100000, 1.0e-1),
        (189, 60300, 0.334),  # This case was found in MOGREPS-G outputs.
    ),
)
def test_for_limited_values(
    temperature,
    pressure,
    humidity,
    temperature_value,
    pressure_value,
    humidity_value,
):
    """Check that for each pair of values, we get the surface temperature and pressure returned
    and that the metadata are as expected."""
    temperature.data = np.full_like(temperature.data, temperature_value)
    pressure.data = np.full_like(pressure.data, pressure_value)
    humidity.data = np.full_like(humidity.data, humidity_value)
    result = CloudCondensationLevel()([temperature, pressure, humidity])
    metadata_ok(result, temperature)
    assert np.isclose(result[0].data, temperature_value, atol=1e-2).all()
    assert np.isclose(result[1].data, pressure_value, atol=1e-0).all()


@pytest.mark.parametrize("model_id_attr", ("mosg__model_configuration", None))
def test_model_id_attr(temperature, pressure, humidity, model_id_attr):
    """Check that tests pass if model_id_attr is set on inputs and is applied or not"""
    temperature.attributes["mosg__model_configuration"] = "gl_ens"
    pressure.attributes["mosg__model_configuration"] = "gl_ens"
    humidity.attributes["mosg__model_configuration"] = "gl_ens"
    result = CloudCondensationLevel(model_id_attr=model_id_attr)(
        [temperature, pressure, humidity]
    )
    metadata_ok(result, temperature, model_id_attr=model_id_attr)


@pytest.mark.parametrize("has_height_coord", (True, False))
def test_with_height_coord(temperature, pressure, humidity, has_height_coord):
    """Check that tests pass if a scalar height coord is present on the temperature cube"""
    if has_height_coord:
        temperature.add_aux_coord(
            AuxCoord(
                np.array([1.65], dtype=np.float32), standard_name="height", units="m"
            )
        )
    result = CloudCondensationLevel()([temperature, pressure, humidity])
    metadata_ok(result, temperature)
