# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Tests for the CloudCondensationLevel plugin"""
import re
from typing import List

import numpy as np
import pytest
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


def metadata_ok(ccl: Cube, baseline: Cube, model_id_attr=None) -> None:
    """
    Checks cloud_condensation_level Cube long_name, units and dtype are as expected.
    Compares cloud_condensation_level Cube with baseline to make sure everything else matches.

    Args:
        ccl: Result of CloudCondensationLevel plugin
        baseline: A temperature or similar cube with the same coordinates and attributes.

    Raises:
        AssertionError: If anything doesn't match
    """
    assert ccl.long_name == "temperature_at_cloud_condensation_level"
    assert ccl.units == "K"
    assert ccl.dtype == np.float32
    ccl.coord("air_pressure")  # Fails if this coord is absent.
    for coord in ccl.coords():
        if coord.standard_name == "air_pressure":
            coord_dims = ccl.coord_dims(coord)
            assert len(coord_dims) == ccl.ndim
        else:
            base_coord = baseline.coord(coord.name())
            assert ccl.coord_dims(coord) == baseline.coord_dims(base_coord)
            assert coord == base_coord
    for attr in MANDATORY_ATTRIBUTES:
        assert ccl.attributes[attr] == baseline.attributes[attr]
    all_attr_keys = list(ccl.attributes.keys())
    if model_id_attr:
        assert ccl.attributes[model_id_attr] == baseline.attributes[model_id_attr]
        mandatory_attr_keys = [k for k in all_attr_keys if k != model_id_attr]
    else:
        mandatory_attr_keys = all_attr_keys
    assert sorted(mandatory_attr_keys) == sorted(MANDATORY_ATTRIBUTES)


@pytest.mark.parametrize(
    "temperature_value, pressure_value, humidity_value, expected_t, expected_p",
    (
        (293, 100000, 2.7e-3, 264.92, 70275),
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
    assert np.isclose(result.data, expected_t, atol=1e-2).all()
    assert np.isclose(result.coord("air_pressure").points, expected_p, atol=1e-0).all()


@pytest.mark.parametrize("reverse_order", (False, True))
def test_unit_conversion_and_cube_order(temperature, pressure, humidity, reverse_order):
    """Check that for one set of values (temperature=300, pressure=100000, humidity=1e-2),
    we get the expected result
    even if the input units are changed and the input cube order is reversed.
    Repeats check on output metadata too."""
    temperature.data = np.full_like(temperature.data, 26.85)  # 300 K
    temperature.units = "Celsius"
    pressure.data = np.full_like(pressure.data, 1000)  # 100000 Pa
    pressure.units = "hPa"
    humidity.data = np.full_like(humidity.data, 10)  # 1e-2 Pa
    humidity.units = "g kg-1"
    expected_t = 284.22
    expected_p = 82757
    cubes = [temperature, pressure, humidity]
    if reverse_order:
        cubes.reverse()
    result = CloudCondensationLevel()(cubes)
    metadata_ok(result, temperature)
    assert np.isclose(result.data, expected_t, atol=1e-2).all()
    assert np.isclose(result.coord("air_pressure").points, expected_p, atol=1e-0).all()


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


def remove_a_cube(cubes: List[Cube]):
    """Removes the last cube from the cube list."""
    del cubes[-1]


def add_unexpected_cube(cubes: List[Cube]):
    """Copies first cube, renames it and adds it to the list of cubes."""
    cube = cubes[0].copy()
    cube.rename("kittens")
    cubes.append(cube)


def spatial_shift(cubes: List[Cube]):
    """Adjusts x-coord of first cube so it no longer matches the second cube."""
    coord = cubes[0].coord(axis="x")
    coord = coord.copy(coord.points + 1)
    cubes[0].replace_coord(coord)


def units_to_kg(cube: Cube):
    """Sets units of cube to kg"""
    cube.units = "kg"


def t_has_time_bounds(cubes: List[Cube]):
    """Sets time bounds info onto temperature cube"""
    cubes[0].coord("time").bounds = [
        [p, p - 3600] for p in cubes[0].coord("time").points
    ]


def t_time_shifted(cubes: List[Cube]):
    """Moves temperature time point back by one hour"""
    cubes[0].coord("time").points = cubes[0].coord("time").points - 3600


def t_frt_shifted(cubes: List[Cube]):
    """Moves temperature forecast_reference_time point back by one hour"""
    cubes[0].coord("forecast_reference_time").points = (
        cubes[0].coord("forecast_reference_time").points - 3600
    )


def set_mismatched_model_ids(cubes: List[Cube]):
    """Sets model_ids to input cubes that do not match"""
    cubes[1].attributes["mosg__model_configuration"] = "gl_ens"
    cubes[0].attributes["mosg__model_configuration"] = "uk_ens"


@pytest.mark.parametrize(
    "modifier,error_match",
    (
        (lambda l: l[0].rename("kittens"), "Expected to find cubes of "),
        (lambda l: l[1].rename("poodles"), "Expected to find cubes of "),
        (remove_a_cube, "Expected to find cubes of "),
        (add_unexpected_cube, re.escape("Unexpected Cube(s) found in inputs: "),),
        (spatial_shift, "Spatial coords of input Cubes do not match: "),
        (lambda l: units_to_kg(l[0]), "Unable to convert from"),
        (lambda l: units_to_kg(l[1]), "Unable to convert from"),
        (t_has_time_bounds, "air_temperature must not have time bounds"),
        (t_time_shifted, "time coordinates do not match"),
        (t_frt_shifted, "forecast_reference_time coordinates do not match"),
        (
            set_mismatched_model_ids,
            "Attribute mosg__model_configuration does not match on input cubes",
        ),
    ),
)
def test_exceptions(
    temperature, pressure, humidity, modifier: callable, error_match: str
):
    """Check for things we know we should reject"""
    temperature.attributes["mosg__model_configuration"] = "gl_ens"
    pressure.attributes["mosg__model_configuration"] = "gl_ens"
    humidity.attributes["mosg__model_configuration"] = "gl_ens"
    cube_list = [temperature, pressure, humidity]
    modifier(cube_list)
    with pytest.raises(ValueError, match=error_match):
        CloudCondensationLevel(model_id_attr="mosg__model_configuration")(cube_list)
