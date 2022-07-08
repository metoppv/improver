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
"""Tests for the HumidityMixingRatio plugin"""
import re
from typing import List

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
        data, name="air_pressure", units="Pa", attributes=LOCAL_MANDATORY_ATTRIBUTES,
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
    Checks cloud_condensation_level Cube long_name, units and dtype are as expected.
    Compares cloud_condensation_level Cube with baseline to make sure everything else matches.

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
    else:
        mandatory_attr_keys = all_attr_keys
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


@pytest.mark.parametrize("reverse_order", (False, True))
def test_unit_conversion_and_cube_order(
    temperature, pressure, rel_humidity, reverse_order
):
    """Check that for one set of values (temperature=300, pressure=100000, humidity=1e-2),
    we get the expected result
    even if the input units are changed and the input cube order is reversed.
    Repeats check on output metadata too."""
    temperature.data = np.full_like(temperature.data, 26.85)  # 300 K
    temperature.units = "Celsius"
    pressure.data = np.full_like(pressure.data, 1000)  # 100000 Pa
    pressure.units = "hPa"
    rel_humidity.data = np.full_like(rel_humidity.data, 10)  # 0.1
    rel_humidity.units = "%"
    expected = 2.23855e-3
    cubes = [temperature, pressure, rel_humidity]
    if reverse_order:
        cubes.reverse()
    result = HumidityMixingRatio()(cubes)
    metadata_ok(result, temperature)
    assert np.isclose(result.data, expected, atol=1e-2).all()


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
        (lambda l: l[0].rename("kittens"), "Expected to find cube of "),
        (lambda l: l[1].rename("poodles"), "Expected to find cube of "),
        (remove_a_cube, "Expected to find cube of "),
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
    temperature, pressure, rel_humidity, modifier: callable, error_match: str
):
    """Check for things we know we should reject"""
    temperature.attributes["mosg__model_configuration"] = "gl_ens"
    pressure.attributes["mosg__model_configuration"] = "gl_ens"
    rel_humidity.attributes["mosg__model_configuration"] = "gl_ens"
    cube_list = [temperature, pressure, rel_humidity]
    modifier(cube_list)
    with pytest.raises(ValueError, match=error_match):
        HumidityMixingRatio(model_id_attr="mosg__model_configuration")(cube_list)
