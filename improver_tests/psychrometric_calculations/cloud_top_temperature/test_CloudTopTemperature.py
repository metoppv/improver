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
"""Tests for the CloudTopTemperature plugin"""
import re
from datetime import datetime
from typing import List

import numpy as np
import pytest
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.cube import Cube

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTES
from improver.psychrometric_calculations.cloud_top_temperature import (
    CloudTopTemperature,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


@pytest.fixture(name="ccl")
def ccl_cube_fixture() -> Cube:
    """Set up a r, y, x cube of Cloud condensation level data with pressure coordinate"""
    data = np.zeros((2, 2, 2), dtype=np.float32)
    ccl_cube = set_up_variable_cube(
        data,
        name="temperature_at_cloud_condensation_level",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    ccl_cube.add_aux_coord(
        AuxCoord(standard_name="air_pressure", points=data.copy(), units="Pa"),
        (0, 1, 2),
    )
    return ccl_cube


@pytest.fixture(name="pressure_points")
def pressure_coord_fixture() -> np.ndarray:
    """Generate a list of pressure values"""
    return np.arange(100000, 29999, -10000)


@pytest.fixture(name="temperature")
def t_cube_fixture(pressure_points) -> Cube:
    """Set up a r, p, y, x cube of Temperature on pressure level data"""
    temperatures = np.array([300, 286, 280, 274, 267, 262, 257, 245], dtype=np.float32)
    data = np.broadcast_to(
        temperatures.reshape((1, len(temperatures), 1, 1)), (2, len(temperatures), 2, 2)
    )
    t_cube = set_up_variable_cube(
        data,
        pressure=True,
        height_levels=pressure_points,
        name="temperature_on_pressure_levels",
        units="K",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return t_cube


def metadata_ok(cct: Cube, baseline: Cube, model_id_attr=None) -> None:
    """
    Checks convective cloud top temperature Cube long_name, units and dtype are as expected.
    Compares with baseline to make sure everything else matches.

    Args:
        cct: Result of ConvectiveCloudTop plugin
        baseline: A cube with the expected coordinates and attributes.

    Raises:
        AssertionError: If anything doesn't match
    """
    assert cct.long_name == "temperature_at_convective_cloud_top"
    assert cct.units == "K"
    assert cct.dtype == np.float32
    for coord in cct.coords():
        base_coord = baseline.coord(coord.name())
        assert cct.coord_dims(coord) == baseline.coord_dims(base_coord)
        assert coord == base_coord
    for attr in MANDATORY_ATTRIBUTES:
        assert cct.attributes[attr] == baseline.attributes[attr]
    all_attr_keys = list(cct.attributes.keys())
    if model_id_attr:
        assert cct.attributes[model_id_attr] == baseline.attributes[model_id_attr]
        mandatory_attr_keys = [k for k in all_attr_keys if k != model_id_attr]
    else:
        mandatory_attr_keys = all_attr_keys
    assert sorted(mandatory_attr_keys) == sorted(MANDATORY_ATTRIBUTES)


@pytest.mark.parametrize("ccl_t, ccl_p", ((290, 95000), (288.12, 90000)))
def test_basic(ccl, temperature, ccl_t, ccl_p):
    """Check that for each pair of CCL values, and the same atmosphere profile,
    we get the expected result and that the metadata are as expected."""
    ccl.data = np.full_like(ccl.data, ccl_t)
    ccl.coord("air_pressure").points = np.full_like(
        ccl.coord("air_pressure").points, fill_value=ccl_p
    )
    expected_value = 264.575
    result = CloudTopTemperature()([ccl, temperature])
    metadata_ok(result, ccl)
    assert np.isclose(result.data, expected_value, atol=1e-2).all()


@pytest.mark.parametrize("reverse_order", (False, True))
def test_unit_conversion_and_cube_order(cape, precip, reverse_order):
    """Check that for one pair of values (CAPE=500, precip=10), we get the expected result
    even if the input units are changed and the input cube order is reversed.
    Repeats check on output metadata too."""
    cape.data = np.full_like(cape.data, 0.5)  # 500 J kg-1
    cape.units = "J g-1"
    precip.data = np.full_like(precip.data, 2.7778e-6)  # 10 mm h-1
    precip.units = "m s-1"
    expected_value = 7.906 + 5.813
    cubes = [cape, precip]
    if reverse_order:
        cubes.reverse()
    result = VerticalUpdraught()(cubes)
    metadata_ok(result, precip)
    assert np.isclose(result.data, expected_value, rtol=1e-4).all()


@pytest.mark.parametrize("model_id_attr", ("mosg__model_configuration", None))
def test_model_id_attr(cape, precip, model_id_attr):
    """Check that tests pass if model_id_attr is set on inputs and is applied or not"""
    cape.attributes["mosg__model_configuration"] = "gl_ens"
    precip.attributes["mosg__model_configuration"] = "gl_ens"
    result = VerticalUpdraught(model_id_attr=model_id_attr)([cape, precip])
    metadata_ok(result, precip, model_id_attr=model_id_attr)


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


def cape_has_time_bounds(cubes: List[Cube]):
    """Copies precip cube time point and bounds info onto CAPE cube"""
    cubes[0].coord("time").points = cubes[1].coord("time").points
    cubes[0].coord("time").bounds = cubes[1].coord("time").bounds


def cape_time_shifted(cubes: List[Cube]):
    """Moves CAPE time point back by one hour"""
    cubes[0].coord("time").points = cubes[0].coord("time").points - 3600


def cape_frt_shifted(cubes: List[Cube]):
    """Moves CAPE forecast_reference_time point back by one hour"""
    cubes[0].coord("forecast_reference_time").points = (
        cubes[0].coord("forecast_reference_time").points - 3600
    )


def precip_has_no_time_bounds(cubes: List[Cube]):
    """Removes precip cube time bounds"""
    cubes[1].coord("time").bounds = None


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
        (cape_has_time_bounds, "CAPE cube must not have time bounds"),
        (cape_time_shifted, "CAPE time must match precip cube's lower time bound"),
        (cape_frt_shifted, "Forecast reference times do not match"),
        (precip_has_no_time_bounds, "Precip cube must have time bounds"),
        (
            set_mismatched_model_ids,
            "Attribute mosg__model_configuration does not match on input cubes",
        ),
    ),
)
def test_exceptions(cape, precip, modifier: callable, error_match: str):
    """Check for things we know we should reject"""
    cape.attributes["mosg__model_configuration"] = "gl_ens"
    precip.attributes["mosg__model_configuration"] = "gl_ens"
    cube_list = [cape, precip]
    modifier(cube_list)
    with pytest.raises(ValueError, match=error_match):
        VerticalUpdraught(model_id_attr="mosg__model_configuration")(cube_list)
