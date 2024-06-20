# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the VerticalUpdraught plugin"""
import re
from datetime import datetime
from typing import List
from unittest.mock import patch, sentinel

import numpy as np
import pytest
from iris.coords import CellMethod
from iris.cube import Cube

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTES
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.wind_calculations.vertical_updraught import VerticalUpdraught

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


class HaltExecution(Exception):
    pass


@patch("improver.wind_calculations.vertical_updraught.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    mock_as_cubelist.side_effect = HaltExecution
    try:
        VerticalUpdraught(model_id_attr=sentinel.model_id_attr)(
            sentinel.cape, sentinel.precip
        )
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(sentinel.cape, sentinel.precip)


@pytest.fixture(name="cape")
def cape_cube_fixture() -> Cube:
    """Set up a r, y, x cube of CAPE data"""
    data = np.zeros((2, 2, 2), dtype=np.float32)
    cape_cube = set_up_variable_cube(
        data,
        name="atmosphere_convective_available_potential_energy",
        units="J kg-1",
        time=datetime(2017, 11, 10, 3, 0),
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return cape_cube


@pytest.fixture(name="precip")
def precip_cube_fixture() -> Cube:
    """Set up a r, y, x cube of precipitation rate data"""
    data = np.zeros((2, 2, 2), dtype=np.float32)
    precip_cube = set_up_variable_cube(
        data,
        name="lwe_precipitation_rate_max",
        units="mm h-1",
        time_bounds=(datetime(2017, 11, 10, 3, 0), datetime(2017, 11, 10, 4, 0)),
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    precip_cube.cell_methods = [CellMethod(coords=["time"], method="max")]
    return precip_cube


def metadata_ok(updraught: Cube, baseline: Cube, model_id_attr=None) -> None:
    """
    Checks updraught Cube long_name, units and dtype are as expected.
    Compares updraught Cube with baseline to make sure everything else matches.

    Args:
        updraught: Result of VerticalUpdraught plugin
        baseline: A Precip or similar cube with the same coordinates and attributes.

    Raises:
        AssertionError: If anything doesn't match
    """
    assert updraught.long_name == "maximum_vertical_updraught"
    assert updraught.units == "m s-1"
    assert updraught.dtype == np.float32
    for coord in updraught.coords():
        base_coord = baseline.coord(coord.name())
        assert updraught.coord_dims(coord) == baseline.coord_dims(base_coord)
        assert coord == base_coord
    for attr in MANDATORY_ATTRIBUTES:
        assert updraught.attributes[attr] == baseline.attributes[attr]
    all_attr_keys = list(updraught.attributes.keys())
    if model_id_attr:
        assert updraught.attributes[model_id_attr] == baseline.attributes[model_id_attr]
        mandatory_attr_keys = [k for k in all_attr_keys if k != model_id_attr]
    else:
        mandatory_attr_keys = all_attr_keys
    assert sorted(mandatory_attr_keys) == sorted(MANDATORY_ATTRIBUTES)


@pytest.mark.parametrize(
    "cape_value,cape_result",
    (
        (0.0, 0.0),
        (9.9, 0.0),
        (10.1, 1.1236),
        (500.0, 7.906),
        (600.0, 8.660),
        (2000.0, 15.811),
        (3000.0, 19.365),
        (4000.0, 22.361),
    ),
)
@pytest.mark.parametrize(
    "precip_value,precip_result",
    (
        (0.0, 0.0),
        (4.9, 0.0),
        (5.1, 5.012),
        (10.0, 5.813),
        (50.0, 8.282),
        (200.0, 11.236),
    ),
)
def test_basic(cape, precip, cape_value, cape_result, precip_value, precip_result):
    """Check that for each pair of values, we get the expected result
    and that the metadata are as expected. Note that the method being tested deals with
    cape and precip separately and that the resulting updraught is the sum of these."""
    cape.data = np.full_like(cape.data, cape_value)
    precip.data = np.full_like(precip.data, precip_value)
    expected_value = cape_result + precip_result
    result = VerticalUpdraught()([cape, precip])
    metadata_ok(result, precip)
    assert np.isclose(result.data, expected_value, rtol=1e-4).all()


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
