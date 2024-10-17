# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Test methods in lightning.LightningFromCapePrecip"""
from datetime import datetime
from unittest.mock import patch, sentinel

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.lightning import LightningFromCapePrecip
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
    set_up_variable_cube,
)


class HaltExecution(Exception):
    pass


@patch("improver.lightning.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    mock_as_cubelist.side_effect = HaltExecution
    try:
        LightningFromCapePrecip()(sentinel.cube1, sentinel.cube2, sentinel.cube3)
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(
        sentinel.cube1, sentinel.cube2, sentinel.cube3
    )


@pytest.fixture(name="cape_cube")
def cape_cube_fixture() -> Cube:
    """
    Set up a CAPE cube for use in tests.
    Has 2 realizations, 7 latitudes spanning from 60S to 60N and 3 longitudes.
    """

    data = np.full((2, 7, 3), dtype=np.float32, fill_value=400)
    cube = set_up_variable_cube(
        data,
        name="atmosphere_convective_available_potential_energy",
        units="J kg-1",
        time=datetime(2017, 11, 10, 4, 0),
        time_bounds=None,
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-60, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    return cube


@pytest.fixture(name="precip_cube")
def precip_cube_fixture() -> Cube:
    """
    Set up a precipitation rate cube for use in tests.
    Has 2 realizations, 7 latitudes spanning from 60S to 60N and 3 longitudes.
    Contains the value of 3 mm h-1 at all points (in SI units)
    """

    data = np.full((2, 7, 3), dtype=np.float32, fill_value=3 / 3.6e6)
    cube = set_up_variable_cube(
        data,
        name="precipitation_rate_max-PT01H",
        units="m s-1",
        time=datetime(2017, 11, 10, 5, 0),
        time_bounds=(datetime(2017, 11, 10, 4, 0), datetime(2017, 11, 10, 5, 0)),
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-60, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    return cube


@pytest.fixture(name="expected_cube")
def expected_cube_fixture() -> Cube:
    """
    Set up the Lightning cube that we expect to get from the plugin
    """

    data = np.ones((1, 7, 3), dtype=np.float32)
    data[:, 2:-2, :] = 0
    cube = set_up_probability_cube(
        data,
        thresholds=[0.0],
        variable_name="number_of_lightning_flashes_per_unit_area",
        threshold_units="m-2",
        time=datetime(2017, 11, 10, 5, 0),
        time_bounds=(datetime(2017, 11, 10, 4, 0), datetime(2017, 11, 10, 5, 0)),
        attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        domain_corner=(-60, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    cube = add_coordinate(
        cube,
        coord_name="realization",
        coord_points=[0, 1],
        coord_units="1",
        dtype=np.int32,
    )

    return cube


def test_basic(cape_cube, precip_cube, expected_cube):
    """Run the plugin and check the result cube matches the expected_cube"""
    result = LightningFromCapePrecip()(CubeList([cape_cube, precip_cube]))
    assert result.xml().splitlines(keepends=True) == expected_cube.xml().splitlines(
        keepends=True
    )
    assert np.allclose(result.data, expected_cube.data)


def test_3h_cubes(cape_cube, precip_cube, expected_cube):
    """Run the plugin again with 3h cubes"""
    cape_cube.coord("time").points = cape_cube.coord("time").points - 2 * 3600
    bounds = precip_cube.coord("time").bounds
    precip_cube.coord("time").bounds = (bounds[0][0] - 2 * 3600, bounds[0][1])
    precip_cube.rename("precipitation_rate_max-PT03H")
    expected_cube.coord("time").bounds = (bounds[0][0] - 2 * 3600, bounds[0][1])
    result = LightningFromCapePrecip()(CubeList([cape_cube, precip_cube]))
    assert result.xml().splitlines(keepends=True) == expected_cube.xml().splitlines(
        keepends=True
    )
    assert np.allclose(result.data, expected_cube.data)


def test_with_model_attribute(cape_cube, precip_cube, expected_cube):
    """Run the plugin with model_id_attr and check the result cube matches the expected_cube"""
    expected_cube.attributes["mosg__model_configuration"] = "gl_ens"
    result = LightningFromCapePrecip(model_id_attr="mosg__model_configuration")(
        CubeList([cape_cube, precip_cube])
    )
    assert result.xml().splitlines(keepends=True) == expected_cube.xml().splitlines(
        keepends=True
    )
    assert np.allclose(result.data, expected_cube.data)


def break_time_point(cape_cube, precip_cube):
    """Modifies cape_cube time points to be incremented by 1 second and
    returns the error message this will trigger"""
    cape_cube.coord("time").points = cape_cube.coord("time").points + 1
    return r"CAPE cube time .* should be valid at the precipitation_rate_max cube lower bound .*"


def break_time_bound(cape_cube, precip_cube):
    """Modifies upper bound on precip_cube time coord to be incremented by 1 second and
    returns the error message this will trigger"""
    bounds = precip_cube.coord("time").bounds
    precip_cube.coord("time").bounds = (bounds[0][0], bounds[0][1] + 1)
    return r"Precipitation_rate_max cube time window must be one or three hours, not .*"


def break_reference_time(cape_cube, precip_cube):
    """Modifies precip_cube forecast_reference_time points to be incremented by 1 second
    and returns the error message this will trigger"""
    precip_cube.coord("forecast_reference_time").points = (
        precip_cube.coord("forecast_reference_time").points + 1
    )
    return r"Supplied cubes must have the same forecast reference times"


def break_latitude_point(cape_cube, precip_cube):
    """Modifies the first latitude point on the precip_cube (adds one degree)
    and returns the error message this will trigger"""
    points = list(precip_cube.coord("latitude").points)
    points[0] = points[0] + 1
    precip_cube.coord("latitude").points = points
    return "Supplied cubes do not have the same spatial coordinates"


def break_units(cape_cube, precip_cube):
    """Modifies the units of the precip_cube to something incompatible with "mm h-1"
    and returns the error message this will trigger"""
    precip_cube.units = "m"
    return r"Unable to convert from 'Unit\('m'\)' to 'Unit\('mm h-1'\)'."


def break_precip_name(cape_cube, precip_cube):
    """Modifies the name of precip_cube and returns the error message this will trigger"""
    precip_cube.rename("precipitation_rate")
    return "No cube named precipitation_rate_max found in .*"


def break_cape_name(cape_cube, precip_cube):
    """Modifies the name of cape_cube and returns the error message this will trigger"""
    cape_cube.rename("CAPE")
    return "No cube named atmosphere_convective_available_potential_energy found in .*"


@pytest.mark.parametrize(
    "breaking_function",
    (
        break_time_point,
        break_time_bound,
        break_reference_time,
        break_latitude_point,
        break_units,
        break_precip_name,
        break_cape_name,
    ),
)
def test_exceptions(cape_cube, precip_cube, breaking_function):
    """Tests that a suitable exception is raised when the precip cube meta-data does
    not match the cape cube"""
    error_msg = breaking_function(cape_cube, precip_cube)
    with pytest.raises(ValueError, match=error_msg):
        LightningFromCapePrecip()(CubeList([cape_cube, precip_cube]))
