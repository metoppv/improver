# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Test methods in lightning.LightningMultivariateProbability_USAF2024"""

from datetime import datetime

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.lightning import LightningMultivariateProbability_USAF2024
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.spatial import create_vicinity_coord


@pytest.fixture(name="cape_cube")
def cape_cube_fixture() -> Cube:
    """
    Set up a CAPE cube for use in tests over a variety of conditions.
    """

    data = np.array([[4000, 100], [0, 4000]], dtype=np.float32)
    data = np.repeat(data[np.newaxis, :, :], 2, axis=0)
    cube = set_up_variable_cube(
        data,
        name="atmosphere_specific_convective_available_potential_energy",
        units="J kg-1",
        time=datetime(2017, 11, 10, 3, 0),
        time_bounds=None,
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-20, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    return cube


@pytest.fixture(name="cin_cube")
def cin_cube_fixture() -> Cube:
    """
    Set up a liftidx cube for use in tests over a variety of conditions.
    """

    data = np.array([[0, -50], [-100, 0]], dtype=np.float32)
    data = np.repeat(data[np.newaxis, :, :], 2, axis=0)
    cube = set_up_variable_cube(
        data,
        name="atmosphere_specific_convective_inhibition",
        units="J kg-1",
        time=datetime(2017, 11, 10, 3, 0),
        time_bounds=None,
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-20, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    return cube


@pytest.fixture(name="liftidx_cube")
def liftidx_cube_fixture() -> Cube:
    """
    Set up a liftidx cube for use in tests over a variety of conditions.
    """

    data = np.array([[10, -5], [2, 10]], dtype=np.float32)
    data = np.repeat(data[np.newaxis, :, :], 2, axis=0)
    cube = set_up_variable_cube(
        data,
        name="temperature_difference_between_ambient_air_and_air_lifted_adiabatically",
        units="K",
        time=datetime(2017, 11, 10, 3, 0),
        time_bounds=None,
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-20, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    return cube


@pytest.fixture(name="pwat_cube")
def pwat_cube_fixture() -> Cube:
    """
    Set up a pwat cube for use in tests over a variety of conditions.
    """

    data = np.array([[3, 20], [20, 40]], dtype=np.float32)
    data = np.repeat(data[np.newaxis, :, :], 2, axis=0)
    cube = set_up_variable_cube(
        data,
        name="precipitable_water",
        units="kg m-2",
        time=datetime(2017, 11, 10, 3, 0),
        time_bounds=None,
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-20, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    return cube


@pytest.fixture(name="apcp_cube")
def apcp_cube_fixture() -> Cube:
    """
    Set up a apcp cube for use in tests over a variety of conditions.
    """

    data = np.array([[6, 0], [1, 10]], dtype=np.float32)
    data = np.repeat(data[np.newaxis, :, :], 2, axis=0)
    cube = set_up_variable_cube(
        data,
        name="precipitation_amount",
        units="kg m-2",
        time=datetime(2017, 11, 10, 6, 0),
        time_bounds=(datetime(2017, 11, 10, 3, 0), datetime(2017, 11, 10, 6, 0)),
        attributes=None,
        standard_grid_metadata="gl_ens",
        domain_corner=(-20, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )
    return cube


@pytest.fixture(name="expected_cube")
def expected_cube_fixture() -> Cube:
    """
    Set up the Lightning cube that we expect to get from the plugin.
    """

    data = np.array([[0.14111012, 0.02940975], [0.09720507, 0.95]], dtype=np.float32)
    data.resize(1, 2, 2)
    cube = set_up_probability_cube(
        data,
        thresholds=[0.0],
        variable_name="number_of_lightning_flashes_per_unit_area_in_vicinity",
        threshold_units="m-2",
        time=datetime(2017, 11, 10, 6, 0),
        time_bounds=(datetime(2017, 11, 10, 3, 0), datetime(2017, 11, 10, 6, 0)),
        attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        domain_corner=(-20, 0),
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

    vic_coord = create_vicinity_coord(20000)
    cube.add_aux_coord(vic_coord)

    return cube


def test_basic(cape_cube, liftidx_cube, pwat_cube, cin_cube, apcp_cube, expected_cube):
    """Run the plugin and check the result cube matches the expected_cube"""
    result = LightningMultivariateProbability_USAF2024()(
        CubeList([cape_cube, liftidx_cube, pwat_cube, cin_cube, apcp_cube]), None
    )
    assert result.xml().splitlines(keepends=True) == expected_cube.xml().splitlines(
        keepends=True
    )
    assert np.allclose(result.data, expected_cube.data)


def break_cape_name(cape_cube, precip_cube, cin_cube, li_cube, pw_cube):
    """Modifies cape_cube name to be changed to appear missing and
    returns the error message this will trigger"""
    cape_cube.rename("CAPE")
    return r"No cube named atmosphere_specific_convective_available_potential_energy found in .*"


def break_precip_name(cape_cube, precip_cube, cin_cube, li_cube, pw_cube):
    """Modifies precip_cube name to be changed to appear missing and
    returns the error message this will trigger"""
    precip_cube.rename("precip")
    return r"No cube named precipitation_amount found in .*"


def break_cin_name(cape_cube, precip_cube, cin_cube, li_cube, pw_cube):
    """Modifies cin_cube name to be changed to appear missing and
    returns the error message this will trigger"""
    cin_cube.rename("CIN")
    return r"No cube named atmosphere_specific_convective_inhibition found in .*"


def break_li_name(cape_cube, precip_cube, cin_cube, li_cube, pw_cube):
    """Modifies li_cube name to be changed to appear missing and
    returns the error message this will trigger"""
    li_cube.rename("LI")
    return r"No cube named temperature_difference_between_ambient_air_and_air_lifted_adiabatic.*"


def break_pw_name(cape_cube, precip_cube, cin_cube, li_cube, pw_cube):
    """Modifies pw_cube name to be changed to appear missing and
    returns the error message this will trigger"""
    pw_cube.rename("PW")
    return r"No cube named precipitable_water found in .*"


def break_cape_time(cape_cube, precip_cube, cin_cube, li_cube, pw_cube):
    """Modifies cape_cube time points to be incremented by 1 second and
    returns the error message this will trigger"""
    cape_cube.coord("time").points = cape_cube.coord("time").points + 1
    return r"The .*energy time .* should be valid at the precipitation_accumulation cube lower .*"


def break_precip_window(cape_cube, precip_cube, cin_cube, li_cube, pw_cube):
    """Modifies upper bound on precip_cube time coord to be incremented by 1 second and
    returns the error message this will trigger"""
    bounds = precip_cube.coord("time").bounds
    precip_cube.coord("time").bounds = (bounds[0][0], bounds[0][1] + 1)
    return r"Precipitation_accumulation cube time window must be three hours, not .*"


def break_reference_time(cape_cube, precip_cube, cin_cube, li_cube, pw_cube):
    """Modifies precip_cube time points to be incremented by 1 second and
    returns the error message this will trigger"""
    precip_cube.coord("forecast_reference_time").points = (
        precip_cube.coord("forecast_reference_time").points + 1
    )
    return r".* and .* do not have the same forecast reference time"


def break_units(cape_cube, precip_cube, cin_cube, li_cube, pw_cube):
    """Modifies cape_cube units to be incorrect as K and
    returns the error message this will trigger"""
    cape_cube.units = "K"
    return r"The .* units are incorrect, expected units as .* but received .*"


def break_coordinates(cape_cube, precip_cube, cin_cube, li_cube, pw_cube):
    """Modifies the first latitude point on the precip_cube (adds one degree)
    and returns the error message this will trigger"""
    points = list(precip_cube.coord("latitude").points)
    points[0] = points[0] + 1
    precip_cube.coord("latitude").points = points
    return ".* and .* do not have the same spatial coordinates"


@pytest.mark.parametrize(
    "breaking_function",
    (
        break_cape_name,
        break_precip_name,
        break_cin_name,
        break_li_name,
        break_pw_name,
        break_cape_time,
        break_precip_window,
        break_reference_time,
        break_units,
        break_coordinates,
    ),
)
def test_exceptions(
    cape_cube, apcp_cube, cin_cube, liftidx_cube, pwat_cube, breaking_function
):
    """Tests that a suitable exception is raised when the  cube meta-data does
    not match what is expected"""
    error_msg = breaking_function(
        cape_cube, apcp_cube, cin_cube, liftidx_cube, pwat_cube
    )
    with pytest.raises(ValueError, match=error_msg):
        LightningMultivariateProbability_USAF2024()(
            CubeList([cape_cube, apcp_cube, cin_cube, liftidx_cube, pwat_cube])
        )
