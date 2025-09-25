# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Test methods in turbulence.TurbulenceIndexAbove1500m_USAF.
It uses Pytest Fixtures to provide a fixed baseline for tests to run on top of."""
from datetime import datetime
import numpy as np
import pytest
import iris
from iris.cube import Cube, CubeList
from improver.turbulence import TurbulenceIndexAbove1500m_USAF

from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    set_up_variable_cube,
)

@pytest.fixture(name="u_wind_high_pres_cube")
def u_wind_high_pres_cube_fixture() -> Cube:
    """
    Set up a wind cube for use in tests over a variety of conditions.
    """
    pressure_mb = int(600)
    name = f"UWindComponentAt{pressure_mb}mb"
    data = np.array([[10, 20], [-10, 30]], dtype=np.float32)
    data = np.repeat(data[np.newaxis, :, :], 2, axis=0)
    cube = set_up_variable_cube(
        data,
        name=name,
        units="m s-1",
        time=datetime(2025, 5, 16, 1, 0),
        time_bounds=None,
        attributes=None,
        domain_corner=(-20, 0),
        x_grid_spacing=0.25,
        y_grid_spacing=0.25,
    )

    new_aux_coord = iris.coords.DimCoord(pressure_mb*100, long_name="pressure", var_name="pressure", units='Pa')
    cube.add_aux_coord(new_aux_coord)
    return cube

@pytest.fixture(name="v_wind_high_pres_cube")
def v_wind_high_pres_cube_fixture(u_wind_high_pres_cube) -> Cube:
    c = u_wind_high_pres_cube.copy()
    p = c.coord(var_name="pressure").cell(0).point
    name = f"VWindComponentAt{p//100}mb"
    # Rename will set either the standard_name attribute if a valid CF name, or the long_name oif not.
    # It always clears the var_name value as part of this function call.
    c.rename(name)
    return c

@pytest.fixture(name="u_wind_low_pres_cube")
def u_wind_low_pres_cube_fixture(u_wind_high_pres_cube) -> Cube:
    c = u_wind_high_pres_cube.copy()
    c.data *= 0.9
    p = c.coord(var_name="pressure").cell(0).point
    p -= 50*100
    name = f"UWindComponentAt{p//100}mb"
    # You can't set the value of cell(0).point directly. You need to remove it then put in new one.
    c.remove_coord(c.coord(var_name="pressure"))
    new_aux_coord = iris.coords.DimCoord(p, long_name="pressure", var_name="pressure", units='Pa')
    c.add_aux_coord(new_aux_coord)
    c.rename(name)
    return c

@pytest.fixture(name="v_wind_low_pres_cube")
def v_wind_low_pres_cube_fixture(v_wind_high_pres_cube) -> Cube:
    c = v_wind_high_pres_cube.copy()
    p = c.coord(var_name="pressure").cell(0).point
    p -= 50*100
    name = f"VWindComponentAt{p//100}mb"
    c.remove_coord(c.coord(var_name="pressure"))
    new_aux_coord = iris.coords.DimCoord(p, long_name="pressure", var_name="pressure", units='Pa')
    c.add_aux_coord(new_aux_coord)
    c.rename(name)
    return c

@pytest.fixture(name="geopotential_high_pres_cube")
def geopotential_high_pres_cube_fixture() -> Cube:
    """
    Set up a geopotential height cube for use in tests over a variety of conditions.
    """
    pressure_mb = int(600)
    name = f"GeopotentialHeightAt{pressure_mb}mb"
    data = np.array([[4800, 4870], [4871, 4865]], dtype=np.float32)
    data = np.repeat(data[np.newaxis, :, :], 2, axis=0)
    cube = set_up_variable_cube(
        data,
        name=name,
        units="m",
        time=datetime(2025, 5, 16, 1, 0),
        time_bounds=None,
        attributes=None,
        domain_corner=(-20, 0),
        x_grid_spacing=0.25,
        y_grid_spacing=0.25,
    )

    new_aux_coord = iris.coords.DimCoord(pressure_mb*100, long_name="pressure", var_name="pressure", units='Pa')
    cube.add_aux_coord(new_aux_coord)
    return cube

@pytest.fixture(name="geopotential_low_pres_cube")
def geopotential_low_pres_cube_fixture(geopotential_high_pres_cube) -> Cube:
    c = geopotential_high_pres_cube.copy()
    c.data += 100
    p = c.coord(var_name="pressure").cell(0).point
    p -= 50*100
    name = f"GeopotentialHeightAt{p//100}mb"
    c.remove_coord(c.coord(var_name="pressure"))
    new_aux_coord = iris.coords.DimCoord(p, long_name="pressure", var_name="pressure", units='Pa')
    c.add_aux_coord(new_aux_coord)
    c.rename(name)
    return c


@pytest.fixture(name="expected_cube")
def expected_cube_fixture() -> Cube:

    data = np.array([[1.81762553e-5, 2.8520788e-6], [1.8155379e-5, 4.283036e-6]], dtype=np.float32)
    data.resize(1, 2, 2)
    pressure = 60000
    cube = set_up_probability_cube(
        data,
        thresholds=[0.0],
        variable_name=f"TurbulenceIndexAbove1500mAt{pressure//100}mb",
        threshold_units="s-2",
        time=datetime(2025, 5, 16, 1, 0),
        time_bounds=None,
        attributes=None,
        domain_corner=(-20, 0),
        x_grid_spacing=20,
        y_grid_spacing=20,
    )

    return cube


def test_basic(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube, expected_cube):
    """Run the plugin and check the result cube matches the expected_cube"""
    result = TurbulenceIndexAbove1500m_USAF()(
        CubeList([u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube])    )
    print(result)

    assert np.allclose(result.data, expected_cube.data)


def break_num_u_winds(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """By renaming the cube named below, it will appear that not all the required
    cube types needed for a calculation are available."""
    u_wind_high_pres_cube.rename("foobar")
    return r"Only two cubes of UWindComponents should be passed, 1 provided."

def break_num_v_winds(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """By renaming the cube named below, it will appear that not all the required
    cube types needed for a calculation are available."""
    v_wind_high_pres_cube.rename("foobar")
    return r"Only two cubes of VWindComponents should be passed, 1 provided."

def break_num_geopot(u_wind_high_pres_cube, u_wind_low_pres_cube,
                      v_wind_high_pres_cube, v_wind_low_pres_cube,
                      geopotential_high_pres_cube, geopotential_low_pres_cube):
    """By renaming the cube named below, it will appear that not all the required
    cube types needed for a calculation are available."""
    geopotential_high_pres_cube.rename("foobar")
    return r"Only two cubes of GeopotentialHeight should be passed, 1 provided."

def break_u_winds_levels(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """Set pressure levels for two cubes that should be at different pressure levels to generate exception."""
    # Place at same pressure level
    p = u_wind_low_pres_cube.coord(var_name="pressure").cell(0).point
    u_wind_high_pres_cube.remove_coord(u_wind_high_pres_cube.coord(var_name="pressure"))
    new_aux_coord = iris.coords.DimCoord(p, long_name="pressure", var_name="pressure", units='Pa')
    u_wind_high_pres_cube.add_aux_coord(new_aux_coord)
    return "Passed UWindComponents should be at two different pressure levels."

def break_v_winds_levels(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """Set pressure levels for two cubes that should be at different pressure levels to generate exception."""
    # Place at same pressure level
    p = v_wind_low_pres_cube.coord(var_name="pressure").cell(0).point
    v_wind_high_pres_cube.remove_coord(v_wind_high_pres_cube.coord(var_name="pressure"))
    new_aux_coord = iris.coords.DimCoord(p, long_name="pressure", var_name="pressure", units='Pa')
    v_wind_high_pres_cube.add_aux_coord(new_aux_coord)
    return "Passed VWindComponents should be at two different pressure levels."

def break_geopot_levels(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """Set pressure levels for two cubes that should be at different pressure levels to generate exception."""
    # Place at same pressure level
    p = geopotential_high_pres_cube.coord(var_name="pressure").cell(0).point
    geopotential_low_pres_cube.remove_coord(geopotential_low_pres_cube.coord(var_name="pressure"))
    new_aux_coord = iris.coords.DimCoord(p, long_name="pressure", var_name="pressure", units='Pa')
    geopotential_low_pres_cube.add_aux_coord(new_aux_coord)
    return "Passed GeopotentialHeight should be at two different pressure levels."

def break_winds_unique_levels(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """Alter one of the cube's pressure level such that it is doe not match the either of the other passed levels.
    Only two district pressure levels should be provided."""
    # Place at unique pressure level
    p = u_wind_high_pres_cube.coord(var_name="pressure").cell(0).point
    p += 1 # Make sure one of the geopototential levels does not match any of the U levels.
    v_wind_high_pres_cube.remove_coord(v_wind_high_pres_cube.coord(var_name="pressure"))
    new_aux_coord = iris.coords.DimCoord(p, long_name="pressure", var_name="pressure", units='Pa')
    v_wind_high_pres_cube.add_aux_coord(new_aux_coord)

    return r"Passed VWindComponents pressure levels .* inconsistent "\
           r"with UWindComponents pressure levels .*."

def break_geopot_unique_levels(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """Alter one of the cube's pressure level such that it is doe not match the either of the other passed levels.
    Only two district pressure levels should be provided."""
    # Place at unique pressure level
    p = geopotential_low_pres_cube.coord(var_name="pressure").cell(0).point
    p += 1 # Make sure one of the geopototential levels does not match any of the U levels.
    geopotential_high_pres_cube.remove_coord(geopotential_high_pres_cube.coord(var_name="pressure"))
    new_aux_coord = iris.coords.DimCoord(p, long_name="pressure", var_name="pressure", units='Pa')
    geopotential_high_pres_cube.add_aux_coord(new_aux_coord)

    return r"Passed GeopotentialHeight pressure levels .* inconsistent "\
           r"with UWindComponents pressure levels .*."

def break_u_latitude_coords(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """Alter the bounds to generate an exception on passed coordinates which
    should be consistent for all passed cubes."""
    bounds = u_wind_high_pres_cube.coord("latitude").bounds
    u_wind_high_pres_cube.coord("latitude").bounds = bounds + 1

    return r"Incompatible coordinates: latitude"

def break_u_longitude_coords(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """Alter the bounds to generate an exception on passed coordinates which
    should be consistent for all passed cubes."""
    bounds = u_wind_high_pres_cube.coord("longitude").bounds
    u_wind_high_pres_cube.coord("longitude").bounds = bounds + 1

    return r"Incompatible coordinates: longitude"

def break_forecast_period(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """Alter the bounds to generate an exception on passed coordinates which
    should be consistent for all passed cubes."""
    u_wind_high_pres_cube.coord("forecast_period").points = (
            u_wind_high_pres_cube.coord("forecast_period").points + 1)

    return r"Incompatible coordinates: forecast_period"

def break_forecast_reference_time(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """Alter the bounds to generate an exception on passed coordinates which
        should be consistent for all passed cubes."""
    u_wind_high_pres_cube.coord("forecast_reference_time").points = (
            u_wind_high_pres_cube.coord("forecast_reference_time").points + 1)

    return r"Incompatible coordinates: forecast_reference_time"

def break_time(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube):
    """Alter the bounds to generate an exception on passed coordinates which
    should be consistent for all passed cubes."""
    u_wind_high_pres_cube.coord("time").points = u_wind_high_pres_cube.coord("time").points + 1

    return r"Incompatible coordinates: time"

@pytest.mark.parametrize(
    "breaking_function",
    (
        break_num_u_winds,
        break_num_v_winds,
        break_num_geopot,
        break_u_winds_levels,
        break_v_winds_levels,
        break_geopot_levels,
        break_winds_unique_levels,
        break_geopot_unique_levels,
        break_u_latitude_coords,
        break_u_longitude_coords,
        break_forecast_period,
        break_forecast_reference_time,
        break_time,
    ),
)
def test_exceptions(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube,
                breaking_function
):
    """Test that suitable exceptions are raised when the cube meta-data does
    not match what is expected"""
    error_msg = breaking_function(u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube)

    with pytest.raises(ValueError, match=error_msg):
        TurbulenceIndexAbove1500m_USAF()(
            CubeList([u_wind_high_pres_cube, u_wind_low_pres_cube,
               v_wind_high_pres_cube, v_wind_low_pres_cube,
               geopotential_high_pres_cube, geopotential_low_pres_cube])
        )
