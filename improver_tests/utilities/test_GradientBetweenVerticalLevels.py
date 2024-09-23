# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the GradientBetweenVerticalLevels plugin."""


import iris
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.gradient_between_vertical_levels import (
    GradientBetweenVerticalLevels,
)


@pytest.fixture
def temperature_at_screen_level():
    """Set up a temperature at screen level cube"""
    data = np.linspace(0, 3, 4).reshape(2, 2).astype(np.float32)
    cube = set_up_variable_cube(data)
    cube.add_aux_coord(iris.coords.AuxCoord(1.5, long_name="height", units="m"))
    return cube


@pytest.fixture
def temperature_at_850hPa():
    """Set up a temperature at 850hPa cube"""
    data = np.linspace(3, 0, 4).reshape(2, 2).astype(np.float32)
    cube = set_up_variable_cube(data)
    cube.add_aux_coord(iris.coords.AuxCoord(85000, long_name="pressure", units="Pa"))
    return cube


@pytest.fixture
def orography():
    """Set up an orography cube"""
    data = np.full((2, 2), 8.5, dtype=np.float32)
    cube = set_up_variable_cube(data, name="surface_altitude", units="m")
    return cube


@pytest.fixture
def height_of_pressure_levels():
    """Set up a height of pressure levels cube"""
    data = np.array([np.full((2, 2), 10), np.full((2, 2), 110)])
    cube = set_up_variable_cube(
        data,
        height_levels=[100000, 85000],
        pressure=True,
        name="geopotential_height",
        units="m",
    )
    return cube


@pytest.mark.parametrize(
    "height_or_pressure", ["height_both", "pressure_both", "mixed"]
)
def test_height_and_pressure(
    temperature_at_screen_level,
    temperature_at_850hPa,
    orography,
    height_of_pressure_levels,
    height_or_pressure,
):
    """Test that the plugin produces the expected result with cubes defined either
    both on height or pressure levels. Also check the plugin produces the expected
    result with one cube defined on height levels and the other on pressure levels."""
    cubes=[orography,height_of_pressure_levels]
    if height_or_pressure == "height_both":
        temperature_at_850hPa.add_aux_coord(
            iris.coords.AuxCoord(101.5, long_name="height", units="m")
        )
        temperature_at_850hPa.remove_coord("pressure")
        cubes=[]
    elif height_or_pressure == "pressure_both":
        temperature_at_screen_level.add_aux_coord(
            iris.coords.AuxCoord(100000, long_name="pressure", units="Pa")
        )
        temperature_at_screen_level.remove_coord("height")
        cubes=[height_of_pressure_levels]

    cubes.append([temperature_at_850hPa, temperature_at_screen_level])

    expected = [[0.03, 0.01], [-0.01, -0.03]]
    result = GradientBetweenVerticalLevels()(
        iris.cube.CubeList(
            cubes 
        )
    )
    np.testing.assert_array_almost_equal(result.data, expected)
    assert result.name() == "gradient_of_air_temperature"
    assert result.units == "K/m"


def test_height_diff_0(
    temperature_at_screen_level,
    temperature_at_850hPa,
    orography,
    height_of_pressure_levels,
):
    """Test that if the height difference a some grid square is 0 the point is masked."""
    height_of_pressure_levels.data[1, 0, 0] = 10
    expected = [[np.inf, 0.01], [-0.01, -0.03]]
    expected_mask = [[True, False], [False, False]]

    result = GradientBetweenVerticalLevels()(
        iris.cube.CubeList(
            [
                temperature_at_850hPa,
                temperature_at_screen_level,
                height_of_pressure_levels,
                orography,
            ]
        )
    )
    np.testing.assert_array_almost_equal(result.data, expected)
    np.testing.assert_array_almost_equal(result.data.mask, expected_mask)
    assert result.name() == "gradient_of_air_temperature"
    assert result.units == "K/m"


def test_height_and_no_orography(
    temperature_at_screen_level, temperature_at_850hPa, height_of_pressure_levels
):
    """Test an error is raised if a height coordinate is present but no orography cube
    is present."""

    with pytest.raises(
        ValueError, match="No orography cube provided",
    ):
        GradientBetweenVerticalLevels()(
            iris.cube.CubeList(
                [
                    temperature_at_screen_level,
                    temperature_at_850hPa,
                    height_of_pressure_levels,
                ]
            )
        )


def test_pressure_coord_and_no_pressure_levels(
    temperature_at_screen_level, temperature_at_850hPa, orography
):
    """Test an error is raised if a pressure coordinate is present but no geopotential_height
    cube is present."""

    with pytest.raises(
        ValueError, match="No geopotential height",
    ):
        GradientBetweenVerticalLevels()(
            iris.cube.CubeList(
                [temperature_at_screen_level, temperature_at_850hPa, orography]
            )
        )
