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
""" Tests of GradientBetweenAdjacentGridSquares plugin."""

import numpy as np
import pytest
from iris.cube import Cube

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import GradientBetweenAdjacentGridSquares


EXAMPLE_DATA = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.float32)

EQUAL_AREA_GRID_SPACING = 1000  # Meters
LATLON_GRID_SPACING = 10  # Degrees
X_GRID_SPACING_AT_EQUATOR = 1111949  # Meters
X_GRID_SPACING_AT_10_DEGREES_NORTH = 1095014  # Meters
X_GRID_SPACING_AT_20_DEGREES_NORTH = 1044735  # Meters
Y_GRID_SPACING = 1111949  # Meters


@pytest.fixture(name="make_input")
def make_wind_speed_fixture() -> callable:
    """Factory as fixture for generating a wind speed cube as test input."""

    def _make_input(spatial_grid, grid_spacing) -> Cube:
        """Wind speed in m/s"""
        cube = set_up_variable_cube(
            EXAMPLE_DATA, name="wind_speed", units="m s^-1", spatial_grid=spatial_grid, grid_spacing=grid_spacing, domain_corner=(0.0, 0.0)
        )
        # for axis in ["x", "y"]:
        #     print(cube.coord(axis=axis).points)
        return cube

    return _make_input


@pytest.fixture(name="make_expected")
def make_expected_fixture() -> callable:
    """Factory as fixture for generating a cube of varying size."""

    def _make_expected(values, spatial_grid_type, grid_spacing) -> Cube:
        """Create a cube filled with data of a specific shape and value."""
        data = np.array(values, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="gradient_of_wind_speed",
            units="s^-1",
            spatial_grid=spatial_grid_type,
            grid_spacing=grid_spacing,
            attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        )
        return cube

    return _make_expected


@pytest.mark.parametrize(
    "grid",
    (
        {"regrid": False, "xshape": (3, 2), "yshape": (2, 3)},
        {"regrid": True, "xshape": (3, 3), "yshape": (3, 3)},
    ),
)
def test_gradient_equal_area_coords(make_input, make_expected, grid):
    """Check calculating the gradient with and without regridding for equal area coordinate systems"""
    # print(wind_speed)
    # print("\n\n\n")
    wind_speed = make_input("equalarea", EQUAL_AREA_GRID_SPACING)
    expected_x = make_expected(np.full(grid["xshape"], 1 / EQUAL_AREA_GRID_SPACING), "equalarea", EQUAL_AREA_GRID_SPACING)
    expected_y = make_expected(np.full(grid["yshape"], 2 / EQUAL_AREA_GRID_SPACING), "equalarea", EQUAL_AREA_GRID_SPACING)
    gradient_x, gradient_y = GradientBetweenAdjacentGridSquares(regrid=grid["regrid"])(
        wind_speed
    )
    for result, expected in zip((gradient_x, gradient_y), (expected_x, expected_y)):
        assert result.name() == expected.name()
        assert result.attributes == expected.attributes
        assert result.units == expected.units
        np.testing.assert_allclose(result.data, expected.data, rtol=1e-5, atol=1e-8)


def get_gradient_between_points(param_array, x_separations, y_separations):
    """
    Calculates the gradient of a 2d numpy array along the x and y axes, accounting for distance between the points.
    Gradients are calculated between grid points, meaning that the resulting arrays will be smaller by one dimension
    along the axis of differentiation.
    """
    x_diff = np.diff(param_array, axis=1)
    x_grad = x_diff / x_separations
    y_diff = np.diff(param_array, axis=0)
    y_grad = y_diff / y_separations
    return x_grad, y_grad

#TODO: How do we handle the edges??
# def get_gradient_at_points(param_array, x_separations, y_separations):
#     """
#     Calculates the gradient of a 2d numpy array along the x and y axes, accounting for distance between the points.
#     Gradients are first calculated between the points, and then linear interpolation is used to re-apply the gradients
#     to the original points, meaning the output array has the same dimensions as the input array. Todo: how are edges handled??
#     """
#     x_grad_between_points, y_grad_between_points = get_gradient_between_points(param_array, x_separations, y_separations)




@pytest.mark.parametrize(
    "grid",
    (
        {"regrid": False, "xshape": (3, 2), "yshape": (2, 3)},
        # {"regrid": True, "xshape": (3, 3), "yshape": (3, 3)}, # ToDo
    ),
)
def test_gradient_lat_lon_coords(make_input, make_expected, grid):
    """Check calculating the gradient with and without regridding for global latitude/longitude coordinate system"""
    wind_speed = make_input("latlon", LATLON_GRID_SPACING)
    # expected_data_values = [1 / X_GRID_SPACING_AT_EQUATOR, 1 / X_GRID_SPACING_AT_10_DEGREES, 1 / X_GRID_SPACING_AT_20_DEGREES]  # Todo: Check if this is right.
    x_separations = np.array(
        [
            [X_GRID_SPACING_AT_20_DEGREES_NORTH, X_GRID_SPACING_AT_20_DEGREES_NORTH],
            [X_GRID_SPACING_AT_10_DEGREES_NORTH, X_GRID_SPACING_AT_10_DEGREES_NORTH],
            [X_GRID_SPACING_AT_EQUATOR, X_GRID_SPACING_AT_EQUATOR]
        ]
    )
    y_separations = np.full((EXAMPLE_DATA.shape[0] - 1, EXAMPLE_DATA.shape[1]), Y_GRID_SPACING)
    expected_x_gradients, expected_y_gradients = get_gradient_between_points(EXAMPLE_DATA, x_separations, y_separations) # TODO: consider moving this into _make_expected()

    expected_x = make_expected(expected_x_gradients, "latlon", LATLON_GRID_SPACING)
    expected_y = make_expected(expected_y_gradients, "latlon", LATLON_GRID_SPACING)        #np.full(grid["yshape"], 2 / Y_GRID_SPACING))
    gradient_x, gradient_y = GradientBetweenAdjacentGridSquares(regrid=grid["regrid"])(
        wind_speed
    )
    for result, expected in zip((gradient_x, gradient_y), (expected_x, expected_y)):
        assert result.name() == expected.name()
        assert result.attributes == expected.attributes
        assert result.units == expected.units
        np.testing.assert_allclose(result.data, expected.data, rtol=1e-5, atol=1e-8)
