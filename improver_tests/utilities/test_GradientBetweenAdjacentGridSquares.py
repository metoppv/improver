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
from iris.analysis import Linear
from iris.coords import DimCoord

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import GradientBetweenAdjacentGridSquares


EXAMPLE_INPUT_DATA = np.array(
    [
        [0,  1,  2],
        [1095014, 0, 1095014], #[40, 30, 400],
        [4,  5,  6]
    ], dtype=np.float32)
# EXAMPLE_INPUT_DATA_DIM_COORDS = [
#     DimCoord(np.array([0, 10, 20]), standard_name="latitude", units="degrees"),
#     DimCoord(np.array([0, 10, 20]), standard_name="longitude", units="degrees")
# ]

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
            EXAMPLE_INPUT_DATA,
            name="wind_speed",
            units="m s^-1",
            spatial_grid=spatial_grid,
            grid_spacing=grid_spacing,
            domain_corner=(0.0, 0.0),
            # dim_coords_and_dims=EXAMPLE_INPUT_DATA_DIM_COORDS
        )
        return cube

    return _make_input


@pytest.fixture(name="make_expected")
def make_expected_fixture() -> callable:
    """Factory as fixture for generating a cube of varying size."""

    def _make_expected(values, spatial_grid_type, grid_spacing, regrid=False) -> Cube:
        """Create a cube filled with data of a specific shape and value."""
        data = np.array(values, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="gradient_of_wind_speed",
            units="s^-1",
            spatial_grid=spatial_grid_type,
            grid_spacing=grid_spacing,
            attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
            domain_corner=(0.0, 0.0),
        )
        if regrid:
            # regridder = Linear().regridder(cube,
            #                                Cube(
            #                                    EXAMPLE_INPUT_DATA.shape,
            #                                    name="gradient_of_wind_speed",
            #                                    units="s^-1",
            #                                    spatial_grid=spatial_grid_type,
            #                                    grid_spacing=grid_spacing,
            #                                    attributes=MANDATORY_ATTRIBUTE_DEFAULTS)
            #                                )
            cube = cube.regrid(set_up_variable_cube(np.zeros(EXAMPLE_INPUT_DATA.shape, dtype=np.float32), units="s^-1", spatial_grid=spatial_grid_type, grid_spacing=grid_spacing), scheme=Linear)

        return cube

    return _make_expected


def get_expected_gradient_between_points(param_array, x_separations, y_separations, regrid=False):
    """
    Calculates the gradient of a 2d numpy array along the x and y axes, accounting for distance between the points.
    Gradients are calculated between grid points, meaning that the resulting arrays will be smaller by one dimension
    along the axis of differentiation.
    """
    x_diff = np.diff(param_array, axis=1)
    x_grad = x_diff / x_separations
    y_diff = np.diff(param_array, axis=0)
    y_grad = y_diff / y_separations
    if regrid:
        # Use linear interpolation to regrid expected data
        x_diff = np.diff(x_grad, axis=1)
        first_col = x_grad[:, [0]]
        second_col = x_grad[:, [1]]
        a = first_col + x_diff / 2
        b = (first_col + second_col) / 2
        c = second_col - x_diff / 2
        x_grad = np.hstack((a,b,c))

        y_diff = np.diff(y_grad, axis=0)
        a = y_grad[0] + y_diff / 2
        b = np.sum(y_grad, axis=0) / 2 #(y_grad[0] + y_grad[1]) # TODO: This doesn't work.
        c = y_grad[1] - y_diff / 2
        y_grad = np.vstack((a,b,c))
    return x_grad, y_grad


@pytest.mark.parametrize(
    "grid",
    (
        {"regrid": False, "xshape": (3, 2), "yshape": (2, 3)},
        {"regrid": True, "xshape": (3, 3), "yshape": (3, 3)},
    ),
)
def test_gradient_equal_area_coords(make_input, make_expected, grid):
    """Check calculating the gradient with and without regridding for equal area coordinate systems"""
    # TODO: make tests work for regridded option. Will have to look at iris linear interp docs to see what we're expecting at edges and if we think it's sensible.
    # expected_x_gradients = np.array(
    #     [
    #         [1 / EQUAL_AREA_GRID_SPACING, 1 / EQUAL_AREA_GRID_SPACING],
    #         [-10 / EQUAL_AREA_GRID_SPACING, 370 / EQUAL_AREA_GRID_SPACING],
    #         [1 / EQUAL_AREA_GRID_SPACING, 1 / EQUAL_AREA_GRID_SPACING]
    #     ]
    # )
    # expected_y_gradients = np.array(
    #     [
    #         [40 / EQUAL_AREA_GRID_SPACING, 29 / EQUAL_AREA_GRID_SPACING, 398 / EQUAL_AREA_GRID_SPACING],
    #         [-36 / EQUAL_AREA_GRID_SPACING, -25 / EQUAL_AREA_GRID_SPACING, -394 / EQUAL_AREA_GRID_SPACING]
    #     ]
    # )
    x_distances = np.full((EXAMPLE_INPUT_DATA.shape[0], EXAMPLE_INPUT_DATA.shape[1] - 1), EQUAL_AREA_GRID_SPACING)
    y_distances = np.full((EXAMPLE_INPUT_DATA.shape[0] - 1, EXAMPLE_INPUT_DATA.shape[1]), EQUAL_AREA_GRID_SPACING)
    expected_x_gradients, expected_y_gradients = get_expected_gradient_between_points(EXAMPLE_INPUT_DATA, x_distances, y_distances)
    wind_speed = make_input("equalarea", EQUAL_AREA_GRID_SPACING)
    expected_x = make_expected(expected_x_gradients, "equalarea", EQUAL_AREA_GRID_SPACING, grid["regrid"])
    expected_y = make_expected(expected_y_gradients, "equalarea", EQUAL_AREA_GRID_SPACING, grid["regrid"])
    gradient_x, gradient_y = GradientBetweenAdjacentGridSquares(regrid=grid["regrid"])(
        wind_speed
    )
    for result, expected in zip((gradient_x, gradient_y), (expected_x, expected_y)):
        assert result.name() == expected.name()
        assert result.attributes == expected.attributes
        assert result.units == expected.units
        np.testing.assert_allclose(expected.data, result.data, rtol=1e-5, atol=1e-8)




#TODO: How do we handle the edges??
@pytest.mark.parametrize(
    "grid",
    (
        {"regrid": False, "xshape": (3, 2), "yshape": (2, 3)},
        {"regrid": True, "xshape": (3, 3), "yshape": (3, 3)}, # ToDo
    ),
)
def test_gradient_lat_lon_coords(make_input, make_expected, grid):
    """Check calculating the gradient with and without regridding for global latitude/longitude coordinate system"""
    wind_speed = make_input("latlon", LATLON_GRID_SPACING)
    # expected_data_values = [1 / X_GRID_SPACING_AT_EQUATOR, 1 / X_GRID_SPACING_AT_10_DEGREES, 1 / X_GRID_SPACING_AT_20_DEGREES]  # Todo: Check if this is right.
    x_separations = np.array(
        [
            [X_GRID_SPACING_AT_EQUATOR, X_GRID_SPACING_AT_EQUATOR],
            [X_GRID_SPACING_AT_10_DEGREES_NORTH, X_GRID_SPACING_AT_10_DEGREES_NORTH],
            [X_GRID_SPACING_AT_20_DEGREES_NORTH, X_GRID_SPACING_AT_20_DEGREES_NORTH],
        ]
    )
    y_separations = np.full((EXAMPLE_INPUT_DATA.shape[0] - 1, EXAMPLE_INPUT_DATA.shape[1]), Y_GRID_SPACING)
    expected_x_gradients, expected_y_gradients = get_expected_gradient_between_points(EXAMPLE_INPUT_DATA, x_separations, y_separations, regrid=grid["regrid"]) # TODO: consider moving this into _make_expected()

    expected_x = make_expected(expected_x_gradients, "latlon", LATLON_GRID_SPACING)
    expected_y = make_expected(expected_y_gradients, "latlon", LATLON_GRID_SPACING)
    gradient_x, gradient_y = GradientBetweenAdjacentGridSquares(regrid=grid["regrid"])(
        wind_speed
    )
    for result, expected in zip((gradient_x, gradient_y), (expected_x, expected_y)):
        assert result.name() == expected.name()
        assert result.attributes == expected.attributes
        assert result.units == expected.units
        np.testing.assert_allclose(expected.data, result.data, rtol=1e-5, atol=1e-8)
