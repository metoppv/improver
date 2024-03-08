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
""" Tests of DifferenceBetweenAdjacentGridSquares plugin."""

import unittest

import iris
import numpy as np
# from iris.coords import CellMethod
from iris.cube import Cube
from iris.coords import DimCoord
from iris.coord_systems import GeogCS, LambertAzimuthalEqualArea
# from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import DistanceBetweenGridSquares


# Distances covered when travelling 10 degrees east/west at different latitudes:
X_GRID_SPACING_AT_EQUATOR = 1111949  # Meters
X_GRID_SPACING_AT_10_DEGREES_NORTH = 1095014  # Meters
X_GRID_SPACING_AT_20_DEGREES_NORTH = 1044735  # Meters
Y_GRID_SPACING = 1111949  # Meters




#Todo: Need to think about how to handle equalarea cubes. Considerations are:
'''
Calculation already exists for equalarea cube (spatial.calculate_grid_spacing()) but this returns a float, whereas I need a cube.
Options are:
1. Use this function in my Distance class... seems hacky.
2. Copy the logic from this function into my class :'(
3. Make my class just handle the latlon projection and have the gradient class call the function for equalarea then boost the float up to a cube.
4. Move the function logic into the class and then have the function use the class for backwards compatibility.   - Maybe the best option. Function logic could be in a public static method so that both function and class can use it.

ToDo:
Move function logic into static method on class. Make sure the function's tests all still pass.
Add logic for class to check which projection we're using and route accordingly.
Make sure latlong tests still pass.
Write test for equal area projection.
Implement logic in class to make this test pass.
Add edge case tests and bugfix as needed.

Cup of tea.

Go back to gradient class and make it use the distance class. Existing test should give a good starting point.
Edge case tests for gradient.
Tidying up.
'''

def make_equalarea_cube(shape, grid_spacing):
    data = np.ones(shape, dtype=np.float32)
    cube = set_up_variable_cube(data, spatial_grid="equalarea", grid_spacing=grid_spacing)
    return cube

def make_latlon_test_cube(shape, latitudes, longitudes):
    example_data = np.ones(shape, dtype=np.float32)
    dimcoords = [(DimCoord(latitudes, standard_name="latitude", units="degrees", coord_system=GeogCS), 0),
                 (DimCoord(longitudes, standard_name="longitude", units="degrees", coord_system=GeogCS), 1)]
    cube = Cube(example_data, standard_name="wind_speed", units="m s^-1",
                dim_coords_and_dims=dimcoords)  # TODO: Do I need a domain corner?
    return cube


def test_latlon_cube():
    input_cube = make_latlon_test_cube((3,3), latitudes=[0, 10, 20], longitudes=[0, 10, 20])
    expected_x_distances = np.array([
        [   X_GRID_SPACING_AT_EQUATOR,          X_GRID_SPACING_AT_EQUATOR           ],
        [   X_GRID_SPACING_AT_10_DEGREES_NORTH, X_GRID_SPACING_AT_10_DEGREES_NORTH  ],
        [   X_GRID_SPACING_AT_20_DEGREES_NORTH, X_GRID_SPACING_AT_20_DEGREES_NORTH  ]
    ])
    expected_y_distances = np.full((2,3), Y_GRID_SPACING)
    calculated_x_distances_cube, calculated_y_distances_cube = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip((calculated_x_distances_cube, calculated_y_distances_cube),
                                (expected_x_distances, expected_y_distances)):
        assert result.units == "meters"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-3, atol=0)  # Allowing 0.2% error for difference between the spherical earth assumption used by the implementation and the full haversine equation used to generate the test data.

def test_latlon_cube_unequal_xy_dims():
    input_cube = make_latlon_test_cube((3, 2), latitudes=[0, 10, 20], longitudes=[0, 10])
    expected_x_distances = np.array([
        [X_GRID_SPACING_AT_EQUATOR],
        [X_GRID_SPACING_AT_10_DEGREES_NORTH],
        [X_GRID_SPACING_AT_20_DEGREES_NORTH]
    ])
    expected_y_distances = np.full((2, 2), Y_GRID_SPACING)
    calculated_x_distances_cube, calculated_y_distances_cube = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip((calculated_x_distances_cube, calculated_y_distances_cube),
                                (expected_x_distances, expected_y_distances)):
        assert result.units == "meters"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-3,
                                   atol=0)  # Allowing 0.2% error for difference between the spherical earth assumption used by the implementation and the full haversine equation used to generate the test data.


def test_latlon_cube_nonuniform_spacing():
    input_cube = make_latlon_test_cube((2, 3), latitudes=[0, 20], longitudes=[0, 10, 20])
    expected_x_distances = np.array([
        [X_GRID_SPACING_AT_EQUATOR,         X_GRID_SPACING_AT_EQUATOR],
        [X_GRID_SPACING_AT_20_DEGREES_NORTH, X_GRID_SPACING_AT_20_DEGREES_NORTH]
    ])
    expected_y_distances = np.full((1, 3), 2 * Y_GRID_SPACING)
    calculated_x_distances_cube, calculated_y_distances_cube = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip((calculated_x_distances_cube, calculated_y_distances_cube),
                                (expected_x_distances, expected_y_distances)):
        assert result.units == "meters"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-3,
                                   atol=0)  # Allowing 0.2% error for difference between the spherical earth assumption used by the implementation and the full haversine equation used to generate the test data.


def test_equalarea_cube():
    input_cube = make_equalarea_cube((3, 3), grid_spacing=1000)
    expected_x_distances = np.full((3,2), 1000)
    expected_y_distances = np.full((2, 3), 1000)
    calculated_x_distances_cube, calculated_y_distances_cube = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip((calculated_x_distances_cube, calculated_y_distances_cube),
                                (expected_x_distances, expected_y_distances)):
        assert result.units == "meters"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-5, atol=0)