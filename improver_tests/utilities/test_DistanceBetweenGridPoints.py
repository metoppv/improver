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
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import DistanceBetweenGridSquares



def make_test_cube(spatial_grid, grid_spacing):
    EXAMPLE_DATA = np.array([[0, 1, 10], [2, 3, 10], [4, 5, 10]], dtype=np.float32)
    cube = set_up_variable_cube(
        EXAMPLE_DATA, name="wind_speed", units="m s^-1", spatial_grid=spatial_grid, grid_spacing=grid_spacing,
        domain_corner=(0.0, 0.0)
    )
    return cube


def test_latlon_cube():
    input_cube = make_test_cube("latlon", 10)
    expected_x_distances = np.array([[1111949, 1111949],
                                     [1095014, 1095014],
                                     [1044735, 1044735]])
    expected_y_distances = np.full((2,3), 1111949)
    calculated_x_distances_cube, calculated_y_distances_cube = DistanceBetweenGridSquares()(input_cube)
    for result, expected in reversed(list(zip((calculated_x_distances_cube, calculated_y_distances_cube), (expected_x_distances, expected_y_distances)))):
        assert result.units == "meters" # TODO: is this correct?
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-3, atol=0)  # Allowing 0.2% error for difference between the spherical earth assumption used by the implementation and the full haversine equation used to generate the test data.




def test_latlon_cube_unequal_xy_dims():
    raise NotImplementedError
def test_latlon_cube_nonuniform_spacing():
    raise NotImplementedError