# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for nbhood.square_kernel.create_cube_with_halo."""

import unittest

import numpy as np
import iris
from iris.tests import IrisTest

from improver.utilities.pad_spatial import create_cube_with_halo
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_create_cube_with_halo(IrisTest):
    """Tests for the create_cube_with_halo function"""

    def setUp(self):
        """Set up a realistic input cube with lots of metadata.  Input cube
        grid is 1000x1000 km with points spaced 100 km apart."""
        attrs = {'history': '2018-12-10Z: StaGE Decoupler',
                 'title': 'Temperature on UK 2 km Standard Grid',
                 'source': 'Met Office Unified Model'}

        self.cube = set_up_variable_cube(
            np.ones((11, 11), dtype=np.float32), spatial_grid='equalarea',
            standard_grid_metadata='uk_det', attributes=attrs)
        self.grid_spacing = 100000

    def test_basic(self):
        """Test function returns a cube with expected metadata"""
        halo_size_km = 162.
        result = create_cube_with_halo(self.cube, halo_size_km)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), 'grid_with_halo')
        self.assertFalse(result.attributes)

    def test_values(self):
        """Test coordinate values with standard halo radius (rounds down to 1
        grid cell)"""
        halo_size_km = 162.

        x_min = self.cube.coord(axis='x').points[0] - 2*self.grid_spacing
        x_max = self.cube.coord(axis='x').points[-1] + 2*self.grid_spacing
        expected_x_points = np.arange(x_min, x_max+1, self.grid_spacing)
        y_min = self.cube.coord(axis='y').points[0] - 2*self.grid_spacing
        y_max = self.cube.coord(axis='y').points[-1] + 2*self.grid_spacing
        expected_y_points = np.arange(y_min, y_max+1, self.grid_spacing)

        result = create_cube_with_halo(self.cube, halo_size_km)
        self.assertSequenceEqual(result.data.shape, (13, 13))
        self.assertArrayAlmostEqual(
            result.coord(axis='x').points, expected_x_points)
        self.assertArrayAlmostEqual(
            result.coord(axis='y').points, expected_y_points)

        # check explicitly that the original grid remains an exact subset of
        # the output cube (ie that padding hasn't shifted the existing grid)
        self.assertArrayAlmostEqual(result.coord(axis='x').points[2:-2],
                                    self.cube.coord(axis='x').points)
        self.assertArrayAlmostEqual(result.coord(axis='y').points[2:-2],
                                    self.cube.coord(axis='y').points)


if __name__ == '__main__':
    unittest.main()
