# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Unit tests for the the stand alone functions in generate_ancillary.py"""

import unittest

import numpy as np
from cf_units import Unit
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from iris.fileformats.pp import EARTH_RADIUS
from iris.tests import IrisTest

from improver.generate_ancillaries.generate_ancillary import _make_mask_cube


def _make_test_cube(long_name):
    """Make a cube to run tests upon"""
    cs = GeogCS(EARTH_RADIUS)
    data = np.array([[1., 1., 1.],
                     [0., 0., 0.],
                     [1., 0., 1.]])
    cube = Cube(data, long_name=long_name)
    x_coord = DimCoord(np.linspace(-45.0, 45.0, 3), 'latitude',
                       units='degrees', coord_system=cs)
    y_coord = DimCoord(np.linspace(120, 180, 3), 'longitude',
                       units='degrees', coord_system=cs)
    cube.add_dim_coord(x_coord, 0)
    cube.add_dim_coord(y_coord, 1)
    return cube


class Test__make_mask_cube(IrisTest):
    """Test private function to make cube from generated mask"""

    def setUp(self):
        """setting up test data"""
        premask = np.array([[0., 3., 2.],
                            [0.5, 0., 1.5],
                            [0.2, 0., 0]])
        self.mask = np.ma.masked_where(premask > 1., premask)

        self.x_coord = DimCoord([1, 2, 3], long_name='longitude')
        self.y_coord = DimCoord([1, 2, 3], long_name='latitude')
        self.coords = [self.x_coord, self.y_coord]
        self.upper = 100.
        self.lower = 0.
        self.units = "m"

    def test_wrong_number_of_bounds(self):
        """test checking that an exception is raised when the _make_mask_cube
        method is called with an incorrect number of bounds."""
        emsg = "should have only an upper and lower limit"
        with self.assertRaisesRegex(TypeError, emsg):
            _make_mask_cube(self.mask, self.coords, [0], self.units)
        with self.assertRaisesRegex(TypeError, emsg):
            _make_mask_cube(self.mask,
                            self.coords,
                            [0, 2, 4], self.units)

    def test_upperbound_fails(self):
        """test checking that an exception is raised when the _make_mask_cube
        method is called with only an upper bound."""
        emsg = "should have both an upper and lower limit"
        with self.assertRaisesRegex(TypeError, emsg):
            _make_mask_cube(self.mask,  self.coords,
                            [None, self.upper], self.units)

    def test_bothbounds(self):
        """test creating cube with both thresholds set"""
        result = _make_mask_cube(self.mask, self.coords,
                                 [self.lower, self.upper], self.units)
        self.assertEqual(result.coord('topographic_zone').bounds[0][1],
                         self.upper)
        self.assertEqual(result.coord('topographic_zone').bounds[0][0],
                         self.lower)
        self.assertEqual(result.coord('topographic_zone').points,
                         np.mean([self.lower, self.upper]))
        self.assertEqual(result.coord('topographic_zone').units, Unit('m'))

    def test_cube_attribute_no_seapoints(self):
        """Test the new attribute is added to the cube."""
        result = _make_mask_cube(self.mask, self.coords,
                                 [self.lower, self.upper], self.units)
        self.assertEqual(
            result.attributes["topographic_zones_include_seapoints"], "False")

    def test_cube_attribute_include_seapoints(self):
        """Test the new attribute is added to the cube when seapoints
           included."""
        result = _make_mask_cube(self.mask, self.coords,
                                 [self.lower, self.upper], self.units,
                                 sea_points_included="True")
        self.assertEqual(
            result.attributes["topographic_zones_include_seapoints"], "True")


if __name__ == "__main__":
    unittest.main()
