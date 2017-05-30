# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Unit tests for the weights.ChooseDefaultWeightsLinear plugin."""


import unittest

from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
import iris
import numpy as np

from improver.weights import ChooseDefaultWeightsLinear as LinearWeights


def add_realizations(cube, num):
    """ Create num realizations of input cube
        Args:
            cube =iris.cube.Cube - input cube
            num = integer - Number of realizations
        Returns
            cubeout = iris.cube.Cube - copy of cube with num realizations added
    """
    cubelist = iris.cube.CubeList()
    for i in range(0, num):
        newcube = cube.copy()
        new_ensemble_coord = iris.coords.AuxCoord(i,
                                                  standard_name='realization')
        newcube.add_aux_coord(new_ensemble_coord)
        cubelist.append(newcube)
    cubeout = cubelist.merge_cube()
    return cubeout


class TestChooseDefaultWeightsLinear(IrisTest):
    """ Test the Default Linear Weights plugin """

    def setUp(self):
        data = np.zeros((2, 2, 2))
        data[0][:][:] = 0.0
        data[1][:][:] = 1.0
        cube = Cube(data, standard_name="precipitation_amount",
                    units="kg m^-2 s^-1")
        cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                                    units='degrees'), 1)
        cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                    units='degrees'), 2)
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_aux_coord(AuxCoord([402192.5, 402193.5],
                                    "time", units=tunit), 0)
        dummy_scalar_coord = iris.coords.AuxCoord(1,
                                                  long_name='scalar_coord',
                                                  units='no_unit')
        cube.add_aux_coord(dummy_scalar_coord)
        self.cube = cube

    def test_basic(self):
        """ Test that the plugin retuns an array of weights """
        coord = "time"
        plugin = LinearWeights()
        result = plugin.process(self.cube, coord)
        self.assertIsInstance(result, np.ndarray)

    def test_array_sum_equals_one(self):
        """ Test that the resulting weights add up to one """
        coord = "time"
        plugin = LinearWeights()
        result = plugin.process(self.cube, coord)
        self.assertAlmostEquals(result.sum(), 1.0)

    def test_fails_coord_not_in_cube(self):
        """Test it raises a Value Error if coord not in the cube."""
        coord = "notset"
        plugin = LinearWeights()
        msg = ('The coord for this plugin must be '
               'an existing coordinate in the input cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, coord)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Value Error if not supplied with a cube."""
        coord = "time"
        plugin = LinearWeights()
        notacube = 0.0
        msg = ('The first argument must be an instance of '
               'iris.cube.Cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(notacube, coord)

    def test_fails_y0val_lessthan_zero(self):
        """ Test it raises a Value Error if y0val less than zero """
        coord = "time"
        plugin = LinearWeights(y0val=-10.0)
        msg = ('y0val must be a float > 0.0')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, coord)

    def test_fails_ynval_and_slope_set(self):
        """ Test it raises a Value Error if slope and ynval set """
        coord = "time"
        plugin = LinearWeights(y0val=10.0, slope=-5.0, ynval=5.0)
        msg = ('Relative end point weight or slope must be set'
               ' but not both.')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, coord)

    def test_fails_weights_negative(self):
        """ Test it raises a Value Error if weights become negative """
        coord = "realization"
        plugin = LinearWeights(y0val=10.0, slope=-5.0)
        cubenew = add_realizations(self.cube, 6)
        msg = 'Weights must be positive, at least one value < 0.0'
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(cubenew, coord)

    def test_works_scalar_coord(self):
        """Test it works if scalar coordinate."""
        coord = 'scalar_coord'
        plugin = LinearWeights()
        result = plugin.process(self.cube, coord)
        self.assertArrayAlmostEqual(result, np.array([1.0]))

    def test_works_defaults_used(self):
        """Test it works if scalar coordinate."""
        coord = "time"
        plugin = LinearWeights()
        result = plugin.process(self.cube, coord)
        expected_result = np.array([0.90909091, 0.09090909])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_y0val_and_slope_set(self):
        """Test it works if y0val and slope_set."""
        coord = "time"
        plugin = LinearWeights(y0val=10.0, slope=-5.0)
        result = plugin.process(self.cube, coord)
        expected_result = np.array([0.66666667, 0.33333333])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_y0val_and_ynval_set(self):
        """Test it works if scalar coordinate."""
        coord = "time"
        plugin = LinearWeights(y0val=10.0, ynval=5.0)
        result = plugin.process(self.cube, coord)
        expected_result = np.array([0.66666667, 0.33333333])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_with_larger_num(self):
        """Test it works with larger num_of_vals"""
        coord = "realization"
        plugin = LinearWeights(y0val=10.0, ynval=5.0)
        cubenew = add_realizations(self.cube, 6)
        result = plugin.process(cubenew, coord)
        expected_result = np.array([0.22222222, 0.2,
                                    0.17777778, 0.15555556,
                                    0.13333333, 0.11111111])
        self.assertArrayAlmostEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
