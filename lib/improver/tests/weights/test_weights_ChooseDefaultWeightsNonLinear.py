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

from improver.weights import ChooseDefaultWeightsNonLinear as NonLinearWeights


def add_realizations(cube, num):
    """Create num realizations of input cube.
        Args:
            cube : iris.cube.Cube
                   input cube.
            num : integer
                   Number of realizations.
        Returns:
            cubeout : iris.cube.Cube
                      copy of cube with num realizations added.
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


class TestChooseDefaultWeightsNonLinear(IrisTest):
    """Test the Default non-Linear Weights plugin. """

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
        """Test that the plugin returns an array of weights. """
        coord = "time"
        plugin = NonLinearWeights()
        result = plugin.process(self.cube, coord)
        self.assertIsInstance(result, np.ndarray)

    def test_array_sum_equals_one(self):
        """Test that the resulting weights add up to one. """
        coord = "time"
        plugin = NonLinearWeights()
        result = plugin.process(self.cube, coord)
        self.assertAlmostEquals(result.sum(), 1.0)

    def test_fails_coord_not_in_cube(self):
        """Test it raises a Value Error if coord not in the cube. """
        coord = "notset"
        plugin = NonLinearWeights()
        msg = ('The coord for this plugin must be '
               'an existing coordinate in the input cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, coord)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Value Error if not supplied with a cube. """
        coord = "time"
        plugin = NonLinearWeights()
        notacube = 0.0
        msg = ('The first argument must be an instance of '
               'iris.cube.Cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(notacube, coord)

    def test_fails_if_cval_not_valid(self):
        """Test it raises a Value Error if cval is not in range,
            cval must be greater than 0.0 and less
            than or equal to 1.0
        """
        coord = "time"
        plugin = NonLinearWeights(cval=-1.0)
        msg = ('cval must be greater than 0.0 and less '
               'than or equal to 1.0')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, coord)
        plugin2 = NonLinearWeights(cval=1.1)
        with self.assertRaisesRegexp(ValueError, msg):
            plugin2.process(self.cube, coord)

    def test_works_if_scalar_coord(self):
        """Test it works if scalar coordinate. """
        coord = "scalar_coord"
        plugin = NonLinearWeights()
        result = plugin.process(self.cube, coord)
        self.assertArrayAlmostEqual(result, np.array([1.0]))

    def test_works_with_default_cval(self):
        """Test it works with default cval. """
        coord = "time"
        plugin = NonLinearWeights()
        result = plugin.process(self.cube, coord)
        expected_result = np.array([0.54054054, 0.45945946])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_with_cval_equal_one(self):
        """Test it works with cval = 1.0, i.e. equal weights. """
        coord = "time"
        plugin = NonLinearWeights(cval=1.0)
        result = plugin.process(self.cube, coord)
        expected_result = np.array([0.5, 0.5])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_with_larger_num(self):
        """Test it works with larger num_of_vals. """
        coord = "realization"
        plugin = NonLinearWeights(cval=0.5)
        cubenew = add_realizations(self.cube, 6)
        result = plugin.process(cubenew, coord)
        expected_result = np.array([0.50793651, 0.25396825,
                                    0.12698413, 0.06349206,
                                    0.03174603, 0.01587302])
        self.assertArrayAlmostEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
