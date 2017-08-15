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
"""Unit tests for the weighted_blend.WeightedBlend plugin."""


import unittest
import warnings

from cf_units import Unit
import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.weighted_blend import WeightedBlend
from improver.tests.weighted_blend.test_PercentileBlendingAggregator import (
    percentile_cube, BLENDED_PERCENTILE_DATA1, BLENDED_PERCENTILE_DATA2)


def example_coord_adjust(pnts):
    """ Example function for coord_adjust.
        A Function to apply to the coordinate after
        collapsing the cube to correct the values.

        Args:
            pnts : numpy.ndarray
    """
    return pnts[len(pnts)-1]


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeightedBlend('time'))
        msg = '<WeightedBlend: coord = time>'
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the Basic Weighted Average plugin."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        data = np.zeros((2, 2, 2))
        data[0][:][:] = 1.0
        data[1][:][:] = 2.0
        cube = Cube(data, standard_name="precipitation_amount",
                    units="kg m^-2 s^-1")
        cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                                    units='degrees'), 1)
        cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                    units='degrees'), 2)
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                    "time", units=tunit), 0)
        self.cube = cube
        new_scalar_coord = iris.coords.AuxCoord(1,
                                                long_name='dummy_scalar_coord',
                                                units='no_unit')
        cube_with_scalar = cube.copy()
        cube_with_scalar.add_aux_coord(new_scalar_coord)
        self.cube_with_scalar = cube_with_scalar

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        coord = "time"
        plugin = WeightedBlend(coord)
        result = plugin.process(self.cube)
        self.assertIsInstance(result, Cube)

    def test_fails_coord_not_in_cube(self):
        """Test it raises a Value Error if coord not in the cube."""
        coord = "notset"
        plugin = WeightedBlend(coord)
        msg = ('The coord for this plugin must be ' +
               'an existing coordinate in the input cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Value Error if not supplied with a cube."""
        coord = "time"
        plugin = WeightedBlend(coord)
        notacube = 0.0
        msg = ('The first argument must be an instance of ' +
               'iris.cube.Cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(notacube)

    def test_fails_perc_coord_not_dim(self):
        """Test it raises a Value Error if not percentile coord not a dim."""
        coord = "time"
        plugin = WeightedBlend(coord)
        new_cube = self.cube.copy()
        new_cube.add_aux_coord(AuxCoord([10.0],
                                        long_name="percentile_over_time"))
        msg = ('The percentile coord must be a dimension '
               'of the cube.')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(new_cube)

    def test_fails_only_one_percentile_value(self):
        """Test it raises a Value Error if there is only one percentile."""
        coord = "time"
        plugin = WeightedBlend(coord)
        new_cube = Cube([[0.0]])
        new_cube.add_dim_coord(DimCoord([10.0],
                                        long_name="percentile_over_time"), 0)
        new_cube.add_dim_coord(DimCoord([10.0],
                                        long_name="time"), 1)
        msg = ('Percentile coordinate does not have enough points'
               ' in order to blend. Must have at least 2 percentiles.')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(new_cube)

    def test_fails_more_than_one_perc_coord(self):
        """Test it raises a Value Error if more than one percentile coord."""
        coord = "time"
        plugin = WeightedBlend(coord)
        new_cube = percentile_cube()
        new_cube.add_aux_coord(AuxCoord([10.0],
                                        long_name="percentile_over_dummy"))
        msg = ('There should only be one percentile coord '
               'on the cube.')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(new_cube)

    def test_fails_weights_shape(self):
        """Test it raises a Value Error if weights shape does not match
           coord shape."""
        coord = "time"
        plugin = WeightedBlend(coord)
        weights = [0.1, 0.2, 0.7]
        msg = ('The weights array must match the shape ' +
               'of the coordinate in the input cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, weights)

    def test_coord_adjust_set(self):
        """Test it works with coord adjust set."""
        coord = "time"
        coord_adjust = example_coord_adjust
        plugin = WeightedBlend(coord, coord_adjust)
        result = plugin.process(self.cube)
        self.assertAlmostEquals(result.coord(coord).points, [402193.5])

    def test_scalar_coord(self):
        """Test it works on scalar coordinate
           and check that a warning has been raised
           if the dimension that you want to blend on
           is a scalar coordinate.
        """
        coord = "dummy_scalar_coord"
        plugin = WeightedBlend(coord)
        weights = np.array([1.0])
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = plugin.process(self.cube_with_scalar, weights)
            self.assertTrue(any(item.category == UserWarning
                                for item in warning_list))
            warning_msg = "Trying to blend across a scalar coordinate"
            self.assertTrue(any(warning_msg in str(item)
                                for item in warning_list))
            self.assertArrayAlmostEqual(result.data, self.cube.data)

    def test_weights_equal_none(self):
        """Test it works with weights set to None."""
        coord = "time"
        plugin = WeightedBlend(coord)
        weights = None
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.5
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_weights_equal_list(self):
        """Test it work with weights set to list [0.2, 0.8]."""
        coord = "time"
        plugin = WeightedBlend(coord)
        weights = [0.2, 0.8]
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.8
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_weights_equal_array(self):
        """Test it works with weights set to array (0.8, 0.2)."""
        coord = "time"
        plugin = WeightedBlend(coord)
        weights = np.array([0.8, 0.2])
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.2
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_percentiles_weights_equal_none(self):
        """Test it works for percentiles with weights set to None."""
        coord = "time"
        plugin = WeightedBlend(coord)
        weights = None
        perc_cube = percentile_cube()
        result = plugin.process(perc_cube, weights)
        expected_result_array = np.reshape(BLENDED_PERCENTILE_DATA1,
                                           (6, 2, 2))
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_percentiles_weights_equal_list(self):
        """Test it works for percentiles with weights [0.8, 0.2]."""
        coord = "time"
        plugin = WeightedBlend(coord)
        weights = np.array([0.8, 0.2])
        perc_cube = percentile_cube()
        result = plugin.process(perc_cube, weights)
        expected_result_array = np.reshape(BLENDED_PERCENTILE_DATA2,
                                           (6, 2, 2))
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_blend_percentile_cube1(self):
        """Test blending percentile cube works with equal weights."""
        coord = "time"
        weights = np.array([0.5, 0.5])
        perc_cube = percentile_cube()
        plugin = WeightedBlend(coord)
        result = plugin.process(perc_cube, weights)
        expected_result_array = np.reshape(BLENDED_PERCENTILE_DATA1,
                                           (6, 2, 2))
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_blend_percentile_cube2(self):
        """Test blending percentile cube works with unequal weights."""
        coord = "time"
        weights = np.array([0.8, 0.2])
        perc_cube = percentile_cube()
        plugin = WeightedBlend(coord)
        result = plugin.process(perc_cube, weights)
        expected_result_array = np.reshape(BLENDED_PERCENTILE_DATA2,
                                           (6, 2, 2))
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_basic_weighted_average(self):
        """Test basic_weighted_average works."""
        coord = "time"
        weights = [0.8, 0.2]
        plugin = WeightedBlend(coord)
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.2
        self.assertArrayAlmostEqual(result.data, expected_result_array)


if __name__ == '__main__':
    unittest.main()
