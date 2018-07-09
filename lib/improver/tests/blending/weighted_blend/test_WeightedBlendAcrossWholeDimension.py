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
"""Unit tests for the
   weighted_blend.WeightedBlendAcrossWholeDimension plugin."""


import unittest

from cf_units import Unit
import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
from iris.exceptions import CoordinateNotFoundError
import numpy as np

from improver.blending.weighted_blend import WeightedBlendAcrossWholeDimension
from improver.tests.blending.weighted_blend.test_PercentileBlendingAggregator \
    import (percentile_cube, BLENDED_PERCENTILE_DATA1,
            BLENDED_PERCENTILE_DATA2)
from improver.utilities.warnings_handler import ManageWarnings


def example_coord_adjust(pnts):
    """ Example function for coord_adjust.
        A Function to apply to the coordinate after
        collapsing the cube to correct the values.

        Args:
            pnts : numpy.ndarray
    """
    return pnts[len(pnts)-1]


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = (WeightedBlendAcrossWholeDimension('time', 'weighted_mean'))
        self.assertEqual(plugin.coord, 'time')
        self.assertEqual(plugin.mode, 'weighted_mean')
        self.assertEqual(plugin.coord_adjust, None)

    def test_raises_expression(self):
        """Test that the __init__ raises an error when appropriate."""
        message = ("weighting_mode: not_a_method is not recognised, "
                   "must be either weighted_maximum or weighted_mean")
        with self.assertRaisesRegex(ValueError, message):
            WeightedBlendAcrossWholeDimension('time', 'not_a_method')


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeightedBlendAcrossWholeDimension('time',
                                                       'weighted_mean'))
        msg = ('<WeightedBlendAcrossWholeDimension: coord = time,'
               ' weighting_mode = weighted_mean, coord_adjust = None>')
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
        cube.add_aux_coord(AuxCoord([402190.0, 402191.0],
                                    "forecast_reference_time",
                                    units=tunit),
                           data_dims=0)
        cube.add_aux_coord(AuxCoord([3.0, 4.0],
                                    "forecast_period",
                                    units=tunit),
                           data_dims=0)

        self.cube = cube
        new_scalar_coord = iris.coords.AuxCoord(1,
                                                long_name='dummy_scalar_coord',
                                                units='no_unit')
        cube_with_scalar = cube.copy()
        cube_with_scalar.add_aux_coord(new_scalar_coord)
        self.cube_with_scalar = cube_with_scalar
        data_threshold = np.zeros((2, 2, 2, 2))
        data_threshold[:, 0, :, :] = 0.5
        data_threshold[:, 1, :, :] = 0.8
        cube_threshold = Cube(data_threshold,
                              long_name="probability_of_precipitation_amount")
        cube_threshold.add_dim_coord(DimCoord([0.4, 1.0],
                                              long_name="threshold",
                                              units="kg m^-2 s^-1"), 0)
        cube_threshold.add_dim_coord(DimCoord([402192.5, 402193.5],
                                              "time", units=tunit), 1)
        cube_threshold.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2),
                                              'latitude',
                                              units='degrees'), 2)
        cube_threshold.add_dim_coord(DimCoord(np.linspace(120, 180, 2),
                                              'longitude',
                                              units='degrees'), 3)
        cube_threshold.add_aux_coord(
            AuxCoord([402190.0, 402191.0], "forecast_reference_time",
                     units=tunit), data_dims=0)
        cube_threshold.add_aux_coord(
            AuxCoord([3.0, 4.0], "forecast_period", units=tunit), data_dims=0)
        cube_threshold.attributes.update({'relative_to_threshold': 'below'})
        self.cube_threshold = cube_threshold

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        result = plugin.process(self.cube)
        self.assertIsInstance(result, Cube)

    def test_fails_coord_not_in_cube(self):
        """Test it raises CoordinateNotFoundError if coord not in the cube."""
        coord = "notset"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        msg = ('Coordinate to be collapsed not found in cube.')
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            plugin.process(self.cube)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Type Error if not supplied with a cube."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        notacube = 0.0
        msg = ('The first argument must be an instance of ' +
               'iris.cube.Cube')
        with self.assertRaisesRegex(TypeError, msg):
            plugin.process(notacube)

    def test_fails_perc_coord_not_dim(self):
        """Test it raises a Value Error if not percentile coord not a dim."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        new_cube = self.cube.copy()
        new_cube.add_aux_coord(AuxCoord([10.0],
                                        long_name="percentile_over_time"))
        msg = ('The percentile coord must be a dimension '
               'of the cube.')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(new_cube)

    def test_fails_only_one_percentile_value(self):
        """Test it raises a Value Error if there is only one percentile."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        new_cube = Cube([[0.0]])
        new_cube.add_dim_coord(DimCoord([10.0],
                                        long_name="percentile_over_time"), 0)
        new_cube.add_dim_coord(DimCoord([10.0],
                                        long_name="time"), 1)
        msg = ('Percentile coordinate does not have enough points'
               ' in order to blend. Must have at least 2 percentiles.')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(new_cube)

    def test_fails_weights_shape(self):
        """Test it raises a Value Error if weights shape does not match
           coord shape."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = [0.1, 0.2, 0.7]
        msg = ('The weights array must match the shape ' +
               'of the coordinate in the input cube')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube, weights)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_coord_adjust_set(self):
        """Test it works with coord adjust set."""
        coord = "time"
        coord_adjust = example_coord_adjust
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean',
                                                   coord_adjust)
        result = plugin.process(self.cube)
        self.assertAlmostEqual(result.coord(coord).points, [402193.5])

    def test_forecast_reference_time_exception(self):
        """Test that a ValueError is raised if the coordinate to be blended
        is forecast_reference_time and the points on the time coordinate are
        not equal."""
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        msg = ('For blending using the forecast_reference_time')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube)

    @ManageWarnings(record=True)
    def test_scalar_coord(self, warning_list=None):
        """Test it works on scalar coordinate
           and check that a warning has been raised
           if the dimension that you want to blend on
           is a scalar coordinate.
        """
        coord = "dummy_scalar_coord"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = np.array([1.0])
        result = plugin.process(self.cube_with_scalar, weights)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Trying to blend across a scalar coordinate"
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertArrayAlmostEqual(result.data, self.cube.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_weights_equal_none(self):
        """Test it works with weights set to None."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = None
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.5
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_weights_equal_list(self):
        """Test it work with weights set to list [0.2, 0.8]."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = [0.2, 0.8]
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.8
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_weights_equal_array(self):
        """Test it works with weights set to array (0.8, 0.2)."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = np.array([0.8, 0.2])
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.2
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def tests_threshold_splicing_works_weighted_mean(self):
        """Test weighted_mean works with a threshold dimension."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = np.array([0.8, 0.2])
        result = plugin.process(self.cube_threshold, weights)
        expected_result_array = np.ones((2, 2, 2))*0.56
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def tests_threshold_splicing_works_with_threshold(self):
        """Test splicing works when the blending is over threshold."""
        coord = "threshold"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = np.array([0.8, 0.2])
        self.cube_threshold.data[0, :, :, :] = 0.5
        self.cube_threshold.data[1, :, :, :] = 0.8
        result = plugin.process(self.cube_threshold, weights)
        expected_result_array = np.ones((2, 2, 2))*0.56
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_percentiles_weights_none(self):
        """Test it works for percentiles with weights set to None."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = None
        perc_cube = percentile_cube()
        result = plugin.process(perc_cube, weights)
        expected_result_array = np.reshape(BLENDED_PERCENTILE_DATA1,
                                           (6, 2, 2))
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_percentiles_non_equal_weights_list(self):
        """Test it works for percentiles with weights [0.8, 0.2]
           given as a list."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = [0.8, 0.2]
        perc_cube = percentile_cube()
        result = plugin.process(perc_cube, weights)
        expected_result_array = np.reshape(BLENDED_PERCENTILE_DATA2,
                                           (6, 2, 2))
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_percentiles_different_coordinate_orders(self):
        """Test the result of the percentile aggregation is the same
        regardless of the coordinate order in the input cube. Most
        importantly, the result should be the same regardless of on which
        side of the collapsing coordinate the percentile coordinate falls."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = None
        percentile_leading = percentile_cube()
        time_leading = percentile_cube()
        time_leading.transpose([1, 0, 2, 3])
        result_percentile_leading = plugin.process(percentile_leading, weights)
        result_time_leading = plugin.process(time_leading, weights)
        expected_result_array = np.reshape(BLENDED_PERCENTILE_DATA1,
                                           (6, 2, 2))
        self.assertArrayAlmostEqual(result_percentile_leading.data,
                                    expected_result_array)
        self.assertArrayAlmostEqual(result_time_leading.data,
                                    expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_weighted_max_weights_none(self):
        """Test it works for weighted max with weights set to None."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_maximum')
        weights = None
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_weighted_max_non_equal_weights_list(self):
        """Test it works for weighted_max with weights [0.2, 0.8]
           given as a list."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_maximum')
        weights = [0.2, 0.8]
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.6
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def tests_threshold_splicing_works_weighted_max(self):
        """Test weighted_max works with a threshold dimension."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_maximum')
        weights = np.array([0.8, 0.2])
        result = plugin.process(self.cube_threshold, weights)
        expected_result_array = np.ones((2, 2, 2))*0.4
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_weighted_max_non_equal_weights_array(self):
        """Test it works for weighted_max with weights [0.2, 0.8]
           given as a array."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_maximum')
        weights = np.array([0.2, 0.8])
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.6
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_source_realizations_attribute_added(self):
        """Test that when a realization coordinate is collapsed, a new
        source_realizations attribute is added to record the contributing
        realizations."""
        coord = "realization"
        self.cube.coord('time').rename(coord)
        self.cube.coord(coord).points = [1, 4]
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = None
        result = plugin.process(self.cube, weights)
        expected = [1, 4]
        self.assertArrayEqual(result.attributes['source_realizations'],
                              expected)


if __name__ == '__main__':
    unittest.main()
