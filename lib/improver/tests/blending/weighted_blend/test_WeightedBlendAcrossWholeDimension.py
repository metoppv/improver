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
            BLENDED_PERCENTILE_DATA2, BLENDED_PERCENTILE_DATA2_EQUAL_WEIGHTS)
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


class Test_weighted_blend(IrisTest):

    """Test the Basic Weighted Average plugin."""

    def setUp(self):
        """Create a cube with a single non-zero point."""

        time_origin = "seconds since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)

        times = np.array([1447891200, 1447894800, 1447898400])

        time_coord = DimCoord(times, "time", units=tunit)
        frt_coord = AuxCoord(times - 3600, "forecast_reference_time",
                             units=tunit)
        fp_coord = AuxCoord([3600] * 3, "forecast_period", units='seconds')

        lat_coord = DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                             units='degrees')
        lon_coord = DimCoord(np.linspace(120, 180, 2), 'longitude',
                             units='degrees')

        data = np.zeros((3, 2, 2))
        data[0][:][:] = 1.0
        data[1][:][:] = 2.0
        data[2][:][:] = 3.0
        cube = Cube(
            data, standard_name="precipitation_amount", units="kg m^-2 s^-1",
            dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1),
                                (lon_coord, 2)])
        cube.add_aux_coord(frt_coord, data_dims=0)
        cube.add_aux_coord(fp_coord, data_dims=0)
        self.cube = cube

        new_scalar_coord = AuxCoord(1, long_name='dummy_scalar_coord',
                                    units='no_unit')
        cube_with_scalar = cube.copy()
        cube_with_scalar.add_aux_coord(new_scalar_coord)
        self.cube_with_scalar = cube_with_scalar

        data_threshold = np.zeros((2, 3, 2, 2))
        data_threshold[:, 0, :, :] = 0.5
        data_threshold[:, 1, :, :] = 0.8
        data_threshold[:, 2, :, :] = 0.9

        cube_threshold = Cube(data_threshold,
                              long_name="probability_of_precipitation_amount")
        cube_threshold.add_dim_coord(DimCoord([0.4, 1.0],
                                              long_name="threshold",
                                              units="kg m^-2 s^-1"), 0)
        cube_threshold.add_dim_coord(time_coord, 1)
        cube_threshold.add_dim_coord(lat_coord, 2)
        cube_threshold.add_dim_coord(lon_coord, 3)
        cube_threshold.add_aux_coord(frt_coord, data_dims=1)
        cube_threshold.add_aux_coord(fp_coord, data_dims=1)

        cube_threshold.attributes.update({'relative_to_threshold': 'below'})
        self.cube_threshold = cube_threshold

        # Weights cubes
        # 1D varying with time.
        weights1d = np.array([0.6, 0.3, 0.1], dtype=np.float32)
        # 2D varying in space.
        weights2d = np.array([[0.5, 0.5],
                              [0.5, 0.5]], dtype=np.float32)
        # 3D varying in space and time.
        weights3d = np.array([[[0.1, 0.3],
                               [0.2, 0.4]],
                              [[0.1, 0.3],
                               [0.2, 0.4]],
                              [[0.8, 0.4],
                               [0.6, 0.2]]], dtype=np.float32)

        self.weights1d = Cube(
            weights1d, long_name='weights',
            dim_coords_and_dims=[(time_coord, 0)],
            aux_coords_and_dims=[(frt_coord , 0), (fp_coord, 0)])

        self.weights2d = Cube(
            weights2d, long_name='weights',
            dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)])
        self.weights2d.add_aux_coord(time_coord[0])
        self.weights2d.add_aux_coord(frt_coord[0])
        self.weights2d.add_aux_coord(fp_coord[0])

        self.weights3d = Cube(
            weights3d, long_name='weights',
            dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1),
                                 (lon_coord, 2)],
            aux_coords_and_dims=[(frt_coord, 0), (fp_coord, 0)])


class Test_is_cube_percentile_data(Test_weighted_blend):

    """Test the percentile checking function."""

    def test_fails_perc_coord_not_dim(self):
        """Test it raises a Value Error if percentile coord not a dim."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        new_cube = self.cube.copy()
        new_cube.add_aux_coord(AuxCoord([10.0],
                                        long_name="percentile_over_time"))
        msg = ('The percentile coord must be a dimension '
               'of the cube.')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.is_cube_percentile_data(new_cube)

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
            plugin.is_cube_percentile_data(new_cube)

    def test_fails_percentile_data_max_mode(self):
        """Test a Value Error is raised if the maximum mode is applied to
        percentile data."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_maximum')
        new_cube = percentile_cube()
        msg = ('The "weighted_maximum" mode cannot be used with percentile '
               'data.')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.is_cube_percentile_data(new_cube)


class Test_compatible_time_points(Test_weighted_blend):

    """Test the time point compatibility checking function."""

    def test_forecast_reference_time_exception(self):
        """Test that a ValueError is raised if the coordinate to be blended
        is forecast_reference_time and the points on the time coordinate are
        not equal."""
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        msg = ('For blending using the forecast_reference_time')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_compatible_time_points(self.cube)


class Test_shape_weights(Test_weighted_blend):

    """Test the shape weights function is able to create a valid a set of
    weights, or raises an error."""

    def test_1D_weights_3D_cube_weighted_mean(self):
        """Test a 1D cube of weights results in a 3D array of weights of the
        same shape as the data cube."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = None
        result = plugin.shape_weights(self.cube, self.weights1d)
        self.assertEqual(self.cube.shape, result.shape)
        self.assertArrayEqual(self.weights1d.data, result[:, 0, 0])

    def test_3D_weights_3D_cube_weighted_mean(self):
        """Test a 3D cube of weights results in a 3D array of weights of the
        same shape as the data cube."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = None
        result = plugin.shape_weights(self.cube, self.weights3d)
        self.assertEqual(self.cube.shape, result.shape)
        self.assertArrayEqual(self.weights3d.data, result)

    def test_incompatible_weights_and_data_cubes(self):
        """Test an exception is raised if the weights cube and the data cube
        are of incompatible shapes."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = None
        weights = np.linspace(0, 1, 5)
        msg = "Weights cube is not a compatible shape with the data cube"

        with self.assertRaisesRegex(ValueError, msg):
            plugin.shape_weights(self.cube, weights)


class Test_non_percentile_weights(Test_weighted_blend):

    """Test the non_percentile_weights function."""

    def test_no_weights_cube_provided(self):
        """Test that if a weights cube is not provided, the function generates
        a weights array that will equally weight all fields along the blending
        coordinate."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = None
        result = plugin.non_percentile_weights(self.cube, None)
        blending_coord_length, = self.cube.coord(coord).shape
        expected = (np.ones(blending_coord_length)/
                    blending_coord_length).astype(np.float32)
        self.assertEqual(self.cube.shape, result.shape)
        self.assertArrayEqual(expected, result[:, 0, 0])

    def test_1D_weights_3D_cube_custom_aggregator(self):
        """Test a 1D cube of weights results in a 3D array of weights. With the
        weighted maximum method (custom_aggregator=True) the shape will be
        almost the same as the data cube, but the blending coordinate will have
        been moved to the -1 position."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = None
        expected_shape = (2, 2, 3)
        result = plugin.non_percentile_weights(
            self.cube, self.weights1d, custom_aggregator=True)
        self.assertEqual(expected_shape, result.shape)
        self.assertArrayEqual(self.weights1d.data, result[0, 0, :])

    def test_3D_weights_3D_cube_custom_aggregator(self):
        """Test a 3D cube of weights results in a 3D array of weights. With the
        weighted maximum method (custom_aggregator=True) the shape will be
        almost the same as the data cube, but the blending coordinate will have
        been moved to the -1 position."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = None
        expected_shape = (2, 2, 3)
        expected_weights = np.moveaxis(self.weights3d.data, 0, -1)
        result = plugin.non_percentile_weights(
            self.cube, self.weights3d, custom_aggregator=True)
        self.assertEqual(expected_shape, result.shape)
        self.assertArrayEqual(expected_weights, result)


class Test_percentile_weights(Test_weighted_blend):

    """Test the percentile_weights function."""

    def test_no_weights_cube_provided(self):
        """Test that if a weights cube is not provided, the function generates
        a weights array that will equally weight all fields along the blending
        coordinate."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = None
        result = plugin.non_percentile_weights(self.cube, None)
        blending_coord_length, = self.cube.coord(coord).shape
        expected = (np.ones(blending_coord_length)/
                    blending_coord_length).astype(np.float32)
        self.assertEqual(self.cube.shape, result.shape)
        self.assertArrayEqual(expected, result[:, 0, 0])

    def test_1D_weights_3D_cube_percentile_weighted_mean(self):
        """Test a 1D cube of weights results in a 3D array of weights of an
        appropriate shape for a percentile cube. In this case the collapse
        coordinate is moved to the leading position and the percentile
        coordinate is second."""

        perc_cube = percentile_cube()
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = perc_cube.coord('percentile_over_realization')
        coord_dim, = perc_cube.coord_dims(coord)
        perc_dim, =  perc_cube.coord_dims(plugin.perc_coord)
        expected = np.empty_like(perc_cube.data)
        expected = np.moveaxis(expected, [coord_dim, perc_dim], [0, 1])

        result = plugin.percentile_weights(perc_cube, self.weights1d)
        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(self.weights1d.data, result[:, 0, 0, 0])

    def test_3D_weights_3D_cube_percentile_weighted_mean(self):
        """Test a 3D cube of weights results in a 3D array of weights of the
        same shape as the data cube. In this case the cube is a percentile
        cube and so has an additional dimension that must be accounted for"""

        perc_cube = percentile_cube()
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = perc_cube.coord('percentile_over_realization')
        coord_dim, = perc_cube.coord_dims(coord)
        perc_dim, =  perc_cube.coord_dims(plugin.perc_coord)
        expected = np.empty_like(perc_cube.data)
        expected = np.moveaxis(expected, [coord_dim, perc_dim], [0, 1])

        result = plugin.percentile_weights(perc_cube, self.weights3d)
        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(self.weights3d.data, result[:, 0, :, :])


class Test_check_weights(Test_weighted_blend):

    """Test the check_weights function."""

    def test_weights_sum_to_1(self):
        """Test that if a weights cube is provided which is properly
        normalised, i.e. the weights sum to one over the blending
        dimension, no exception is raised."""

        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        _ = plugin.check_weights(self.weights3d.data, 0)

    def test_weights_do_not_sum_to_1_error(self):
        """Test that if a weights cube is provided which is not properly
        normalised, i.e. the weights do not sum to one over the blending
        dimension, an error is raised."""

        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = self.weights3d.data
        weights[0, 0, 1] = 1.
        msg = 'Weights do not sum to 1 over the blending coordinate.'
        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_weights(weights, 0)


class Test_percentile_weighted_mean(Test_weighted_blend):

    """Test the percentile_weighted_mean function."""

    def test_with_weights(self):
        """Test function when a data cube and a weights cube are provided."""

        perc_cube = percentile_cube()
        print('cube data', perc_cube.data.shape)
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = perc_cube.coord('percentile_over_realization')
        result = plugin.percentile_weighted_mean(perc_cube, self.weights1d,
                                                 plugin.perc_coord)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data,
                                    BLENDED_PERCENTILE_DATA2)

    def test_without_weights(self):
        """Test function when a data cube is provided, but no weights cube
        which should result in equal weightings."""

        perc_cube = percentile_cube()
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = perc_cube.coord('percentile_over_realization')
        result = plugin.percentile_weighted_mean(perc_cube, None,
                                                 plugin.perc_coord)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.data, BLENDED_PERCENTILE_DATA2_EQUAL_WEIGHTS)

class Test_weighted_mean(Test_weighted_blend):

    """Test the weighted_mean function."""

    def test_with_weights(self):
        """Test function when a data cube and a weights cube are provided."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = None
        result = plugin.weighted_mean(self.cube, self.weights1d)
        expected = np.full((2, 2), 1.5)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_without_weights(self):
        """Test function when a data cube is provided, but no weights cube
        which should result in equal weightings."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        plugin.perc_coord = None
        result = plugin.weighted_mean(self.cube, None)
        expected = np.full((2, 2), 2.)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)


class Test_weighted_maximum(Test_weighted_blend):

    """Test the weighted_maximum function."""

    def test_with_weights(self):
        """Test function when a data cube and a weights cube are provided."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_maximum')
        plugin.perc_coord = None
        result = plugin.weighted_maximum(self.cube, self.weights1d)
        expected = np.full((2, 2), 0.6)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_without_weights(self):
        """Test function when a data cube is provided, but no weights cube
        which should result in equal weightings."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_maximum')
        plugin.perc_coord = None
        result = plugin.weighted_maximum(self.cube, None)
        expected = np.full((2, 2), 1.)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)


class Test_process(Test_weighted_blend):

    """Test the process method."""

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

    def test_fails_weights_shape(self):
        """Test it raises a Value Error if weights cube shape does not match
           coord shape."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = Cube([0.1, 0.2, 0.7], long_name='weights')
        msg = ('The weights cube must match the shape ' +
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

    def test_scalar_coord(self):
        """Test plugin throws an error if trying to blending across a scalar
        coordinate.
        """
        coord = "dummy_scalar_coord"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = ([1.0])
        msg = 'has no associated dimension'
        with self.assertRaisesRegex(ValueError, msg):
            _ = plugin.process(self.cube_with_scalar, weights)

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
    def test_weights_equal_cube(self):
        """Test it work with weights set to [0.2, 0.8] in a cube."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = Cube([0.2, 0.8], long_name='weights')
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.8
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def tests_threshold_splicing_works_weighted_mean(self):
        """Test weighted_mean works with a threshold dimension."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = Cube([0.8, 0.2], long_name='weights')
        result = plugin.process(self.cube_threshold, weights)
        expected_result_array = np.ones((2, 2, 2))*0.56
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def tests_threshold_splicing_works_with_threshold(self):
        """Test splicing works when the blending is over threshold."""
        coord = "threshold"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = Cube([0.8, 0.2], long_name='weights')
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
    def test_percentiles_non_equal_weights_cube(self):
        """Test it works for percentiles with weights [0.8, 0.2]
           given as a cube."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_mean')
        weights = Cube([0.8, 0.2], long_name='weights')
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
    def test_weighted_max_non_equal_weights_cube(self):
        """Test it works for weighted_max with weights [0.2, 0.8]
           given as a cube."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_maximum')
        weights = Cube([0.2, 0.8], long_name='weights')
        result = plugin.process(self.cube, weights)
        expected_result_array = np.ones((2, 2))*1.6
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def tests_threshold_splicing_works_weighted_max(self):
        """Test weighted_max works with a threshold dimension."""
        coord = "time"
        plugin = WeightedBlendAcrossWholeDimension(coord, 'weighted_maximum')
        weights = Cube([0.8, 0.2], long_name='weights')
        result = plugin.process(self.cube_threshold, weights)
        expected_result_array = np.ones((2, 2, 2))*0.4
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
