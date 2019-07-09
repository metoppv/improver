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
"""Unit tests for the
   weighted_blend.WeightedBlendAcrossWholeDimension plugin."""


import unittest

from cf_units import date2num
from datetime import datetime

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
from iris.exceptions import CoordinateNotFoundError
import numpy as np

from improver.blending.weighted_blend import WeightedBlendAcrossWholeDimension
from improver.tests.blending.weighted_blend.test_PercentileBlendingAggregator \
    import (PERCENTILE_DATA, BLENDED_PERCENTILE_DATA,
            BLENDED_PERCENTILE_DATA_EQUAL_WEIGHTS,
            BLENDED_PERCENTILE_DATA_SPATIAL_WEIGHTS)
from improver.utilities.warnings_handler import ManageWarnings
from improver.utilities.cube_manipulation import merge_cubes


def time_coords_for_test_cubes():
    """Set up time coordinates in human-readable format"""
    tunit = "seconds since 1970-01-01 00:00:00"
    calendar = "gregorian"

    time_point_seconds = np.round(
        date2num(datetime(2015, 11, 19, 2), tunit, calendar)).astype(np.int64)

    frt_points = []
    for i in range(3):
        frt_points.append(np.round(date2num(datetime(2015, 11, 19, i), tunit,
                                            calendar)).astype(np.int64))

    time_coord = AuxCoord([time_point_seconds], "time", units=tunit)
    frt_coord = DimCoord(frt_points, "forecast_reference_time", units=tunit)
    fp_coord = AuxCoord([7200, 3600, 0], "forecast_period", units="seconds")

    return time_coord, frt_coord, fp_coord


def percentile_cube():
    """Create a percentile cube for testing."""
    data = np.reshape(PERCENTILE_DATA, (6, 3, 2, 2))

    # construct dim coords
    perc_coord = DimCoord([0, 20, 40, 60, 80, 100],
                          long_name="percentile")
    y_coord = DimCoord(np.linspace(-45.0, 45.0, 2).astype(np.float32),
                       'latitude', units='degrees')
    x_coord = DimCoord(np.linspace(120, 180, 2).astype(np.float32),
                       'longitude', units='degrees')

    # construct time coords
    time_coord, frt_coord, fp_coord = time_coords_for_test_cubes()

    # build cube
    dim_coords = [(perc_coord, 0), (frt_coord, 1),
                  (y_coord, 2), (x_coord, 3)]
    aux_coords = [(fp_coord, 1), (time_coord, None)]

    cube = Cube(data, standard_name="air_temperature", units="C",
                dim_coords_and_dims=dim_coords, aux_coords_and_dims=aux_coords)

    return cube


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = (WeightedBlendAcrossWholeDimension(
            'time', cycletime='20171101T0300Z'))
        self.assertEqual(plugin.coord, 'time')
        self.assertEqual(plugin.cycletime, '20171101T0300Z')

    def test_threshold_blending_unsupported(self):
        """Test that the __init__ raises an error if trying to blend over
        thresholds (which is neither supported nor sensible)."""
        msg = "Blending over thresholds is not supported"
        with self.assertRaisesRegex(ValueError, msg):
            WeightedBlendAcrossWholeDimension('threshold')


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeightedBlendAcrossWholeDimension('time'))
        msg = ('<WeightedBlendAcrossWholeDimension: coord = time, '
               'cycletime = None, timeblending: False>')
        self.assertEqual(result, msg)


class Test_weighted_blend(IrisTest):

    """A shared setup for tests in the WeightedBlendAcrossWholeDimension
    plugin."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        time_coord, frt_coord, fp_coord = time_coords_for_test_cubes()

        lat_coord = DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                             units='degrees')
        lon_coord = DimCoord(np.linspace(120, 180, 2), 'longitude',
                             units='degrees')
        threshold = DimCoord([0.4, 1.0], long_name="precipitation_amount",
                             units="kg m^-2 s^-1", var_name="threshold")

        data = np.zeros((3, 2, 2))
        data[0][:][:] = 1.0
        data[1][:][:] = 2.0
        data[2][:][:] = 3.0
        cube = Cube(
            data, standard_name="precipitation_amount", units="kg m^-2 s^-1",
            dim_coords_and_dims=[(frt_coord, 0), (lat_coord, 1),
                                 (lon_coord, 2)])
        cube.add_aux_coord(time_coord)
        cube.add_aux_coord(fp_coord, data_dims=0)
        self.attributes = {
            'mosg__grid_domain': 'global',
            'mosg__grid_type': 'standard',
            'mosg__grid_version': '1.2.0',
            'mosg__model_configuration': 'gl_det',
            'title': 'Operational ENGL Model Forecast'
            }
        cube.attributes = self.attributes
        self.cube = cube

        new_scalar_coord = AuxCoord(1, long_name='dummy_scalar_coord',
                                    units='no_unit')
        cube_with_scalar = cube.copy()
        cube_with_scalar.add_aux_coord(new_scalar_coord)
        self.cube_with_scalar = cube_with_scalar

        data_threshold = np.zeros((2, 3, 2, 2))
        data_threshold[0, 0, :, :] = 0.2
        data_threshold[0, 1, :, :] = 0.4
        data_threshold[0, 2, :, :] = 0.6
        data_threshold[1, 0, :, :] = 0.4
        data_threshold[1, 1, :, :] = 0.6
        data_threshold[1, 2, :, :] = 0.8

        cube_threshold = Cube(
            data_threshold,
            long_name="probability_of_precipitation_amount_below_threshold")
        cube_threshold.add_dim_coord(threshold, 0)
        cube_threshold.add_dim_coord(frt_coord, 1)
        cube_threshold.add_dim_coord(lat_coord, 2)
        cube_threshold.add_dim_coord(lon_coord, 3)
        cube_threshold.add_aux_coord(time_coord)
        cube_threshold.add_aux_coord(fp_coord, data_dims=1)

        cube_threshold.coord(var_name="threshold").attributes.update(
            {'spp__relative_to_threshold': 'below'})
        cube_threshold.attributes = self.attributes
        self.cube_threshold = cube_threshold

        # Weights cubes
        # 1D varying with forecast reference time.
        weights1d = np.array([0.6, 0.3, 0.1], dtype=np.float32)
        # 1D varying with threshold.
        weights_threshold = np.array([0.8, 0.2], dtype=np.float32)
        # 3D varying in space and forecast reference time.
        weights3d = np.array([[[0.1, 0.3],
                               [0.2, 0.4]],
                              [[0.1, 0.3],
                               [0.2, 0.4]],
                              [[0.8, 0.4],
                               [0.6, 0.2]]], dtype=np.float32)

        self.weights1d = Cube(
            weights1d, long_name='weights',
            dim_coords_and_dims=[(frt_coord, 0)],
            aux_coords_and_dims=[(fp_coord, 0)])
        self.weights1d.add_aux_coord(time_coord)

        self.weights_threshold = Cube(
            weights_threshold, long_name='weights',
            dim_coords_and_dims=[(threshold, 0)])

        self.weights3d = Cube(
            weights3d, long_name='weights',
            dim_coords_and_dims=[(frt_coord, 0), (lat_coord, 1),
                                 (lon_coord, 2)],
            aux_coords_and_dims=[(fp_coord, 0), (time_coord, None)])


class Test_check_percentile_coord(Test_weighted_blend):

    """Test the percentile coord checking function."""

    def test_fails_perc_coord_not_dim(self):
        """Test it raises a Value Error if percentile coord not a dim."""
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        new_cube = self.cube.copy()
        new_cube.add_aux_coord(AuxCoord([10.0],
                                        long_name="percentile"))
        msg = ('The percentile coord must be a dimension '
               'of the cube.')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_percentile_coord(new_cube)

    def test_fails_only_one_percentile_value(self):
        """Test it raises a Value Error if there is only one percentile."""
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        new_cube = Cube([[0.0]])
        new_cube.add_dim_coord(DimCoord([10.0],
                                        long_name="percentile"), 0)
        new_cube.add_dim_coord(
            DimCoord([10.0], long_name="forecast_reference_time"), 1)
        msg = ('Percentile coordinate does not have enough points'
               ' in order to blend. Must have at least 2 percentiles.')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_percentile_coord(new_cube)


class Test_check_compatible_time_points(Test_weighted_blend):

    """Test the time point compatibility checking function."""

    def test_unmatched_validity_time_exception(self):
        """Test that a ValueError is raised if the validity time of the
        slices over the blending coordinate differ. We should only be blending
        data at equivalent times unless we are triangular time blending."""
        self.cube.remove_coord('time')
        self.cube.coord('forecast_reference_time').rename('time')
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        msg = ('Attempting to blend data for different validity times.')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_compatible_time_points(self.cube)

    def test_unmatched_validity_time_exemption(self):
        """Test that no ValueError is raised for unmatched validity times if
        we use the timeblending=True flag. As long as no exception is raised
        this test passes."""
        self.cube.remove_coord('time')
        self.cube.coord('forecast_reference_time').rename('time')
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord, timeblending=True)
        plugin.check_compatible_time_points(self.cube)


class Test_shape_weights(Test_weighted_blend):

    """Test the shape weights function is able to create a valid a set of
    weights, or raises an error."""

    def test_1D_weights_3D_cube_weighted_mean(self):
        """Test a 1D cube of weights results in a 3D array of weights of the
        same shape as the data cube."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        result = plugin.shape_weights(self.cube, self.weights1d)
        self.assertEqual(self.cube.shape, result.shape)
        self.assertArrayEqual(self.weights1d.data, result[:, 0, 0])

    def test_3D_weights_3D_cube_weighted_mean(self):
        """Test a 3D cube of weights results in a 3D array of weights of the
        same shape as the data cube."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        result = plugin.shape_weights(self.cube, self.weights3d)
        self.assertEqual(self.cube.shape, result.shape)
        self.assertArrayEqual(self.weights3d.data, result)

    def test_3D_weights_3D_cube_weighted_mean_wrong_order(self):
        """Test a 3D cube of weights results in a 3D array of weights of the
        same shape as the data cube. In this test the weights cube has the
        same coordinates but slightly differently ordered. These should be
        reordered to match the cube."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        expected = self.weights3d.copy().data
        self.weights3d.transpose([1, 0, 2])
        result = plugin.shape_weights(self.cube, self.weights3d)
        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(expected.data, result)

    def test_3D_weights_4D_cube_weighted_mean_wrong_order(self):
        """Test a 4D cube of weights results in a 3D array of weights of the
        same shape as the data cube. In this test the input cube has the
        same coordinates but slightly differently ordered. The weights cube
        should be reordered to match the cube."""
        # Add a new axis to input cube to make it 4D
        coord = "forecast_reference_time"
        cube = iris.util.new_axis(self.cube, scalar_coord="time")
        cube.transpose([3, 2, 0, 1])
        plugin = WeightedBlendAcrossWholeDimension(coord)
        # Create an expected array which has been transposed to match
        # input cube with the extra axis added.
        expected = self.weights3d.copy()
        expected.transpose([2, 1, 0])
        expected = np.expand_dims(expected.data, axis=2)
        result = plugin.shape_weights(cube, self.weights3d)
        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(expected, result)

    def test_3D_weights_3D_cube_weighted_mean_unmatched_coordinate(self):
        """Test a 3D cube of weights results in a 3D array of weights of the
        same shape as the data cube. In this test the weights cube has the
        same shape but different coordinates to the diagnostic cube, raising
        a ValueError as a result."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        weights = self.weights3d.copy()
        weights.coord('longitude').rename('projection_x_coordinate')
        msg = (
            "projection_x_coordinate is a coordinate on the weights cube "
            "but it is not found on the cube we are trying to collapse.")
        with self.assertRaisesRegex(ValueError, msg):
            plugin.shape_weights(self.cube, weights)

    def test_incompatible_weights_and_data_cubes(self):
        """Test an exception is raised if the weights cube and the data cube
        have incompatible coordinates."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        msg = (
            "precipitation_amount is a coordinate on the weights cube but it "
            "is not found on the cube we are trying to collapse.")
        with self.assertRaisesRegex(ValueError, msg):
            plugin.shape_weights(self.cube, self.weights_threshold)


class Test_percentile_weights(Test_weighted_blend):

    """Test the percentile_weights function."""

    def test_no_weights_cube_provided(self):
        """Test that if a weights cube is not provided, the function generates
        a weights array that will equally weight all fields along the blending
        coordinate."""

        perc_cube = percentile_cube()
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        perc_coord = perc_cube.coord('percentile')
        coord_dim, = perc_cube.coord_dims(coord)
        perc_dim, = perc_cube.coord_dims(perc_coord)

        blending_coord_length, = perc_cube.coord(coord).shape
        expected = np.empty_like(perc_cube.data)
        expected = np.moveaxis(expected, [coord_dim, perc_dim], [0, 1])
        expected = np.full_like(expected, 1./blending_coord_length)

        result = plugin.percentile_weights(perc_cube, None, perc_coord)

        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(expected, result)

    def test_1D_weights_3D_cube_percentile_weighted_mean(self):
        """Test a 1D cube of weights results in a 3D array of weights of an
        appropriate shape for a percentile cube. In this case the collapse
        coordinate is moved to the leading position and the percentile
        coordinate is second."""

        perc_cube = percentile_cube()
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        perc_coord = perc_cube.coord('percentile')
        coord_dim, = perc_cube.coord_dims(coord)
        perc_dim, = perc_cube.coord_dims(perc_coord)
        expected = np.empty_like(perc_cube.data)
        expected = np.moveaxis(expected, [coord_dim, perc_dim], [0, 1])

        result = plugin.percentile_weights(perc_cube, self.weights1d,
                                           perc_coord)
        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(self.weights1d.data, result[:, 0, 0, 0])

    def test_3D_weights_3D_cube_percentile_weighted_mean(self):
        """Test a 3D cube of weights results in a 3D array of weights of the
        same shape as the data cube. In this case the cube is a percentile
        cube and so has an additional dimension that must be accounted for."""

        perc_cube = percentile_cube()
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        perc_coord = perc_cube.coord('percentile')
        coord_dim, = perc_cube.coord_dims(coord)
        perc_dim, = perc_cube.coord_dims(perc_coord)
        expected = np.empty_like(perc_cube.data)
        expected = np.moveaxis(expected, [coord_dim, perc_dim], [0, 1])

        result = plugin.percentile_weights(perc_cube, self.weights3d,
                                           perc_coord)
        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(self.weights3d.data, result[:, 0, :, :])


class Test_non_percentile_weights(Test_weighted_blend):

    """Test the non_percentile_weights function."""

    def test_no_weights_cube_provided(self):
        """Test that if a weights cube is not provided, the function generates
        a weights array that will equally weight all fields along the blending
        coordinate."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        result = plugin.non_percentile_weights(self.cube, None)
        blending_coord_length, = self.cube.coord(coord).shape
        expected = (np.ones(blending_coord_length) /
                    blending_coord_length).astype(np.float32)
        self.assertEqual(self.cube.shape, result.shape)
        self.assertArrayEqual(expected, result[:, 0, 0])


class Test_check_weights(Test_weighted_blend):

    """Test the check_weights function."""

    def test_weights_sum_to_1(self):
        """Test that if a weights cube is provided which is properly
        normalised, i.e. the weights sum to one over the blending
        dimension, no exception is raised."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        plugin.check_weights(self.weights3d.data, 0)

    def test_weights_do_not_sum_to_1_error(self):
        """Test that if a weights cube is provided which is not properly
        normalised, i.e. the weights do not sum to one over the blending
        dimension, an error is raised."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        weights = self.weights3d.data
        weights[0, 0, 1] = 1.
        msg = 'Weights do not sum to 1 over the blending coordinate.'
        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_weights(weights, 0)


class Test_percentile_weighted_mean(Test_weighted_blend):

    """Test the percentile_weighted_mean function."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_with_weights(self):
        """Test function when a data cube and a weights cube are provided."""

        perc_cube = percentile_cube()
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        perc_coord = perc_cube.coord('percentile')
        result = plugin.percentile_weighted_mean(perc_cube, self.weights1d,
                                                 perc_coord)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data,
                                    BLENDED_PERCENTILE_DATA)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_with_spatially_varying_weights(self):
        """Test function when a data cube and a multi dimensional weights cube
        are provided. This tests spatially varying weights, where each x-y
        position is weighted differently in each slice along the blending
        coordinate."""

        perc_cube = percentile_cube()
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        perc_coord = perc_cube.coord('percentile')
        result = plugin.percentile_weighted_mean(perc_cube, self.weights3d,
                                                 perc_coord)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data,
                                    BLENDED_PERCENTILE_DATA_SPATIAL_WEIGHTS)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_without_weights(self):
        """Test function when a data cube is provided, but no weights cube
        which should result in equal weightings."""

        perc_cube = percentile_cube()
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        perc_coord = perc_cube.coord('percentile')
        result = plugin.percentile_weighted_mean(perc_cube, None,
                                                 perc_coord)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.data, BLENDED_PERCENTILE_DATA_EQUAL_WEIGHTS)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_percentiles_different_coordinate_orders(self):
        """Test the result of the percentile aggregation is the same
        regardless of the coordinate order in the input cube. Most
        importantly, the result should be the same regardless of on which
        side of the collapsing coordinate the percentile coordinate falls."""
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        weights = None
        percentile_leading = percentile_cube()
        time_leading = percentile_cube()
        time_leading.transpose([1, 0, 2, 3])

        pl_perc_coord = percentile_leading.coord('percentile')
        tl_perc_coord = time_leading.coord('percentile')

        result_percentile_leading = plugin.percentile_weighted_mean(
            percentile_leading, weights, pl_perc_coord)
        result_time_leading = plugin.percentile_weighted_mean(
            time_leading, weights, tl_perc_coord)

        self.assertArrayAlmostEqual(result_percentile_leading.data,
                                    BLENDED_PERCENTILE_DATA_EQUAL_WEIGHTS)
        self.assertArrayAlmostEqual(result_time_leading.data,
                                    BLENDED_PERCENTILE_DATA_EQUAL_WEIGHTS)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_percentiles_another_coordinate_order(self):
        """Test the result of the percentile aggregation is the same
        regardless of the coordinate order in the input cube. In this case a
        spatial coordinate is moved to the leading position. The resulting
        array is of a different shape, but the data is the same. We reorder
        the expected values to confirm this."""
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        weights = None
        longitude_leading = percentile_cube()
        longitude_leading.transpose([3, 0, 1, 2])
        perc_coord = longitude_leading.coord('percentile')

        result = plugin.percentile_weighted_mean(
            longitude_leading, weights, perc_coord)

        self.assertArrayAlmostEqual(
            result.data,
            np.moveaxis(BLENDED_PERCENTILE_DATA_EQUAL_WEIGHTS, [2], [0]))


class Test_weighted_mean(Test_weighted_blend):

    """Test the weighted_mean function."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_with_weights(self):
        """Test function when a data cube and a weights cube are provided."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        result = plugin.weighted_mean(self.cube, self.weights1d)
        expected = np.full((2, 2), 1.5)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_with_spatially_varying_weights(self):
        """Test function when a data cube and a multi dimensional weights cube
        are provided. This tests spatially varying weights, where each x-y
        position is weighted differently in each slice along the blending
        coordinate."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        result = plugin.weighted_mean(self.cube, self.weights3d)
        expected = np.array([[2.7, 2.1], [2.4, 1.8]])

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_without_weights(self):
        """Test function when a data cube is provided, but no weights cube
        which should result in equal weightings."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        result = plugin.weighted_mean(self.cube, None)
        expected = np.full((2, 2), 2.)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)


class Test_process(Test_weighted_blend):

    """Test the process method."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube with metadata that
        matches the input cube where appropriate."""
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        result = plugin.process(self.cube)

        # conform metadata means the output FRT and forecast period are the
        # most recent on the input cubes.
        expected_frt = int(
            self.cube.coord('forecast_reference_time').points[-1])
        expected_forecast_period = int(
            self.cube.coord('forecast_period').points[-1])

        self.assertIsInstance(result, Cube)
        self.assertEqual(result.attributes, self.attributes)
        self.assertEqual(result.coord('forecast_reference_time').points,
                         expected_frt)
        self.assertEqual(result.coord('forecast_period').points,
                         expected_forecast_period)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_alternative_title(self):
        """Test that the plugin returns an iris.cube.Cube with metadata that
        matches the input cube where appropriate. In this case the title is
        removed from the input cube, resulting in a default title being applied
        to the result cube."""

        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        self.cube.attributes.pop('title')
        expected = "IMPROVER Model Forecast"
        result = plugin.process(self.cube)

        self.assertEqual(result.attributes['title'], expected)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_specific_cycletime(self):
        """Test that the plugin setup with a specific cycletime returns a cube
        in which the forecast reference time has been changed to match the
        given cycletime. The forecast period should also have been adjusted to
        be given relative to this time.

        For this we need a single time in our cube and so to blend over
        something else. In this case we create a "model_id" coordinate as if we
        are model blending."""

        coord_name = "model_id"

        cube1 = self.cube[0].copy()
        model_crd1 = iris.coords.DimCoord([0], long_name=coord_name, units=1)
        cube1.add_aux_coord(model_crd1)

        cube2 = self.cube[0].copy()
        model_crd2 = iris.coords.DimCoord([1], long_name=coord_name, units=1)
        cube2.add_aux_coord(model_crd2)

        cubes = iris.cube.CubeList([cube1, cube2])
        cube = merge_cubes(cubes)

        plugin = WeightedBlendAcrossWholeDimension(
            coord_name, cycletime='20151118T0900Z')
        expected_frt = 1447837200
        expected_forecast_period = 61200
        result = plugin.process(cube)

        self.assertEqual(result.coord('forecast_reference_time').points,
                         expected_frt)
        self.assertEqual(result.coord('forecast_period').points,
                         expected_forecast_period)
        self.assertEqual(result.coord('time').points,
                         cube.coord('time').points)

    def test_fails_coord_not_in_cube(self):
        """Test it raises CoordinateNotFoundError if the blending coord is not
        found in the cube."""
        coord = "notset"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        msg = ('Coordinate to be collapsed not found in cube.')
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            plugin.process(self.cube)

    def test_fails_coord_not_in_weights_cube(self):
        """Test it raises CoordinateNotFoundError if the blending coord is not
        found in the weights cube."""
        coord = "forecast_reference_time"
        self.weights1d.remove_coord("forecast_reference_time")
        plugin = WeightedBlendAcrossWholeDimension(coord)
        msg = ('Coordinate to be collapsed not found in weights cube.')
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            plugin.process(self.cube, self.weights1d)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Type Error if not supplied with a cube."""
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        notacube = 0.0
        msg = ('The first argument must be an instance of iris.cube.Cube')
        with self.assertRaisesRegex(TypeError, msg):
            plugin.process(notacube)

    def test_scalar_coord(self):
        """Test plugin throws an error if trying to blending across a scalar
        coordinate."""

        coord = "dummy_scalar_coord"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        weights = ([1.0])
        msg = 'has no associated dimension'
        with self.assertRaisesRegex(ValueError, msg):
            _ = plugin.process(self.cube_with_scalar, weights)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def tests_threshold_cube_with_weights_weighted_mean(self):
        """Test weighted_mean method works collapsing a cube with a threshold
        dimension when the blending is over a different coordinate. Note that
        this test is in process to include the slicing."""
        coord = "forecast_reference_time"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        result = plugin.process(self.cube_threshold, self.weights1d)
        expected_result_array = np.ones((2, 2, 2))*0.3
        expected_result_array[1, :, :] = 0.5

        expected_frt = int(
            self.cube.coord('forecast_reference_time').points[-1])
        expected_forecast_period = int(
            self.cube.coord('forecast_period').points[-1])

        self.assertArrayAlmostEqual(result.data, expected_result_array)
        self.assertEqual(result.attributes, self.attributes)
        self.assertEqual(result.coord('time').points, expected_frt)
        self.assertEqual(result.coord('forecast_period').points,
                         expected_forecast_period)


if __name__ == '__main__':
    unittest.main()
