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
""" Unit tests for the nowcasting.OpticalFlow plugin """

import unittest
from datetime import datetime, timedelta

import iris
import numpy as np
from iris.coords import DimCoord
from iris.exceptions import InvalidCubeError
from iris.tests import IrisTest

from improver.nowcasting.optical_flow import OpticalFlow
from improver.utilities.warnings_handler import ManageWarnings

from ...set_up_test_cubes import set_up_variable_cube


class Test__init__(IrisTest):
    """Test OpticalFlow class initialisation"""

    def test_basic(self):
        """Test initialisation and types"""
        plugin = OpticalFlow()
        self.assertIsInstance(plugin.data_smoothing_radius_km, float)
        self.assertIsInstance(plugin.data_smoothing_method, str)
        self.assertIsInstance(plugin.iterations, int)
        self.assertIsInstance(plugin.point_weight, float)
        self.assertIsNone(plugin.data1)
        self.assertIsNone(plugin.data2)
        self.assertIsNone(plugin.shape)


class Test__repr__(IrisTest):
    """Test string representation"""

    def test_basic(self):
        """Test string representation"""
        expected_string = ('<OpticalFlow: data_smoothing_radius_km: 14.0, '
                           'data_smoothing_method: box, iterations: 100, '
                           'point_weight: 0.1>')
        result = str(OpticalFlow())
        self.assertEqual(result, expected_string)


class Test_makekernel(IrisTest):
    """Test makekernel function"""

    def test_basic(self):
        """Test for correct output type"""
        result = OpticalFlow().makekernel(2)
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output values"""
        expected_output = np.array([[0., 0., 0., 0., 0.],
                                    [0., 0.0625, 0.1250, 0.0625, 0.],
                                    [0., 0.1250, 0.2500, 0.1250, 0.],
                                    [0., 0.0625, 0.1250, 0.0625, 0.],
                                    [0., 0., 0., 0., 0.]])
        result = OpticalFlow().makekernel(2)
        self.assertArrayAlmostEqual(result, expected_output)


class OpticalFlowUtilityTest(IrisTest):
    """Class with shared plugin definition for small utility tests"""

    def setUp(self):
        """Set up dummy plugin and populate data members"""
        self.plugin = OpticalFlow()
        self.plugin.data1 = np.array([[1., 2., 3., 4., 5.],
                                      [0., 1., 2., 3., 4.],
                                      [0., 0., 1., 2., 3.]])

        self.plugin.data2 = np.array([[0., 1., 2., 3., 4.],
                                      [0., 0., 1., 2., 3.],
                                      [0., 0., 0., 1., 2.]])

        self.plugin.shape = self.plugin.data1.shape


class Test_interp_to_midpoint(OpticalFlowUtilityTest):
    """Test interp_to_midpoint averaging function"""

    def test_basic(self):
        """Test result is of correct type and shape"""
        result = self.plugin.interp_to_midpoint(self.plugin.data1)
        self.assertIsInstance(result, np.ndarray)
        self.assertSequenceEqual(result.shape, (2, 4))

    def test_values(self):
        """Test output values"""
        expected_output = np.array([[1., 2., 3., 4.],
                                    [0.25, 1., 2., 3.]])
        result = self.plugin.interp_to_midpoint(self.plugin.data1)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_first_axis(self):
        """Test averaging over first axis"""
        expected_output = np.array([[0.5, 1.5, 2.5, 3.5, 4.5],
                                    [0.0, 0.5, 1.5, 2.5, 3.5]])
        result = self.plugin.interp_to_midpoint(self.plugin.data1, axis=0)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_second_axis(self):
        """Test averaging over second axis"""
        expected_output = np.array([[1.5, 2.5, 3.5, 4.5],
                                    [0.5, 1.5, 2.5, 3.5],
                                    [0.0, 0.5, 1.5, 2.5]])
        result = self.plugin.interp_to_midpoint(self.plugin.data1, axis=1)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_array_too_small(self):
        """Test returns empty array if averaging over an axis of length 1"""
        small_array = self.plugin.data1[0, :].reshape((1, 5))
        result = self.plugin.interp_to_midpoint(small_array)
        self.assertEqual(result.size, 0.)

    def test_small_array_single_axis(self):
        """Test sensible output if averaging over one valid axis"""
        expected_output = np.array([[1.5, 2.5, 3.5, 4.5]])
        small_array = self.plugin.data1[0, :].reshape((1, 5))
        result = self.plugin.interp_to_midpoint(small_array, axis=1)
        self.assertArrayAlmostEqual(result, expected_output)


class Test__partial_derivative_spatial(OpticalFlowUtilityTest):
    """Test _partial_derivative_spatial function"""

    def test_basic(self):
        """Test for correct output type and shape"""
        result = self.plugin._partial_derivative_spatial(axis=0)
        self.assertIsInstance(result, np.ndarray)
        self.assertSequenceEqual(result.shape, self.plugin.shape)

    def test_first_axis(self):
        """Test output values for axis=0"""
        expected_output = np.array([[-0.1875, -0.4375, -0.5, -0.5, -0.25],
                                    [-0.2500, -0.6875, -0.9375, -1.0, -0.50],
                                    [-0.0625, -0.2500, -0.4375, -0.5, -0.25]])
        result = self.plugin._partial_derivative_spatial(axis=0)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_second_axis(self):
        """Test output values for axis=1"""
        expected_output = np.array([[0.1875, 0.4375, 0.5000, 0.5, 0.25],
                                    [0.2500, 0.6875, 0.9375, 1.0, 0.50],
                                    [0.0625, 0.2500, 0.4375, 0.5, 0.25]])
        result = self.plugin._partial_derivative_spatial(axis=1)
        self.assertArrayAlmostEqual(result, expected_output)


class Test__partial_derivative_temporal(OpticalFlowUtilityTest):
    """Test _partial_derivative_temporal function"""

    def test_basic(self):
        """Test for correct output type and shape"""
        result = self.plugin._partial_derivative_temporal()
        self.assertIsInstance(result, np.ndarray)
        self.assertSequenceEqual(result.shape, self.plugin.shape)

    def test_values(self):
        """Test output values.  Note this is NOT the same function as
        _partial_derivative_spatial(axis=0), the output arrays are the same
        as a result of the choice of data."""
        expected_output = np.array([[-0.1875, -0.4375, -0.5, -0.5, -0.25],
                                    [-0.2500, -0.6875, -0.9375, -1.0, -0.50],
                                    [-0.0625, -0.2500, -0.4375, -0.5, -0.25]])
        result = self.plugin._partial_derivative_temporal()
        self.assertArrayAlmostEqual(result, expected_output)


class Test__make_subboxes(OpticalFlowUtilityTest):
    """Test _make_subboxes function"""

    def test_basic(self):
        """Test for correct output types"""
        self.plugin.boxsize = 2
        boxes, weights = self.plugin._make_subboxes(self.plugin.data1)
        self.assertIsInstance(boxes, list)
        self.assertIsInstance(boxes[0], np.ndarray)
        self.assertIsInstance(weights, np.ndarray)

    def test_box_list(self):
        """Test function carves up array as expected"""
        expected_boxes = \
            [np.array([[1., 2.], [0., 1.]]), np.array([[3., 4.], [2., 3.]]),
             np.array([[5.], [4.]]), np.array([[0., 0.]]),
             np.array([[1., 2.]]), np.array([[3.]])]
        self.plugin.boxsize = 2
        boxes, _ = self.plugin._make_subboxes(self.plugin.data1)
        for box, ebox in zip(boxes, expected_boxes):
            self.assertArrayAlmostEqual(box, ebox)

    def test_weights_values(self):
        """Test output weights values"""
        expected_weights = np.array([0.54216664, 0.95606307, 0.917915, 0.,
                                     0.46473857, 0.54216664])
        self.plugin.boxsize = 2
        _, weights = self.plugin._make_subboxes(self.plugin.data1)
        self.assertArrayAlmostEqual(weights, expected_weights)


class OpticalFlowDisplacementTest(IrisTest):
    """Class with shared plugin definition for smoothing and regridding
    tests"""

    def setUp(self):
        """Define input matrices and dummy plugin"""
        self.umat = np.array([[1., 0., 0., 0., 0.],
                              [1., 1., 0., 0., 0.],
                              [2., 1., 1., 0., 0.],
                              [3., 2., 1., 1., 0.]])

        self.vmat = np.array([[3., 2., 1., 0., 0.],
                              [2., 1., 0., 0., 0.],
                              [1., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0.]])

        self.weights = 0.3*np.multiply(self.umat, self.vmat)
        self.plugin = OpticalFlow(iterations=20)
        self.plugin.data_smoothing_radius = 3
        self.plugin.boxsize = 3
        # NOTE data dimensions are NOT exact multiples of box size
        self.plugin.data1 = np.zeros((11, 14))
        self.plugin.shape = self.plugin.data1.shape


class Test__box_to_grid(OpticalFlowDisplacementTest):
    """Test _box_to_grid function"""

    def test_basic(self):
        """Test for correct output types"""
        umat = self.plugin._box_to_grid(self.umat)
        self.assertIsInstance(umat, np.ndarray)
        self.assertSequenceEqual(umat.shape, (11, 14))

    def test_values(self):
        """Test output matrix values"""
        expected_umat = np.array(
            [[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
             [2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
             [2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
             [2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
             [3., 3., 3., 2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0.],
             [3., 3., 3., 2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0.]])

        umat = self.plugin._box_to_grid(self.umat)
        self.assertArrayAlmostEqual(umat, expected_umat)


class Test_smooth(OpticalFlowDisplacementTest):
    """Test simple smooth function"""

    def test_basic(self):
        """Test for correct output types"""
        output = self.plugin.smooth(self.umat, 2)
        self.assertIsInstance(output, np.ndarray)

    def test_box_smooth(self):
        """Test smooth over square box (default)"""
        expected_output = np.array([[0.84, 0.60, 0.36, 0.12, 0.04],
                                    [1.20, 0.92, 0.60, 0.28, 0.12],
                                    [1.56, 1.24, 0.84, 0.44, 0.20],
                                    [1.92, 1.56, 1.08, 0.60, 0.28]])

        output = self.plugin.smooth(self.umat, 2)
        self.assertArrayAlmostEqual(output, expected_output)

    def test_kernel_smooth(self):
        """Test smooth over circular kernel"""
        expected_output = np.array([[0.8125, 0.3750, 0.0625, 0., 0.],
                                    [1.1250, 0.7500, 0.3125, 0.0625, 0.],
                                    [1.8125, 1.3125, 0.7500, 0.3125, 0.0625],
                                    [2.5000, 1.8125, 1.1250, 0.6250, 0.1875]])

        output = self.plugin.smooth(self.umat, 2, method='kernel')
        self.assertArrayAlmostEqual(output, expected_output)

    def test_null_behaviour(self):
        """Test smooth with a kernel radius of 1 has no effect"""
        output = self.plugin.smooth(self.umat, 1, method='kernel')
        self.assertArrayAlmostEqual(output, self.umat)


class Test__smart_smooth(OpticalFlowDisplacementTest):
    """Test _smart_smooth function"""

    def test_basic(self):
        """Test for correct output types"""
        umat = self.plugin._smart_smooth(self.umat, self.umat, self.weights)
        self.assertIsInstance(umat, np.ndarray)
        self.assertSequenceEqual(umat.shape, self.umat.shape)

    def test_values(self):
        """Test output matrices have expected values"""
        expected_umat = np.array([[1., 1., 1., 0., 0.],
                                  [1.25352113, 1.19354839, 1., 0.08333333, 0.],
                                  [1.48780488, 1.50000000, 1., 1.00000000, 1.],
                                  [2., 2., 1., 1., 1.]])
        umat = self.plugin._smart_smooth(self.umat, self.umat, self.weights)
        self.assertArrayAlmostEqual(umat, expected_umat)


class Test__smooth_advection_fields(OpticalFlowDisplacementTest):
    """Test smoothing of advection displacements"""

    def test_basic(self):
        """Test for correct output types"""
        vmat = self.plugin._smooth_advection_fields(self.vmat,
                                                    self.weights)
        self.assertIsInstance(vmat, np.ndarray)
        self.assertSequenceEqual(vmat.shape, (11, 14))

    def test_values(self):
        """Test output matrices have expected values"""
        first_row_v = np.array(
            [2.451711, 2.451711, 2.451711, 2.341303, 2.341303, 2.341303,
             2.028805, 2.028805, 2.028805, 1.694845, 1.694845, 1.694845,
             1.503583, 1.503583])
        vmat = self.plugin._smooth_advection_fields(self.vmat,
                                                    self.weights)
        self.assertArrayAlmostEqual(vmat[0], first_row_v)


class Test_solve_for_uv(IrisTest):
    """Test solve_for_uv function"""

    def setUp(self):
        """Define input matrices"""
        self.I_xy = np.array([[2., 3.],
                              [1., -2.]])
        self.I_t = np.array([-8., 3.])

    def test_basic(self):
        """Test for correct output types"""
        u, v = OpticalFlow().solve_for_uv(self.I_xy, self.I_t)
        self.assertIsInstance(u, float)
        self.assertIsInstance(v, float)

    def test_values(self):
        """Test output values"""
        u, v = OpticalFlow().solve_for_uv(self.I_xy, self.I_t)
        self.assertAlmostEqual(u, 1.)
        self.assertAlmostEqual(v, 2.)


class Test_extreme_value_check(IrisTest):
    """Test extreme_value_check function"""

    def setUp(self):
        """Define some test velocity matrices"""
        self.umat = 0.2*np.arange(12).reshape((3, 4))
        self.vmat = -0.1*np.ones((3, 4), dtype=float)
        self.weights = np.full((3, 4), 0.5)

    def test_basic(self):
        """Test for correct output types"""
        OpticalFlow().extreme_value_check(self.umat, self.vmat, self.weights)
        self.assertIsInstance(self.umat, np.ndarray)
        self.assertIsInstance(self.vmat, np.ndarray)
        self.assertIsInstance(self.weights, np.ndarray)

    def test_values(self):
        """Test extreme data values are infilled with zeros"""
        expected_umat = np.array([[0., 0.2, 0.4, 0.6],
                                  [0.8, 0., 0., 0.],
                                  [0., 0., 0., 0.]])
        expected_vmat = np.array([[-0.1, -0.1, -0.1, -0.1],
                                  [-0.1, 0., 0., 0.],
                                  [0., 0., 0., 0.]])
        expected_weights = np.array([[0.5, 0.5, 0.5, 0.5],
                                     [0.5, 0., 0., 0.],
                                     [0., 0., 0., 0.]])
        OpticalFlow().extreme_value_check(self.umat, self.vmat, self.weights)
        self.assertArrayAlmostEqual(self.umat, expected_umat)
        self.assertArrayAlmostEqual(self.vmat, expected_vmat)
        self.assertArrayAlmostEqual(self.weights, expected_weights)

    def test_null_behaviour(self):
        """Test reasonable data values are preserved"""
        umat = 0.5*np.ones((3, 4), dtype=float)
        expected_umat = np.copy(umat)
        expected_vmat = np.copy(self.vmat)
        expected_weights = np.copy(self.weights)
        OpticalFlow().extreme_value_check(umat, self.vmat, self.weights)
        self.assertArrayAlmostEqual(umat, expected_umat)
        self.assertArrayAlmostEqual(self.vmat, expected_vmat)
        self.assertArrayAlmostEqual(self.weights, expected_weights)


class Test_calculate_displacement_vectors(IrisTest):
    """Test calculation of advection displacement vectors"""

    def setUp(self):
        """Set up plugin options and input rainfall-like matrices that produce
        non-singular outputs.  Large matrices with zeros are needed for the
        smoothing algorithms to behave sensibly."""

        self.plugin = OpticalFlow(iterations=20)
        self.plugin.data_smoothing_radius = 3
        self.plugin.boxsize = 3

        rainfall_block = np.array([[1., 1., 1., 1., 1., 1., 1.],
                                   [1., 2., 2., 2., 2., 1., 1.],
                                   [1., 2., 3., 3., 2., 1., 1.],
                                   [1., 2., 3., 3., 2., 1., 1.],
                                   [1., 2., 2., 2., 2., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1.]])

        first_input = np.zeros((10, 10), dtype=np.float32)
        first_input[1:8, 2:9] = rainfall_block
        self.plugin.data1 = first_input
        self.plugin.shape = first_input.shape

        second_input = np.zeros((10, 10), dtype=np.float32)
        second_input[2:9, 1:8] = rainfall_block
        self.plugin.data2 = second_input

        self.partial_dx = self.plugin._partial_derivative_spatial(axis=1)
        self.partial_dy = self.plugin._partial_derivative_spatial(axis=0)
        self.partial_dt = self.plugin._partial_derivative_temporal()

    def test_basic(self):
        """Test outputs are of the correct type"""
        umat, _ = self.plugin.calculate_displacement_vectors(
            self.partial_dx, self.partial_dy, self.partial_dt)
        self.assertIsInstance(umat, np.ndarray)
        self.assertSequenceEqual(umat.shape, self.plugin.shape)

    def test_values(self):
        """Test output values"""
        umat, vmat = self.plugin.calculate_displacement_vectors(
            self.partial_dx, self.partial_dy, self.partial_dt)
        self.assertAlmostEqual(np.mean(umat), np.float32(-0.124607998))
        self.assertAlmostEqual(np.mean(vmat), np.float32(0.124607998))


class Test__zero_advection_velocities_warning(IrisTest):
    """Test the _zero_advection_velocities_warning."""

    def setUp(self):
        """Set up arrays of advection velocities"""
        self.plugin = OpticalFlow()
        rain = np.ones((3, 3))
        self.rain_mask = np.where(rain > 0)

    @ManageWarnings(record=True)
    def test_warning_raised(self, warning_list=None):
        """Test that a warning is raised if an excess number of zero values
        are present within the input array."""
        greater_than_10_percent_zeroes_array = (
            np.array([[3., 5., 7.],
                      [0., 2., 1.],
                      [1., 1., 1.]]))
        warning_msg = "cells within the domain have zero advection"
        self.plugin._zero_advection_velocities_warning(
            greater_than_10_percent_zeroes_array, self.rain_mask)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    @ManageWarnings(record=True)
    def test_no_warning_raised_if_no_zeroes(self, warning_list=None):
        """Test that no warning is raised if the number of zero values in the
        array is below the threshold used to define an excessive number of
        zero values."""
        nonzero_array = np.array([[3., 5., 7.],
                                  [2., 2., 1.],
                                  [1., 1., 1.]])
        self.plugin._zero_advection_velocities_warning(nonzero_array,
                                                       self.rain_mask)
        self.assertTrue(len(warning_list) == 0)

    @ManageWarnings(record=True)
    def test_no_warning_raised_if_fewer_zeroes_than_threshold(
            self, warning_list=None):
        """Test that no warning is raised if the number of zero values in the
        array is below the threshold used to define an excessive number of
        zero values when at least one zero exists within the array."""
        rain = np.ones((5, 5))
        less_than_10_percent_zeroes_array = (
            np.array([[1., 3., 5., 7., 1.],
                      [0., 2., 1., 1., 1.],
                      [1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1.]]))
        self.plugin._zero_advection_velocities_warning(
            less_than_10_percent_zeroes_array, np.where(rain > 0))
        self.assertTrue(len(warning_list) == 0)

    @ManageWarnings(record=True)
    def test_no_warning_raised_for_modified_threshold(
            self, warning_list=None):
        """Test that no warning is raised if the number of zero values in the
        array is below the threshold used to define an excessive number of
        zero values when the threshold is modified."""
        less_than_30_percent_zeroes_array = (
            np.array([[3., 5., 7.],
                      [0., 2., 1.],
                      [0., 1., 1.]]))
        self.plugin._zero_advection_velocities_warning(
            less_than_30_percent_zeroes_array, self.rain_mask,
            zero_vel_threshold=0.3)
        self.assertTrue(len(warning_list) == 0)

    @ManageWarnings(record=True)
    def test_no_warning_raised_outside_rain(self, warning_list=None):
        """Test warning ignores zeros outside the rain area mask"""
        rain = np.array([[0, 0, 1],
                         [0, 1, 1],
                         [1, 1, 1]])
        wind = np.array([[0, 0, 1],
                         [0, 1, 1],
                         [1, 1, 1]])
        self.plugin._zero_advection_velocities_warning(
            wind, np.where(rain > 0))
        self.assertTrue(len(warning_list) == 0)


class Test_process_dimensionless(IrisTest):
    """Test the process_dimensionless method"""

    def setUp(self):
        """Set up plugin options and input rainfall-like matrices that produce
        non-singular outputs.  Large matrices with zeros are needed for the
        smoothing algorithms to behave sensibly."""

        self.plugin = OpticalFlow(iterations=20)
        self.plugin.boxsize = 3
        self.smoothing_kernel = 3

        rainfall_block = np.array([[1., 1., 1., 1., 1., 1., 1.],
                                   [1., 2., 2., 2., 2., 1., 1.],
                                   [1., 2., 3., 3., 2., 1., 1.],
                                   [1., 2., 3., 3., 2., 1., 1.],
                                   [1., 2., 2., 2., 2., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1.]])

        self.first_input = np.zeros((16, 16), dtype=np.float32)
        self.first_input[1:8, 2:9] = rainfall_block

        self.second_input = np.zeros((16, 16), dtype=np.float32)
        self.second_input[2:9, 1:8] = rainfall_block

    def test_basic(self):
        """Test outputs are of the correct type and value"""
        ucomp, vcomp = self.plugin.process_dimensionless(
            self.first_input, self.second_input, 0, 1, self.smoothing_kernel)
        self.assertIsInstance(ucomp, np.ndarray)
        self.assertIsInstance(vcomp, np.ndarray)
        self.assertAlmostEqual(np.mean(ucomp), 0.97735876)
        self.assertAlmostEqual(np.mean(vcomp), -0.97735894)

    def test_axis_inversion(self):
        """Test inverting x and y axis indices gives the correct result"""
        ucomp, vcomp = self.plugin.process_dimensionless(
            self.first_input, self.second_input, 1, 0, self.smoothing_kernel)
        self.assertAlmostEqual(np.mean(ucomp), -0.97735894)
        self.assertAlmostEqual(np.mean(vcomp), 0.97735876)


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up plugin and input rainfall-like cubes"""
        self.plugin = OpticalFlow(iterations=20)
        self.plugin.data_smoothing_radius_km = np.float32(6.)

        coord_points = 2000*np.arange(16, dtype=np.float32)  # in metres
        rainfall_block = np.array([[1., 1., 1., 1., 1., 1., 1.],
                                   [1., 2., 2., 2., 2., 1., 1.],
                                   [1., 2., 3., 3., 2., 1., 1.],
                                   [1., 2., 3., 3., 2., 1., 1.],
                                   [1., 2., 2., 2., 2., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1.]],
                                  dtype=np.float32)

        data1 = np.zeros((16, 16), dtype=np.float32)
        data1[1:8, 2:9] = rainfall_block
        self.cube1 = set_up_variable_cube(
            data1, name="rainfall_rate", units="mm h-1",
            spatial_grid="equalarea", time=datetime(2018, 2, 20, 4, 0),
            frt=datetime(2018, 2, 20, 4, 0))
        self.cube1.coord(axis='x').points = coord_points
        self.cube1.coord(axis='y').points = coord_points

        data2 = np.zeros((16, 16), dtype=np.float32)
        data2[2:9, 1:8] = rainfall_block
        self.cube2 = set_up_variable_cube(
            data2, name="rainfall_rate", units="mm h-1",
            spatial_grid="equalarea", time=datetime(2018, 2, 20, 4, 15),
            frt=datetime(2018, 2, 20, 4, 15))
        self.cube2.coord(axis='x').points = coord_points
        self.cube2.coord(axis='y').points = coord_points

    def test_basic(self):
        """Test correct output types and metadata"""
        ucube, vcube = self.plugin.process(self.cube1, self.cube2, boxsize=3)
        for cube in [ucube, vcube]:
            self.assertIsInstance(cube, iris.cube.Cube)
            self.assertEqual(cube.coord("time")[0],
                             self.cube2.coord("time")[0])
            self.assertEqual(cube.units, "m s-1")
            self.assertIn("precipitation_advection", cube.name())
            self.assertIn("velocity", cube.name())

    def test_values(self):
        """Test velocity values are as expected (in m/s)"""
        ucube, vcube = self.plugin.process(self.cube1, self.cube2, boxsize=3)
        self.assertAlmostEqual(
            np.mean(ucube.data), -2.1719084)
        self.assertAlmostEqual(np.mean(vcube.data), 2.1719084)

    def test_values_with_precip_rate_in_m_per_s(self):
        """Test velocity values are as expected (in m/s) when the input
        precipitation rates are in units of m/s rather than the expected
        mm/hr."""
        self.cube1.convert_units('m s-1')
        self.cube2.convert_units('m s-1')
        ucube, vcube = self.plugin.process(self.cube1, self.cube2, boxsize=3)
        self.assertAlmostEqual(
            np.mean(ucube.data), -2.1719084)
        self.assertAlmostEqual(np.mean(vcube.data), 2.1719084)

    def test_values_with_masked_data(self):
        """Test velocity values are as expected when masked cubes are used as
        input to the tests. This test is to capture behaviour whereby mask
        fill values were being used as valid data. This resulted in far from
        correct velocities being calculated by the optical flow code. Notably
        the velocity fields did not reflect the position of precipitation in
        the input precipitation fields, and the returned velocities were too
        low.

        In this test masked cubes are used and comparable unmasked cubes in
        which there the fill values are included in the field. We expect
        the results to be different, with higher velocities returned for the
        masked cubes.
        """
        mask = np.zeros((16, 16))
        mask[:2, :] = 1
        mask[:, :2] = 1

        # Ensure the masked data points contain a high fill value.
        data1 = self.cube1.data
        data2 = self.cube2.data
        data1[:2, :] = 1.0E36
        data1[:, :2] = 1.0E36
        data2[:2, :] = 1.0E36
        data2[:, :2] = 1.0E36

        masked1 = np.ma.MaskedArray(self.cube1.data, mask=mask)
        masked2 = np.ma.MaskedArray(self.cube2.data, mask=mask)

        masked_cube1 = self.cube1.copy(data=masked1)
        masked_cube2 = self.cube2.copy(data=masked2)
        unmasked_cube1 = self.cube1.copy(data=data1)
        unmasked_cube2 = self.cube2.copy(data=data2)

        ucube_masked, vcube_masked = self.plugin.process(
            masked_cube1, masked_cube2, boxsize=3)
        ucube_unmasked, vcube_unmasked = self.plugin.process(
            unmasked_cube1, unmasked_cube2, boxsize=3)

        self.assertAlmostEqual(
            np.mean(ucube_masked.data), -1.4995803)
        self.assertAlmostEqual(
            np.mean(vcube_masked.data), 1.4995805)
        self.assertAlmostEqual(
            np.mean(ucube_unmasked.data), -0.2869996)
        self.assertAlmostEqual(
            np.mean(vcube_unmasked.data), 0.28699964)

    def test_error_for_unconvertable_units(self):
        """Test that an exception is raised if the input precipitation cubes
        have units that cannot be converted to mm/hr."""
        self.cube1.units = 'm'
        self.cube2.units = 'm'

        msg = "Input data are in units that cannot be converted to mm/hr"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(self.cube1, self.cube2, boxsize=3)

    def test_input_cubes_unchanged(self):
        """Test the input precipitation rate cubes are unchanged by use in the
        optical flow plugin. One of the cubes is converted to rates in ms-1
        before use to ensure the cube remains in these units despite the
        default working units within optical flow being mm/hr."""
        self.cube1.convert_units("m s-1")
        cube1_ref = self.cube1.copy()
        cube2_ref = self.cube2.copy()
        _, _ = self.plugin.process(self.cube1, self.cube2, boxsize=3)
        self.assertEqual(self.cube1, cube1_ref)
        self.assertEqual(self.cube2, cube2_ref)

    def test_decrease_time_interval(self):
        """Test that decreasing the time interval between radar frames below
        15 minutes does not alter the smoothing radius. To test this the time
        interval is halved, which should give an answer identical to the values
        test above multiplied by a factor of two."""
        time_unit = self.cube2.coord("time").units
        new_time = time_unit.num2date(self.cube2.coord("time").points[0])
        new_time -= timedelta(seconds=450)
        self.cube2.remove_coord("time")
        time_coord = DimCoord(time_unit.date2num(new_time),
                              standard_name="time", units=time_unit)
        self.cube2.add_aux_coord(time_coord)

        ucube, vcube = self.plugin.process(self.cube1, self.cube2, boxsize=3)
        self.assertAlmostEqual(
            np.mean(ucube.data), -2.1719084 * 2.)
        self.assertAlmostEqual(np.mean(vcube.data), 2.1719084 * 2.)

    def test_increase_time_interval(self):
        """Test that increasing the time interval between radar frames above
        15 minutes leads to an increase in the data smoothing radius. In this
        test this will result in a smoothing radius larger than the box size,
        which is not allowed and will raise an exception. The updated radius
        value in this case is 12 km (6 grid squares), exceeding the 3 square
        box size."""
        time_unit = self.cube2.coord("time").units
        new_time = time_unit.num2date(self.cube2.coord("time").points[0])
        new_time += timedelta(seconds=900)
        self.cube2.remove_coord("time")
        time_coord = DimCoord(time_unit.date2num(new_time),
                              standard_name="time", units=time_unit)
        self.cube2.add_aux_coord(time_coord)
        msg = "Box size ([0-9]+) too small"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(self.cube1, self.cube2, boxsize=3)

    def test_error_small_kernel(self):
        """Test failure if data smoothing radius is too small"""
        self.plugin.data_smoothing_radius_km = 3.
        msg = "Input data smoothing radius 1 too small "
        with self.assertRaisesRegex(ValueError, msg):
            _ = self.plugin.process(self.cube1, self.cube2)

    def test_error_small_box(self):
        """Test failure if box size is smaller than data smoothing radius"""
        msg = "Box size 2 too small"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(self.cube1, self.cube2, boxsize=2)

    def test_error_unmatched_coords(self):
        """Test failure if cubes are provided on unmatched grids"""
        cube2 = self.cube2.copy()
        for ax in ["x", "y"]:
            cube2.coord(axis=ax).points = 4*np.arange(16)
        msg = "Input cubes on unmatched grids"
        with self.assertRaisesRegex(InvalidCubeError, msg):
            _ = self.plugin.process(self.cube1, cube2)

    def test_error_no_time_difference(self):
        """Test failure if two cubes are provided with the same time"""
        msg = "Expected positive time difference "
        with self.assertRaisesRegex(InvalidCubeError, msg):
            _ = self.plugin.process(self.cube1, self.cube1)

    def test_error_negative_time_difference(self):
        """Test failure if cubes are provided in the wrong order"""
        msg = "Expected positive time difference "
        with self.assertRaisesRegex(InvalidCubeError, msg):
            _ = self.plugin.process(self.cube2, self.cube1)

    @ManageWarnings(record=True)
    def test_warning_zero_inputs(self, warning_list=None):
        """Test code raises a warning and sets advection velocities to zero
        if there is no rain in the input cubes."""
        null_data = np.zeros(self.cube1.shape)
        cube1 = self.cube1.copy(data=null_data)
        cube2 = self.cube2.copy(data=null_data)
        ucube, vcube = self.plugin.process(cube1, cube2)
        warning_msg = "No non-zero data in input fields"
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertArrayAlmostEqual(ucube.data, null_data)
        self.assertArrayAlmostEqual(vcube.data, null_data)

    def test_error_nonmatching_inputs(self):
        """Test failure if cubes are of different data types"""
        self.cube1.rename("snowfall_rate")
        msg = "Input cubes contain different data types"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(self.cube1, self.cube2)

    @ManageWarnings(record=True)
    def test_warning_nonprecip_inputs(self, warning_list=None):
        """Test code raises a warning if input cubes have
        non-rain variable names"""
        self.cube1.rename("snowfall_rate")
        self.cube2.rename("snowfall_rate")
        _, _ = self.plugin.process(self.cube1, self.cube2, boxsize=3)
        warning_msg = "Input data are of non-precipitation type"
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))


if __name__ == '__main__':
    unittest.main()
