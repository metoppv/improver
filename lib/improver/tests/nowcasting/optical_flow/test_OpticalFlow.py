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
""" Unit tests for the optical_flow.OpticalFlow plugin """

import datetime
import unittest
import warnings
import numpy as np

import iris
from iris.tests import IrisTest

from improver.nowcasting.optical_flow import OpticalFlow


class Test__init__(IrisTest):
    """Test class initialisation"""

    def test_basic(self):
        """Test initialisation and types"""
        plugin = OpticalFlow()
        self.assertIsInstance(plugin.data_smoothing_radius, int)
        self.assertIsInstance(plugin.data_smoothing_method, str)
        self.assertIsInstance(plugin.boxsize, int)
        self.assertIsInstance(plugin.iterations, int)
        self.assertIsInstance(plugin.point_weight, float)
        self.assertIsInstance(plugin.small_kernel, np.ndarray)
        self.assertIsNone(plugin.data1)
        self.assertIsNone(plugin.data2)
        self.assertIsNone(plugin.shape)
        self.assertIsNone(plugin.ucomp)
        self.assertIsNone(plugin.vcomp)


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


class Test_corner(OpticalFlowUtilityTest):
    """Test corner averaging function"""

    def test_basic(self):
        """Test result is of correct type and shape"""
        result = self.plugin.corner(self.plugin.data1)
        self.assertIsInstance(result, np.ndarray)
        self.assertSequenceEqual(result.shape, (2, 4))

    def test_values(self):
        """Test output values"""
        expected_output = np.array([[1., 2., 3., 4.],
                                    [0.25, 1., 2., 3.]])
        result = self.plugin.corner(self.plugin.data1)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_first_axis(self):
        """Test averaging over first axis"""
        expected_output = np.array([[0.5, 1.5, 2.5, 3.5, 4.5],
                                    [0.0, 0.5, 1.5, 2.5, 3.5]])
        result = self.plugin.corner(self.plugin.data1, axis=0)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_second_axis(self):
        """Test averaging over second axis"""
        expected_output = np.array([[1.5, 2.5, 3.5, 4.5],
                                    [0.5, 1.5, 2.5, 3.5],
                                    [0.0, 0.5, 1.5, 2.5]])
        result = self.plugin.corner(self.plugin.data1, axis=1)
        self.assertArrayAlmostEqual(result, expected_output)


class Test_mdiff_spatial(OpticalFlowUtilityTest):
    """Test mdiff_spatial function"""

    def test_basic(self):
        """Test for correct output type"""
        result = self.plugin.mdiff_spatial(axis=0)
        self.assertIsInstance(result, np.ndarray)

    def test_default(self):
        """Test output values for axis=0"""
        expected_output = np.array([[-0.1875, -0.4375, -0.5,    -0.5, -0.25],
                                    [-0.2500, -0.6875, -0.9375, -1.0, -0.50],
                                    [-0.0625, -0.2500, -0.4375, -0.5, -0.25]])
        result = self.plugin.mdiff_spatial(axis=0)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_transpose(self):
        """Test output values for axis=1"""
        expected_output = np.array([[0.1875, 0.4375, 0.5000, 0.5, 0.25],
                                    [0.2500, 0.6875, 0.9375, 1.0, 0.50],
                                    [0.0625, 0.2500, 0.4375, 0.5, 0.25]])
        result = self.plugin.mdiff_spatial(axis=1)
        self.assertArrayAlmostEqual(result, expected_output)


class Test_mdiff_temporal(OpticalFlowUtilityTest):
    """Test mdiff_temporal function"""

    def test_basic(self):
        """Test for correct output type"""
        result = self.plugin.mdiff_temporal()
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output values"""
        expected_output = np.array([[-0.1875, -0.4375, -0.5,    -0.5, -0.25],
                                    [-0.2500, -0.6875, -0.9375, -1.0, -0.50],
                                    [-0.0625, -0.2500, -0.4375, -0.5, -0.25]])
        result = self.plugin.mdiff_temporal()
        self.assertArrayAlmostEqual(result, expected_output)


class Test_makesubboxes(OpticalFlowUtilityTest):
    """Test makesubboxes function"""

    def test_basic(self):
        """Test for correct output types"""
        field = np.ones(shape=self.plugin.data1.shape)
        boxes, weights = self.plugin.makesubboxes(field, 2)
        self.assertIsInstance(boxes, list)
        self.assertIsInstance(weights, np.ndarray)

    def test_weights(self):
        """Test output weights values"""
        expected_weights = np.array([0.54216664, 0.95606307, 0.917915, 0.,
                                     0.46473857, 0.54216664])
        field = np.ones(shape=self.plugin.data1.shape)
        _, weights = self.plugin.makesubboxes(field, 2)
        self.assertArrayAlmostEqual(weights, expected_weights)


class OpticalFlowVelocityTest(IrisTest):
    """Class with shared plugin definition for velocity smoothing tests"""

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
        self.plugin = OpticalFlow(boxsize=3, kernel=3, iterations=10)
        self.plugin.data1 = np.zeros((12, 15))
        self.plugin.shape = self.plugin.data1.shape


class Test_regrid_velocities(OpticalFlowVelocityTest):
    """Test regrid_velocities function"""

    def test_basic(self):
        """Test for correct output types"""
        umat, _ = self.plugin.regrid_velocities(self.umat, self.vmat)
        self.assertIsInstance(umat, np.ndarray)
        self.assertSequenceEqual(umat.shape, (12, 15))

    def test_values(self):
        """Test output matrices have expected values"""
        expected_umat = np.array(
            [[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
             [2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
             [2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
             [3., 3., 3., 2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
             [3., 3., 3., 2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
             [3., 3., 3., 2., 2., 2., 1., 1., 1., 1., 1., 1., 0., 0., 0.]])

        expected_vmat = np.array(
            [[3., 3., 3., 2., 2., 2., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
             [3., 3., 3., 2., 2., 2., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
             [3., 3., 3., 2., 2., 2., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
             [2., 2., 2., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [2., 2., 2., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [2., 2., 2., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.]])

        umat, vmat = self.plugin.regrid_velocities(self.umat, self.vmat)

        self.assertArrayAlmostEqual(umat, expected_umat)
        self.assertArrayAlmostEqual(vmat, expected_vmat)


class Test_smoothing(OpticalFlowVelocityTest):
    """Test simple smoothing function"""

    def test_basic(self):
        """Test for correct output types"""
        output = self.plugin.smoothing(self.umat, 2)
        self.assertIsInstance(output, np.ndarray)

    def test_box_smoothing(self):
        """Test smoothing over square box (default)"""
        expected_output = np.array([[0.84, 0.60, 0.36, 0.12, 0.04],
                                    [1.20, 0.92, 0.60, 0.28, 0.12],
                                    [1.56, 1.24, 0.84, 0.44, 0.20],
                                    [1.92, 1.56, 1.08, 0.60, 0.28]])

        output = self.plugin.smoothing(self.umat, 2)
        self.assertArrayAlmostEqual(output, expected_output)

    def test_kernel_smoothing(self):
        """Test smoothing over circular kernel"""
        expected_output = np.array([[0.8125, 0.3750, 0.0625, 0., 0.],
                                    [1.1250, 0.7500, 0.3125, 0.0625, 0.],
                                    [1.8125, 1.3125, 0.7500, 0.3125, 0.0625],
                                    [2.5000, 1.8125, 1.1250, 0.6250, 0.1875]])

        output = self.plugin.smoothing(self.umat, 2, method='kernel')
        self.assertArrayAlmostEqual(output, expected_output)

    def test_null_behaviour(self):
        """Test smoothing with a kernel radius of 1 has no effect"""
        output = self.plugin.smoothing(self.umat, 1, method='kernel')
        self.assertArrayAlmostEqual(output, self.umat)


class Test_smart_smoothing(OpticalFlowVelocityTest):
    """Test smart smoothing function"""

    def test_basic(self):
        """Test for correct output types"""
        umat, _ = self.plugin.smart_smoothing(
            self.umat, self.vmat, self.umat, self.vmat, self.weights)
        self.assertIsInstance(umat, np.ndarray)
        self.assertSequenceEqual(umat.shape, self.umat.shape)

    def test_values(self):
        """Test output matrices have expected values"""
        expected_umat = np.array([[1., 1., 1., 0., 0.],
                                  [1.25352113, 1.19354839, 1., 0.08333333, 0.],
                                  [1.48780488, 1.50000000, 1., 1.00000000, 1.],
                                  [2., 2., 1., 1., 1.]])

        expected_vmat = np.array([[2.69230769, 2.53846154, 1., 0.25000000, 0.],
                                  [2.04225352, 1.96774194, 1., 0.08333333, 0.],
                                  [1.43902439, 1.25000000, 1., 1., 1.],
                                  [1., 1., 1., 1., 1.]])

        umat, vmat = self.plugin.smart_smoothing(
            self.umat, self.vmat, self.umat, self.vmat, self.weights)
        self.assertArrayAlmostEqual(umat, expected_umat)
        self.assertArrayAlmostEqual(vmat, expected_vmat)


class Test_smooth_advection_velocities(OpticalFlowVelocityTest):
    """Test smoothing of advection velocities"""

    def test_basic(self):
        """Test for correct output types"""
        umat, _ = self.plugin.smooth_advection_velocities(
            self.umat, self.vmat, self.weights)
        self.assertIsInstance(umat, np.ndarray)
        self.assertSequenceEqual(umat.shape, (12, 15))

    def test_values(self):
        """Test output matrices have expected values"""
        first_row_u = np.array(
            [1.124620, 1.124620, 1.124620, 1.145532, 1.145532,
             1.145532, 1.192604, 1.192604, 1.192604, 1.050985,
             1.050985, 1.050985, 0.967760, 0.967760, 0.967760])

        first_row_v = np.array(
            [2.455172, 2.455172, 2.455172, 2.345390, 2.345390,
             2.345390, 2.032608, 2.032608, 2.032608, 1.589809,
             1.589809, 1.589809, 1.331045, 1.331045, 1.331045])

        umat, vmat = self.plugin.smooth_advection_velocities(
            self.umat, self.vmat, self.weights)
        self.assertArrayAlmostEqual(umat[0], first_row_u)
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


class Test_calculate_advection_velocities(IrisTest):
    """Test calculation of advection velocity fields"""

    def setUp(self):
        """Set up plugin options and input rainfall-like matrices that produce
        non-singular outputs.  Large matrices with zeros are needed for the
        smoothing algorithms to behave sensibly."""

        self.plugin = OpticalFlow(kernel=3, boxsize=3, iterations=10)

        rainfall_block = np.array([[1., 1., 1., 1., 1., 1., 1.],
                                   [1., 2., 2., 2., 2., 1., 1.],
                                   [1., 2., 3., 3., 2., 1., 1.],
                                   [1., 2., 3., 3., 2., 1., 1.],
                                   [1., 2., 2., 2., 2., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1.]])

        first_input = np.zeros((10, 10))
        first_input[1:8, 2:9] = rainfall_block
        self.plugin.data1 = first_input
        self.plugin.shape = first_input.shape

        second_input = np.zeros((10, 10))
        second_input[2:9, 1:8] = rainfall_block
        self.plugin.data2 = second_input

        # NOTE fix x/y axis inversion - coord naming...
        self.partial_dx = self.plugin.mdiff_spatial(axis=0)
        self.partial_dy = self.plugin.mdiff_spatial(axis=1)
        self.partial_dt = self.plugin.mdiff_temporal()


    def test_basic(self):
        """Test outputs are of the correct type"""
        umat, _ = self.plugin.calculate_advection_velocities(
            self.partial_dx, self.partial_dy, self.partial_dt)
        self.assertIsInstance(umat, np.ndarray)
        self.assertSequenceEqual(umat.shape, self.plugin.shape)

    def test_values(self):
        """Test output values"""
        umat, vmat = self.plugin.calculate_advection_velocities(
            self.partial_dx, self.partial_dy, self.partial_dt)
        self.assertAlmostEqual(np.mean(umat), 0.121514428331)
        self.assertAlmostEqual(np.mean(vmat), -0.121514428331)


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up plugin options and input rainfall-like matrices that produce
        non-singular outputs.  Large matrices with zeros are needed for the
        smoothing algorithms to behave sensibly."""

        self.plugin = OpticalFlow(kernel=3, boxsize=3, iterations=10)

        rainfall_block = np.array([[1., 1., 1., 1., 1., 1., 1.],
                                   [1., 2., 2., 2., 2., 1., 1.],
                                   [1., 2., 3., 3., 2., 1., 1.],
                                   [1., 2., 3., 3., 2., 1., 1.],
                                   [1., 2., 2., 2., 2., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1.]])

        self.first_input = np.zeros((16, 16))
        self.first_input[1:8, 2:9] = rainfall_block

        self.second_input = np.zeros((16, 16))
        self.second_input[2:9, 1:8] = rainfall_block


    def test_basic(self):
        """Test outputs are of the correct type and value"""
        self.plugin.process(self.first_input, self.second_input)
        self.assertIsInstance(self.plugin.ucomp, np.ndarray)
        self.assertIsInstance(self.plugin.vcomp, np.ndarray)
        # NOTE x/y axis inversion - will need to fix this...
        self.assertAlmostEqual(np.mean(self.plugin.ucomp), 0.95435266462)
        self.assertAlmostEqual(np.mean(self.plugin.vcomp), -0.95435266462)


if __name__ == '__main__':
    unittest.main()
