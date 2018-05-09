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

from improver.optical_flow.optical_flow import OpticalFlow


class OpticalFlowUtilityTest(IrisTest):
    """Class with shared plugin definition for utility tests"""

    def setUp(self):
        """Set up dummy plugin and populate data members"""
        self.plugin = OpticalFlow()
        self.plugin.data1 = np.array([[1., 2., 3., 4., 5.],
                                      [0., 1., 2., 3., 4.],
                                      [0., 0., 1., 2., 3.]])
        self.plugin.data2 = np.array([[0., 1., 2., 3., 4.],
                                      [0., 0., 1., 2., 3.],
                                      [0., 0., 0., 1., 2.]])


class OpticalFlowTest(IrisTest):
    """Class with shared matrix and plugin definitions for tests
    requiring larger input arrays"""

    def setUp(self):
        """Set up plugin options and input rainfall-like matrices that produce
        non-singular outputs.  Large matrices with zeros are needed for the
        smoothing algorithms to behave sensibly."""

        self.plugin = OpticalFlow(kernel=3, boxsize=3, iterations=10)

        self.first_input = np.zeros((16, 16))
        self.first_input[1:8, 2:9] = 1.
        self.first_input[2:6, 3:7] = 2.
        self.first_input[3:5, 4:6] = 3.

        self.second_input = np.zeros((16, 16))
        self.second_input[2:9, 1:8] = 1.
        self.second_input[3:7, 2:6] = 2.
        self.second_input[4:6, 3:5] = 3.


class Test__init__(IrisTest):
    """Test class initialisation"""

    def test_basic(self):
        """Test initialisation and types"""
        plugin = OpticalFlow(kernel=3, boxsize=3, iterations=10)
        self.assertIsInstance(plugin.kernel, int)
        self.assertIsInstance(plugin.boxsize, int)
        self.assertIsInstance(plugin.iterations, int)
        self.assertIsInstance(plugin.pointweight, float)
        self.assertIsNone(plugin.ucomp)
        self.assertIsNone(plugin.vcomp)


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


class Test_solve_for_uv(IrisTest):
    """Test solve_for_uv function"""

    def setUp(self):
        """Define input matrices"""
        self.I_xy = np.array([[2., 3.,],
                              [1., -2.,]])
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


class Test_smooth_advection_velocities(OpticalFlowTest):
    """Test smoothing of advection velocities"""

    def test_basic(self):
        """Test for correct output types"""
        # TODO
        pass



class Test_calculate_advection_velocities(IrisTest):
    """Test calculation of advection velocity fields"""
    # TODO
    pass


class Test_process(OpticalFlowTest):
    """Test the process method"""

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
