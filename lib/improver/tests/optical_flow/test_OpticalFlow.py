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


class OpticalFlowUtilityTest(IrisTest):
    """Class with shared matrix definitions for utility tests"""

    def setUp(self):
        """Set up input matrices and dummy plugin"""
        self.first_input = np.array([[1., 2., 3., 4., 5.],
                                     [0., 1., 2., 3., 4.],
                                     [0., 0., 1., 2., 3.]])

        self.second_input = np.array([[0., 1., 2., 3., 4.],
                                      [0., 0., 1., 2., 3.],
                                      [0., 0., 0., 1., 2.]])


class Test_mdiff_spatial(OpticalFlowUtilityTest):
    """Test mdiff_spatial function"""

    def test_basic(self):
        """Test for correct output type"""
        result = OpticalFlow().mdiff_spatial(self.first_input,
                                             self.second_input, 0)
        self.assertIsInstance(result, np.ndarray)

    def test_default(self):
        """Test output values for axis=0"""
        expected_output = np.array([[0., 0., 0., 0., 0.],
                                    [0., -0.75, -1., -1., -1.],
                                    [0., -0.25, -0.75, -1., -1.]])
        result = OpticalFlow().mdiff_spatial(self.first_input,
                                             self.second_input, 0)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_transpose(self):
        """Test output values for axis=1"""
        expected_output = np.array([[0., 0.00, 0.00, 0., 0.],
                                    [0., 0.75, 1.00, 1., 1.],
                                    [0., 0.25, 0.75, 1., 1.]])
        result = OpticalFlow().mdiff_spatial(self.first_input,
                                             self.second_input, 1)
        self.assertArrayAlmostEqual(result, expected_output)


class Test_totx(OpticalFlowUtilityTest):
    """Test calculation of partial derivatives"""

    def test_basic(self):
        """Test output type"""
        result = OpticalFlow().totx(self.first_input, self.second_input, 0)
        self.assertIsInstance(result, np.ndarray)

    def test_x_axis(self):
        """Test derivative along the x-axis (TODO hardcoded 0)"""
        expected_output = np.array([[-0.1875, -0.4375, -0.5,    -0.5, -0.25],
                                    [-0.2500, -0.6875, -0.9375, -1.0, -0.50],
                                    [-0.0625, -0.2500, -0.4375, -0.5, -0.25]])
        result = OpticalFlow().totx(self.first_input, self.second_input, 0)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_y_axis(self):
        """Test derivative along the y-axis (TODO hardcoded 1)"""
        expected_output = np.array([[0.1875, 0.4375, 0.5000, 0.5, 0.25],
                                    [0.2500, 0.6875, 0.9375, 1.0, 0.50],
                                    [0.0625, 0.2500, 0.4375, 0.5, 0.25]])
        result = OpticalFlow().totx(self.first_input, self.second_input, 1)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_t_axis(self):
        """Test derivative in time"""
        expected_output = np.array([[0.1875, 0.4375, 0.5000, 0.5, 0.25],
                                    [0.2500, 0.6875, 0.9375, 1.0, 0.50],
                                    [0.0625, 0.2500, 0.4375, 0.5, 0.25]])
        result = OpticalFlow().totx(self.first_input, self.second_input, 2)



class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up plugin options and input rainfall-like matrices that produce
        non-singular outputs.  Large matrices with zeros are needed for the
        algorithm to be stable."""

        self.plugin = OpticalFlow(kernel=3, boxsize=3, iterations=10)

        self.first_input = np.zeros((16, 16))
        self.first_input[1:8, 2:9] = 1.
        self.first_input[2:6, 3:7] = 2.
        self.first_input[3:5, 4:6] = 3.
        #print self.first_input
        #print "\n"
        self.second_input = np.zeros((16, 16))
        self.second_input[2:9, 1:8] = 1.
        self.second_input[3:7, 2:6] = 2.
        self.second_input[4:6, 3:5] = 3.
        #print self.second_input
        #print "\n"

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
