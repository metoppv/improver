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
"""Unit tests for the rescale function from rescale.py."""

import unittest
import numpy as np

from iris.tests import IrisTest

from improver.utilities.rescale import rescale
from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


class Test_rescale(IrisTest):

    """Test the utilities.rescale rescale function."""

    def setUp(self):
        """
        Create a cube with a single non-zero point.
        Trap standard output
        """
        self.cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube())

    def test_basic(self):
        """Test that the method returns the expected array type"""
        result = rescale(self.cube.data)
        self.assertIsInstance(result, np.ndarray)

    def test_zero_range_input(self):
        """Test that the method returns the expected error"""
        msg = "Cannot rescale a zero input range"
        with self.assertRaisesRegex(ValueError, msg):
            rescale(self.cube.data, data_range=[0, 0])

    def test_zero_range_output(self):
        """Test that the method returns the expected error"""
        msg = "Cannot rescale a zero output range"
        with self.assertRaisesRegex(ValueError, msg):
            rescale(self.cube.data, scale_range=[4, 4])

    def test_rescaling_inrange(self):
        """Test that the method returns the expected values when in range"""
        expected = self.cube.data.copy()
        expected[...] = 110.
        expected[0, 0, 7, 7] = 100.
        result = rescale(self.cube.data, data_range=(0., 1.),
                         scale_range=(100., 110.))
        self.assertArrayAlmostEqual(result, expected)

    def test_rescaling_outrange(self):
        """Test that the method gives the expected values when out of range"""
        expected = self.cube.data.copy()
        expected[...] = 108.
        expected[0, 0, 7, 7] = 98.
        result = rescale(self.cube.data, data_range=(0.2, 1.2),
                         scale_range=(100., 110.))
        self.assertArrayAlmostEqual(result, expected)

    def test_clip(self):
        """Test that the method clips values when out of range"""
        expected = self.cube.data.copy()
        expected[...] = 108.
        expected[0, 0, 7, 7] = 100.
        result = rescale(self.cube.data, data_range=(0.2, 1.2),
                         scale_range=(100., 110.), clip=True)
        self.assertArrayAlmostEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
