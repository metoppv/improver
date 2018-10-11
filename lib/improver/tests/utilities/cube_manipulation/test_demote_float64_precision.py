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
"""
Unit tests for the utilities within the
"cube_manipulation.demote_float64_precision" module.
"""

import unittest

from iris.tests import IrisTest
import numpy as np

from improver.utilities.cube_manipulation import demote_float64_precision

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube


class Test_demote_float64_precision(IrisTest):
    """ Test the demote_float64_precision utility."""

    def setUp(self):
        """Create two temperature cubes to test with."""
        self.cube1 = set_up_temperature_cube(dtype=np.float64)
        self.cube2 = set_up_temperature_cube(dtype=np.float64)
        self.assertEqual(self.cube1.dtype, np.float64)

    def test_basic(self):
        """Test that the function will return a single iris.cube.Cube with
           float32 precision."""
        result1 = self.cube1
        demote_float64_precision(result1)
        self.assertEqual(result1.dtype, np.float32)

    def test_process_list(self):
        """Test that the function will return a list of cubes with
           float32 precision."""
        result1 = self.cube1
        result2 = self.cube2
        demote_float64_precision([result1, result2])
        self.assertEqual(result1.dtype, np.float32)
        self.assertEqual(result2.dtype, np.float32)

    def test_process_none(self):
        """Test that the function fails with non-cube types."""
        result1 = self.cube1
        result2 = None
        msg = "Object is not a cube"
        with self.assertRaisesRegex(AssertionError, msg):
            demote_float64_precision([result1, result2])

    def test_ignores_int(self):
        """Test that the function does not change int data types."""
        result1 = self.cube1
        result1.data = result1.data.astype(np.int32)
        demote_float64_precision(result1)
        self.assertEqual(result1.dtype, np.int32)


if __name__ == '__main__':
    unittest.main()
