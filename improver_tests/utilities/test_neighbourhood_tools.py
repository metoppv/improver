# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Unit tests for spatial padding utilities"""

import unittest
import numpy as np
from iris.tests import IrisTest

from improver.utilities.neighbourhood_tools import rolling_window


class Test_rolling_window(IrisTest):

    """Test the padding of a coordinate."""

    def setUp(self):
        """Set up a 10 * 10 * 10 array."""
        self.array = np.arange(25).astype(np.int32).reshape((5, 5))

    def test_neighbourhood_size_2(self):
        """Test producing a 2 * 2 neighbourhood."""
        windows = rolling_window(self.array, (2, 2))
        expected = np.zeros((4, 4, 2, 2), dtype=np.int32)
        for i in range(4):
            for j in range(4):
                expected[i, j] = self.array[i : i + 2, j : j + 2]
        self.assertArrayEqual(windows, expected)

    def test_exception_too_many_dims(self):
        """Test an exception is raised if shape has too many dimensions."""
        msg = "Number of dimensions of input_array must be greater than or equal to length of shape"
        with self.assertRaisesRegex(ValueError, msg):
            rolling_window(self.array, (2, 2, 2))

    def test_exception_dims_too_large(self):
        """Test an exception is raised if dimensions of shape are larger than corresponding dimensions of input array."""
        msg = "New shape contains a dimension that is negative or zero"
        with self.assertRaisesRegex(RuntimeError, msg):
            rolling_window(self.array, (2, 6))


if __name__ == "__main__":
    unittest.main()
