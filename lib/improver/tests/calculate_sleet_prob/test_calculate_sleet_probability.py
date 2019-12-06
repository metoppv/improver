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
"""Unit tests for the calculate_sleet_probability plugin."""

import unittest

import numpy as np
from iris.tests import IrisTest
from improver.calculate_sleet_prob import calculate_sleet_probability
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_calculate_sleet_probability(IrisTest):
    """ Tests the calculate sleet probability function."""

    def setUp(self):
        """Create cubes to input into the function."""

        rain_prob = np.array([[0.5, 0.1, 1.0],
                              [0.0, 0.2, 0.5],
                              [0.1, 0.1, 0.3]], dtype=np.float32)
        self.rain_prob_cube = set_up_variable_cube(rain_prob)

        snow_prob = np.array([[0.0, 0.4, 0.0],
                              [0.5, 0.3, 0.1],
                              [0.0, 0.4, 0.3]], dtype=np.float32)
        self.snow_prob_cube = set_up_variable_cube(snow_prob)

    def test_basic_calculation(self):
        """Test the basic sleet calculation works."""
        expected_result = np.array([[0.5, 0.5, 0.0],
                                    [0.5, 0.5, 0.4],
                                    [0.9, 0.5, 0.4]], dtype=np.float32)
        result = calculate_sleet_probability(
            self.rain_prob_cube, self.snow_prob_cube)
        self.assertArrayAlmostEqual(result.data, expected_result)


if __name__ == '__main__':
    unittest.main()
