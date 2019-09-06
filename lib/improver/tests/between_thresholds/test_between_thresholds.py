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
"""Tests for the OccurrenceBetweenThresholds plugin"""

import unittest
import numpy as np

import iris
from iris.tests import IrisTest

from improver.tests.set_up_test_cubes import set_up_probability_cube
from improver.between_thresholds import OccurrenceBetweenThresholds


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up a test cube with probability data"""
        data = np.array([
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[0.9, 0.9, 0.9], [0.8, 0.8, 0.8], [0.7, 0.7, 0.7]],
            [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
            [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.1, 0.2, 0.2]]],
            dtype=np.float32)
        thresholds = np.array([279, 280, 281, 282], dtype=np.float32)
        self.temp_cube = set_up_probability_cube(data, thresholds)

        vis_thresholds = np.array([100, 1000, 5000, 10000], dtype=np.float32)
        self.vis_cube = set_up_probability_cube(
            np.flip(data, axis=0), vis_thresholds, variable_name='visibility',
            threshold_units='m', spp__relative_to_threshold='below')

    def test_above_threshold(self):
        """Test values from an "above threshold" cube"""
        threshold_ranges = [[280, 281], [281, 282]]
        plugin = OccurrenceBetweenThresholds(threshold_ranges)
        result = plugin.process(self.temp_cube)
        print(result)













if __name__ == '__main__':
    unittest.main()
