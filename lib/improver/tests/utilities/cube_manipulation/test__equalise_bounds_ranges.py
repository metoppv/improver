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
Unit tests for the function "cube_manipulation._equalise_bounds_ranges".
"""

import unittest
import numpy as np
from datetime import datetime as dt

import iris
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import _equalise_bounds_ranges
from improver.tests.set_up_test_cubes import set_up_probability_cube


class Test__equalise_bounds_ranges(IrisTest):
    """Test for the _equalise_bounds_ranges function"""

    def setUp(self):
        """Set up some cubes with different bounds ranges"""
        frt = dt(2017, 11, 9, 21, 0)
        times = [dt(2017, 11, 10, 3, 0),
                 dt(2017, 11, 10, 4, 0),
                 dt(2017, 11, 10, 5, 0)]
        time_bounds = np.array([
            [dt(2017, 11, 10, 2, 0), dt(2017, 11, 10, 3, 0)],
            [dt(2017, 11, 10, 3, 0), dt(2017, 11, 10, 4, 0)],
            [dt(2017, 11, 10, 4, 0), dt(2017, 11, 10, 5, 0)]])

        cubes = iris.cube.CubeList([])
        for tpoint, tbounds in zip(times, time_bounds):
            cube = set_up_probability_cube(
                0.6*np.ones((2, 3, 3), dtype=np.float32),
                np.array([278., 280.], dtype=np.float32),
                time=tpoint, frt=frt, time_bounds=tbounds)
            cubes.append(cube)
        self.matched_cube = cubes.merge_cube()

        time_bounds[2, 0] = dt(2017, 11, 10, 2, 0)
        cubes = iris.cube.CubeList([])
        for tpoint, tbounds in zip(times, time_bounds):
            cube = set_up_probability_cube(
                0.6*np.ones((2, 3, 3), dtype=np.float32),
                np.array([278., 280.], dtype=np.float32),
                time=tpoint, frt=frt, time_bounds=tbounds)
            cubes.append(cube)
        self.unmatched_cube = cubes.merge_cube()

    def test_basic(self):
        """Test no error when bounds match"""
        _equalise_bounds_ranges(
            self.matched_cube, ["time", "forecast_period"])

    def test_error(self):
        """Test error when bounds do not match"""
        msg = 'Cube with mismatching time bounds ranges'
        with self.assertRaisesRegex(ValueError, msg):
            _equalise_bounds_ranges(
                self.unmatched_cube, ["time", "forecast_period"])


if __name__ == '__main__':
    unittest.main()
