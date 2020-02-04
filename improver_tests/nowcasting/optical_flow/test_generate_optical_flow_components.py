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
"""Unit tests for generation of optical flow components"""

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.tests import IrisTest

from improver.nowcasting.optical_flow import generate_optical_flow_components
from improver.utilities.warnings_handler import ManageWarnings

from ...set_up_test_cubes import set_up_variable_cube


class Test_generate_optical_flow_components(IrisTest):
    """Tests for the generate_optical_flow_components method"""

    def setUp(self):
        """Set up test cubes.  Optical flow velocity values are tested within
        the Test_optical_flow module; this class tests timestamps only."""
        self.iterations = 20
        self.ofc_box_size = 10

        dummy_cube = set_up_variable_cube(
            np.zeros((30, 30), dtype=np.float32),
            name="lwe_precipitation_rate", units="mm h-1",
            spatial_grid="equalarea", time=datetime(2018, 2, 20, 4, 0),
            frt=datetime(2018, 2, 20, 4, 0))
        coord_points = 2000*np.arange(30, dtype=np.float32)  # in metres
        dummy_cube.coord(axis='x').points = coord_points
        dummy_cube.coord(axis='y').points = coord_points

        self.first_cube = dummy_cube.copy()
        self.second_cube = dummy_cube.copy()
        # 15 minutes later, in seconds
        self.second_cube.coord("time").points = (
            self.second_cube.coord("time").points + 15*60)

        self.third_cube = dummy_cube.copy()
        # 30 minutes later, in seconds
        self.third_cube.coord("time").points = (
            self.third_cube.coord("time").points + 30*60)

        self.expected_time = self.third_cube.coord("time").points[0]

    @ManageWarnings(ignored_messages=[
        "No non-zero data in input fields",
        "Collapsing a non-contiguous coordinate"])
    def test_basic(self):
        """Test output is a tuple of cubes"""
        cubelist = [self.first_cube, self.second_cube, self.third_cube]
        result = generate_optical_flow_components(
            cubelist, self.ofc_box_size, self.iterations)
        for cube in result:
            self.assertIsInstance(cube, iris.cube.Cube)
            self.assertAlmostEqual(
                cube.coord("time").points[0], self.expected_time)

    @ManageWarnings(ignored_messages=[
        "No non-zero data in input fields",
        "Collapsing a non-contiguous coordinate"])
    def test_time_ordering(self):
        """Test output timestamps are insensitive to input cube order"""
        cubelist = [self.second_cube, self.third_cube, self.first_cube]
        result = generate_optical_flow_components(
            cubelist, self.ofc_box_size, self.iterations)
        for cube in result:
            self.assertAlmostEqual(
                cube.coord("time").points[0], self.expected_time)

    @ManageWarnings(ignored_messages=[
        "No non-zero data in input fields",
        "Collapsing a non-contiguous coordinate"])
    def test_fewer_inputs(self):
        """Test routine can produce output from a shorter list of inputs"""
        result = generate_optical_flow_components(
            [self.second_cube, self.third_cube], self.ofc_box_size,
            self.iterations)
        for cube in result:
            self.assertAlmostEqual(
                cube.coord("time").points[0], self.expected_time)


if __name__ == '__main__':
    unittest.main()
