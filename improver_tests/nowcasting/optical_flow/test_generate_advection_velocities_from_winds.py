# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Unit tests for generation of optical flow components from a background flow"""

import unittest
from datetime import datetime

import iris
import numpy as np
import pytest
from iris.tests import IrisTest

from improver.nowcasting.optical_flow import generate_advection_velocities_from_winds
from improver.synthetic_data.set_up_test_cubes import add_coordinate
from improver.utilities.warnings_handler import ManageWarnings

from . import set_up_test_cube


class Test_generate_advection_velocities_from_winds(IrisTest):
    """Tests for the generate_advection_velocities_from_winds function.
    Optical flow velocity values are tested within the Test_optical_flow module;
    this class tests metadata only."""

    def setUp(self):
        """Set up test input cubes"""
        # Skip if pysteps not available
        pytest.importorskip("pysteps")

        shape = (30, 30)
        earlier_cube = set_up_test_cube(
            np.zeros(shape, dtype=np.float32),
            name="lwe_precipitation_rate",
            units="m s-1",
            time=datetime(2018, 2, 20, 4, 15),
        )
        later_cube = set_up_test_cube(
            np.zeros(shape, dtype=np.float32),
            name="lwe_precipitation_rate",
            units="m s-1",
            time=datetime(2018, 2, 20, 4, 30),
        )
        self.cubes = iris.cube.CubeList([earlier_cube, later_cube])

        wind_u = set_up_test_cube(
            np.ones(shape, dtype=np.float32),
            name="grid_eastward_wind",
            units="m s-1",
            time=datetime(2018, 2, 20, 4, 0),
        )
        wind_v = wind_u.copy()
        wind_v.rename("grid_northward_wind")
        self.steering_flow = iris.cube.CubeList([wind_u, wind_v])

        orogenh = set_up_test_cube(
            np.zeros(shape, dtype=np.float32),
            name="orographic_enhancement",
            units="m s-1",
            time=datetime(2018, 2, 20, 3, 0),
        )
        time_points = []
        for i in range(3):
            time_points.append(datetime(2018, 2, 20, 3 + i))
        self.orogenh = add_coordinate(orogenh, time_points, "time", is_datetime=True)

    @ManageWarnings(ignored_messages=["No non-zero data in input fields"])
    def test_basic(self):
        """Test function returns a cubelist with the expected components"""
        result = generate_advection_velocities_from_winds(
            self.cubes, self.steering_flow, self.orogenh
        )
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result[0].name(), "precipitation_advection_x_velocity")
        self.assertEqual(result[1].name(), "precipitation_advection_y_velocity")

    @ManageWarnings(ignored_messages=["No non-zero data in input fields"])
    def test_time(self):
        """Test output time coordinates are as expected"""
        current_time = self.cubes[1].coord("time").points[0]
        result = generate_advection_velocities_from_winds(
            self.cubes, self.steering_flow, self.orogenh
        )
        for cube in result:
            self.assertIsInstance(cube, iris.cube.Cube)
            self.assertEqual(cube.coord("time").points[0], current_time)

    @ManageWarnings(ignored_messages=["No non-zero data in input fields"])
    def test_input_sort(self):
        """Test output time coordinates are correct if the inputs are in the wrong
        order"""
        current_time = self.cubes[1].coord("time").points[0]
        reversed_cubelist = iris.cube.CubeList([self.cubes[1], self.cubes[0]])
        result = generate_advection_velocities_from_winds(
            reversed_cubelist, self.steering_flow, self.orogenh
        )
        for cube in result:
            self.assertIsInstance(cube, iris.cube.Cube)
            self.assertEqual(cube.coord("time").points[0], current_time)


if __name__ == "__main__":
    unittest.main()
