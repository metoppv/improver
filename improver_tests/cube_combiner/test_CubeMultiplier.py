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
"""Unit tests for the cube_combiner.CubeMultiplier plugin."""
import unittest
from copy import deepcopy
from datetime import datetime

import iris
import numpy as np
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver.cube_combiner import CubeMultiplier
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver_tests.cube_combiner.test_CubeCombiner import CombinerTest


class Test_process(CombinerTest):
    """Test process method of CubeMultiplier"""

    def test_broadcast_coord(self):
        """Test that plugin broadcasts to threshold coord without changing inputs.
        Using the broadcast_to_coords argument including a value of "threshold"
        will result in the returned cube maintaining the probabilistic elements
        of the name of the first input cube."""
        cube = self.cube4[:, 0, ...].copy()
        cube.data = np.ones_like(cube.data)
        cube.remove_coord("lwe_thickness_of_precipitation_amount")
        cubelist = iris.cube.CubeList([self.cube4.copy(), cube])
        input_copy = deepcopy(cubelist)
        result = CubeMultiplier()(
            cubelist, "new_cube_name", broadcast_to_threshold=True
        )
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "probability_of_new_cube_name_above_threshold")
        self.assertEqual(result.coord(var_name="threshold").name(), "new_cube_name")
        self.assertArrayAlmostEqual(result.data, self.cube4.data)
        self.assertCubeListEqual(input_copy, cubelist)

    def test_vicinity_names(self):
        """Test plugin names the cube and threshold coordinate correctly for a
        vicinity diagnostic"""
        input = "lwe_thickness_of_precipitation_amount_in_vicinity"
        output = "thickness_of_rainfall_amount_in_vicinity"
        self.cube4.rename(f"probability_of_{input}_above_threshold")
        cube = self.cube4[:, 0, ...].copy()
        cube.data = np.ones_like(cube.data)
        cube.remove_coord("lwe_thickness_of_precipitation_amount")
        cubelist = iris.cube.CubeList([self.cube4.copy(), cube])
        input_copy = deepcopy(cubelist)
        result = CubeMultiplier()(cubelist, output, broadcast_to_threshold=True)
        self.assertEqual(result.name(), f"probability_of_{output}_above_threshold")
        self.assertEqual(
            result.coord(var_name="threshold").name(), "thickness_of_rainfall_amount"
        )

    def test_error_broadcast_coord_not_found(self):
        """Test that plugin throws an error if asked to broadcast to a threshold coord
        that is not present on the first cube"""
        cube = self.cube4[:, 0, ...].copy()
        cube.data = np.ones_like(cube.data)
        cube.remove_coord("lwe_thickness_of_precipitation_amount")
        cubelist = iris.cube.CubeList([cube, self.cube4.copy()])
        msg = (
            "Cannot find coord threshold in "
            "<iris 'Cube' of probability_of_lwe_thickness_of_precipitation_amount_above_threshold / \(1\) "
            "\(realization: 3; latitude: 2; longitude: 2\)> to broadcast to"
        )
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            CubeMultiplier()(cubelist, "new_cube_name", broadcast_to_threshold=True)

    def test_error_broadcast_coord_is_auxcoord(self):
        """Test that plugin throws an error if asked to broadcast to a threshold coord
        that already exists on later cubes"""
        cube = self.cube4[:, 0, ...].copy()
        cube.data = np.ones_like(cube.data)
        cubelist = iris.cube.CubeList([self.cube4.copy(), cube])
        msg = "Cannot broadcast to coord threshold as it already exists as an AuxCoord"
        with self.assertRaisesRegex(TypeError, msg):
            CubeMultiplier()(cubelist, "new_cube_name", broadcast_to_threshold=True)

    def test_multiply_preserves_bounds(self):
        """Test specific case for precipitation type, where multiplying a
        precipitation accumulation by a point-time probability of snow retains
        the bounds on the original accumulation."""
        validity_time = datetime(2015, 11, 19, 0)
        time_bounds = [datetime(2015, 11, 18, 23), datetime(2015, 11, 19, 0)]
        forecast_reference_time = datetime(2015, 11, 18, 22)
        precip_accum = set_up_variable_cube(
            np.full((2, 3, 3), 1.5, dtype=np.float32),
            name="lwe_thickness_of_precipitation_amount",
            units="mm",
            time=validity_time,
            time_bounds=time_bounds,
            frt=forecast_reference_time,
        )
        snow_prob = set_up_variable_cube(
            np.full(precip_accum.shape, 0.2, dtype=np.float32),
            name="probability_of_snow",
            units="1",
            time=validity_time,
            frt=forecast_reference_time,
        )
        result = CubeMultiplier()(
            [precip_accum, snow_prob], "lwe_thickness_of_snowfall_amount",
        )
        self.assertArrayAlmostEqual(result.data, np.full((2, 3, 3), 0.3))
        self.assertArrayEqual(result.coord("time"), precip_accum.coord("time"))

    def test_exception_for_single_entry_cubelist(self):
        """Test that the plugin raises an exception if a cubelist containing
        only one cube is passed in."""
        plugin = CubeMultiplier()
        msg = "Expecting 2 or more cubes in cube_list"
        cubelist = iris.cube.CubeList([self.cube1])
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cubelist, "new_cube_name")


if __name__ == "__main__":
    unittest.main()
