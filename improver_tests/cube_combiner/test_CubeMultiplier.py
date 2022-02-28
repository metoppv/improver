# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
from iris.coords import CellMethod
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver.cube_combiner import Combine, CubeMultiplier
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver_tests.cube_combiner.test_CubeCombiner import CombinerTest


class MultiplierTest(CombinerTest):
    """Add specific test cubes for testing the Multiplier functionality."""

    def setUp(self):
        """Set-up the plugin for testing."""
        super().setUp()
        self.cube = self.cube4.copy()
        cell_methods = [
            CellMethod(
                "sum",
                coords="time",
                comments="of lwe_thickness_of_precipitation_amount",
            ),
        ]
        self.cube.cell_methods = cell_methods

        self.multiplier = self.cube4[:, 0, ...].copy(
            data=np.ones_like(self.cube4[:, 0, ...].data)
        )
        self.threshold_aux = self.multiplier.coord(
            "lwe_thickness_of_precipitation_amount"
        )
        self.multiplier.remove_coord("lwe_thickness_of_precipitation_amount")


class Test_process(MultiplierTest):
    """Test process method of CubeMultiplier"""

    def test_broadcast_coord(self):
        """Test that plugin broadcasts to threshold coord without changing inputs.
        Using the broadcast_to_coords argument including a value of "threshold"
        will result in the returned cube maintaining the probabilistic elements
        of the name of the first input cube."""
        cubelist = iris.cube.CubeList([self.cube.copy(), self.multiplier])
        input_copy = deepcopy(cubelist)
        result = CubeMultiplier(broadcast_to_threshold=True)(cubelist, "new_cube_name")
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "probability_of_new_cube_name_above_threshold")
        self.assertEqual(result.coord(var_name="threshold").name(), "new_cube_name")
        self.assertArrayAlmostEqual(result.data, self.cube.data)
        self.assertCubeListEqual(input_copy, cubelist)

    def test_vicinity_names(self):
        """Test plugin names the cube and threshold coordinate correctly for a
        vicinity diagnostic"""
        input = "lwe_thickness_of_precipitation_amount_in_vicinity"
        output = "thickness_of_rainfall_amount_in_vicinity"
        self.cube.rename(f"probability_of_{input}_above_threshold")
        cubelist = iris.cube.CubeList([self.cube.copy(), self.multiplier])
        result = CubeMultiplier(broadcast_to_threshold=True)(cubelist, output)
        self.assertEqual(result.name(), f"probability_of_{output}_above_threshold")
        self.assertEqual(
            result.coord(var_name="threshold").name(), "thickness_of_rainfall_amount"
        )

    def test_error_broadcast_coord_not_found(self):
        """Test that plugin throws an error if asked to broadcast to a threshold coord
        that is not present on the first cube"""
        cubelist = iris.cube.CubeList([self.multiplier, self.cube.copy()])
        msg = (
            r"Cannot find coord threshold in "
            r"<iris 'Cube' of "
            r"probability_of_lwe_thickness_of_precipitation_amount_above_threshold / \(1\) "
            r"\(realization: 3; latitude: 2; longitude: 2\)> to broadcast to"
        )
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            CubeMultiplier(broadcast_to_threshold=True)(cubelist, "new_cube_name")

    def test_error_broadcast_coord_is_auxcoord(self):
        """Test that plugin throws an error if asked to broadcast to a threshold coord
        that already exists on later cubes"""
        self.multiplier.add_aux_coord(self.threshold_aux)
        cubelist = iris.cube.CubeList([self.cube.copy(), self.multiplier])
        msg = "Cannot broadcast to coord threshold as it already exists as an AuxCoord"
        with self.assertRaisesRegex(TypeError, msg):
            CubeMultiplier(broadcast_to_threshold=True)(cubelist, "new_cube_name")

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

    def test_update_cell_methods(self):
        """Test that plugin updates cell methods where required when a new
        diagnostic name is provided."""
        cubelist = iris.cube.CubeList([self.cube, self.multiplier])

        new_cube_name = "new_cube_name"
        expected = CellMethod("sum", coords="time", comments=f"of {new_cube_name}")

        result = CubeMultiplier(broadcast_to_threshold=True)(cubelist, new_cube_name)
        self.assertEqual(result.cell_methods[0], expected)

    def test_unmodified_cell_methods(self):
        """Test that plugin leaves cell methods that are diagnostic name
        agnostic unchanged."""

        cell_methods = list(self.cube.cell_methods)
        additional_cell_method_1 = CellMethod("sum", coords="longitude")
        additional_cell_method_2 = CellMethod(
            "sum", coords="latitude", comments="Kittens are great"
        )
        cell_methods.extend([additional_cell_method_1, additional_cell_method_2])

        self.cube.cell_methods = cell_methods
        cubelist = iris.cube.CubeList([self.cube, self.multiplier])

        new_cube_name = "new_cube_name"
        expected = [
            CellMethod("sum", coords="time", comments=f"of {new_cube_name}"),
            additional_cell_method_1,
            additional_cell_method_2,
        ]

        result = CubeMultiplier(broadcast_to_threshold=True)(cubelist, new_cube_name)
        self.assertArrayEqual(result.cell_methods, expected)

    def test_with_Combine(self):
        """Test plugin works through the Combine plugin"""
        cubelist = iris.cube.CubeList([self.cube.copy(), self.multiplier])
        result = Combine("*", broadcast_to_threshold=True, new_name="new_cube_name")(
            cubelist
        )
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "probability_of_new_cube_name_above_threshold")
        self.assertEqual(result.coord(var_name="threshold").name(), "new_cube_name")
        self.assertArrayAlmostEqual(result.data, self.cube.data)


if __name__ == "__main__":
    unittest.main()
