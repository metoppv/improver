# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Unit tests for the cube_combiner.CubeCombiner plugin."""
import unittest
from copy import deepcopy
from datetime import datetime

import iris
import numpy as np
from iris.coords import CellMethod
from iris.cube import Cube
from iris.tests import IrisTest

from improver.cube_combiner import Combine, CubeCombiner
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
    set_up_variable_cube
)
from improver_tests import ImproverTest
from iris.exceptions import CoordinateNotFoundError

class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = CubeCombiner("+")
        self.assertEqual(plugin.operator, np.add)

    def test_raise_error_wrong_operation(self):
        """Test __init__ raises a ValueError for invalid operation"""
        msg = "Unknown operation "
        with self.assertRaisesRegex(ValueError, msg):
            CubeCombiner("%")


class CombinerTest(ImproverTest):
    """Set up a common set of test cubes for subsequent test classes."""

    def setUp(self):
        """ Set up cubes for testing. """
        data = np.full((1, 2, 2), 0.5, dtype=np.float32)
        self.cube1 = set_up_probability_cube(
            data,
            np.array([0.001], dtype=np.float32),
            variable_name="lwe_thickness_of_precipitation_amount",
            time=datetime(2015, 11, 19, 0),
            time_bounds=(datetime(2015, 11, 18, 23), datetime(2015, 11, 19, 0)),
            frt=datetime(2015, 11, 18, 22),
        )

        data = np.full((1, 2, 2), 0.6, dtype=np.float32)
        self.cube2 = set_up_probability_cube(
            data,
            np.array([0.001], dtype=np.float32),
            variable_name="lwe_thickness_of_precipitation_amount",
            time=datetime(2015, 11, 19, 1),
            time_bounds=(datetime(2015, 11, 19, 0), datetime(2015, 11, 19, 1)),
            frt=datetime(2015, 11, 18, 22),
        )

        data = np.full((1, 2, 2), 0.1, dtype=np.float32)
        self.cube3 = set_up_probability_cube(
            data,
            np.array([0.001], dtype=np.float32),
            variable_name="lwe_thickness_of_precipitation_amount",
            time=datetime(2015, 11, 19, 1),
            time_bounds=(datetime(2015, 11, 19, 0), datetime(2015, 11, 19, 1)),
            frt=datetime(2015, 11, 18, 22),
        )

        data = np.full((2, 2, 2), 0.1, dtype=np.float32)
        self.cube4 = set_up_probability_cube(
            data,
            np.array([1.0, 2.0], dtype=np.float32),
            variable_name="lwe_thickness_of_precipitation_amount",
            time=datetime(2015, 11, 19, 1),
            time_bounds=(datetime(2015, 11, 19, 0), datetime(2015, 11, 19, 1)),
            frt=datetime(2015, 11, 18, 22),
        )
        self.cube4 = add_coordinate(
            iris.util.squeeze(self.cube4), np.arange(3), "realization", coord_units="1"
        )

        self.cube5 = self.cube4.copy()
        cell_methods = [
            CellMethod(
                "sum",
                coords="time",
                comments="of lwe_thickness_of_precipitation_amount",
            ),
        ]
        self.cube5.cell_methods = cell_methods

        self.multiplier = self.cube4[:, 0, ...].copy(
            data=np.ones_like(self.cube4[:, 0, ...].data)
        )
        self.threshold_aux = self.multiplier.coord(
            "lwe_thickness_of_precipitation_amount"
        )
        self.multiplier.remove_coord("lwe_thickness_of_precipitation_amount")

class Test__get_expanded_coord_names(CombinerTest):
    """Test method to determine coordinates for expansion"""

    def test_basic(self):
        """Test correct names are returned for scalar coordinates with
        different values"""
        expected_coord_set = {"time", "forecast_period"}
        result = CubeCombiner("+")._get_expanded_coord_names(
            [self.cube1, self.cube2, self.cube3]
        )
        self.assertIsInstance(result, list)
        self.assertSetEqual(set(result), expected_coord_set)

    def test_identical_inputs(self):
        """Test no coordinates are returned if inputs are identical"""
        result = CubeCombiner("+")._get_expanded_coord_names(
            [self.cube1, self.cube1, self.cube1]
        )
        self.assertFalse(result)

    def test_unmatched_coords_ignored(self):
        """Test coordinates that are not present on all cubes are ignored,
        regardless of input order"""
        expected_coord_set = {"time", "forecast_period"}
        height = iris.coords.AuxCoord([1.5], "height", units="m")
        self.cube1.add_aux_coord(height)
        result = CubeCombiner("+")._get_expanded_coord_names(
            [self.cube1, self.cube2, self.cube3]
        )
        self.assertSetEqual(set(result), expected_coord_set)
        result = CubeCombiner("+")._get_expanded_coord_names(
            [self.cube3, self.cube2, self.cube1]
        )
        self.assertSetEqual(set(result), expected_coord_set)


class Test_process(CombinerTest):

    """Test the plugin combines the cubelist into a cube."""

    def test_basic(self):
        """Test that the plugin returns a Cube and doesn't modify the inputs."""
        plugin = CubeCombiner("+")
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        input_copy = deepcopy(cubelist)
        result = plugin.process(cubelist, "new_cube_name")
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "new_cube_name")
        expected_data = np.full((2, 2), 1.1, dtype=np.float32)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertCubeListEqual(input_copy, cubelist)

    def test_basic_with_Combine(self):
        """Test that the basic test also works through the Combine plugin."""
        plugin = Combine("+", new_name="new_cube_name")
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        input_copy = deepcopy(cubelist)
        result = plugin.process(cubelist)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "new_cube_name")
        expected_data = np.full((2, 2), 1.1, dtype=np.float32)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertCubeListEqual(input_copy, cubelist)

    def test_mean(self):
        """Test that the plugin calculates the mean correctly. """
        plugin = CubeCombiner("mean")
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, "new_cube_name")
        expected_data = np.full((2, 2), 0.55, dtype=np.float32)
        self.assertEqual(result.name(), "new_cube_name")
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_addition_cell_method_coordinate(self):
        """Test that an exception is raised if a cell method coordinate is provided
        and the operation is not max, min or mean."""
        plugin = CubeCombiner("add", cell_method_coordinate="time")
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        msg = "A cell method coordinate has been produced with operation: add"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cubelist, "new_cube_name")

    def test_mean_cell_method_coordinate(self):
        """Test that a cell method is added, if a cell method coordinate is provided
        and a mean operation is undertaken."""
        plugin = CubeCombiner("mean", cell_method_coordinate="time")
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        input_copy = deepcopy(cubelist)
        result = plugin.process(cubelist, "new_cube_name")
        expected_data = np.full((2, 2), 0.55, dtype=np.float32)
        expected_cell_methods = (CellMethod("mean", coords="time"),)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(result.cell_methods, expected_cell_methods)
        self.assertCubeListEqual(input_copy, cubelist)

    def test_max_cell_method_coordinate(self):
        """Test that a cell method is added, if a cell method coordinate is provided
        and a max operation is undertaken."""
        plugin = CubeCombiner("max", cell_method_coordinate="time")
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        input_copy = deepcopy(cubelist)
        result = plugin.process(cubelist, "new_cube_name")
        expected_data = np.full((2, 2), 0.6, dtype=np.float32)
        expected_cell_methods = (CellMethod("maximum", coords="time"),)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(result.cell_methods, expected_cell_methods)
        self.assertCubeListEqual(input_copy, cubelist)

    def test_mixed_dtypes(self):
        """Test that the plugin calculates the sum correctly and doesn't mangle dtypes."""
        plugin = CubeCombiner("add")
        cubelist = iris.cube.CubeList(
            [self.cube1, self.cube2.copy(np.ones_like(self.cube2.data, dtype=np.int8))]
        )
        result = plugin.process(cubelist, "new_cube_name")
        expected_data = np.full((2, 2), 1.5, dtype=np.float32)
        self.assertEqual(result.name(), "new_cube_name")
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertTrue(cubelist[0].dtype == np.float32)
        self.assertTrue(cubelist[1].dtype == np.int8)
        self.assertTrue(result.dtype == np.float32)

    def test_mixed_dtypes_overflow(self):
        """Test the plugin with a dtype combination that results in float64 data."""
        plugin = CubeCombiner("add")
        cubelist = iris.cube.CubeList(
            [self.cube1, self.cube2.copy(np.ones_like(self.cube2.data, dtype=np.int32))]
        )
        msg = "Operation .* results in float64 data"
        with self.assertRaisesRegex(TypeError, msg):
            plugin.process(cubelist, "new_cube_name")

    def test_bounds_expansion(self):
        """Test that the plugin calculates the sum of the input cubes
        correctly and expands the time coordinate bounds on the
        resulting output."""
        plugin = CubeCombiner("add")
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, "new_cube_name")
        expected_data = np.full((2, 2), 1.1, dtype=np.float32)
        self.assertEqual(result.name(), "new_cube_name")
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(result.coord("time").points[0], 1447894800)
        self.assertArrayEqual(result.coord("time").bounds, [[1447887600, 1447894800]])

    def test_unmatched_scalar_coords(self):
        """Test a scalar coordinate that is present on the first cube is
        present unmodified on the output; and if present on a later cube is
        not present on the output."""
        height = iris.coords.AuxCoord([1.5], "height", units="m")
        self.cube1.add_aux_coord(height)
        result = CubeCombiner("add").process([self.cube1, self.cube2], "new_cube_name")
        self.assertEqual(result.coord("height"), height)
        result = CubeCombiner("add").process([self.cube2, self.cube1], "new_cube_name")
        result_coords = [coord.name() for coord in result.coords()]
        self.assertNotIn("height", result_coords)

    def test_mean_multi_cube(self):
        """Test that the plugin calculates the mean for three cubes."""
        plugin = CubeCombiner("mean")
        cubelist = iris.cube.CubeList([self.cube1, self.cube2, self.cube3])
        result = plugin.process(cubelist, "new_cube_name")
        expected_data = np.full((2, 2), 0.4, dtype=np.float32)
        self.assertEqual(result.name(), "new_cube_name")
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_with_mask(self):
        """Test that the plugin preserves the mask if any of the inputs are
        masked"""
        expected_data = np.full((2, 2), 1.2, dtype=np.float32)
        mask = [[False, True], [False, False]]
        self.cube1.data = np.ma.MaskedArray(self.cube1.data, mask=mask)
        plugin = CubeCombiner("add")
        result = plugin.process([self.cube1, self.cube2, self.cube3], "new_cube_name")
        self.assertIsInstance(result.data, np.ma.MaskedArray)
        self.assertArrayAlmostEqual(result.data.data, expected_data)
        self.assertArrayEqual(result.data.mask, mask)

    def test_exception_for_single_entry_cubelist(self):
        """Test that the plugin raises an exception if a cubelist containing
        only one cube is passed in."""
        plugin = CubeCombiner("-")
        msg = "Expecting 2 or more cubes in cube_list"
        cubelist = iris.cube.CubeList([self.cube1])
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cubelist, "new_cube_name")

    def test_broadcast_coord(self):
        """Test that plugin broadcasts to threshold coord without changing inputs.
        Using the broadcast_to_coords argument including a value of "threshold"
        will result in the returned cube maintaining the probabilistic elements
        of the name of the first input cube."""
        cubelist = iris.cube.CubeList([self.cube5.copy(), self.multiplier])
        input_copy = deepcopy(cubelist)
        result = CubeCombiner(operation="*",broadcast="threshold")(cubelist, "new_cube_name")
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "probability_of_new_cube_name_above_threshold")
        self.assertEqual(result.coord(var_name="threshold").name(), "new_cube_name")
        self.assertArrayAlmostEqual(result.data, self.cube5.data)
        self.assertCubeListEqual(input_copy, cubelist)

    def test_broadcast_non_threshold_coord(self):
        """Test that plugin broadcasts to a non-threshold coordinate."""
        validity_time = datetime(2015, 11, 19, 0)
        forecast_reference_time = datetime(2015, 11, 18, 22)
        
        cloud_base_height = set_up_variable_cube(
            np.full((2, 3, 4), 1000 , dtype=np.float32),
            name="cloud_base_altitude_assuming_only_consider_cloud_area_fraction_greater_than_2p5_oktas",
            units="m",
            time=validity_time,
            frt=forecast_reference_time,
        )
        orography=set_up_variable_cube(
            np.full((3, 4), 80 , dtype=np.float32),
            name="orography",
            units="m",
        )
        new_name="cloud_base_height_assuming_only_consider_cloud_area_fraction_greater_than_2p5_oktas"
        result = CubeCombiner(operation="-",broadcast="realization")([cloud_base_height,orography], new_name)
        self.assertArrayAlmostEqual(result.data,np.full_like(cloud_base_height,920))

    def test_vicinity_names(self):
        """Test plugin names the cube and threshold coordinate correctly for a
        vicinity diagnostic"""
        input = "lwe_thickness_of_precipitation_amount_in_vicinity"
        output = "thickness_of_rainfall_amount_in_vicinity"
        self.cube5.rename(f"probability_of_{input}_above_threshold")
        cubelist = iris.cube.CubeList([self.cube5.copy(), self.multiplier])
        result = CubeCombiner(operation="*",broadcast="threshold")(cubelist, output)
        self.assertEqual(result.name(), f"probability_of_{output}_above_threshold")
        self.assertEqual(
            result.coord(var_name="threshold").name(), "thickness_of_rainfall_amount"
        )

    def test_error_broadcast_coord_not_found(self):
        """Test that plugin throws an error if asked to broadcast to a threshold coord
        that is not present on the first cube"""
        cubelist = iris.cube.CubeList([self.multiplier, self.cube5.copy()])
        msg = (
            r"Cannot find coord percentile in "
            r"<iris 'Cube' of "
            r"probability_of_lwe_thickness_of_precipitation_amount_above_threshold / \(1\) "
            r"\(realization: 3; latitude: 2; longitude: 2\)> to broadcast to"
        )
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            CubeCombiner(operation="*",broadcast="percentile")(cubelist, "new_cube_name")

    def test_error_broadcast_coord_is_auxcoord(self):
        """Test that plugin throws an error if asked to broadcast to a threshold coord
        that already exists on later cubes"""
        self.multiplier.add_aux_coord(self.threshold_aux)
        cubelist = iris.cube.CubeList([self.cube5.copy(), self.multiplier])
        msg = "Cannot broadcast to coord threshold as it already exists as an AuxCoord"
        with self.assertRaisesRegex(TypeError, msg):
            CubeCombiner(operation="*",broadcast="threshold")(cubelist, "new_cube_name")

    def test_multiply_preserves_bounds(self):
        """Test specific case for precipitation type, where multiplying a
        precipitation accumulation by a point-time probability of snow retains
        the bounds on the original accumulation."""
        validity_time = datetime(2015, 11, 19, 0)
        time_bounds = (datetime(2015, 11, 18, 23), datetime(2015, 11, 19, 0))
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
        result = CubeCombiner(operation="*",)(
            [precip_accum, snow_prob], "lwe_thickness_of_snowfall_amount",
        )
        self.assertArrayAlmostEqual(result.data, np.full((2, 3, 3), 0.3))
        self.assertArrayEqual(result.coord("time"), precip_accum.coord("time"))

    def test_update_cell_methods_probabilistic(self):
        """Test that plugin updates cell methods where required when a new
        diagnostic name is provided for a probabilistic cube."""
        cubelist = iris.cube.CubeList([self.cube5, self.multiplier])

        new_cube_name = "new_cube_name"
        expected = CellMethod("sum", coords="time", comments=f"of {new_cube_name}")

        result = CubeCombiner(operation="*",broadcast="threshold")(cubelist, new_cube_name)
        self.assertEqual(result.cell_methods[0], expected)

    def test_update_cell_methods_non_probabilistic(self):
        """Test that plugin updates cell methods where required when a new
        diagnostic name is provided for a non-probabilistic cube."""
        cube = set_up_variable_cube(
            np.full_like(self.cube5.data[:, 0], 0.001),
            name="lwe_thickness_of_precipitation_amount",
            units="m",
            time=datetime(2015, 11, 19, 1),
            time_bounds=(datetime(2015, 11, 19, 0), datetime(2015, 11, 19, 1)),
            frt=datetime(2015, 11, 18, 22),
        )
        cube.cell_methods = self.cube5.cell_methods
        cubelist = iris.cube.CubeList([cube, self.multiplier])

        new_cube_name = "new_cube_name"
        expected = CellMethod("sum", coords="time", comments=f"of {new_cube_name}")

        result = CubeCombiner(operation="*",)(cubelist, new_cube_name)
        self.assertEqual(result.cell_methods[0], expected)

    def test_unmodified_cell_methods(self):
        """Test that plugin leaves cell methods that are diagnostic name
        agnostic unchanged."""

        cell_methods = list(self.cube5.cell_methods)
        additional_cell_method_1 = CellMethod("sum", coords="longitude")
        additional_cell_method_2 = CellMethod(
            "sum", coords="latitude", comments="Kittens are great"
        )
        cell_methods.extend([additional_cell_method_1, additional_cell_method_2])

        self.cube5.cell_methods = cell_methods
        cubelist = iris.cube.CubeList([self.cube5, self.multiplier])

        new_cube_name = "new_cube_name"
        expected = [
            CellMethod("sum", coords="time", comments=f"of {new_cube_name}"),
            additional_cell_method_1,
            additional_cell_method_2,
        ]

        result = CubeCombiner(operation="*",broadcast="threshold")(cubelist, new_cube_name)
        self.assertArrayEqual(result.cell_methods, expected)

    def test_exception_mismatched_dimensions(self):
        """Test an error is raised if dimension coordinates do not match"""
        self.multiplier.coord("latitude").rename("projection_y_coordinate")
        new_cube_name = "new_cube_name"
        plugin = CubeCombiner(operation="*",)
        msg = "Cannot combine cubes with different dimensions"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process([self.cube5.copy(), self.multiplier],new_cube_name)

    def test_with_Combine(self):
        """Test plugin works through the Combine plugin"""
        cubelist = iris.cube.CubeList([self.cube5.copy(), self.multiplier])
        result = Combine("*", broadcast="threshold", new_name="new_cube_name")(
            cubelist
        )
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "probability_of_new_cube_name_above_threshold")
        self.assertEqual(result.coord(var_name="threshold").name(), "new_cube_name")
        self.assertArrayAlmostEqual(result.data, self.cube5.data)

if __name__ == "__main__":
    unittest.main()
