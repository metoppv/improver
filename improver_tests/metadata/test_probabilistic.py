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
"""Tests for the improver.metadata.probabilistic module"""

import unittest

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.metadata.probabilistic import (
    find_percentile_coordinate,
    find_threshold_coordinate,
    format_cell_methods_for_diagnostic,
    format_cell_methods_for_probability,
    get_diagnostic_cube_name_from_probability_name,
    get_threshold_coord_name_from_probability_name,
    in_vicinity_name_format,
    is_probability,
    probability_is_above_or_below,
)
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)


class Test_probability_is_or_below(unittest.TestCase):
    """Test that the probability_is_above_or_below function correctly
    identifies whether the spp__relative_to_threshold attribute is above
    or below with the respect to the threshold."""

    def setUp(self):
        """Set up data and thresholds for the cubes."""
        self.data = np.ones((3, 3, 3), dtype=np.float32)
        self.threshold_points = np.array([276, 277, 278], dtype=np.float32)

    def test_above(self):
        """ Tests the case where spp__relative_threshold is above"""
        cube = set_up_probability_cube(
            self.data, self.threshold_points, spp__relative_to_threshold="above"
        )
        result = probability_is_above_or_below(cube)
        self.assertEqual(result, "above")

    def test_below(self):
        """ Tests the case where spp__relative_threshold is below"""
        cube = set_up_probability_cube(
            self.data, self.threshold_points, spp__relative_to_threshold="below"
        )
        result = probability_is_above_or_below(cube)
        self.assertEqual(result, "below")

    def test_greater_than(self):
        """ Tests the case where spp__relative_threshold is greater_than"""
        cube = set_up_probability_cube(
            self.data, self.threshold_points, spp__relative_to_threshold="greater_than"
        )
        result = probability_is_above_or_below(cube)
        self.assertEqual(result, "above")

    def test_greater_than_or_equal_to(self):
        """ Tests the case where spp__relative_threshold is
        greater_than_or_equal_to"""
        cube = set_up_probability_cube(
            self.data,
            self.threshold_points,
            spp__relative_to_threshold="greater_than_or_equal_to",
        )
        result = probability_is_above_or_below(cube)
        self.assertEqual(result, "above")

    def test_less_than(self):
        """ Tests the case where spp__relative_threshold is less_than"""
        cube = set_up_probability_cube(
            self.data, self.threshold_points, spp__relative_to_threshold="less_than"
        )
        result = probability_is_above_or_below(cube)
        self.assertEqual(result, "below")

    def test_less_than_or_equal_to(self):
        """ Tests the case where spp__relative_threshold is
        less_than_or_equal_to"""
        cube = set_up_probability_cube(
            self.data,
            self.threshold_points,
            spp__relative_to_threshold="less_than_or_equal_to",
        )
        result = probability_is_above_or_below(cube)
        self.assertEqual(result, "below")

    def test_no_spp__relative_to_threshold(self):
        """Tests it returns None if there is no spp__relative_to_threshold
        attribute."""
        cube = set_up_probability_cube(self.data, self.threshold_points,)
        cube.coord("air_temperature").attributes = {
            "relative_to_threshold": "greater_than"
        }
        result = probability_is_above_or_below(cube)
        self.assertEqual(result, None)

    def test_incorrect_attribute(self):
        """Tests it returns None if the spp__relative_to_threshold
        attribute has an invalid value."""
        cube = set_up_probability_cube(self.data, self.threshold_points,)
        cube.coord("air_temperature").attributes = {
            "spp__relative_to_threshold": "higher"
        }
        result = probability_is_above_or_below(cube)
        self.assertEqual(result, None)


class Test_in_vicinity_name_format(unittest.TestCase):
    """Test that the 'in_vicinity' above/below threshold probability
    cube naming function produces the correctly formatted names."""

    def setUp(self):
        """Set up test cube"""
        data = np.ones((3, 3, 3), dtype=np.float32)
        threshold_points = np.array([276, 277, 278], dtype=np.float32)
        self.cube = set_up_probability_cube(data, threshold_points)
        self.cube.rename("probability_of_X_above_threshold")

    def test_in_vicinity_name_format(self):
        """Test that 'in_vicinity' is added correctly to the name for both
        above and below threshold cases"""
        correct_name_above = "probability_of_X_in_vicinity_above_threshold"
        new_name_above = in_vicinity_name_format(self.cube.name())
        self.cube.rename("probability_of_X_below_threshold")
        correct_name_below = "probability_of_X_in_vicinity_below_threshold"
        new_name_below = in_vicinity_name_format(self.cube.name())
        self.assertEqual(new_name_above, correct_name_above)
        self.assertEqual(new_name_below, correct_name_below)

    def test_between_thresholds(self):
        """Test for "between_thresholds" suffix"""
        self.cube.rename("probability_of_visibility_between_thresholds")
        correct_name = "probability_of_visibility_in_vicinity_between_thresholds"
        new_name = in_vicinity_name_format(self.cube.name())
        self.assertEqual(new_name, correct_name)

    def test_no_above_below_threshold(self):
        """Test the case of name without above/below_threshold is handled
        correctly"""
        self.cube.rename("probability_of_X")
        correct_name_no_threshold = "probability_of_X_in_vicinity"
        new_name_no_threshold = in_vicinity_name_format(self.cube.name())
        self.assertEqual(new_name_no_threshold, correct_name_no_threshold)

    def test_in_vicinity_already_exists(self):
        """Test the case of 'in_vicinity' already existing in the cube name"""
        self.cube.rename("probability_of_X_in_vicinity")
        result = in_vicinity_name_format(self.cube.name())
        self.assertEqual(result, "probability_of_X_in_vicinity")


class Test_get_threshold_coord_name_from_probability_name(unittest.TestCase):
    """Test utility to derive threshold coordinate name from probability cube name"""

    def test_above_threshold(self):
        """Test correct name is returned from a standard (above threshold)
        probability field"""
        result = get_threshold_coord_name_from_probability_name(
            "probability_of_air_temperature_above_threshold"
        )
        self.assertEqual(result, "air_temperature")

    def test_below_threshold(self):
        """Test correct name is returned from a probability below threshold"""
        result = get_threshold_coord_name_from_probability_name(
            "probability_of_air_temperature_below_threshold"
        )
        self.assertEqual(result, "air_temperature")

    def test_between_thresholds(self):
        """Test correct name is returned from a probability between thresholds
        """
        result = get_threshold_coord_name_from_probability_name(
            "probability_of_visibility_in_air_between_thresholds"
        )
        self.assertEqual(result, "visibility_in_air")

    def test_in_vicinity(self):
        """Test correct name is returned from an "in vicinity" probability.
        Name "cloud_height" is used in this test to illustrate why suffix
        cannot be removed with "rstrip"."""
        diagnostic = "cloud_height"
        result = get_threshold_coord_name_from_probability_name(
            f"probability_of_{diagnostic}_in_vicinity_above_threshold"
        )
        self.assertEqual(result, diagnostic)

    def test_error_not_probability(self):
        """Test exception if input is not a probability cube name"""
        with self.assertRaises(ValueError):
            get_threshold_coord_name_from_probability_name("lwe_precipitation_rate")


class Test_get_diagnostic_cube_name_from_probability_name(unittest.TestCase):
    """Test utility to derive diagnostic cube name from probability cube name"""

    def test_basic(self):
        """Test correct name is returned from a point probability field"""
        diagnostic = "air_temperature"
        result = get_diagnostic_cube_name_from_probability_name(
            f"probability_of_{diagnostic}_above_threshold"
        )
        self.assertEqual(result, diagnostic)

    def test_in_vicinity(self):
        """Test the full vicinity name is returned from a vicinity probability
        field"""
        diagnostic = "precipitation_rate"
        result = get_diagnostic_cube_name_from_probability_name(
            f"probability_of_{diagnostic}_in_vicinity_above_threshold"
        )
        self.assertEqual(result, f"{diagnostic}_in_vicinity")

    def test_error_not_probability(self):
        """Test exception if input is not a probability cube name"""
        with self.assertRaises(ValueError):
            get_diagnostic_cube_name_from_probability_name("lwe_precipitation_rate")


class Test_is_probability(IrisTest):
    """Test the is_probability function"""

    def setUp(self):
        """Set up test data"""
        self.data = np.ones((3, 3, 3), dtype=np.float32)
        self.threshold_points = np.array([276, 277, 278], dtype=np.float32)
        self.prob_cube = set_up_probability_cube(self.data, self.threshold_points)

    def test_true(self):
        """Test a probability cube evaluates as true"""
        result = is_probability(self.prob_cube)
        self.assertTrue(result)

    def test_scalar_threshold_coord(self):
        """Test a probability cube with a single threshold evaluates as true"""
        cube = iris.util.squeeze(self.prob_cube[0])
        result = is_probability(cube)
        self.assertTrue(result)

    def test_false(self):
        """Test cube that does not contain thresholded probabilities
        evaluates as false"""
        cube = set_up_variable_cube(
            self.data, name="probability_of_rain_at_surface", units="1"
        )
        result = is_probability(cube)
        self.assertFalse(result)


class Test_find_threshold_coordinate(IrisTest):
    """Test the find_threshold_coordinate function"""

    def setUp(self):
        """Set up test probability cubes with old and new threshold coordinate
        naming conventions"""
        data = np.ones((3, 3, 3), dtype=np.float32)
        self.threshold_points = np.array([276, 277, 278], dtype=np.float32)
        cube = set_up_probability_cube(data, self.threshold_points)

        self.cube_new = cube.copy()
        self.cube_old = cube.copy()
        self.cube_old.coord("air_temperature").rename("threshold")

    def test_basic(self):
        """Test function returns an iris.coords.Coord"""
        threshold_coord = find_threshold_coordinate(self.cube_new)
        self.assertIsInstance(threshold_coord, iris.coords.Coord)

    def test_old_convention(self):
        """Test function recognises threshold coordinate with name "threshold"
        """
        threshold_coord = find_threshold_coordinate(self.cube_old)
        self.assertEqual(threshold_coord.name(), "threshold")
        self.assertArrayAlmostEqual(threshold_coord.points, self.threshold_points)

    def test_new_convention(self):
        """Test function recognises threshold coordinate with standard
        diagnostic name and "threshold" as var_name"""
        threshold_coord = find_threshold_coordinate(self.cube_new)
        self.assertEqual(threshold_coord.name(), "air_temperature")
        self.assertEqual(threshold_coord.var_name, "threshold")
        self.assertArrayAlmostEqual(threshold_coord.points, self.threshold_points)

    def test_fails_if_not_cube(self):
        """Test error if given a non-cube argument"""
        msg = "Expecting data to be an instance of iris.cube.Cube"
        with self.assertRaisesRegex(TypeError, msg):
            find_threshold_coordinate([self.cube_new])

    def test_fails_if_no_threshold_coord(self):
        """Test error if no threshold coordinate is present"""
        self.cube_new.coord("air_temperature").var_name = None
        msg = "No threshold coord found"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            find_threshold_coordinate(self.cube_new)


class Test_find_percentile_coordinate(IrisTest):

    """Test whether the cube has a percentile coordinate."""

    def setUp(self):
        """Create a wind-speed and wind-gust cube with percentile coord."""
        data = np.zeros((2, 3, 3), dtype=np.float32)
        percentiles = np.array([50.0, 90.0], dtype=np.float32)
        self.cube_wg = set_up_percentile_cube(data, percentiles)

    def test_basic(self):
        """Test that the function returns a Coord."""
        perc_coord = find_percentile_coordinate(self.cube_wg)
        self.assertIsInstance(perc_coord, iris.coords.Coord)
        self.assertEqual(perc_coord.name(), "percentile")

    def test_fails_if_data_is_not_cube(self):
        """Test it raises a Type Error if cube is not a cube."""
        msg = "Expecting data to be an instance of iris.cube.Cube "
        with self.assertRaisesRegex(TypeError, msg):
            find_percentile_coordinate(50.0)

    def test_fails_if_no_perc_coord(self):
        """Test it raises an Error if there is no percentile coord."""
        msg = "No percentile coord found on"
        cube = self.cube_wg
        cube.remove_coord("percentile")
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            find_percentile_coordinate(cube)

    def test_fails_if_too_many_perc_coord(self):
        """Test it raises a Value Error if there are too many perc coords."""
        msg = "Too many percentile coords found"
        cube = self.cube_wg
        new_perc_coord = iris.coords.AuxCoord(
            1, long_name="percentile", units="no_unit"
        )
        cube.add_aux_coord(new_perc_coord)
        with self.assertRaisesRegex(ValueError, msg):
            find_percentile_coordinate(cube)


class Test_format_cell_methods_for_probability(unittest.TestCase):
    """Test addition of coordinate information to probability cell methods"""

    def setUp(self):
        """Set up a test input cube"""
        self.cube = set_up_probability_cube(
            np.zeros((3, 3, 3), dtype=np.float32),
            np.array([298, 300, 302], dtype=np.float32),
        )

    def test_one_method(self):
        """Test when the input cube has one cell method"""
        input = iris.coords.CellMethod("max", coords="time", intervals="1 hour")
        self.cube.add_cell_method(input)
        format_cell_methods_for_probability(self.cube, "air_temperature")
        result = self.cube.cell_methods[0]
        self.assertEqual(result.method, input.method)
        self.assertEqual(result.coord_names, input.coord_names)
        self.assertEqual(result.intervals, input.intervals)
        self.assertEqual(result.comments, ("of air_temperature",))

    def test_multiple_methods(self):
        """Test a list of methods returns the expected string"""
        input1 = iris.coords.CellMethod("max", coords="time")
        input2 = iris.coords.CellMethod("min", coords=("latitude", "longitude"))
        for method in [input1, input2]:
            self.cube.add_cell_method(method)
        format_cell_methods_for_probability(self.cube, "air_temperature")
        for method in self.cube.cell_methods:
            self.assertEqual(method.comments, ("of air_temperature",))


class Test_format_cell_methods_for_diagnostic(unittest.TestCase):
    """Test removal of coordinate information from probability cell methods"""

    def setUp(self):
        """Set up a test input cube"""
        self.cube = set_up_probability_cube(
            np.zeros((3, 3, 3), dtype=np.float32),
            np.array([298, 300, 302], dtype=np.float32),
        )
        self.cube.add_cell_method(
            iris.coords.CellMethod("max", coords="time", comments="of air_temperature")
        )

    def test_one_method(self):
        """Test the output list of cell methods is as expected"""
        format_cell_methods_for_diagnostic(self.cube)
        self.assertFalse(self.cube.cell_methods[0].comments)


if __name__ == "__main__":
    unittest.main()
