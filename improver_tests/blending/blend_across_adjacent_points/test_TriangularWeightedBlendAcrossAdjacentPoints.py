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
"""Unit tests for the
   weighted_blend.TriangularWeightedBlendAcrossAdjacentPoints plugin."""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
from iris.tests import IrisTest

from improver.blending.blend_across_adjacent_points import (
    TriangularWeightedBlendAcrossAdjacentPoints,
)
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.warnings_handler import ManageWarnings


def set_up_cubes_for_process_tests():
    """Set up some cubes with data for testing the "process" and
    "find_central_point" functions"""
    central_cube = set_up_variable_cube(
        np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        time=dt(2017, 1, 10, 3, 0),
        frt=dt(2017, 1, 10, 3, 0),
        time_bounds=(dt(2017, 1, 10, 0, 0), dt(2017, 1, 10, 3, 0)),
    )
    another_cube = set_up_variable_cube(
        np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        time=dt(2017, 1, 10, 4, 0),
        frt=dt(2017, 1, 10, 3, 0),
        time_bounds=(dt(2017, 1, 10, 1, 0), dt(2017, 1, 10, 4, 0)),
    )
    cube = iris.cube.CubeList([central_cube, another_cube]).merge_cube()
    return central_cube, cube


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        width = 3.0
        forecast_period = 1
        result = str(
            TriangularWeightedBlendAcrossAdjacentPoints(
                "time", forecast_period, "hours", width
            )
        )
        msg = (
            "<TriangularWeightedBlendAcrossAdjacentPoints:"
            " coord = time, central_point = 1.00, "
            "parameter_units = hours, width = 3.00"
        )
        self.assertEqual(result, msg)


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        width = 3.0
        forecast_period = 1
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "time", forecast_period, "hours", width
        )
        expected_coord = "time"
        expected_width = 3.0
        expected_parameter_units = "hours"
        self.assertEqual(plugin.coord, expected_coord)
        self.assertEqual(plugin.width, expected_width)
        self.assertEqual(plugin.parameter_units, expected_parameter_units)


class Test__find_central_point(IrisTest):
    """Test the _find_central_point."""

    def setUp(self):
        """Set up a test cubes."""
        self.central_cube, self.cube = set_up_cubes_for_process_tests()
        self.forecast_period = self.central_cube.coord("forecast_period").points[0]
        self.width = 1.0

    def test_central_point_available(self):
        """Test that the central point is available within the input cube."""
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "forecast_period", self.forecast_period, "hours", self.width
        )
        central_cube = plugin._find_central_point(self.cube)
        self.assertEqual(
            self.central_cube.coord("forecast_period"),
            central_cube.coord("forecast_period"),
        )
        self.assertEqual(self.central_cube.coord("time"), central_cube.coord("time"))
        self.assertArrayEqual(self.central_cube.data, central_cube.data)

    def test_central_point_not_available(self):
        """Test that the central point is not available within the
           input cube."""
        forecast_period = 2
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "forecast_period", forecast_period, "hours", self.width
        )
        msg = "The central point 2 in units of hours"
        with self.assertRaisesRegex(ValueError, msg):
            plugin._find_central_point(self.cube)


class Test_process(IrisTest):
    """Test the process method."""

    def setUp(self):
        """Set up test cubes."""
        self.central_cube, self.cube = set_up_cubes_for_process_tests()
        self.forecast_period = self.central_cube.coord("forecast_period").points[0]

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_triangle_width_1(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 1. This is equivalent to no blending."""
        width = 1.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "forecast_period", self.forecast_period, "hours", width
        )
        result = plugin(self.cube)
        self.assertEqual(
            self.central_cube.coord("forecast_period"), result.coord("forecast_period")
        )
        self.assertEqual(self.central_cube.coord("time"), result.coord("time"))
        self.assertArrayEqual(self.central_cube.data, result.data)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_triangle_width_2(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 2 and there is some blending."""
        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "forecast_period", self.forecast_period, "hours", width
        )
        result = plugin(self.cube)
        expected_data = np.array([[1.333333, 1.333333], [1.333333, 1.333333]])
        self.assertEqual(
            self.central_cube.coord("forecast_period"), result.coord("forecast_period")
        )
        self.assertEqual(self.central_cube.coord("time"), result.coord("time"))
        self.assertArrayAlmostEqual(expected_data, result.data)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_central_point_not_in_allowed_range(self):
        """Test that an exception is generated when the central cube is not
           within the allowed range."""
        width = 1.0
        forecast_period = 2
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "forecast_period", forecast_period, "hours", width
        )
        msg = "The central point 2 in units of hours"
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.cube)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_alternative_parameter_units(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 7200 seconds. """
        forecast_period = 0
        width = 7200.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "forecast_period", forecast_period, "seconds", width
        )
        result = plugin(self.cube)
        expected_data = np.array([[1.333333, 1.333333], [1.333333, 1.333333]])
        self.assertEqual(
            self.central_cube.coord("forecast_period"), result.coord("forecast_period")
        )
        self.assertEqual(self.central_cube.coord("time"), result.coord("time"))
        self.assertArrayAlmostEqual(expected_data, result.data)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_input_cube_no_change(self):
        """Test that the plugin does not change the original input cube."""

        # Add height axis to standard input cube.
        cube = add_coordinate(self.cube.copy(), [0], "height", coord_units="m")
        original_cube = cube.copy()

        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "forecast_period", self.forecast_period, "hours", width
        )
        _ = plugin(cube)

        # Test that the input cube is unchanged by the function.
        self.assertEqual(cube, original_cube)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_promoted_dimension(self):
        """Test that the plugin retains the single height point from the input
           cube and promotes it to a dim coord."""

        # Creates a cube containing the expected outputs.
        fill_value = 1 + 1 / 3.0
        data = np.full((2, 2), fill_value)

        # Take a slice of the time coordinate.
        expected_cube = self.cube[0].copy(data.astype(np.float32))

        # Add height axis to expected output cube.
        expected_cube = add_coordinate(expected_cube, [0.5], "height", coord_units="m")
        expected_cube = iris.util.new_axis(expected_cube, "height")

        # Add height axis to standard input cube.
        cube = add_coordinate(self.cube.copy(), [0.5], "height", coord_units="m")
        cube = iris.util.new_axis(cube, "height")

        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "forecast_period", self.forecast_period, "hours", width
        )
        result = plugin(cube)

        # Test that the result cube retains height co-ordinates
        # from original cube.
        self.assertEqual(expected_cube.coord("height"), result.coord("height"))
        self.assertArrayEqual(expected_cube.data, result.data)
        self.assertEqual(expected_cube, result)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_extra_dimension(self):
        """Test that the plugin retains the single height point from the input
           cube."""

        # Creates a cube containing the expected outputs.
        fill_value = 1 + 1 / 3.0
        data = np.full((2, 2), fill_value)

        # Take a slice of the time coordinate.
        expected_cube = self.cube[0].copy(data.astype(np.float32))

        # Add height axis to expected output cube.
        expected_cube = add_coordinate(expected_cube, [0.5], "height", coord_units="m")

        # Add height axis to standard input cube.
        cube = add_coordinate(self.cube.copy(), [0.5], "height", coord_units="m")
        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "forecast_period", self.forecast_period, "hours", width
        )
        result = plugin(cube)

        # Test that the result cube retains height co-ordinates
        # from original cube.
        self.assertEqual(expected_cube.coord("height"), result.coord("height"))
        self.assertArrayEqual(expected_cube.data, result.data)
        self.assertEqual(expected_cube, result)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_works_two_thresh(self):
        """Test that the plugin works with a cube that contains multiple
           thresholds."""
        width = 2.0
        thresh_cube = self.cube.copy()
        thresh_cubes = add_coordinate(
            thresh_cube, [0.25, 0.5, 0.75], "precipitation_amount", coord_units="mm"
        )
        thresh_cubes.coord("precipitation_amount").var_name = "threshold"

        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            "forecast_period", self.forecast_period, "hours", width
        )
        result = plugin(thresh_cubes)

        # Test that the result cube retains threshold co-ordinates
        # from original cube.
        self.assertEqual(
            thresh_cubes.coord("precipitation_amount"),
            result.coord("precipitation_amount"),
        )


if __name__ == "__main__":
    unittest.main()
