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
"""Unit tests for the AdjustLandSeaPoints class"""

import unittest

import iris
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.grids import ELLIPSOID
from improver.regrid.landsea import AdjustLandSeaPoints
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.spatial import OccurrenceWithinVicinity
from improver.utilities.warnings_handler import ManageWarnings

# The warning messages are internal to the iris.analysis module v2.2.0
IGNORED_MESSAGES = ["Using a non-tuple sequence for multidimensional indexing"]
WARNING_TYPES = [FutureWarning]


class Test__init__(IrisTest):
    """Tests for the initiation of the AdjustLandSeaPoints class."""

    def test_basic(self):
        """Test that instantiating the class results in an object with
        expected variables."""
        expected_members = {
            "nearest_cube": None,
            "input_land": None,
            "output_land": None,
            "output_cube": None,
        }
        result = AdjustLandSeaPoints()
        members = {
            attr: getattr(result, attr)
            for attr in dir(result)
            if not attr.startswith("_")
        }
        non_methods = {key: val for key, val in members.items() if not callable(val)}
        regridder = non_methods.pop("regridder")
        vicinity = members.pop("vicinity")
        self.assertDictEqual(non_methods, expected_members)
        self.assertTrue(isinstance(regridder, iris.analysis.Nearest))
        self.assertTrue(isinstance(vicinity, OccurrenceWithinVicinity))

    def test_extrap_arg(self):
        """Test with extrapolation_mode argument."""
        result = AdjustLandSeaPoints(extrapolation_mode="mask")
        regridder = getattr(result, "regridder")
        self.assertTrue(isinstance(regridder, iris.analysis.Nearest))

    def test_extrap_arg_error(self):
        """Test with invalid extrapolation_mode argument."""
        msg = "Extrapolation mode 'not_valid' not supported"
        with self.assertRaisesRegex(ValueError, msg):
            AdjustLandSeaPoints(extrapolation_mode="not_valid")

    def test_vicinity_arg(self):
        """Test with vicinity_radius argument."""
        result = AdjustLandSeaPoints(vicinity_radius=30000.0)
        vicinity = getattr(result, "vicinity")
        self.assertTrue(isinstance(vicinity, OccurrenceWithinVicinity))
        self.assertEqual(vicinity.distance, 30000.0)

    def test_vicinity_arg_error(self):
        """Test with invalid vicinity_radius argument.
        This is not possible as OccurrenceWithinVicinity does not check the
        input value."""
        pass


class Test_correct_where_input_true(IrisTest):
    """Tests the correct_where_input_true method of the AdjustLandSeaPoints
    class."""

    def setUp(self):
        """Create a class-object containing the necessary cubes.
        All cubes are on the target grid. Here this is defined as a 3x3 grid.
        The grid contains ones everywhere except the centre point (a zero).
        The output_cube has a value of 0.5 at [0, 1].
        The move_sea_point cube has the zero value at [0, 1] instead of [1, 1],
        this allows it to be used in place of input_land to trigger the
        expected behaviour in the function.
        """
        self.plugin = AdjustLandSeaPoints(vicinity_radius=2200.0)
        data = np.ones((3, 3), dtype=np.float32)
        data[1, 1] = 0.0
        cube = set_up_variable_cube(
            data, "precipitation_amount", "kg m^-2", "equalarea",
        )
        self.plugin.input_land = cube.copy()
        self.plugin.output_land = cube.copy()
        self.plugin.nearest_cube = cube.copy()
        self.plugin.nearest_cube.data[0, 1] = 0.5
        self.plugin.output_cube = self.plugin.nearest_cube.copy()
        data_move_sea = np.ones((3, 3), dtype=np.float32)
        data_move_sea[0, 1] = 0.0
        self.move_sea_point = cube.copy(data=data_move_sea)

    def test_basic_sea(self):
        """Test that nothing changes with argument zero (sea)."""
        input_land = self.plugin.input_land.copy()
        output_land = self.plugin.output_land.copy()
        output_cube = self.plugin.output_cube.copy()
        self.plugin.correct_where_input_true(0)
        self.assertArrayEqual(input_land.data, self.plugin.input_land.data)
        self.assertArrayEqual(output_land.data, self.plugin.output_land.data)
        self.assertArrayEqual(output_cube.data, self.plugin.output_cube.data)

    def test_basic_land(self):
        """Test that nothing changes with argument one (land)."""
        input_land = self.plugin.input_land.copy()
        output_land = self.plugin.output_land.copy()
        output_cube = self.plugin.output_cube.copy()
        self.plugin.correct_where_input_true(1)
        self.assertArrayEqual(input_land.data, self.plugin.input_land.data)
        self.assertArrayEqual(output_land.data, self.plugin.output_land.data)
        self.assertArrayEqual(output_cube.data, self.plugin.output_cube.data)

    def test_work_sea(self):
        """Test for expected change with argument zero (sea)."""
        self.plugin.input_land = self.move_sea_point
        output_cube = self.plugin.output_cube.copy()
        # The output sea point should have been changed to the value from the
        # input sea point in the same grid.
        output_cube.data[1, 1] = 0.5
        self.plugin.correct_where_input_true(0)
        self.assertArrayEqual(output_cube.data, self.plugin.output_cube.data)

    def test_work_land(self):
        """Test for expected change with argument one (land)."""
        self.plugin.input_land = self.move_sea_point
        output_cube = self.plugin.output_cube.copy()
        # The input sea point should have been changed to the value from an
        # input land point in the same grid.
        output_cube.data[0, 1] = 1.0
        self.plugin.correct_where_input_true(1)
        self.assertArrayEqual(output_cube.data, self.plugin.output_cube.data)

    def test_work_sealand_eq_landsea(self):
        """Test result is independent of order of sea/land handling."""
        self.plugin.input_land = self.move_sea_point.copy()
        reset_cube = self.plugin.output_cube.copy()
        self.plugin.correct_where_input_true(0)
        self.plugin.correct_where_input_true(1)
        attempt_01 = self.plugin.output_cube.data.copy()

        self.plugin.output_cube = reset_cube
        self.plugin.correct_where_input_true(1)
        self.plugin.correct_where_input_true(0)
        attempt_10 = self.plugin.output_cube.data.copy()

        self.assertArrayEqual(attempt_01, attempt_10)

    def test_not_in_vicinity(self):
        """Test for no change if the matching point is too far away."""
        # We need larger arrays for this.
        # Define 5 x 5 arrays with output sea point at [1, 1] and input sea
        # point at [4, 4]. The alternative value of 0.5 at [4, 4] should not
        # be selected with a small vicinity_radius.
        self.plugin = AdjustLandSeaPoints(vicinity_radius=2200.0)
        data = np.ones((5, 5), dtype=np.float32)
        data[1, 1] = 0.0
        cube = set_up_variable_cube(
            data, "precipitation_amount", "kg m^-2", "equalarea",
        )
        self.plugin.output_land = cube.copy()
        self.plugin.nearest_cube = cube.copy()
        self.plugin.nearest_cube.data[4, 4] = 0.5
        self.plugin.output_cube = self.plugin.nearest_cube.copy()
        land_data = np.ones((5, 5), dtype=np.float32)
        land_data[4, 4] = 0.0
        self.plugin.input_land = cube.copy(data=land_data)

        output_cube = self.plugin.output_cube.copy()
        self.plugin.correct_where_input_true(0)
        self.assertArrayEqual(output_cube.data, self.plugin.output_cube.data)

    def test_no_matching_points(self):
        """Test code runs and makes no changes if no sea points are present."""
        self.plugin.input_land.data = np.ones_like(self.plugin.input_land.data)
        self.plugin.output_land.data = np.ones_like(self.plugin.output_land.data)
        output_cube = self.plugin.output_cube.copy()
        self.plugin.correct_where_input_true(0)
        self.assertArrayEqual(output_cube.data, self.plugin.output_cube.data)

    def test_all_matching_points(self):
        """Test code runs and makes no changes if all land points are
        present."""
        self.plugin.input_land.data = np.ones_like(self.plugin.input_land.data)
        self.plugin.output_land.data = np.ones_like(self.plugin.output_land.data)
        output_cube = self.plugin.output_cube.copy()
        self.plugin.correct_where_input_true(1)
        self.assertArrayEqual(output_cube.data, self.plugin.output_cube.data)


class Test_process(IrisTest):
    """Tests the process method of the AdjustLandSeaPoints class."""

    def setUp(self):
        """Create a class-object containing the necessary cubes.
        All cubes are on the target grid. Here this is defined as a 5x5 grid.
        The cubes have values of one everywhere except:
        input_land: zeroes (sea points) at [0, 1], [4, 4]
        output_land: zeroes (sea points) at [0, 0], [1, 1]
        input_cube: 0. at [1, 1]; 0.5 at [0, 1]; 0.1 at [4, 4]
        These should trigger all the behavior we expect.
        """
        data_input_land = np.ones((5, 5), dtype=np.float32)
        data_input_land[0, 1] = 0.0
        data_input_land[4, 4] = 0.0

        data_output_land = np.ones((5, 5), dtype=np.float32)
        data_output_land[0, 0] = 0.0
        data_output_land[1, 1] = 0.0

        data_cube = np.ones((5, 5), dtype=np.float32)
        data_cube[1, 1] = 0.0
        data_cube[0, 1] = 0.5
        data_cube[4, 4] = 0.1

        self.plugin = AdjustLandSeaPoints(vicinity_radius=2200.0)

        self.cube = set_up_variable_cube(
            data_cube,
            "precipitation_amount",
            "kg m^-2",
            "equalarea",
            grid_spacing=2000,
            domain_corner=(0, -50000),
        )

        self.output_land = self.cube.copy(data=data_output_land)
        self.input_land = self.cube.copy(data=data_input_land)

        # Lat-lon coords for reprojection
        # These coords result in a 1:1 regridding with the above cubes.
        x_coord = DimCoord(
            np.linspace(-3.281, -3.153, 5),
            standard_name="longitude",
            units="degrees",
            coord_system=ELLIPSOID,
        )
        y_coord = DimCoord(
            np.linspace(54.896, 54.971, 5),
            standard_name="latitude",
            units="degrees",
            coord_system=ELLIPSOID,
        )
        self.input_land_ll = Cube(
            self.input_land.data,
            long_name="land_sea_mask",
            units="1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
        )

    @ManageWarnings(
        ignored_messages=["Using a non-tuple sequence for "],
        warning_types=[FutureWarning],
    )
    def test_basic(self):
        """Test that the expected changes occur and meta-data are unchanged."""
        expected = self.cube.data.copy()
        # Output sea-point populated with data from input sea-point:
        expected[0, 0] = 0.5
        # Output sea-point populated with data from input sea-point:
        expected[1, 1] = 0.5
        # Output land-point populated with data from input land-point:
        expected[0, 1] = 1.0
        # Output land-point populated with data from input sea-point due to
        # vicinity-constraint:
        expected[4, 4] = 1.0
        result = self.plugin.process(self.cube, self.input_land, self.output_land)
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.data, expected)
        self.assertDictEqual(result.attributes, self.cube.attributes)
        self.assertEqual(result.name(), self.cube.name())

    @ManageWarnings(ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_with_regridding(self):
        """Test when input grid is on a different projection."""
        input_land = self.input_land_ll
        expected = self.cube.data.copy()
        # Output sea-point populated with data from input sea-point:
        expected[0, 0] = 0.5
        # Output sea-point populated with data from input sea-point:
        expected[1, 1] = 0.5
        # Output land-point populated with data from input land-point:
        expected[0, 1] = 1.0
        # Output land-point populated with data from input sea-point due to
        # vicinity-constraint:
        expected[4, 4] = 1.0
        result = self.plugin.process(self.cube, input_land, self.output_land)
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.data, expected)
        self.assertDictEqual(result.attributes, self.cube.attributes)
        self.assertEqual(result.name(), self.cube.name())

    @ManageWarnings(ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_multi_realization(self):
        """Test that the expected changes occur and meta-data are unchanged
        when handling a multi-realization cube."""
        cube = add_coordinate(self.cube, [0, 1], "realization", coord_units=1)
        expected = cube.data.copy()

        # Output sea-point populated with data from input sea-point:
        expected[:, 0, 0] = 0.5
        # Output sea-point populated with data from input sea-point:
        expected[:, 1, 1] = 0.5
        # Output land-point populated with data from input land-point:
        expected[:, 0, 1] = 1.0
        # Output land-point populated with data from input sea-point due to
        # vicinity-constraint:
        expected[:, 4, 4] = 1.0
        result = self.plugin.process(cube, self.input_land, self.output_land)
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.data, expected)
        self.assertDictEqual(result.attributes, self.cube.attributes)
        self.assertEqual(result.name(), self.cube.name())

    def test_raises_gridding_error(self):
        """Test error raised when cube and output grids don't match."""
        self.cube = self.input_land_ll
        msg = "X and Y coordinates do not match for cubes"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(self.cube, self.input_land, self.output_land)


if __name__ == "__main__":
    unittest.main()
