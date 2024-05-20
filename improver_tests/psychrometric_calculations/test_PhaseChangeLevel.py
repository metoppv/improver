# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for psychrometric_calculations PhaseChangeLevel."""

import unittest

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.cube import CubeList
from iris.tests import IrisTest

from improver.psychrometric_calculations.psychrometric_calculations import (
    PhaseChangeLevel,
)
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.cube_manipulation import sort_coord_in_cube


class Test__init__(IrisTest):

    """Test the init method."""

    def test_snow_sleet(self):
        """Test that the __init__ method configures the plugin as expected
        for the snow-sleet phase change."""

        phase_change = "snow-sleet"
        plugin = PhaseChangeLevel(phase_change, grid_point_radius=3)

        self.assertEqual(plugin.falling_level_threshold, 90.0)
        self.assertEqual(plugin.phase_change_name, "snow_falling")
        self.assertEqual(plugin.grid_point_radius, 3)

    def test_sleet_rain(self):
        """Test that the __init__ method configures the plugin as expected
        for the sleet_rain phase change."""

        phase_change = "sleet-rain"
        plugin = PhaseChangeLevel(phase_change, grid_point_radius=3)

        self.assertEqual(plugin.falling_level_threshold, 202.5)
        self.assertEqual(plugin.phase_change_name, "rain_falling")
        self.assertEqual(plugin.grid_point_radius, 3)

    def test_hail_rain(self):
        """Test that the __init__ method configures the plugin as expected
        for the hail_rain phase change."""

        phase_change = "hail-rain"
        plugin = PhaseChangeLevel(phase_change, grid_point_radius=3)

        self.assertEqual(plugin.falling_level_threshold, 5000)
        self.assertEqual(plugin.phase_change_name, "rain_from_hail_falling")
        self.assertEqual(plugin.grid_point_radius, 3)

    def test_unknown_phase_change(self):
        """Test that the __init__ method raised an exception for an unknown
        phase change argument."""

        phase_change = "kittens-puppies"
        msg = (
            "Unknown phase change 'kittens-puppies' requested.\n"
            "Available options are: snow-sleet, sleet-rain"
        )

        with self.assertRaisesRegex(ValueError, msg):
            PhaseChangeLevel(phase_change)


class Test_find_falling_level(IrisTest):

    """Test the find_falling_level method."""

    def setUp(self):
        """Set up arrays."""
        pytest.importorskip("stratify")
        self.wb_int_data = np.array(
            [
                [[80.0, 80.0], [70.0, 50.0]],
                [[90.0, 100.0], [80.0, 60.0]],
                [[100.0, 110.0], [90.0, 100.0]],
            ]
        )

        self.orog_data = np.array([[0.0, 0.0], [5.0, 3.0]])
        self.height_points = np.array([5.0, 10.0, 20.0])

    def test_basic(self):
        """Test method returns an array with correct data"""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")
        expected = np.array([[10.0, 7.5], [25.0, 20.5]])
        result = plugin.find_falling_level(
            self.wb_int_data, self.orog_data, self.height_points
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)

    def test_outside_range(self):
        """Test method returns an nan if data outside range"""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")
        wb_int_data = self.wb_int_data
        wb_int_data[2, 1, 1] = 70.0
        result = plugin.find_falling_level(
            wb_int_data, self.orog_data, self.height_points
        )
        self.assertTrue(np.isnan(result[1, 1]))


class Test_fill_in_high_phase_change_falling_levels(IrisTest):

    """Test the fill_in_high_phase_change_falling_levels method."""

    def setUp(self):
        """Set up arrays for testing."""
        self.phase_change_level_data = np.array(
            [[1.0, 1.0, 2.0], [1.0, np.nan, 2.0], [1.0, 2.0, 2.0]]
        )
        self.phase_change_data_no_interp = np.array(
            [[np.nan, np.nan, np.nan], [1.0, np.nan, 2.0], [1.0, 2.0, np.nan]]
        )
        self.orog = np.ones((3, 3))
        self.highest_wb_int = np.ones((3, 3))
        self.highest_height = 300.0

    def test_basic(self):
        """Test fills in missing data with orography + highest height"""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")
        self.highest_wb_int[1, 1] = 100.0
        expected = np.array([[1.0, 1.0, 2.0], [1.0, 301.0, 2.0], [1.0, 2.0, 2.0]])
        plugin.fill_in_high_phase_change_falling_levels(
            self.phase_change_level_data,
            self.orog,
            self.highest_wb_int,
            self.highest_height,
        )
        self.assertArrayEqual(self.phase_change_level_data, expected)

    def test_no_fill_if_conditions_not_met(self):
        """Test it doesn't fill in NaN if the heighest wet bulb integral value
        is less than the threshold."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")
        expected = np.array([[1.0, 1.0, 2.0], [1.0, np.nan, 2.0], [1.0, 2.0, 2.0]])
        plugin.fill_in_high_phase_change_falling_levels(
            self.phase_change_level_data,
            self.orog,
            self.highest_wb_int,
            self.highest_height,
        )
        self.assertArrayEqual(self.phase_change_level_data, expected)


class Test_linear_wet_bulb_fit(IrisTest):

    """Test the linear_wet_bulb_fit method."""

    def setUp(self):
        """
        Set up arrays for testing.

        Set up a wet bulb temperature array with a linear trend near sea
        level. Some of the straight line fits of wet bulb temperature will
        cross the height axis above zero and some below.
        """
        data = np.ones((5, 3, 3)) * -0.8
        self.heights = np.array([5, 10, 20, 30, 50])
        for i in range(5):
            data[i] = data[i] * self.heights[i]
        data[:, :, 0] = data[:, :, 0] - 10
        data[:, :, 2] = data[:, :, 2] + 20
        self.wet_bulb_temperature = data
        self.sea_points = np.array(
            [[True, True, True], [False, False, False], [True, True, True]]
        )
        self.expected_gradients = np.array(
            [[-0.8, -0.8, -0.8], [0.0, 0.0, 0.0], [-0.8, -0.8, -0.8]]
        )
        self.expected_intercepts = np.array(
            [[-10, 0.0, 20.0], [0.0, 0.0, 0.0], [-10, 0.0, 20.0]]
        )

    def test_basic(self):
        """Test we find the correct gradient and intercepts for simple case"""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")

        gradients, intercepts = plugin.linear_wet_bulb_fit(
            self.wet_bulb_temperature, self.heights, self.sea_points
        )
        self.assertArrayAlmostEqual(self.expected_gradients, gradients)
        self.assertArrayAlmostEqual(self.expected_intercepts, intercepts)

    def test_land_points(self):
        """Test it returns arrays of zeros if points are land."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")
        sea_points = np.ones((3, 3)) * False
        gradients, intercepts = plugin.linear_wet_bulb_fit(
            self.wet_bulb_temperature, self.heights, sea_points
        )
        self.assertArrayAlmostEqual(np.zeros((3, 3)), gradients)
        self.assertArrayAlmostEqual(np.zeros((3, 3)), intercepts)


class Test_find_extrapolated_falling_level(IrisTest):

    """Test the find_extrapolated_falling_level method."""

    def setUp(self):
        """
        Set up arrays for testing.
        Set up a wet bulb temperature array with a linear trend near sea
        level. Some of the straight line fits of wet bulb temperature will
        cross the height axis above zero and some below.
        """
        self.phase_change_level = np.ones((3, 3)) * np.nan
        self.max_wb_integral = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]
        )
        self.sea_points = np.array(
            [[True, True, True], [False, False, False], [True, True, True]]
        )
        self.gradients = np.array(
            [[-0.8, -0.8, -0.8], [0.0, 0.0, 0.0], [-0.8, -0.8, -0.8]]
        )
        self.intercepts = np.array(
            [[-10, 0.0, 20.0], [0.0, 0.0, 0.0], [-10, 0.0, 20.0]]
        )
        self.expected_phase_change_level = np.array(
            [
                [-27.5, -15.0, -4.154759],
                [np.nan, np.nan, np.nan],
                [-26.642136, -14.142136, -3.722813],
            ]
        )

    def test_basic(self):
        """Test we fill in the correct snow falling levels for a simple case"""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")

        plugin.find_extrapolated_falling_level(
            self.max_wb_integral,
            self.gradients,
            self.intercepts,
            self.phase_change_level,
            self.sea_points,
        )
        self.assertArrayAlmostEqual(
            self.expected_phase_change_level, self.phase_change_level
        )

    def test_gradients_zero(self):
        """Test we do nothing if all gradients are zero"""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")
        gradients = np.zeros((3, 3))
        plugin.find_extrapolated_falling_level(
            self.max_wb_integral,
            gradients,
            self.intercepts,
            self.phase_change_level,
            self.sea_points,
        )
        expected_phase_change_level = np.ones((3, 3)) * np.nan
        self.assertArrayAlmostEqual(
            expected_phase_change_level, self.phase_change_level
        )


class Test_fill_sea_points(IrisTest):

    """Test the fill_in_sea_points method."""

    def setUp(self):
        """Set up arrays for testing."""
        self.phase_change_level = np.ones((3, 3)) * np.nan
        self.max_wb_integral = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]
        )

        self.land_sea = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        data = np.ones((5, 3, 3)) * -0.8
        self.heights = np.array([5, 10, 20, 30, 50])
        for i in range(5):
            data[i] = data[i] * self.heights[i]
        data[:, :, 0] = data[:, :, 0] - 10
        data[:, :, 2] = data[:, :, 2] + 20
        self.wet_bulb_temperature = data
        self.orography = np.zeros((3, 3))
        self.expected_phase_change_level = np.array(
            [
                [-27.5, -15.0, -4.154759],
                [np.nan, np.nan, np.nan],
                [-26.642136, -14.142136, -3.722813],
            ]
        )

    def test_basic(self):
        """Test it fills in the points it's meant to."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")
        plugin.fill_in_sea_points(
            self.phase_change_level,
            self.land_sea,
            self.max_wb_integral,
            self.wet_bulb_temperature,
            self.heights,
            self.orography,
        )
        self.assertArrayAlmostEqual(
            self.phase_change_level.data, self.expected_phase_change_level
        )

    def test_no_sea(self):
        """Test it only fills in sea points, and ignores a land point"""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")
        expected = np.ones((3, 3)) * np.nan
        land_sea = np.ones((3, 3))
        plugin.fill_in_sea_points(
            self.phase_change_level,
            land_sea,
            self.max_wb_integral,
            self.wet_bulb_temperature,
            self.heights,
            self.orography,
        )
        self.assertArrayAlmostEqual(self.phase_change_level.data, expected)

    def test_all_above_threshold(self):
        """Test it doesn't change points that are all above the threshold"""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")
        self.max_wb_integral[0, 1] = 100
        self.phase_change_level[0, 1] = 100
        self.expected_phase_change_level[0, 1] = 100
        plugin.fill_in_sea_points(
            self.phase_change_level,
            self.land_sea,
            self.max_wb_integral,
            self.wet_bulb_temperature,
            self.heights,
            self.orography,
        )
        self.assertArrayAlmostEqual(
            self.phase_change_level.data, self.expected_phase_change_level
        )

    def test_non_zero_orography(self):
        """Test that points with non-zero orography are updated correctly"""
        plugin = PhaseChangeLevel(phase_change="snow-sleet")
        orography = np.array([[-5, 0, 5], [-5, 0, 5], [-5, 0, 5]])
        self.expected_phase_change_level += orography
        plugin.fill_in_sea_points(
            self.phase_change_level,
            self.land_sea,
            self.max_wb_integral,
            self.wet_bulb_temperature,
            self.heights,
            orography,
        )
        self.assertArrayAlmostEqual(
            self.phase_change_level.data, self.expected_phase_change_level
        )


class Test_find_max_in_nbhood_orography(IrisTest):

    """Test the find_max_in_nbhood_orography method"""

    def setUp(self):
        """Set up a cube with x and y coordinates"""
        data = np.array(
            [
                [0, 10, 20, 5, 0],
                [0, 50, 20, 5, 0],
                [0, 80, 90, 0, 0],
                [0, 20, 5, 10, 0],
                [0, 5, 10, 10, 0],
            ]
        )
        self.cube = set_up_variable_cube(
            data,
            name="orographic_height",
            units="m",
            spatial_grid="equalarea",
            x_grid_spacing=2000.0,
            y_grid_spacing=2000.0,
        )
        self.expected_data = [
            [50, 50, 50, 20, 5],
            [80, 90, 90, 90, 5],
            [80, 90, 90, 90, 10],
            [80, 90, 90, 90, 10],
            [20, 20, 20, 10, 10],
        ]
        self.cube_latlon = set_up_variable_cube(
            self.cube.data,
            name="orographic_height",
            units="m",
            spatial_grid="latlon",
            x_grid_spacing=0.01,
            y_grid_spacing=0.01,
        )

    def test_basic(self):
        """Test the function does what it's meant to in a simple case."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet", grid_point_radius=1)
        result = plugin.find_max_in_nbhood_orography(self.cube)
        self.assertArrayAlmostEqual(result.data, self.expected_data)

    def test_null(self):
        """Test the function does nothing when radius is zero."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet", grid_point_radius=0)
        expected_data = self.cube.data.copy()
        result = plugin.find_max_in_nbhood_orography(self.cube)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_null_lat_lon(self):
        """Test the function succeeds and does nothing when radius is zero and grid is
        lat-lon."""
        cube = self.cube_latlon
        plugin = PhaseChangeLevel(phase_change="snow-sleet", grid_point_radius=0)
        expected_data = self.cube.data.copy()
        result = plugin.find_max_in_nbhood_orography(cube)
        self.assertArrayAlmostEqual(result.data, expected_data)


class Test_horizontally_interpolate_phase(IrisTest):

    """Test the PhaseChangeLevel horizontal interpolation."""

    def setUp(self):
        """Set up input data for our different cases.

        The main aim in this set of unit tests is to test interpolation when
        the phase change level data is largely above the maximum neighbourhood
        orography, but also contains isolated NaNs. If the interpolation code
        depends on interpolating low phase change level data (e.g. from
        valleys) that is below the max orography, edge cases can arise where
        the interpolation fails. This tests those edge cases.

        The first case is one of the simplest possible - a 1d case with a NaN
        in the middle, with two options to choose from either side.

        The second case follows a real life example where the phase change
        level was not determinable at a single point with steep gradients.

        The third case is more deliberately contrived, with a large low
        orography set of NaN points surrounded by a high orography set.

        """

        # A simple 1d case.
        self.phase_change_data_1d = np.array(
            [[1000.0, np.nan, 800.0]], dtype=np.float32
        )
        self.orography_1d = np.array([[850.0, 700.0, 500.0]], dtype=np.float32)
        self.max_nbhood_orog_1d = np.array([[850.0, 850.0, 700.0]], dtype=np.float32)
        self.expected_result_1d = np.array([[1000.0, 700.0, 800.0]], dtype=np.float32)

        # A case that mimics a real side-of-mountain failure.
        self.phase_change_data_2d = np.array(
            [
                [1000.0, 1000.0, 950.0, 800.0, 800.0],
                [1000.0, 1000.0, 950.0, 900.0, 800.0],
                [1000.0, 1000.0, 950.0, np.nan, 800.0],
                [1000.0, 1000.0, 950.0, 900.0, 800.0],
                [1000.0, 1000.0, 950.0, 800.0, 800.0],
            ],
            dtype=np.float32,
        )
        self.orography_2d = np.array(
            [
                [400.0, 500.0, 500.0, 500.0, 400.0],
                [500.0, 700.0, 750.0, 700.0, 500.0],
                [500.0, 700.0, 850.0, 700.0, 500.0],
                [500.0, 700.0, 750.0, 700.0, 500.0],
                [400.0, 500.0, 500.0, 500.0, 400.0],
            ],
            dtype=np.float32,
        )
        self.max_nbhood_orog_2d = np.array(
            [
                [700.0, 700.0, 700.0, 700.0, 700.0],
                [700.0, 850.0, 850.0, 850.0, 700.0],
                [700.0, 850.0, 850.0, 850.0, 700.0],
                [700.0, 850.0, 850.0, 850.0, 700.0],
                [700.0, 700.0, 700.0, 700.0, 700.0],
            ],
            dtype=np.float32,
        )
        self.expected_result_2d = np.array(
            [
                [1000.0, 1000.0, 950.0, 800.0, 800.0],
                [1000.0, 1000.0, 950.0, 900.0, 800.0],
                [1000.0, 1000.0, 950.0, 700.0, 800.0],
                [1000.0, 1000.0, 950.0, 900.0, 800.0],
                [1000.0, 1000.0, 950.0, 800.0, 800.0],
            ],
            dtype=np.float32,
        )

        # A 'NaN crater' with low orography circled by high orography.
        self.phase_change_data_2d_crater = np.full((9, 9), 1000.0, dtype=np.float32)
        self.phase_change_data_2d_crater[2:7, 2:7] = np.nan
        self.orography_2d_crater = np.full((9, 9), 900.0, dtype=np.float32)
        self.orography_2d_crater[2:7, 2:7] = 600.0
        self.max_nbhood_orog_2d_crater = np.full((9, 9), 900.0, dtype=np.float32)
        self.max_nbhood_orog_2d_crater[3:6, 3:6] = 600.0
        self.expected_result_2d_crater = np.full((9, 9), 1000.0, dtype=np.float32)
        self.expected_result_2d_crater[2:7, 2:7] = self.orography_2d_crater[2:7, 2:7]

    def test_interpolate_edge_case_1d(self):
        """Test that we fill in missing areas under a 1d peaked edge case."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet", grid_point_radius=1)
        result = plugin._horizontally_interpolate_phase(
            self.phase_change_data_1d, self.orography_1d, self.max_nbhood_orog_1d
        )
        self.assertArrayAlmostEqual(result, self.expected_result_1d)

    def test_interpolate_edge_case_2d(self):
        """Test that we fill in missing areas under a peaked edge case."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet", grid_point_radius=1)
        result = plugin._horizontally_interpolate_phase(
            self.phase_change_data_2d, self.orography_2d, self.max_nbhood_orog_2d
        )
        self.assertArrayAlmostEqual(result, self.expected_result_2d)

    def test_interpolate_edge_case_2d_grid_point_radius_2(self):
        """Test filling in missing areas under a radius 2 peaked edge case.

        In this case, due to the higher max nbhood orog at the edges, we get
        successful interpolation using the edge points as valid points, with
        the orography at the nan point used as the limit.

        """
        plugin = PhaseChangeLevel(phase_change="snow-sleet", grid_point_radius=2)
        max_nbhood_orog = np.full(
            (5, 5), 850.0
        )  # Determined from the grid point radius increase.
        result = plugin._horizontally_interpolate_phase(
            self.phase_change_data_2d, self.orography_2d, max_nbhood_orog
        )
        expected_result = self.expected_result_2d.copy()
        expected_result[2][3] = self.orography_2d[2][
            3
        ]  # This uses the orography as the limit.
        self.assertArrayAlmostEqual(result, expected_result)

    def test_interpolate_edge_case_2d_nan_peak(self):
        """Test that we fill in missing areas under a nan-peaked edge case."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet", grid_point_radius=1)
        phase_change_data = self.phase_change_data_2d.copy()
        phase_change_data[2][2] = np.nan  # Peak is also nan.
        result = plugin._horizontally_interpolate_phase(
            phase_change_data, self.orography_2d, self.max_nbhood_orog_2d
        )
        expected_result = self.expected_result_2d.copy()
        expected_result[2][2] = self.orography_2d[2][2]
        expected_result[2][3] = self.orography_2d[2][3]
        self.assertArrayAlmostEqual(result, expected_result)

    def test_interpolate_edge_case_2d_nan_peakonly(self):
        """Test that we fill in missing areas under only-nan-peaked edge case."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet", grid_point_radius=1)
        phase_change_data = self.phase_change_data_2d.copy()
        phase_change_data[2][2] = np.nan
        phase_change_data[2][3] = 950.0  # Just the peak is nan.
        result = plugin._horizontally_interpolate_phase(
            phase_change_data, self.orography_2d, self.max_nbhood_orog_2d
        )
        expected_result = self.expected_result_2d.copy()
        expected_result[2][2] = self.orography_2d[2][2]
        expected_result[2][3] = 950.0
        self.assertArrayAlmostEqual(result, expected_result)

    def test_interpolate_edge_case_2d_crater(self):
        """Test that we fill in missing areas under a nan crater edge case."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet", grid_point_radius=1)
        result = plugin._horizontally_interpolate_phase(
            self.phase_change_data_2d_crater,
            self.orography_2d_crater,
            self.max_nbhood_orog_2d_crater,
        )
        self.assertArrayAlmostEqual(result, self.expected_result_2d_crater)

    def test_interpolate_edge_case_2d_crater_grid_point_radius_2(self):
        """Test filling in missing areas under a radius 2 nan crater edge case."""
        plugin = PhaseChangeLevel(phase_change="snow-sleet", grid_point_radius=2)
        max_nbhood_orog = np.full(
            (9, 9), 900.0
        )  # Determined from the grid point radius increase.
        max_nbhood_orog[4, 4] = 600.0
        result = plugin._horizontally_interpolate_phase(
            self.phase_change_data_2d_crater, self.orography_2d_crater, max_nbhood_orog
        )
        self.assertArrayAlmostEqual(result, self.expected_result_2d_crater)


class Test_process(IrisTest):

    """Test the PhaseChangeLevel processing works"""

    def setUp(self):
        """Set up orography and land-sea mask cubes. Also create temperature,
        pressure, and relative humidity cubes that contain multiple height
        levels."""
        pytest.importorskip("stratify")
        self.setup_cubes_for_process()

    def setup_cubes_for_process(self, spatial_grid="equalarea"):
        data = np.ones((5, 5), dtype=np.float32)
        data[2, 2] = 100.0
        self.orog = set_up_variable_cube(
            data, name="surface_altitude", units="m", spatial_grid=spatial_grid
        )
        self.land_sea = set_up_variable_cube(
            np.ones_like(data, dtype=np.int8),
            name="land_binary_mask",
            units=1,
            spatial_grid=spatial_grid,
        )
        # Note the values below are ordered at [5, 195, 200] m.
        wbt_0 = np.full_like(data, fill_value=271.46216)
        wbt_0[2, 2] = 270.20343
        wbt_1 = np.full_like(data, fill_value=274.4207)
        wbt_1[2, 2] = 271.46216
        wbt_2 = np.full_like(data, fill_value=275.0666)
        wbt_2[2, 2] = 274.4207
        wbt_data = np.array(
            [
                np.broadcast_to(wbt_0, (3, 5, 5)),
                np.broadcast_to(wbt_1, (3, 5, 5)),
                np.broadcast_to(wbt_2, (3, 5, 5)),
            ],
            dtype=np.float32,
        )
        # Note the values below are ordered at [5, 195] m.
        wbti_0 = np.full_like(data, fill_value=128.68324)
        wbti_0[2, 2] = 3.1767120
        wbti_0[1:4, 1:4] = 100.0
        wbti_1 = np.full_like(data, fill_value=7.9681854)
        wbti_1[2, 2] = 3.1767120
        wbti_data = np.array(
            [np.broadcast_to(wbti_0, (3, 5, 5)), np.broadcast_to(wbti_1, (3, 5, 5))],
            dtype=np.float32,
        )
        height_points = [5.0, 195.0, 200.0]
        height_attribute = {"positive": "up"}
        wet_bulb_temperature = set_up_variable_cube(
            data, spatial_grid=spatial_grid, name="wet_bulb_temperature"
        )
        wet_bulb_temperature = add_coordinate(
            wet_bulb_temperature, [0, 1, 2], "realization"
        )
        self.wet_bulb_temperature_cube = add_coordinate(
            wet_bulb_temperature,
            height_points,
            "height",
            coord_units="m",
            attributes=height_attribute,
        )
        self.wet_bulb_temperature_cube.data = wbt_data
        # Note that the iris cubelist merge_cube operation sorts the coordinate
        # being merged into ascending order. The cube created below is thus
        # in the incorrect height order, i.e. [5, 195] instead of [195, 5].
        # There is a function in the the PhaseChangeLevel plugin that ensures
        # the height coordinate is in descending order. This is tested here by
        # creating test cubes with both orders.
        height_attribute = {"positive": "down"}
        wet_bulb_integral = set_up_variable_cube(
            data,
            spatial_grid=spatial_grid,
            name="wet_bulb_temperature_integral",
            units="K m",
        )
        wet_bulb_integral = add_coordinate(wet_bulb_integral, [0, 1, 2], "realization")
        self.wet_bulb_integral_cube_inverted = add_coordinate(
            wet_bulb_integral,
            height_points[0:2],
            "height",
            coord_units="m",
            attributes=height_attribute,
        )
        self.wet_bulb_integral_cube_inverted.data = wbti_data
        self.wet_bulb_integral_cube = sort_coord_in_cube(
            self.wet_bulb_integral_cube_inverted, "height", descending=True
        )
        self.expected_snow_sleet = np.full(
            (3, 5, 5), fill_value=66.88566, dtype=np.float32
        )
        self.expected_snow_sleet[:, 1:4, 1:4] = 26.645035
        self.expected_snow_sleet[:, 2, 2] = 124.623375

    def test_snow_sleet_phase_change(self):
        """Test that process returns a cube with the right name, units and
        values. In this instance the phase change is from snow to sleet. The
        returned level has three values, all above orography."""
        result = PhaseChangeLevel(phase_change="snow-sleet").process(
            CubeList(
                [
                    self.wet_bulb_temperature_cube,
                    self.wet_bulb_integral_cube,
                    self.orog,
                    self.land_sea,
                ]
            )
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "altitude_of_snow_falling_level")
        self.assertEqual(result.units, Unit("m"))
        self.assertArrayAlmostEqual(result.data, self.expected_snow_sleet)
        if hasattr(result.data, "mask"):
            self.assertFalse(result.data.mask.any())

    def test_model_id_attr(self):
        """Test that process returns a cube with the right name, units and
        values when the model_id_attr is provided. In this instance the phase change
        is from snow to sleet. The returned level has three values, all above
        orography."""
        self.wet_bulb_temperature_cube.attributes[
            "mosg__model_configuration"
        ] = "uk_ens"
        self.wet_bulb_integral_cube.attributes["mosg__model_configuration"] = "uk_ens"
        result = PhaseChangeLevel(
            phase_change="snow-sleet", model_id_attr="mosg__model_configuration"
        ).process(
            CubeList(
                [
                    self.wet_bulb_temperature_cube,
                    self.wet_bulb_integral_cube,
                    self.orog,
                    self.land_sea,
                ]
            )
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "altitude_of_snow_falling_level")
        self.assertEqual(result.units, Unit("m"))
        self.assertEqual(result.attributes["mosg__model_configuration"], "uk_ens")
        self.assertArrayAlmostEqual(result.data, self.expected_snow_sleet)
        if hasattr(result.data, "mask"):
            self.assertFalse(result.data.mask.any())

    def test_model_id_attr_exception(self):
        """Test that non-matching model_id_attr values result in an exception."""
        self.wet_bulb_temperature_cube.attributes[
            "mosg__model_configuration"
        ] = "uk_ens"
        self.wet_bulb_integral_cube.attributes["mosg__model_configuration"] = "uk_det"
        msg = "Attribute mosg__model_configuration"
        with self.assertRaisesRegex(ValueError, msg):
            PhaseChangeLevel(
                phase_change="snow-sleet", model_id_attr="mosg__model_configuration"
            ).process(
                CubeList(
                    [
                        self.wet_bulb_temperature_cube,
                        self.wet_bulb_integral_cube,
                        self.orog,
                        self.land_sea,
                    ]
                )
            )

    def test_snow_sleet_phase_change_reorder_cubes(self):
        """Same test as test_snow_sleet_phase_change but the cubes are in a
        different order"""
        result = PhaseChangeLevel(phase_change="snow-sleet").process(
            CubeList(
                [
                    self.wet_bulb_integral_cube,
                    self.wet_bulb_temperature_cube,
                    self.orog,
                    self.land_sea,
                ]
            )
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "altitude_of_snow_falling_level")
        self.assertEqual(result.units, Unit("m"))
        self.assertArrayAlmostEqual(result.data, self.expected_snow_sleet)

    def test_sleet_rain_phase_change(self):
        """Test that process returns a cube with the right name, units and
        values. In this instance the phase change is from sleet to rain. Note
        that the wet bulb temperature integral values are doubled such that the
        rain threshold is reached above the surface.
        The result has an odd pattern of 49.178673 around the edge and at the centre
        point with a value of 1 forming a ring around the centre point. This arises
        because the input data are not entirely realistic in this case. The ring
        [1::4, 1::4] has a sleet-rain-phase-level below the orography (1 m) but the
        centre point is an unrealistic point-hill of 100m which is interpolated
        from the outer ring due to the grid_point_radius default value of 2."""
        self.wet_bulb_integral_cube.data *= 2.0
        result = PhaseChangeLevel(phase_change="sleet-rain").process(
            CubeList(
                [
                    self.wet_bulb_temperature_cube,
                    self.wet_bulb_integral_cube,
                    self.orog,
                    self.land_sea,
                ]
            )
        )
        expected = np.full_like(
            self.expected_snow_sleet, fill_value=49.178673, dtype=np.float32
        )
        expected[:, 1:4, 1:4] = 1.0
        expected[:, 2, 2] = 49.178673
        self.assertIsInstance(result, iris.cube.Cube)
        if hasattr(result.data, "mask"):
            self.assertFalse(result.data.mask.any())
        self.assertEqual(result.name(), "altitude_of_rain_falling_level")
        self.assertEqual(result.units, Unit("m"))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_hail_rain_phase_change(self):
        """Test that process returns a cube with the right name, units and
        values. In this instance the phase change is from hail to rain. The
        wet bulb integral is multiplied by 40 so the threshold for the hail
        to melt is reached before the ground"""

        self.wet_bulb_integral_cube.data *= 40.0

        result = PhaseChangeLevel(phase_change="hail-rain").process(
            CubeList(
                [
                    self.wet_bulb_temperature_cube,
                    self.wet_bulb_integral_cube,
                    self.orog,
                    self.land_sea,
                ]
            )
        )

        expected = np.full_like(
            self.expected_snow_sleet, fill_value=11.797252, dtype=np.float32
        )
        expected[:, 1:4, 1:4] = 1.0
        expected[:, 2, 2] = 11.797252
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "altitude_of_rain_from_hail_falling_level")
        self.assertEqual(result.units, Unit("m"))
        self.assertArrayAlmostEqual(result.data, expected)

        if hasattr(result.data, "mask"):
            self.assertFalse(result.data.mask.any())

    def test_inverted_input_cube(self):
        """Test that the phase change level process returns a cube
        containing the expected data when the height coordinate is in
        ascending order rather than the expected descending order."""
        result = PhaseChangeLevel(phase_change="snow-sleet").process(
            CubeList(
                [
                    self.wet_bulb_temperature_cube,
                    self.wet_bulb_integral_cube,
                    self.orog,
                    self.land_sea,
                ]
            )
        )
        self.assertArrayAlmostEqual(result.data, self.expected_snow_sleet)

    def test_interpolation_from_sea_points(self):
        """Test that the phase change level process returns a cube
        containing the expected data. In this case there is a single
        non-sea-level point in the orography. The snow falling level is below
        the surface of the sea, so for the single high point falling level is
        interpolated from the surrounding sea-level points."""
        orog = self.orog
        orog.data = np.zeros_like(orog.data)
        orog.data[2, 2] = 100.0
        land_sea = self.land_sea
        land_sea.data[1, 1] = 1
        result = PhaseChangeLevel(
            phase_change="snow-sleet", grid_point_radius=1
        ).process(
            CubeList(
                [
                    self.wet_bulb_temperature_cube,
                    self.wet_bulb_integral_cube,
                    orog,
                    land_sea,
                ]
            )
        )
        expected = self.expected_snow_sleet - 1
        expected[:, 2, 2] += 1
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_too_many_cubes(self):
        """Tests that an error is raised if there are too many cubes."""
        msg = "Expected 4"
        with self.assertRaisesRegex(ValueError, msg):
            PhaseChangeLevel(phase_change="snow-sleet").process(
                CubeList(
                    [
                        self.wet_bulb_temperature_cube,
                        self.wet_bulb_integral_cube,
                        self.orog,
                        self.land_sea,
                        self.orog,
                    ]
                )
            )

    def test_empty_cube_list(self):
        """Tests that an error is raised if there is an empty list."""
        msg = "Expected 4"
        with self.assertRaisesRegex(ValueError, msg):
            PhaseChangeLevel(phase_change="snow-sleet").process(CubeList([]))


if __name__ == "__main__":
    unittest.main()
