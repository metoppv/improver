# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the windgust_diagnostic.WindGustDiagnostic plugin."""
import unittest
from datetime import datetime

import iris
import numpy as np
import pytest
from iris.cube import Cube
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_percentile_cube,
)
from improver.wind_calculations.wind_gust_diagnostic import WindGustDiagnostic


def create_wind_percentile_cube(data=None, perc_values=None, name="wind_speed_of_gust"):
    """Create a cube with percentile coordinate and two time slices"""
    if perc_values is None:
        perc_values = [50.0]
    if data is None:
        data = np.zeros((len(perc_values), 2, 2, 2), dtype=np.float32)
        data[:, 0, :, :] = 1.0
        data[:, 1, :, :] = 2.0

    data_times = [datetime(2015, 11, 19, 0, 30), datetime(2015, 11, 19, 1, 30)]
    perc_cube = set_up_percentile_cube(
        data[:, 0, :, :],
        perc_values,
        name=name,
        units="m s-1",
        time=data_times[0],
        frt=datetime(2015, 11, 18, 21),
    )
    cube = add_coordinate(perc_cube, data_times, "time", is_datetime=True)
    cube.data = np.squeeze(data)
    return cube


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = WindGustDiagnostic(50.0, 95.0)
        self.assertEqual(plugin.percentile_gust, 50.0)
        self.assertEqual(plugin.percentile_windspeed, 95.0)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WindGustDiagnostic(50.0, 95.0))
        msg = "<WindGustDiagnostic: wind-gust perc=50.0, wind-speed perc=95.0>"
        self.assertEqual(result, msg)


class Test_add_metadata(IrisTest):

    """Test the add_metadata method."""

    def setUp(self):
        """Create a cube."""
        self.cube_wg = create_wind_percentile_cube()

    def test_basic(self):
        """Test that the function returns a Cube. """
        plugin = WindGustDiagnostic(50.0, 95.0)
        result = plugin.add_metadata(self.cube_wg)
        self.assertIsInstance(result, Cube)

    def test_metadata(self):
        """Test that the metadata is set as expected """
        plugin = WindGustDiagnostic(50.0, 80.0)
        result = plugin.add_metadata(self.cube_wg)
        self.assertEqual(result.standard_name, "wind_speed_of_gust")
        msg = "<WindGustDiagnostic: wind-gust perc=50.0, wind-speed perc=80.0>"
        self.assertEqual(result.attributes["wind_gust_diagnostic"], msg)

    def test_diagnostic_typical_txt(self):
        """Test that the attribute is set as expected for typical gusts"""
        plugin = WindGustDiagnostic(50.0, 95.0)
        result = plugin.add_metadata(self.cube_wg)
        msg = "Typical gusts"
        self.assertEqual(result.attributes["wind_gust_diagnostic"], msg)

    def test_diagnostic_extreme_txt(self):
        """Test that the attribute is set as expected for extreme gusts"""
        plugin = WindGustDiagnostic(95.0, 100.0)
        result = plugin.add_metadata(self.cube_wg)
        msg = "Extreme gusts"
        self.assertEqual(result.attributes["wind_gust_diagnostic"], msg)


class Test_extract_percentile_data(IrisTest):

    """Test the extract_percentile_data method."""

    def setUp(self):
        """Create a wind-speed and wind-gust cube with percentile coord."""
        data = np.zeros((2, 2, 2, 2), dtype=np.float32)
        self.wg_perc = 50.0
        self.ws_perc = 95.0
        self.cube_wg = create_wind_percentile_cube(
            data=data, perc_values=[self.wg_perc, 90.0]
        )

    def test_basic(self):
        """Test that the function returns a Cube and Coord."""
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        result, perc_coord = plugin.extract_percentile_data(
            self.cube_wg, self.wg_perc, "wind_speed_of_gust"
        )
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(perc_coord, iris.coords.Coord)

    def test_fails_if_data_is_not_cube(self):
        """Test it raises a Type Error if cube is not a cube."""
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        msg = (
            "Expecting wind_speed_of_gust data to be an instance of "
            "iris.cube.Cube but is"
            " {0}.".format(type(self.wg_perc))
        )
        with self.assertRaisesRegex(TypeError, msg):
            plugin.extract_percentile_data(
                self.wg_perc, self.wg_perc, "wind_speed_of_gust"
            )

    def test_warning_if_standard_names_do_not_match(self):
        """Test it raises a warning if standard names do not match."""
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        warning_msg = "Warning mismatching name for data expecting"
        with pytest.warns(UserWarning, match=warning_msg):
            result, perc_coord = plugin.extract_percentile_data(
                self.cube_wg, self.wg_perc, "wind_speed"
            )
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(perc_coord, iris.coords.Coord)

    def test_fails_if_req_percentile_not_in_cube(self):
        """Test it raises a Value Error if req_perc not in cube."""
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        msg = "Could not find required percentile"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.extract_percentile_data(self.cube_wg, 20.0, "wind_speed_of_gust")

    def test_returns_correct_cube_and_coord(self):
        """Test it returns the correct Cube and Coord."""
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        result, perc_coord = plugin.extract_percentile_data(
            self.cube_wg, self.wg_perc, "wind_speed_of_gust"
        )
        self.assertEqual(perc_coord.name(), "percentile")
        self.assertEqual(result.coord("percentile").points, [self.wg_perc])


class Test_process(IrisTest):

    """Test the creation of wind-gust diagnostic by the plugin."""

    def setUp(self):
        """Create a wind-speed and wind-gust cube with percentile coord."""
        self.ws_perc = 95.0
        data_ws = np.zeros((1, 2, 2, 2), dtype=np.float32)
        data_ws[0, 0, :, :] = 2.5
        data_ws[0, 1, :, :] = 2.0
        self.cube_ws = create_wind_percentile_cube(
            data=data_ws, perc_values=[self.ws_perc], name="wind_speed"
        )
        data_wg = np.zeros((1, 2, 2, 2), dtype=np.float32)
        data_wg[0, 0, :, :] = 3.0
        data_wg[0, 1, :, :] = 1.5
        self.wg_perc = 50.0
        self.cube_wg = create_wind_percentile_cube(
            data=data_wg, perc_values=[self.wg_perc]
        )

    def test_basic(self):
        """Test that the plugin returns a Cube. """
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        result = plugin(self.cube_wg, self.cube_ws)
        self.assertIsInstance(result, Cube)

    def test_raises_error_for_mismatching_perc_coords(self):
        """Test raises an error for mismatching perc coords. """
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        self.cube_wg.coord("percentile").rename("percentile_dummy")
        msg = (
            "Percentile coord of wind-gust data"
            "does not match coord of wind-speed data"
        )
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.cube_wg, self.cube_ws)

    def test_raises_error_for_no_time_coord(self):
        """Test raises Value Error if cubes have no time coordinate """
        cube_wg = self.cube_wg[:, 0, ::]
        cube_ws = self.cube_ws[:, 0, ::]
        cube_wg.remove_coord("time")
        cube_wg = iris.util.squeeze(cube_wg)
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        msg = "Could not match time coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            plugin(cube_wg, cube_ws)

    def test_raises_error_points_mismatch_and_no_bounds(self):
        """Test raises Value Error if points mismatch and no bounds """
        # offset times by half an hour (in seconds)
        self.cube_wg.coord("time").points = self.cube_wg.coord("time").points + 30 * 60
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        msg = "Could not match time coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.cube_wg, self.cube_ws)

    def test_raises_error_points_mismatch_and_bounds(self):
        """Test raises Value Error if both points and bounds mismatch """
        # offset by 4 hours (in seconds)
        self.cube_wg.coord("time").points = (
            self.cube_wg.coord("time").points + 4 * 60 * 60
        )
        times = self.cube_wg.coord("time").points
        self.cube_wg.coord("time").bounds = [
            [times[0] - 3600, times[0]],
            [times[1] - 3600, times[1]],
        ]
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        msg = "Could not match time coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.cube_wg, self.cube_ws)

    def test_no_error_if_ws_point_in_bounds(self):
        """Test raises no Value Error if wind-speed point in bounds """
        self.cube_wg.coord("time").points = self.cube_wg.coord("time").points + 30 * 60
        times = self.cube_wg.coord("time").points
        self.cube_wg.coord("time").bounds = [
            [times[0] - 3600, times[0]],
            [times[1] - 3600, times[1]],
        ]
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        result = plugin(self.cube_wg, self.cube_ws)
        self.assertIsInstance(result, Cube)

    def test_returns_wind_gust_diagnostic(self):
        """Test that the plugin returns a Cube. """
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        result = plugin(self.cube_wg, self.cube_ws)
        expected_data = np.zeros((2, 2, 2), dtype=np.float32)
        expected_data[0, :, :] = 3.0
        expected_data[1, :, :] = 2.0
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(result.attributes["wind_gust_diagnostic"], "Typical gusts")


if __name__ == "__main__":
    unittest.main()
