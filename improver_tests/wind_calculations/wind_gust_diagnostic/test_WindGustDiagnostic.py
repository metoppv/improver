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
"""Unit tests for the windgust_diagnostic.WindGustDiagnostic plugin."""
import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.utilities.warnings_handler import ManageWarnings
from improver.wind_calculations.wind_gust_diagnostic import WindGustDiagnostic


def create_cube_with_percentile_coord(
        data=None,
        standard_name=None,
        perc_values=None,
        perc_name='percentile',
        units=None):
    """Create a cube with percentile coord."""
    if perc_values is None:
        perc_values = [50.0]
    if data is None:
        data = np.zeros((len(perc_values), 2, 2, 2))
        data[:, 0, :, :] = 1.0
        data[:, 1, :, :] = 2.0
    if standard_name is None:
        standard_name = "wind_speed"
    if units is None:
        units = "m s^-1"

    cube = Cube(data, standard_name=standard_name, units=units)
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                                units='degrees'), 2)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                units='degrees'), 3)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(perc_values,
                                long_name=perc_name,
                                units="%"), 0)
    return cube


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = (WindGustDiagnostic(50.0, 95.0))
        self.assertEqual(plugin.percentile_gust, 50.0)
        self.assertEqual(plugin.percentile_windspeed, 95.0)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WindGustDiagnostic(50.0, 95.0))
        msg = ('<WindGustDiagnostic: wind-gust perc=50.0, '
               'wind-speed perc=95.0>')
        self.assertEqual(result, msg)


class Test_add_metadata(IrisTest):

    """Test the add_metadata method."""

    def setUp(self):
        """Create a cube."""
        self.cube_wg = create_cube_with_percentile_coord()

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
        msg = ('<WindGustDiagnostic: wind-gust perc=50.0, '
               'wind-speed perc=80.0>')
        self.assertEqual(result.attributes['wind_gust_diagnostic'], msg)

    def test_diagnostic_typical_txt(self):
        """Test that the attribute is set as expected for typical gusts"""
        plugin = WindGustDiagnostic(50.0, 95.0)
        result = plugin.add_metadata(self.cube_wg)
        msg = 'Typical gusts'
        self.assertEqual(result.attributes['wind_gust_diagnostic'], msg)

    def test_diagnostic_extreme_txt(self):
        """Test that the attribute is set as expected for extreme gusts"""
        plugin = WindGustDiagnostic(95.0, 100.0)
        result = plugin.add_metadata(self.cube_wg)
        msg = 'Extreme gusts'
        self.assertEqual(result.attributes['wind_gust_diagnostic'], msg)


class Test_extract_percentile_data(IrisTest):

    """Test the extract_percentile_data method."""
    def setUp(self):
        """Create a wind-speed and wind-gust cube with percentile coord."""
        data = np.zeros((2, 2, 2, 2))
        self.wg_perc = 50.0
        self.ws_perc = 95.0
        gust = "wind_speed_of_gust"
        self.cube_wg = (
            create_cube_with_percentile_coord(data=data,
                                              perc_values=[self.wg_perc, 90.0],
                                              standard_name=gust))

    def test_basic(self):
        """Test that the function returns a Cube and Coord."""
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        result, perc_coord = (
            plugin.extract_percentile_data(self.cube_wg,
                                           self.wg_perc,
                                           "wind_speed_of_gust"))
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(perc_coord, iris.coords.Coord)

    def test_fails_if_data_is_not_cube(self):
        """Test it raises a Type Error if cube is not a cube."""
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        msg = ('Expecting wind_speed_of_gust data to be an instance of '
               'iris.cube.Cube but is'
               ' {0}.'.format(type(self.wg_perc)))
        with self.assertRaisesRegex(TypeError, msg):
            plugin.extract_percentile_data(self.wg_perc,
                                           self.wg_perc,
                                           "wind_speed_of_gust")

    @ManageWarnings(record=True)
    def test_warning_if_standard_names_do_not_match(self, warning_list=None):
        """Test it raises a warning if standard names do not match."""
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        warning_msg = ('Warning mismatching name for data expecting')
        result, perc_coord = (
            plugin.extract_percentile_data(self.cube_wg,
                                           self.wg_perc,
                                           "wind_speed"))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(perc_coord, iris.coords.Coord)

    def test_fails_if_req_percentile_not_in_cube(self):
        """Test it raises a Value Error if req_perc not in cube."""
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        msg = ('Could not find required percentile')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.extract_percentile_data(self.cube_wg,
                                           20.0,
                                           "wind_speed_of_gust")

    def test_returns_correct_cube_and_coord(self):
        """Test it returns the correct Cube and Coord."""
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        result, perc_coord = (
            plugin.extract_percentile_data(self.cube_wg,
                                           self.wg_perc,
                                           "wind_speed_of_gust"))
        self.assertEqual(perc_coord.name(), "percentile")
        self.assertEqual(result.coord("percentile").points,
                         [self.wg_perc])


class Test_process(IrisTest):

    """Test the creation of wind-gust diagnostic by the plugin."""

    def setUp(self):
        """Create a wind-speed and wind-gust cube with percentile coord."""
        self.ws_perc = 95.0
        data_ws = np.zeros((1, 2, 2, 2))
        data_ws[0, 0, :, :] = 2.5
        data_ws[0, 1, :, :] = 2.0
        self.cube_ws = (
            create_cube_with_percentile_coord(data=data_ws,
                                              perc_values=[self.ws_perc]))
        data_wg = np.zeros((1, 2, 2, 2))
        data_wg[0, 0, :, :] = 3.0
        data_wg[0, 1, :, :] = 1.5
        self.wg_perc = 50.0
        gust = "wind_speed_of_gust"
        self.cube_wg = (
            create_cube_with_percentile_coord(data=data_wg,
                                              perc_values=[self.wg_perc],
                                              standard_name=gust))

    def test_basic(self):
        """Test that the plugin returns a Cube. """
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        result = plugin(self.cube_wg, self.cube_ws)
        self.assertIsInstance(result, Cube)

    def test_raises_error_for_mismatching_perc_coords(self):
        """Test raises an error for mismatching perc coords. """
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        data_wg = np.zeros((1, 2, 2, 2))
        data_wg[0, 0, :, :] = 3.0
        data_wg[0, 1, :, :] = 1.5
        gust = "wind_speed_of_gust"
        cube_wg = (
            create_cube_with_percentile_coord(data=data_wg,
                                              perc_values=[self.wg_perc],
                                              standard_name=gust,
                                              perc_name='percentile_dummy'))
        msg = ('Percentile coord of wind-gust data'
               'does not match coord of wind-speed data')
        with self.assertRaisesRegex(ValueError, msg):
            plugin(cube_wg, self.cube_ws)

    def test_raises_error_for_no_time_coord(self):
        """Test raises Value Error if cubes have no time coordinate """
        cube_wg = self.cube_wg[:, 0, ::]
        cube_ws = self.cube_ws[:, 0, ::]
        cube_wg.remove_coord('time')
        cube_wg = iris.util.squeeze(cube_wg)
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        msg = ('Could not match time coordinate')
        with self.assertRaisesRegex(ValueError, msg):
            plugin(cube_wg, cube_ws)

    def test_raises_error_points_mismatch_and_no_bounds(self):
        """Test raises Value Error if points mismatch and no bounds """
        cube_wg = self.cube_wg
        cube_wg.coord('time').points = [402192.5, 402194.5]
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        msg = ('Could not match time coordinate')
        with self.assertRaisesRegex(ValueError, msg):
            plugin(cube_wg, self.cube_ws)

    def test_raises_error_points_mismatch_and_bounds(self):
        """Test raises Value Error if both points and bounds mismatch """
        cube_wg = self.cube_wg
        cube_wg.coord('time').points = [402192.0, 402193.0]
        cube_wg.coord('time').bounds = [[402191.0, 402192.0],
                                        [402192.0, 402193.0]]
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        msg = ('Could not match time coordinate')
        with self.assertRaisesRegex(ValueError, msg):
            plugin(cube_wg, self.cube_ws)

    def test_no_raises_error_if_ws_point_in_bounds(self):
        """Test raises no Value Error if wind-speed point in bounds """
        cube_wg = self.cube_wg
        cube_wg.coord('time').points = [402192.0, 402193.0]
        cube_wg.coord('time').bounds = [[402191.5, 402192.5],
                                        [402192.5, 402193.5]]

        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        result = plugin(cube_wg, self.cube_ws)
        self.assertIsInstance(result, Cube)

    def test_returns_wind_gust_diagnostic(self):
        """Test that the plugin returns a Cube. """
        plugin = WindGustDiagnostic(self.wg_perc, self.ws_perc)
        result = plugin(self.cube_wg, self.cube_ws)
        expected_data = np.zeros((2, 2, 2))
        expected_data[0, :, :] = 3.0
        expected_data[1, :, :] = 2.0
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(result.attributes['wind_gust_diagnostic'],
                         'Typical gusts')


if __name__ == '__main__':
    unittest.main()
