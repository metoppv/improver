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
"""Unit tests for weather code utilities."""

import datetime
import os
import pathlib
import unittest
from tempfile import mkdtemp

import iris
import numpy as np
import pytest
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.categorical.utilities import (
    WX_DICT,
    check_tree,
    expand_nested_lists,
    get_parameter_names,
    interrogate_decision_tree,
    update_daynight,
    update_tree_thresholds,
    weather_code_attributes,
)
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver_tests.categorical.decision_tree import set_up_wxcube, wxcode_decision_tree


class Test_wx_dict(IrisTest):
    """ Test WX_DICT set correctly """

    def test_wxcode_values(self):
        """Check wxcode values are set correctly."""
        self.assertEqual(WX_DICT[0], "Clear_Night")
        self.assertEqual(WX_DICT[1], "Sunny_Day")
        self.assertEqual(WX_DICT[2], "Partly_Cloudy_Night")
        self.assertEqual(WX_DICT[3], "Partly_Cloudy_Day")
        self.assertEqual(WX_DICT[4], "Dust")
        self.assertEqual(WX_DICT[5], "Mist")
        self.assertEqual(WX_DICT[6], "Fog")
        self.assertEqual(WX_DICT[7], "Cloudy")
        self.assertEqual(WX_DICT[8], "Overcast")
        self.assertEqual(WX_DICT[9], "Light_Shower_Night")
        self.assertEqual(WX_DICT[10], "Light_Shower_Day")
        self.assertEqual(WX_DICT[11], "Drizzle")
        self.assertEqual(WX_DICT[12], "Light_Rain")
        self.assertEqual(WX_DICT[13], "Heavy_Shower_Night")
        self.assertEqual(WX_DICT[14], "Heavy_Shower_Day")
        self.assertEqual(WX_DICT[15], "Heavy_Rain")
        self.assertEqual(WX_DICT[16], "Sleet_Shower_Night")
        self.assertEqual(WX_DICT[17], "Sleet_Shower_Day")
        self.assertEqual(WX_DICT[18], "Sleet")
        self.assertEqual(WX_DICT[19], "Hail_Shower_Night")
        self.assertEqual(WX_DICT[20], "Hail_Shower_Day")
        self.assertEqual(WX_DICT[21], "Hail")
        self.assertEqual(WX_DICT[22], "Light_Snow_Shower_Night")
        self.assertEqual(WX_DICT[23], "Light_Snow_Shower_Day")
        self.assertEqual(WX_DICT[24], "Light_Snow")
        self.assertEqual(WX_DICT[25], "Heavy_Snow_Shower_Night")
        self.assertEqual(WX_DICT[26], "Heavy_Snow_Shower_Day")
        self.assertEqual(WX_DICT[27], "Heavy_Snow")
        self.assertEqual(WX_DICT[28], "Thunder_Shower_Night")
        self.assertEqual(WX_DICT[29], "Thunder_Shower_Day")
        self.assertEqual(WX_DICT[30], "Thunder")


@pytest.mark.parametrize(
    "accumulation, target_period, expected_value, expected_unit",
    (
        (False, None, 1, "mm/hr"),
        (False, 10800, 1, "mm/hr"),
        (True, 3600, 1, "mm"),
        (True, 10800, 3, "mm"),
    ),
)
def test_update_tree_thresholds(
    accumulation, target_period, expected_value, expected_unit
):
    """Test that updating tree thresholds returns iris AuxCoords with the
    expected value and units. Includes a test that the threshold value is scaled
    if it is defined with an associated period that differs from a user supplied
    target period."""

    tree = wxcode_decision_tree(accumulation=accumulation)
    tree = update_tree_thresholds(tree, target_period=target_period)
    (result,) = tree["heavy_precipitation"]["diagnostic_thresholds"]

    assert isinstance(result, iris.coords.AuxCoord)
    assert result.points[0] == expected_value
    assert result.units == expected_unit


def test_update_tree_thresholds_exception():
    """Test that updating tree thresholds raises an error if the input thresholds
    are defined with an associated period, but no target_period is provided."""

    tree = wxcode_decision_tree(accumulation=True)
    expected = "The decision tree contains thresholds defined"
    with pytest.raises(ValueError, match=expected):
        update_tree_thresholds(tree, target_period=None)


class Test_weather_code_attributes(IrisTest):
    """ Test weather_code_attributes is working correctly """

    def setUp(self):
        """Set up cube """
        data = np.array(
            [0, 1, 5, 11, 20, 5, 9, 10, 4, 2, 0, 1, 29, 30, 1, 5, 6, 6], dtype=np.int32
        ).reshape((2, 3, 3))
        cube = set_up_variable_cube(data, "weather_code", "1",)
        date_times = [
            datetime.datetime(2017, 11, 19, 0, 30),
            datetime.datetime(2017, 11, 19, 1, 30),
        ]
        self.cube = add_coordinate(
            cube, date_times, "time", is_datetime=True, order=[1, 0, 2, 3],
        )
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())
        self.data_directory = mkdtemp()
        self.nc_file = self.data_directory + "/wxcode.nc"
        pathlib.Path(self.nc_file).touch(exist_ok=True)

    def tearDown(self):
        """Remove temporary directories created for testing."""
        os.remove(self.nc_file)
        os.rmdir(self.data_directory)

    def test_values(self):
        """Test attribute values are correctly set."""
        result = weather_code_attributes()
        self.assertArrayEqual(result["weather_code"], self.wxcode)
        self.assertEqual(result["weather_code_meaning"], self.wxmeaning)

    def test_metadata_saves(self):
        """Test that the metadata saves as NetCDF correctly."""
        self.cube.attributes.update(weather_code_attributes())
        save_netcdf(self.cube, self.nc_file)
        result = load_cube(self.nc_file)
        self.assertArrayEqual(result.attributes["weather_code"], self.wxcode)
        self.assertEqual(result.attributes["weather_code_meaning"], self.wxmeaning)


class Test_expand_nested_lists(IrisTest):
    """ Test expand_nested_lists is working correctly """

    def setUp(self):
        """ Set up dictionary for testing """
        self.dictionary = {
            "list": ["a", "a"],
            "list_of_lists": [["a", "a"], ["a", "a"]],
        }

    def test_basic(self):
        """Test that the expand_nested_lists returns a list."""
        result = expand_nested_lists(self.dictionary, "list")
        self.assertIsInstance(result, list)

    def test_simple_list(self):
        """Testexpand_nested_lists returns a expanded list if given a list."""
        result = expand_nested_lists(self.dictionary, "list")
        for val in result:
            self.assertEqual(val, "a")

    def test_list_of_lists(self):
        """Returns a expanded list if given a list of lists."""
        result = expand_nested_lists(self.dictionary, "list_of_lists")
        for val in result:
            self.assertEqual(val, "a")


class Test_update_daynight(IrisTest):
    """Test updating weather cube depending on whether it is day or night"""

    def setUp(self):
        """Set up for update_daynight class"""
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())
        self.cube_data = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
                [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
                [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
                [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                [23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23],
                [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26],
                [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
                [29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29],
            ]
        )

    def test_basic(self):
        """Test that the function returns a weather code cube."""
        cube = set_up_wxcube()
        result = update_daynight(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "weather_code")
        self.assertEqual(result.units, "1")
        self.assertArrayEqual(result.attributes["weather_code"], self.wxcode)
        self.assertEqual(result.attributes["weather_code_meaning"], self.wxmeaning)

    def test_raise_error_no_time_coordinate(self):
        """Test that the function raises an error if no time coordinate."""
        cube = set_up_wxcube()
        cube.coord("time").rename("nottime")
        msg = "cube must have time coordinate"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            update_daynight(cube)

    def test_wxcode_updated(self):
        """Test Correct wxcodes returned for cube."""
        cube = set_up_wxcube()
        cube.data = self.cube_data
        # Only 1,3,10, 14, 17, 20, 23, 26 and 29 change from day to night
        expected_result = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                [9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10],
                [13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14],
                [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                [16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17],
                [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
                [19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20],
                [22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23],
                [25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26],
                [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
                [28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29],
            ]
        )
        result = update_daynight(cube)
        self.assertArrayEqual(result.data, expected_result)
        self.assertEqual(result.shape, (16, 16))

    def test_wxcode_time_different_seconds(self):
        """ Test code works if time coordinate has a difference in the number
        of seconds, which should round to the same time in hours and minutes.
        This was raised by changes to cftime which altered its precision."""
        cube = set_up_wxcube(time=datetime.datetime(2018, 9, 12, 5, 42, 59))
        cube.data = self.cube_data
        expected_result = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                [9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10],
                [13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14],
                [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                [16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17],
                [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
                [19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20],
                [22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23],
                [25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26],
                [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
                [28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29],
            ]
        )
        result = update_daynight(cube)

        self.assertArrayEqual(result.data, expected_result)
        self.assertEqual(result.data.shape, (16, 16))

    def test_wxcode_time_as_array(self):
        """ Test code works if time is an array of dimension > 1 """
        time_points = [
            datetime.datetime(2018, 9, 12, 5),
            datetime.datetime(2018, 9, 12, 6),
            datetime.datetime(2018, 9, 12, 7),
        ]
        cubes = iris.cube.CubeList()
        for time in time_points:
            cubes.append(set_up_wxcube(time=time))
        cube = cubes.merge_cube()

        expected_result = np.ones((3, 16, 16))
        expected_result[0, :, :] = 0
        result = update_daynight(cube)
        self.assertArrayEqual(result.data, expected_result)

    def test_basic_lat_lon(self):
        """Test that the function returns a weather code lat lon cube.."""
        cube = set_up_wxcube(lat_lon=True)
        result = update_daynight(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "weather_code")
        self.assertEqual(result.units, "1")
        self.assertArrayEqual(result.attributes["weather_code"], self.wxcode)
        self.assertEqual(result.attributes["weather_code_meaning"], self.wxmeaning)

    def test_wxcode_updated_on_latlon(self):
        """Test Correct wxcodes returned for lat lon cube."""
        cube = set_up_wxcube(lat_lon=True)
        cube.data = self.cube_data

        expected_result = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                [9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                [13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
                [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                [16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
                [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
                [19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                [22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23],
                [25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26],
                [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
                [28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29],
            ]
        )
        result = update_daynight(cube)
        self.assertArrayEqual(result.data, expected_result)


def test_interrogate_decision_tree():
    """Test that the function returns the right strings."""
    expected = (
        "\u26C5 probability_of_low_and_medium_type_cloud_area_fraction_above_threshold (1): 0.1875, 0.8125\n"  # noqa: E501
        "\u26C5 probability_of_low_type_cloud_area_fraction_above_threshold (1): 0.85\n"
        "\u26C5 probability_of_lwe_graupel_and_hail_fall_rate_in_vicinity_above_threshold (mm hr-1): 0.0\n"  # noqa: E501
        "\u26C5 probability_of_lwe_precipitation_rate_above_threshold (mm hr-1): 0.03, 1.0\n"  # noqa: E501
        "\u26C5 probability_of_lwe_precipitation_rate_in_vicinity_above_threshold (mm hr-1): 0.1, 1.0\n"  # noqa: E501
        "\u26C5 probability_of_lwe_precipitation_rate_max_above_threshold (mm hr-1): 1.0\n"  # noqa: E501
        "\u26C5 probability_of_lwe_sleetfall_rate_above_threshold (mm hr-1): 0.03, 1.0\n"
        "\u26C5 probability_of_lwe_snowfall_rate_above_threshold (mm hr-1): 0.03, 1.0\n"
        "\u26C5 probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_threshold (m-2): 0.0\n"  # noqa: E501
        "\u26C5 probability_of_rainfall_rate_above_threshold (mm hr-1): 0.03, 1.0\n"
        "\u26C5 probability_of_shower_condition_above_threshold (1): 1.0\n"
        "\u26C5 probability_of_visibility_in_air_below_threshold (m): 1000.0, 5000.0\n"
    )
    tree = update_tree_thresholds(wxcode_decision_tree(), None)
    result = interrogate_decision_tree(tree)
    assert result == expected


def test_interrogate_decision_tree_accumulation_1h():
    """Test that the function returns the right strings."""
    expected = (
        "\u26C5 probability_of_low_and_medium_type_cloud_area_fraction_above_threshold (1): 0.1875, 0.8125\n"  # noqa: E501
        "\u26C5 probability_of_low_type_cloud_area_fraction_above_threshold (1): 0.85\n"
        "\u26C5 probability_of_lwe_graupel_and_hail_fall_rate_in_vicinity_above_threshold (mm hr-1): 0.0\n"  # noqa: E501
        "\u26C5 probability_of_lwe_precipitation_rate_above_threshold (mm hr-1): 0.03\n"  # noqa: E501
        "\u26C5 probability_of_lwe_precipitation_rate_in_vicinity_above_threshold (mm hr-1): 0.1, 1.0\n"  # noqa: E501
        "\u26C5 probability_of_lwe_precipitation_rate_max_above_threshold (mm hr-1): 1.0\n"  # noqa: E501
        "\u26C5 probability_of_lwe_sleetfall_rate_above_threshold (mm hr-1): 0.03, 1.0\n"
        "\u26C5 probability_of_lwe_snowfall_rate_above_threshold (mm hr-1): 0.03, 1.0\n"
        "\u26C5 probability_of_lwe_thickness_of_precipitation_amount_above_threshold (mm): 1.0\n"  # noqa: E501
        "\u26C5 probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_threshold (m-2): 0.0\n"  # noqa: E501
        "\u26C5 probability_of_rainfall_rate_above_threshold (mm hr-1): 0.03, 1.0\n"
        "\u26C5 probability_of_shower_condition_above_threshold (1): 1.0\n"
        "\u26C5 probability_of_visibility_in_air_below_threshold (m): 1000.0, 5000.0\n"
    )
    tree = update_tree_thresholds(wxcode_decision_tree(accumulation=True), 3600)
    result = interrogate_decision_tree(tree)
    assert result == expected


def test_interrogate_decision_tree_accumulation_3h():
    """Test that the function returns the right strings."""
    expected = (
        "\u26C5 probability_of_low_and_medium_type_cloud_area_fraction_above_threshold (1): 0.1875, 0.8125\n"  # noqa: E501
        "\u26C5 probability_of_low_type_cloud_area_fraction_above_threshold (1): 0.85\n"
        "\u26C5 probability_of_lwe_graupel_and_hail_fall_rate_in_vicinity_above_threshold (mm hr-1): 0.0\n"  # noqa: E501
        "\u26C5 probability_of_lwe_precipitation_rate_above_threshold (mm hr-1): 0.03\n"  # noqa: E501
        "\u26C5 probability_of_lwe_precipitation_rate_in_vicinity_above_threshold (mm hr-1): 0.1, 1.0\n"  # noqa: E501
        "\u26C5 probability_of_lwe_precipitation_rate_max_above_threshold (mm hr-1): 1.0\n"  # noqa: E501
        "\u26C5 probability_of_lwe_sleetfall_rate_above_threshold (mm hr-1): 0.03, 1.0\n"
        "\u26C5 probability_of_lwe_snowfall_rate_above_threshold (mm hr-1): 0.03, 1.0\n"
        "\u26C5 probability_of_lwe_thickness_of_precipitation_amount_above_threshold (mm): 3.0\n"  # noqa: E501
        "\u26C5 probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_threshold (m-2): 0.0\n"  # noqa: E501
        "\u26C5 probability_of_rainfall_rate_above_threshold (mm hr-1): 0.03, 1.0\n"
        "\u26C5 probability_of_shower_condition_above_threshold (1): 1.0\n"
        "\u26C5 probability_of_visibility_in_air_below_threshold (m): 1000.0, 5000.0\n"
    )
    tree = update_tree_thresholds(wxcode_decision_tree(accumulation=True), 10800)
    result = interrogate_decision_tree(tree)
    assert result == expected


class Test_get_parameter_names(IrisTest):
    """Test the get_parameter_names method."""

    def test_basic(self):
        """Test that the get_parameter_names method does what it says."""
        condition = ["parameter_name_one", "*", "4.0", "+", "parameter_name_two"]
        expected = ["parameter_name_one", "parameter_name_two"]
        result = get_parameter_names(condition)
        self.assertEqual(result, expected)

    def test_nested(self):
        """Test getting parameter names from nested lists."""
        condition = [
            ["parameter_name_one", "*", "4.0", "+", "parameter_name_two"],
            ["parameter_name_three", "parameter_name_four"],
        ]
        expected = [
            ["parameter_name_one", "parameter_name_two"],
            ["parameter_name_three", "parameter_name_four"],
        ]
        result = get_parameter_names(condition)
        self.assertEqual(result, expected)


@pytest.fixture(name="modify_tree")
def modify_tree_fixture(node, key, value):
    """Create a new decision tree and modify it"""
    tree = wxcode_decision_tree()
    tree[node][key] = value
    return tree


@pytest.mark.parametrize(
    "node, key, value, expected",
    (
        (
            "lightning",
            "if_diagnostic_missing",
            "kittens",
            (
                "Node lightning contains an if_diagnostic_missing key that targets "
                "key 'kittens' which is neither 'if_true' nor 'if_false'"
            ),
        ),
        (
            "drizzle_mist",
            "condition_combination",
            "kittens",
            (
                "Node drizzle_mist utilises 2 diagnostic fields but 'kittens' is "
                "not a valid combination condition"
            ),
        ),
        (
            "lightning",
            "condition_combination",
            "AND",
            (
                "Node lightning utilises combination condition 'AND' but does not "
                "use 2 diagnostic fields for combination in this way"
            ),
        ),
        (
            "lightning",
            "threshold_condition",
            ">>",
            "Node lightning uses invalid threshold condition >>",
        ),
        (
            "lightning",
            "diagnostic_conditions",
            ["equal"],
            (
                "Node lightning uses invalid diagnostic condition 'equal'; this "
                "should be 'above' or 'below'"
            ),
        ),
        (
            "lightning_shower",
            "if_true",
            100,
            (
                "Node lightning_shower results in an invalid category of 100 for the "
                "if_true condition"
            ),
        ),
        (
            "lightning_shower",
            "if_false",
            100,
            (
                "Node lightning_shower results in an invalid category of 100 for the "
                "if_false condition"
            ),
        ),
        (
            "lightning_shower",
            "if_false",
            "kittens",
            "Node lightning_shower has an invalid destination of kittens for the if_false "
            "condition",
        ),
        (
            "snow_in_vicinity",
            "diagnostic_fields",
            [
                [
                    [
                        "probability_of_lwe_sleetfall_rate_above_threshold",
                        "+",
                        "probability_of_rainfall_rate_above_threshold",
                        "-",
                        "probability_of_lwe_snowfall_rate_above_threshold",
                    ]
                ]
            ],
            (
                "Node snow_in_vicinity has inconsistent nesting for the "
                "diagnostic_fields, diagnostic_conditions, and diagnostic_thresholds "
                "fields"
            ),
        ),
        (
            "snow_in_vicinity",
            "diagnostic_conditions",
            ["above", "above", "above"],
            (
                "Node snow_in_vicinity has inconsistent nesting for the "
                "diagnostic_fields, diagnostic_conditions, and diagnostic_thresholds "
                "fields"
            ),
        ),
        (
            "snow_in_vicinity",
            "diagnostic_thresholds",
            [[0.03, "mm hr-1"], [0.03, "mm hr-1"], [0.03, "mm hr-1"]],
            (
                "Node snow_in_vicinity has inconsistent nesting for the "
                "diagnostic_fields, diagnostic_conditions, and diagnostic_thresholds "
                "fields"
            ),
        ),
        (
            "lightning",
            "probability_thresholds",
            [0.5, 0.5],
            (
                "Node lightning has a different number of probability thresholds "
                "and diagnostic_fields: [0.5, 0.5], "
                "['probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_threshold']"  # noqa: E501
            ),
        ),
        (
            "lightning",
            "probability_thresholds",
            ["kittens"],
            "Node lightning has a non-numeric probability threshold ['kittens']",
        ),
        (
            "mist_conditions",
            "if_true",
            "no_precipitation_cloud",
            "Unreachable node 'fog_conditions'",
        ),
    ),
)
def test_check_tree(modify_tree, expected):
    """Test that the various possible decision tree problems are identified."""
    result = check_tree(modify_tree)
    assert result == expected


def test_check_tree_invalid_key():
    """Modify a wx decision tree."""
    expected = "Node lightning contains unknown key 'kittens'"
    tree = wxcode_decision_tree()
    tree["lightning"]["kittens"] = 0
    result = check_tree(tree)
    assert result == expected


def test_check_tree_non_dictionary():
    """Check ValueError is raised if non-dictionary is passed to check_tree."""
    expected = "Decision tree is not a dictionary"
    with pytest.raises(ValueError, match=expected):
        check_tree(1.0)


def test_check_tree_list_requirements():
    """
    This test simply checks that the expected wrapper text is returned. The
    listing of the diagnostics is checked in testing the interrogate_decision_tree
    function.
    """
    expected = "Decision tree OK\nRequired inputs are:"
    tree = wxcode_decision_tree()
    result = check_tree(tree)
    assert expected in result


if __name__ == "__main__":
    unittest.main()
