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
"""Utilities for Unit tests for ApplyDecisionTree"""

import datetime
from typing import Any, Dict

import numpy as np

from improver.categorical.utilities import categorical_attributes
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


def set_up_wxcube(
    data=np.ones((16, 16), dtype=np.int8),
    time=datetime.datetime(2018, 9, 12, 5, 43),
    time_bounds=None,
    frt=datetime.datetime(2018, 9, 12, 3),
    lat_lon=False,
):
    """
    Set up a wxcube for a particular time and location, which can cover
    the terminator and test the "update_daynight" functionality

    Args:
        data (numpy array):
            The weather codes with which to populate the array.
        time (datetime.datetime):
            Validity time for cube.
        time_bounds ([datetime.datetime, datetime.datetime]):
            Validity time bounds.
        frt (datetime.datetime):
            Forecast reference time for cube.
        lat_lon (bool):
            If True, returns a cube on a lat-lon grid.
            If False, returns equal area.

    Returns:
        iris.cube.Cube:
            cube of weather codes set to 1
            data shape (time_points, 16, 16)
    """
    decision_tree = wxcode_decision_tree()
    kwargs = {
        "name": "weather_code",
        "units": 1,
        "time": time,
        "time_bounds": time_bounds,
        "frt": frt,
        "attributes": categorical_attributes(
            decision_tree, decision_tree["meta"]["name"]
        ),
        "spatial_grid": "equalarea",
        "domain_corner": (0, -30000),
    }

    if lat_lon:
        kwargs.update(
            {"spatial_grid": "latlon", "domain_corner": (49, -8), "grid_spacing": 1}
        )
    cube = set_up_variable_cube(data, **kwargs)

    return cube


def wxcode_decision_tree(accumulation: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Define an example decision tree to test the weather symbols code.

    Args:
        accumulation:
            A boolean that if true means the "heavy_precipitation" node
            should be modified to use a precipitation accumulation rather than
            a rate.
    Returns:
        A dictionary containing the queries that comprise the decision
        tree.
    """
    queries = {
        "meta": {"name": "weather_code"},
        "lightning": {
            "if_true": "lightning_shower",
            "if_false": "hail",
            "if_diagnostic_missing": "if_false",
            "probability_thresholds": [0.3],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_threshold"  # noqa: E501
            ],
            "diagnostic_thresholds": [[0.0, "m-2"]],
            "diagnostic_conditions": ["above"],
        },
        "lightning_shower": {
            "if_true": "Thunder_Shower_Day",
            "if_false": "Thunder",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "diagnostic_conditions": ["above"],
        },
        "hail": {
            "if_true": "hail_precip",
            "if_false": "heavy_precipitation",
            "if_diagnostic_missing": "if_false",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_lwe_graupel_and_hail_fall_rate_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[0.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "hail_precip": {
            "if_true": "hail_shower",
            "if_false": "heavy_precipitation",
            "if_diagnostic_missing": "if_false",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_lwe_precipitation_rate_max_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "hail_shower": {
            "if_true": "Hail_Shower_Day",
            "if_false": "Hail",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_precipitation": {
            "if_true": "heavy_precipitation_cloud",
            "if_false": "precipitation_in_vicinity",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_lwe_precipitation_rate_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_precipitation_cloud": {
            "if_true": "heavy_snow_shower",
            "if_false": "heavy_snow_continuous",
            "if_diagnostic_missing": "if_true",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_snow_shower": {
            "if_true": "Heavy_Snow_Shower_Day",
            "if_false": "heavy_rain_or_sleet_shower",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [
                    "probability_of_lwe_sleetfall_rate_above_threshold",
                    "+",
                    "probability_of_rainfall_rate_above_threshold",
                    "-",
                    "probability_of_lwe_snowfall_rate_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "heavy_rain_or_sleet_shower": {
            "if_true": "Heavy_Shower_Day",
            "if_false": "Sleet_Shower_Day",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [
                    "probability_of_lwe_sleetfall_rate_above_threshold",
                    "+",
                    "probability_of_lwe_snowfall_rate_above_threshold",
                    "-",
                    "probability_of_rainfall_rate_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "heavy_snow_continuous": {
            "if_true": "Heavy_Snow",
            "if_false": "heavy_rain_or_sleet_continuous",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [
                    "probability_of_lwe_sleetfall_rate_above_threshold",
                    "+",
                    "probability_of_rainfall_rate_above_threshold",
                    "-",
                    "probability_of_lwe_snowfall_rate_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "heavy_rain_or_sleet_continuous": {
            "if_true": "Heavy_Rain",
            "if_false": "Sleet",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [
                    "probability_of_lwe_sleetfall_rate_above_threshold",
                    "+",
                    "probability_of_lwe_snowfall_rate_above_threshold",
                    "-",
                    "probability_of_rainfall_rate_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "precipitation_in_vicinity": {
            "if_true": "snow_in_vicinity",
            "if_false": "drizzle_mist",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_lwe_precipitation_rate_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[0.1, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "snow_in_vicinity": {
            "if_true": "snow_in_vicinity_cloud",
            "if_false": "rain_or_sleet_in_vicinity",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [
                    "probability_of_lwe_sleetfall_rate_above_threshold",
                    "+",
                    "probability_of_rainfall_rate_above_threshold",
                    "-",
                    "probability_of_lwe_snowfall_rate_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[0.03, "mm hr-1"], [0.03, "mm hr-1"], [0.03, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "snow_in_vicinity_cloud": {
            "if_true": "heavy_snow_shower_in_vicinity",
            "if_false": "heavy_snow_continuous_in_vicinity",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_snow_shower_in_vicinity": {
            "if_true": "Heavy_Snow_Shower_Day",
            "if_false": "Light_Snow_Shower_Day",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_lwe_precipitation_rate_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_snow_continuous_in_vicinity": {
            "if_true": "Heavy_Snow",
            "if_false": "Light_Snow",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_lwe_precipitation_rate_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "rain_or_sleet_in_vicinity": {
            "if_true": "rain_in_vicinity_cloud",
            "if_false": "sleet_in_vicinity_cloud",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [
                    "probability_of_lwe_sleetfall_rate_above_threshold",
                    "+",
                    "probability_of_lwe_snowfall_rate_above_threshold",
                    "-",
                    "probability_of_rainfall_rate_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[0.03, "mm hr-1"], [0.03, "mm hr-1"], [0.03, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "rain_in_vicinity_cloud": {
            "if_true": "heavy_rain_shower_in_vicinity",
            "if_false": "heavy_rain_continuous_in_vicinity",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_rain_shower_in_vicinity": {
            "if_true": "Heavy_Shower_Day",
            "if_false": "Light_Shower_Day",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_lwe_precipitation_rate_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_rain_continuous_in_vicinity": {
            "if_true": "Heavy_Rain",
            "if_false": "Light_Rain",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_lwe_precipitation_rate_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "sleet_in_vicinity_cloud": {
            "if_true": "Sleet_Shower_Day",
            "if_false": "Sleet",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "diagnostic_conditions": ["above"],
        },
        "drizzle_mist": {
            "if_true": "drizzle_is_rain",
            "if_false": "drizzle_cloud",
            "probability_thresholds": [0.5, 0.5],
            "threshold_condition": ">=",
            "condition_combination": "AND",
            "diagnostic_fields": [
                "probability_of_lwe_precipitation_rate_above_threshold",
                "probability_of_visibility_in_air_below_threshold",
            ],
            "diagnostic_thresholds": [[0.03, "mm hr-1"], [5000.0, "m"]],
            "diagnostic_conditions": ["above", "below"],
        },
        "drizzle_cloud": {
            "if_true": "drizzle_is_rain",
            "if_false": "mist_conditions",
            "probability_thresholds": [0.5, 0.5],
            "threshold_condition": ">=",
            "condition_combination": "AND",
            "diagnostic_fields": [
                "probability_of_lwe_precipitation_rate_above_threshold",
                "probability_of_low_type_cloud_area_fraction_above_threshold",
            ],
            "diagnostic_thresholds": [[0.03, "mm hr-1"], [0.85, 1]],
            "diagnostic_conditions": ["above", "above"],
        },
        "drizzle_is_rain": {
            "if_true": "Drizzle",
            "if_false": "mist_conditions",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [
                    "probability_of_lwe_sleetfall_rate_above_threshold",
                    "+",
                    "probability_of_lwe_snowfall_rate_above_threshold",
                    "-",
                    "probability_of_rainfall_rate_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[0.03, "mm hr-1"], [0.03, "mm hr-1"], [0.03, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "mist_conditions": {
            "if_true": "fog_conditions",
            "if_false": "no_precipitation_cloud",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": ["probability_of_visibility_in_air_below_threshold"],
            "diagnostic_thresholds": [[5000.0, "m"]],
            "diagnostic_conditions": ["below"],
        },
        "fog_conditions": {
            "if_true": "Fog",
            "if_false": "Mist",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": ["probability_of_visibility_in_air_below_threshold"],
            "diagnostic_thresholds": [[1000.0, "m"]],
            "diagnostic_conditions": ["below"],
        },
        "no_precipitation_cloud": {
            "if_true": "overcast_cloud",
            "if_false": "partly_cloudy",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_low_and_medium_type_cloud_area_fraction_above_threshold"
            ],
            "diagnostic_thresholds": [[0.8125, 1]],
            "diagnostic_conditions": ["above"],
        },
        "overcast_cloud": {
            "if_true": "Overcast",
            "if_false": "Cloudy",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_low_type_cloud_area_fraction_above_threshold"
            ],
            "diagnostic_thresholds": [[0.85, 1]],
            "diagnostic_conditions": ["above"],
        },
        "partly_cloudy": {
            "if_true": "Partly_Cloudy_Day",
            "if_false": "Sunny_Day",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_low_and_medium_type_cloud_area_fraction_above_threshold"
            ],
            "diagnostic_thresholds": [[0.1875, 1]],
            "diagnostic_conditions": ["above"],
        },
        "Clear_Night": {"leaf": 0},
        "Sunny_Day": {"leaf": 1, "if_night": "Clear_Night"},
        "Partly_Cloudy_Night": {"leaf": 2},
        "Partly_Cloudy_Day": {"leaf": 3, "if_night": "Partly_Cloudy_Night"},
        "Dust": {"leaf": 4, "is_unreachable": True},
        "Mist": {"leaf": 5, "group": "visibility"},
        "Fog": {"leaf": 6, "group": "visibility"},
        "Cloudy": {"leaf": 7},
        "Overcast": {"leaf": 8},
        "Light_Shower_Night": {"leaf": 9},
        "Light_Shower_Day": {
            "leaf": 10,
            "if_night": "Light_Shower_Night",
            "group": "rain",
        },
        "Drizzle": {"leaf": 11, "group": "rain"},
        "Light_Rain": {"leaf": 12, "group": "rain"},
        "Heavy_Shower_Night": {"leaf": 13},
        "Heavy_Shower_Day": {
            "leaf": 14,
            "if_night": "Heavy_Shower_Night",
            "group": "rain",
        },
        "Heavy_Rain": {"leaf": 15, "group": "rain"},
        "Sleet_Shower_Night": {"leaf": 16},
        "Sleet_Shower_Day": {
            "leaf": 17,
            "if_night": "Sleet_Shower_Night",
            "group": "sleet",
        },
        "Sleet": {"leaf": 18, "group": "sleet"},
        "Hail_Shower_Night": {"leaf": 19},
        "Hail_Shower_Day": {
            "leaf": 20,
            "if_night": "Hail_Shower_Night",
            "group": "convection",
        },
        "Hail": {"leaf": 21, "group": "convection"},
        "Light_Snow_Shower_Night": {"leaf": 22},
        "Light_Snow_Shower_Day": {
            "leaf": 23,
            "if_night": "Light_Snow_Shower_Night",
            "group": "snow",
        },
        "Light_Snow": {"leaf": 24, "group": "snow"},
        "Heavy_Snow_Shower_Night": {"leaf": 25},
        "Heavy_Snow_Shower_Day": {
            "leaf": 26,
            "if_night": "Heavy_Snow_Shower_Night",
            "group": "snow",
        },
        "Heavy_Snow": {"leaf": 27, "group": "snow"},
        "Thunder_Shower_Night": {"leaf": 28},
        "Thunder_Shower_Day": {
            "leaf": 29,
            "if_night": "Thunder_Shower_Night",
            "group": "convection",
        },
        "Thunder": {"leaf": 30, "group": "convection"},
    }

    if accumulation:
        queries["heavy_precipitation"] = {
            "if_true": "heavy_precipitation_cloud",
            "if_false": "precipitation_in_vicinity",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [
                "probability_of_lwe_thickness_of_precipitation_amount_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm", 3600]],
            "diagnostic_conditions": ["above"],
        }

    return queries
