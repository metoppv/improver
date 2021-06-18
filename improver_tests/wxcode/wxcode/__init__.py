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
"""Utilities for Unit tests for Weather Symbols"""

from typing import Any, Dict


def prob_above_name(diagnostic: str) -> str:
    """Inline function to construct probability cube name"""
    return f"probability_of_{diagnostic}_above_threshold"


LIGHTNING_VICINITY_PROB = prob_above_name(
    "number_of_lightning_flashes_per_unit_area_in_vicinity"
)
CLOUD_NAME = "low_and_medium_type_cloud_area_fraction"
CLOUD_PROB_ABOVE = prob_above_name(CLOUD_NAME)
LOW_CLOUD_PROB_ABOVE = prob_above_name("low_type_cloud_area_fraction")
TEXTURE_PROB_ABOVE = prob_above_name(f"texture_of_{CLOUD_NAME}")
CONVECTION_PROB_ABOVE = prob_above_name("convective_ratio")
PRECIP_PROB_ABOVE = prob_above_name("lwe_precipitation_rate")
PRECIP_VICINITY_PROB_ABOVE = prob_above_name("lwe_precipitation_rate_in_vicinity")
RAIN_PROB_ABOVE = prob_above_name("rainfall_rate")
SLEET_PROB_ABOVE = prob_above_name("lwe_sleetfall_rate")
SNOW_PROB_ABOVE = prob_above_name("lwe_snowfall_rate")
VIS_PROB_BELOW = "probability_of_visibility_in_air_below_threshold"


def wxcode_decision_tree_uk() -> Dict[str, Dict[str, Any]]:
    """
    Define an example UK decision tree to test the weather symbols code.

    Returns:
        A dictionary containing the queries that comprise the decision
        tree.
    """
    queries = {
        "lightning": {
            "succeed": "lightning_cloud",
            "fail": "heavy_precipitation",
            "diagnostic_missing_action": "fail",
            "probability_thresholds": [0.3],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [LIGHTNING_VICINITY_PROB],
            "diagnostic_thresholds": [[0.0, "m-2"]],
            "diagnostic_conditions": ["above"],
        },
        "lightning_cloud": {
            "succeed": 29,
            "fail": 30,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [TEXTURE_PROB_ABOVE],
            "diagnostic_thresholds": [[0.05, 1]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_precipitation": {
            "succeed": "heavy_precipitation_cloud",
            "fail": "precipitation_in_vicinity",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [PRECIP_PROB_ABOVE],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_precipitation_cloud": {
            "succeed": "heavy_snow_shower",
            "fail": "heavy_snow_continuous",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [TEXTURE_PROB_ABOVE],
            "diagnostic_thresholds": [[0.05, 1]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_snow_shower": {
            "succeed": 26,
            "fail": "heavy_rain_or_sleet_shower",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", RAIN_PROB_ABOVE, "-", SNOW_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "heavy_rain_or_sleet_shower": {
            "succeed": 14,
            "fail": 17,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", SNOW_PROB_ABOVE, "-", RAIN_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "heavy_snow_continuous": {
            "succeed": 27,
            "fail": "heavy_rain_or_sleet_continuous",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", RAIN_PROB_ABOVE, "-", SNOW_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "heavy_rain_or_sleet_continuous": {
            "succeed": 15,
            "fail": 18,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", SNOW_PROB_ABOVE, "-", RAIN_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "precipitation_in_vicinity": {
            "succeed": "snow_in_vicinity",
            "fail": "drizzle_mist",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [PRECIP_VICINITY_PROB_ABOVE],
            "diagnostic_thresholds": [[0.1, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "snow_in_vicinity": {
            "succeed": "snow_in_vicinity_cloud",
            "fail": "rain_or_sleet_in_vicinity",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", RAIN_PROB_ABOVE, "-", SNOW_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[0.03, "mm hr-1"], [0.03, "mm hr-1"], [0.03, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "snow_in_vicinity_cloud": {
            "succeed": "heavy_snow_shower_in_vicinity",
            "fail": "heavy_snow_continuous_in_vicinity",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [TEXTURE_PROB_ABOVE],
            "diagnostic_thresholds": [[0.05, 1]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_snow_shower_in_vicinity": {
            "succeed": 26,
            "fail": 23,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [PRECIP_VICINITY_PROB_ABOVE],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_snow_continuous_in_vicinity": {
            "succeed": 27,
            "fail": 24,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [PRECIP_VICINITY_PROB_ABOVE],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "rain_or_sleet_in_vicinity": {
            "succeed": "rain_in_vicinity_cloud",
            "fail": "sleet_in_vicinity_cloud",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", SNOW_PROB_ABOVE, "-", RAIN_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[0.03, "mm hr-1"], [0.03, "mm hr-1"], [0.03, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "rain_in_vicinity_cloud": {
            "succeed": "heavy_rain_shower_in_vicinity",
            "fail": "heavy_rain_continuous_in_vicinity",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [TEXTURE_PROB_ABOVE],
            "diagnostic_thresholds": [[0.05, 1]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_rain_shower_in_vicinity": {
            "succeed": 14,
            "fail": 10,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [PRECIP_VICINITY_PROB_ABOVE],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_rain_continuous_in_vicinity": {
            "succeed": 15,
            "fail": 12,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [PRECIP_VICINITY_PROB_ABOVE],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "sleet_in_vicinity_cloud": {
            "succeed": 17,
            "fail": 18,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [TEXTURE_PROB_ABOVE],
            "diagnostic_thresholds": [[0.05, 1]],
            "diagnostic_conditions": ["above"],
        },
        "drizzle_mist": {
            "succeed": "drizzle_is_rain",
            "fail": "drizzle_cloud",
            "probability_thresholds": [0.5, 0.5],
            "threshold_condition": ">=",
            "condition_combination": "AND",
            "diagnostic_fields": [PRECIP_PROB_ABOVE, VIS_PROB_BELOW],
            "diagnostic_thresholds": [[0.03, "mm hr-1"], [5000.0, "m"]],
            "diagnostic_conditions": ["above", "below"],
        },
        "drizzle_cloud": {
            "succeed": "drizzle_is_rain",
            "fail": "mist_conditions",
            "probability_thresholds": [0.5, 0.5],
            "threshold_condition": ">=",
            "condition_combination": "AND",
            "diagnostic_fields": [PRECIP_PROB_ABOVE, LOW_CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [[0.03, "mm hr-1"], [0.85, 1]],
            "diagnostic_conditions": ["above", "above"],
        },
        "drizzle_is_rain": {
            "succeed": 11,
            "fail": "mist_conditions",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", SNOW_PROB_ABOVE, "-", RAIN_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[0.03, "mm hr-1"], [0.03, "mm hr-1"], [0.03, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "mist_conditions": {
            "succeed": "fog_conditions",
            "fail": "no_precipitation_cloud",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [VIS_PROB_BELOW],
            "diagnostic_thresholds": [[5000.0, "m"]],
            "diagnostic_conditions": ["below"],
        },
        "fog_conditions": {
            "succeed": 6,
            "fail": 5,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [VIS_PROB_BELOW],
            "diagnostic_thresholds": [[1000.0, "m"]],
            "diagnostic_conditions": ["below"],
        },
        "no_precipitation_cloud": {
            "succeed": "overcast_cloud",
            "fail": "partly_cloudy",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [[0.8125, 1]],
            "diagnostic_conditions": ["above"],
        },
        "overcast_cloud": {
            "succeed": 8,
            "fail": 7,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [LOW_CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [[0.85, 1]],
            "diagnostic_conditions": ["above"],
        },
        "partly_cloudy": {
            "succeed": 3,
            "fail": 1,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [[0.1875, 1]],
            "diagnostic_conditions": ["above"],
        },
    }

    return queries


def wxcode_decision_tree_global() -> Dict[str, Dict[str, Any]]:
    """
    Define an example global decision tree to test the weather symbols code.

    Returns:
        A dictionary containing the queries that comprise the decision
        tree.
    """
    queries = {
        "heavy_precipitation": {
            "succeed": "heavy_precipitation_cloud",
            "fail": "light_precipitation",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [PRECIP_PROB_ABOVE],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_precipitation_cloud": {
            "succeed": "heavy_precipitation_convective_ratio",
            "fail": "heavy_snow_shower",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [[0.8125, 1]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_precipitation_convective_ratio": {
            "succeed": "heavy_snow_shower",
            "fail": "heavy_snow_continuous",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CONVECTION_PROB_ABOVE],
            "diagnostic_thresholds": [[0.8, 1]],
            "diagnostic_conditions": ["above"],
        },
        "heavy_snow_shower": {
            "succeed": 26,
            "fail": "heavy_rain_or_sleet_shower",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", RAIN_PROB_ABOVE, "-", SNOW_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "heavy_rain_or_sleet_shower": {
            "succeed": 14,
            "fail": 17,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", SNOW_PROB_ABOVE, "-", RAIN_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "heavy_snow_continuous": {
            "succeed": 27,
            "fail": "heavy_rain_or_sleet_continuous",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", RAIN_PROB_ABOVE, "-", SNOW_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "heavy_rain_or_sleet_continuous": {
            "succeed": 15,
            "fail": 18,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", SNOW_PROB_ABOVE, "-", RAIN_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "light_precipitation": {
            "succeed": "light_precipitation_cloud",
            "fail": "drizzle_mist",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [PRECIP_PROB_ABOVE],
            "diagnostic_thresholds": [[0.1, "mm hr-1"]],
            "diagnostic_conditions": ["above"],
        },
        "light_precipitation_cloud": {
            "succeed": "light_precipitation_convective_ratio",
            "fail": "light_snow_shower",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [[0.8125, 1]],
            "diagnostic_conditions": ["above"],
        },
        "light_precipitation_convective_ratio": {
            "succeed": "light_snow_shower",
            "fail": "light_snow_continuous",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CONVECTION_PROB_ABOVE],
            "diagnostic_thresholds": [[0.8, 1]],
            "diagnostic_conditions": ["above"],
        },
        "light_snow_shower": {
            "succeed": 23,
            "fail": "light_rain_or_sleet_shower",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", RAIN_PROB_ABOVE, "-", SNOW_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[0.1, "mm hr-1"], [0.1, "mm hr-1"], [0.1, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "light_rain_or_sleet_shower": {
            "succeed": 10,
            "fail": 17,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", SNOW_PROB_ABOVE, "-", RAIN_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[0.1, "mm hr-1"], [0.1, "mm hr-1"], [0.1, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "light_snow_continuous": {
            "succeed": 24,
            "fail": "light_rain_or_sleet_continuous",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", RAIN_PROB_ABOVE, "-", SNOW_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[0.1, "mm hr-1"], [0.1, "mm hr-1"], [0.1, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "light_rain_or_sleet_continuous": {
            "succeed": 12,
            "fail": 18,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", SNOW_PROB_ABOVE, "-", RAIN_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[0.1, "mm hr-1"], [0.1, "mm hr-1"], [0.1, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "drizzle_mist": {
            "succeed": "drizzle_is_rain",
            "fail": "drizzle_cloud",
            "probability_thresholds": [0.5, 0.5],
            "threshold_condition": ">=",
            "condition_combination": "AND",
            "diagnostic_fields": [PRECIP_PROB_ABOVE, VIS_PROB_BELOW],
            "diagnostic_thresholds": [[0.03, "mm hr-1"], [5000.0, "m"]],
            "diagnostic_conditions": ["above", "below"],
        },
        "drizzle_cloud": {
            "succeed": "drizzle_is_rain",
            "fail": "mist_conditions",
            "probability_thresholds": [0.5, 0.5],
            "threshold_condition": ">=",
            "condition_combination": "AND",
            "diagnostic_fields": [PRECIP_PROB_ABOVE, LOW_CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [[0.03, "mm hr-1"], [0.85, 1]],
            "diagnostic_conditions": ["above", "above"],
        },
        "drizzle_is_rain": {
            "succeed": 11,
            "fail": "mist_conditions",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
            "condition_combination": "",
            "diagnostic_fields": [
                [SLEET_PROB_ABOVE, "+", SNOW_PROB_ABOVE, "-", RAIN_PROB_ABOVE]
            ],
            "diagnostic_thresholds": [
                [[0.03, "mm hr-1"], [0.03, "mm hr-1"], [0.03, "mm hr-1"]]
            ],
            "diagnostic_conditions": [["above", "above", "above"]],
        },
        "mist_conditions": {
            "succeed": "fog_conditions",
            "fail": "no_precipitation_cloud",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [VIS_PROB_BELOW],
            "diagnostic_thresholds": [[5000.0, "m"]],
            "diagnostic_conditions": ["below"],
        },
        "fog_conditions": {
            "succeed": 6,
            "fail": 5,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [VIS_PROB_BELOW],
            "diagnostic_thresholds": [[1000.0, "m"]],
            "diagnostic_conditions": ["below"],
        },
        "no_precipitation_cloud": {
            "succeed": "overcast_cloud",
            "fail": "partly_cloudy",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [[0.8125, 1]],
            "diagnostic_conditions": ["above"],
        },
        "overcast_cloud": {
            "succeed": 8,
            "fail": 7,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [LOW_CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [[0.85, 1]],
            "diagnostic_conditions": ["above"],
        },
        "partly_cloudy": {
            "succeed": 3,
            "fail": 1,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [[0.1875, 1]],
            "diagnostic_conditions": ["above"],
        },
    }

    return queries
