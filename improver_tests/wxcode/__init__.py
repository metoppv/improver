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
"""Decision tree for testing the weather code plugin."""
from typing import Dict, Any


def wxcode_decision_tree() -> Dict[str, Dict[str, Any]]:
    """
    Define an example decision tree to test the weather symbols code.

    Returns:
        A dictionary containing the queries that comprise the decision
        tree.
    """
    queries = {
        "lightning": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[0.0, "m-2"]],
            "if_diagnostic_missing": "if_false",
            "if_false": "hail",
            "if_true": "lightning_shower",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "drizzle_cloud": {
            "condition_combination": "AND",
            "diagnostic_conditions": ["above", "above"],
            "diagnostic_fields": [
                "probability_of_lwe_thickness_of_precipitation_amount_above_threshold",
                "probability_of_low_type_cloud_area_fraction_above_threshold",
            ],
            "diagnostic_thresholds": [[0.03, "mm", 3600], [0.85, 1]],
            "if_false": "mist_conditions",
            "if_true": "drizzle_is_rain",
            "probability_thresholds": [0.5, 0.5],
            "threshold_condition": ">=",
        },
        "drizzle_is_rain": {
            "condition_combination": "",
            "diagnostic_conditions": [["above", "above", "above"]],
            "diagnostic_fields": [
                [
                    "probability_of_lwe_thickness_of_sleetfall_amount_above_threshold",
                    "+",
                    "probability_of_lwe_thickness_of_snowfall_amount_above_threshold",
                    "-",
                    "probability_of_thickness_of_rainfall_amount_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[0.03, "mm", 3600], [0.03, "mm", 3600], [0.03, "mm", 3600]]
            ],
            "if_false": "mist_conditions",
            "if_true": 11,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
        },
        "drizzle_mist": {
            "condition_combination": "AND",
            "diagnostic_conditions": ["above", "below"],
            "diagnostic_fields": [
                "probability_of_lwe_thickness_of_precipitation_amount_above_threshold",
                "probability_of_visibility_in_air_below_threshold",
            ],
            "diagnostic_thresholds": [[0.03, "mm", 3600], [5000.0, "m"]],
            "if_false": "drizzle_cloud",
            "if_true": "drizzle_is_rain",
            "probability_thresholds": [0.5, 0.5],
            "threshold_condition": ">=",
        },
        "fog_conditions": {
            "condition_combination": "",
            "diagnostic_conditions": ["below"],
            "diagnostic_fields": ["probability_of_visibility_in_air_below_threshold"],
            "diagnostic_thresholds": [[1000.0, "m"]],
            "if_false": 5,
            "if_true": 6,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "hail": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_lwe_graupel_and_hail_fall_rate_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm hr-1"]],
            "if_diagnostic_missing": "if_false",
            "if_false": "heavy_precipitation",
            "if_true": "hail_rain",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "hail_rain": {
            "condition_combination": "",
            "diagnostic_conditions": [["above", "above"]],
            "diagnostic_fields": [
                [
                    "probability_of_lwe_thickness_of_graupel_and_hail_fall_amount_above_threshold",
                    "-",
                    "probability_of_thickness_of_rainfall_amount_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [[[0.1, "mm", 3600], [0.1, "mm", 3600]]],
            "if_diagnostic_missing": "if_false",
            "if_false": "heavy_precipitation",
            "if_true": "hail_sleet",
            "probability_thresholds": [0.0],
            "threshold_condition": ">",
        },
        "hail_shower": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "if_false": 21,
            "if_true": 20,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "hail_sleet": {
            "condition_combination": "",
            "diagnostic_conditions": [["above", "above"]],
            "diagnostic_fields": [
                [
                    "probability_of_lwe_thickness_of_graupel_and_hail_fall_amount_above_threshold",
                    "-",
                    "probability_of_lwe_thickness_of_sleetfall_amount_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [[[0.1, "mm", 3600], [0.1, "mm", 3600]]],
            "if_diagnostic_missing": "if_false",
            "if_false": "heavy_precipitation",
            "if_true": "hail_snow",
            "probability_thresholds": [0.0],
            "threshold_condition": ">",
        },
        "hail_snow": {
            "condition_combination": "",
            "diagnostic_conditions": [["above", "above"]],
            "diagnostic_fields": [
                [
                    "probability_of_lwe_thickness_of_graupel_and_hail_fall_amount_above_threshold",
                    "-",
                    "probability_of_lwe_thickness_of_snowfall_amount_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [[[0.1, "mm", 3600], [0.1, "mm", 3600]]],
            "if_diagnostic_missing": "if_false",
            "if_false": "heavy_precipitation",
            "if_true": "hail_shower",
            "probability_thresholds": [0.0],
            "threshold_condition": ">",
        },
        "heavy_precipitation": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_lwe_thickness_of_precipitation_amount_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm", 3600]],
            "if_false": "precipitation_in_vicinity",
            "if_true": "heavy_precipitation_cloud",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "heavy_precipitation_cloud": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "if_false": "heavy_snow_continuous",
            "if_true": "heavy_snow_shower",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "heavy_rain_continuous_in_vicinity": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_lwe_thickness_of_precipitation_amount_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm", 3600]],
            "if_false": 12,
            "if_true": 15,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "heavy_rain_or_sleet_continuous": {
            "condition_combination": "",
            "diagnostic_conditions": [["above", "above", "above"]],
            "diagnostic_fields": [
                [
                    "probability_of_lwe_thickness_of_sleetfall_amount_above_threshold",
                    "+",
                    "probability_of_lwe_thickness_of_snowfall_amount_above_threshold",
                    "-",
                    "probability_of_thickness_of_rainfall_amount_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm", 3600], [1.0, "mm", 3600], [1.0, "mm", 3600]]
            ],
            "if_false": 18,
            "if_true": 15,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
        },
        "heavy_rain_or_sleet_shower": {
            "condition_combination": "",
            "diagnostic_conditions": [["above", "above", "above"]],
            "diagnostic_fields": [
                [
                    "probability_of_lwe_thickness_of_sleetfall_amount_above_threshold",
                    "+",
                    "probability_of_lwe_thickness_of_snowfall_amount_above_threshold",
                    "-",
                    "probability_of_thickness_of_rainfall_amount_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm", 3600], [1.0, "mm", 3600], [1.0, "mm", 3600]]
            ],
            "if_false": 17,
            "if_true": 14,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
        },
        "heavy_rain_shower_in_vicinity": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_lwe_thickness_of_precipitation_amount_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm", 3600]],
            "if_false": 10,
            "if_true": 14,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "heavy_snow_continuous": {
            "condition_combination": "",
            "diagnostic_conditions": [["above", "above", "above"]],
            "diagnostic_fields": [
                [
                    "probability_of_lwe_thickness_of_sleetfall_amount_above_threshold",
                    "+",
                    "probability_of_thickness_of_rainfall_amount_above_threshold",
                    "-",
                    "probability_of_lwe_thickness_of_snowfall_amount_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm", 3600], [1.0, "mm", 3600], [1.0, "mm", 3600]]
            ],
            "if_false": "heavy_rain_or_sleet_continuous",
            "if_true": 27,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
        },
        "heavy_snow_continuous_in_vicinity": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_lwe_thickness_of_precipitation_amount_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm", 3600]],
            "if_false": 24,
            "if_true": 27,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "heavy_snow_shower": {
            "condition_combination": "",
            "diagnostic_conditions": [["above", "above", "above"]],
            "diagnostic_fields": [
                [
                    "probability_of_lwe_thickness_of_sleetfall_amount_above_threshold",
                    "+",
                    "probability_of_thickness_of_rainfall_amount_above_threshold",
                    "-",
                    "probability_of_lwe_thickness_of_snowfall_amount_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[1.0, "mm", 3600], [1.0, "mm", 3600], [1.0, "mm", 3600]]
            ],
            "if_false": "heavy_rain_or_sleet_shower",
            "if_true": 26,
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
        },
        "heavy_snow_shower_in_vicinity": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_lwe_thickness_of_precipitation_amount_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[1.0, "mm", 3600]],
            "if_false": 23,
            "if_true": 26,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "lightning_shower": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "if_false": 30,
            "if_true": 29,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "mist_conditions": {
            "condition_combination": "",
            "diagnostic_conditions": ["below"],
            "diagnostic_fields": ["probability_of_visibility_in_air_below_threshold"],
            "diagnostic_thresholds": [[5000.0, "m"]],
            "if_false": "no_precipitation_cloud",
            "if_true": "fog_conditions",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "no_precipitation_cloud": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_low_and_medium_type_cloud_area_fraction_above_threshold"
            ],
            "diagnostic_thresholds": [[0.8125, 1]],
            "if_false": "partly_cloudy",
            "if_true": "overcast_cloud",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "overcast_cloud": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_low_type_cloud_area_fraction_above_threshold"
            ],
            "diagnostic_thresholds": [[0.85, 1]],
            "if_false": 7,
            "if_true": 8,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "partly_cloudy": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_low_and_medium_type_cloud_area_fraction_above_threshold"
            ],
            "diagnostic_thresholds": [[0.1875, 1]],
            "if_false": 1,
            "if_true": 3,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "precipitation_in_vicinity": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": [
                "probability_of_lwe_thickness_of_precipitation_amount_in_vicinity_above_threshold"
            ],
            "diagnostic_thresholds": [[0.1, "mm", 3600]],
            "if_false": "drizzle_mist",
            "if_true": "snow_in_vicinity",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "rain_in_vicinity_cloud": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "if_false": "heavy_rain_continuous_in_vicinity",
            "if_true": "heavy_rain_shower_in_vicinity",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "rain_or_sleet_in_vicinity": {
            "condition_combination": "",
            "diagnostic_conditions": [["above", "above", "above"]],
            "diagnostic_fields": [
                [
                    "probability_of_lwe_thickness_of_sleetfall_amount_above_threshold",
                    "+",
                    "probability_of_lwe_thickness_of_snowfall_amount_above_threshold",
                    "-",
                    "probability_of_thickness_of_rainfall_amount_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[0.03, "mm", 3600], [0.03, "mm", 3600], [0.03, "mm", 3600]]
            ],
            "if_false": "sleet_in_vicinity_cloud",
            "if_true": "rain_in_vicinity_cloud",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
        },
        "sleet_in_vicinity_cloud": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "if_false": 18,
            "if_true": 17,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
        "snow_in_vicinity": {
            "condition_combination": "",
            "diagnostic_conditions": [["above", "above", "above"]],
            "diagnostic_fields": [
                [
                    "probability_of_lwe_thickness_of_sleetfall_amount_above_threshold",
                    "+",
                    "probability_of_thickness_of_rainfall_amount_above_threshold",
                    "-",
                    "probability_of_lwe_thickness_of_snowfall_amount_above_threshold",
                ]
            ],
            "diagnostic_thresholds": [
                [[0.03, "mm", 3600], [0.03, "mm", 3600], [0.03, "mm", 3600]]
            ],
            "if_false": "rain_or_sleet_in_vicinity",
            "if_true": "snow_in_vicinity_cloud",
            "probability_thresholds": [0.0],
            "threshold_condition": "<",
        },
        "snow_in_vicinity_cloud": {
            "condition_combination": "",
            "diagnostic_conditions": ["above"],
            "diagnostic_fields": ["probability_of_shower_condition_above_threshold"],
            "diagnostic_thresholds": [[1.0, 1]],
            "if_false": "heavy_snow_continuous_in_vicinity",
            "if_true": "heavy_snow_shower_in_vicinity",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
        },
    }

    return queries
