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
"""Module containing weather symbol decision tree for Global data"""

from typing import Any, Dict

from improver.wxcode import (
    CLOUD_PROB_ABOVE,
    CONVECTION_PROB_ABOVE,
    LOW_CLOUD_PROB_ABOVE,
    PRECIP_PROB_ABOVE,
    RAIN_PROB_ABOVE,
    SLEET_PROB_ABOVE,
    SNOW_PROB_ABOVE,
    VIS_PROB_BELOW,
)

# Start node for the Global wxcode decision tree.
START_NODE_GLOBAL = "heavy_precipitation"


def wxcode_decision_tree_global() -> Dict[str, Dict[str, Any]]:
    """
    Define queries that comprise the weather symbol decision tree.
    Each query contains the following elements:

        * succeed: The next query to call if the diagnostic being queried
              satisfies the current query.
        * fail: The next query to call if the diagnostic being queried
              does not satisfy the current query.
        * probability_thresholds: A list of probability thresholds that the
              query requires. One entry is provided for each diagnostic
              field being tested.
        * threshold_condition: The condition the diagnostic must satisfy
              relative to the probability threshold (e.g. greater than (>)
              the probability threshold).
        * condition_combination: The way (AND, OR) in which multiple
              conditions should be combined;
              e.g. rainfall > 0.5 AND snowfall > 0.5
        * diagnostics_fields: The diagnostics which are being used in the
              query. If this is a list of lists, the fields are manipulated
              according to the operators in the list and then compared
              with the probability threshold.
        * diagnostic_thresholds: The thresholding that is expected to have
              been applied to the input data; this is used to extract the
              appropriate data from the input cubes.
        * diagnostic_conditions: The condition that is expected to have
              been applied to the input data; this can be used to ensure
              the thresholding is as expected.

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
            "diagnostic_thresholds": [(1.0, "mm hr-1")],
            "diagnostic_conditions": ["above"],
        },
        "heavy_precipitation_cloud": {
            "succeed": "heavy_precipitation_convective_ratio",
            "fail": "heavy_snow_shower",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [(0.8125, 1)],
            "diagnostic_conditions": ["above"],
        },
        "heavy_precipitation_convective_ratio": {
            "succeed": "heavy_snow_shower",
            "fail": "heavy_snow_continuous",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CONVECTION_PROB_ABOVE],
            "diagnostic_thresholds": [(0.8, 1)],
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
                [(1.0, "mm hr-1"), (1.0, "mm hr-1"), (1.0, "mm hr-1")]
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
                [(1.0, "mm hr-1"), (1.0, "mm hr-1"), (1.0, "mm hr-1")]
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
                [(1.0, "mm hr-1"), (1.0, "mm hr-1"), (1.0, "mm hr-1")]
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
                [(1.0, "mm hr-1"), (1.0, "mm hr-1"), (1.0, "mm hr-1")]
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
            "diagnostic_thresholds": [(0.1, "mm hr-1")],
            "diagnostic_conditions": ["above"],
        },
        "light_precipitation_cloud": {
            "succeed": "light_precipitation_convective_ratio",
            "fail": "light_snow_shower",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [(0.8125, 1)],
            "diagnostic_conditions": ["above"],
        },
        "light_precipitation_convective_ratio": {
            "succeed": "light_snow_shower",
            "fail": "light_snow_continuous",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CONVECTION_PROB_ABOVE],
            "diagnostic_thresholds": [(0.8, 1)],
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
                [(0.1, "mm hr-1"), (0.1, "mm hr-1"), (0.1, "mm hr-1")]
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
                [(0.1, "mm hr-1"), (0.1, "mm hr-1"), (0.1, "mm hr-1")]
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
                [(0.1, "mm hr-1"), (0.1, "mm hr-1"), (0.1, "mm hr-1")]
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
                [(0.1, "mm hr-1"), (0.1, "mm hr-1"), (0.1, "mm hr-1")]
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
            "diagnostic_thresholds": [(0.03, "mm hr-1"), (5000.0, "m")],
            "diagnostic_conditions": ["above", "below"],
        },
        "drizzle_cloud": {
            "succeed": "drizzle_is_rain",
            "fail": "mist_conditions",
            "probability_thresholds": [0.5, 0.5],
            "threshold_condition": ">=",
            "condition_combination": "AND",
            "diagnostic_fields": [PRECIP_PROB_ABOVE, LOW_CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [(0.03, "mm hr-1"), (0.85, 1)],
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
                [(0.03, "mm hr-1"), (0.03, "mm hr-1"), (0.03, "mm hr-1")]
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
            "diagnostic_thresholds": [(5000.0, "m")],
            "diagnostic_conditions": ["below"],
        },
        "fog_conditions": {
            "succeed": 6,
            "fail": 5,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [VIS_PROB_BELOW],
            "diagnostic_thresholds": [(1000.0, "m")],
            "diagnostic_conditions": ["below"],
        },
        "no_precipitation_cloud": {
            "succeed": "overcast_cloud",
            "fail": "partly_cloudy",
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [(0.8125, 1)],
            "diagnostic_conditions": ["above"],
        },
        "overcast_cloud": {
            "succeed": 8,
            "fail": 7,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [LOW_CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [(0.85, 1)],
            "diagnostic_conditions": ["above"],
        },
        "partly_cloudy": {
            "succeed": 3,
            "fail": 1,
            "probability_thresholds": [0.5],
            "threshold_condition": ">=",
            "condition_combination": "",
            "diagnostic_fields": [CLOUD_PROB_ABOVE],
            "diagnostic_thresholds": [(0.1875, 1)],
            "diagnostic_conditions": ["above"],
        },
    }

    return queries
