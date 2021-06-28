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
"""Plugin to calculate whether precipitation is showery"""

from typing import Dict, List, Tuple

from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.wxcode.weather_symbols import WeatherSymbols


class ShowerCondition(BasePlugin):
    """Plugin to calculate whether precipitation is showery based on input
    cloud, texture and / or convective ratio probability fields"""

    def __init__(self) -> None:
        """
        Set up fixed conditions from which to diagnose showers from different
        input fields.

        Shower condition from UK diagnostics:
        - Probability of cloud texture above 0.05 >= 0.5

        Shower condition from global diagnostics:
        - Probability of cloud area fraction above 6.5 oktas < 0.5 AND
        - Probability of convective ratio above 0.8 >= 0.5
        """
        self.conditions_uk = {
            "texture_of_low_and_medium_type_cloud_area_fraction": {
                "succeed": 1,
                "fail": 0,
                "probability_thresholds": [0.5],
                "threshold_condition": ">=",
                "condition_combination": "",
                "diagnostic_fields": [
                    "probability_of_texture_of_low_and_medium_type_cloud_area_fraction_above_threshold"
                ],
                "diagnostic_thresholds": [[0.05, 1]],
                "diagnostic_conditions": ["above"],
            },
        }

        self.conditions_global = {
            "low_and_medium_type_cloud_area_fraction": {
                "succeed": "convective_ratio",
                "fail": 1,
                "probability_thresholds": [0.5],
                "threshold_condition": ">=",
                "condition_combination": "",
                "diagnostic_fields": [
                    "probability_of_low_and_medium_type_cloud_area_"
                    "fraction_above_threshold",
                ],
                "diagnostic_thresholds": [[0.8125, 1]],
                "diagnostic_conditions": ["above"],
            },
            "convective_ratio": {
                "succeed": 1,
                "fail": 0,
                "probability_thresholds": [0.5],
                "threshold_condition": ">=",
                "condition_combination": "",
                "diagnostic_fields": [
                    "probability_of_convective_ratio_above_threshold"
                ],
                "diagnostic_thresholds": [[0.8, 1]],
                "diagnostic_conditions": ["above"],
            },
        }

        self.cubes = []
        self.tree = None

    def _output_metadata(self) -> Tuple[Cube, Dict]:
        """Returns template cube and mandatory attributes for result"""
        template = next(
            self.cubes[0].slices_over(find_threshold_coordinate(self.cubes[0]))
        )
        template.remove_coord(find_threshold_coordinate(self.cubes[0]))
        attributes = generate_mandatory_attributes(self.cubes)
        return template, attributes

    def _extract_cubes(self, conditions: List[str], cubes: CubeList) -> bool:
        """For a given set of conditions, put all matching cubes onto self.cubes and
        put conditions onto self.tree. If ALL conditions are not satisfied, the function
        exits without updating self.cubes or self.tree."""
        matched_cubes = []
        for name in conditions:
            found_cubes = cubes.extract(f"probability_of_{name}_above_threshold")
            if not found_cubes:
                return False
            (found_cube,) = found_cubes  # We expect exactly one cube here
            matched_cubes.append(found_cube)
        self.cubes = CubeList(matched_cubes)
        self.tree = conditions
        return True

    def process(self, cubes: CubeList) -> Cube:
        """
        Determine the shower condition from global or UK data depending
        on input fields. Expected inputs for UK:
        cloud_texture:
        probability_of_texture_of_low_and_medium_type_cloud_area_fraction_above_threshold,
        and for global cloud:
        probability_of_low_and_medium_type_cloud_area_fraction_above_threshold
        conv_ratio: probability_of_convective_ratio_above_threshold

        Args:
            cubes:
                List of input cubes

        Returns:
            Binary (0/1) "precipitation is showery"

        Raises:
            ValueError: If inputs are incomplete
        """
        for conditions in [self.conditions_uk, self.conditions_global]:
            if self._extract_cubes(conditions, cubes):
                break
        else:  # no break
            raise ValueError(
                "Incomplete inputs: must include either cloud_texture for the UK,"
                "or cloud and conv_ratio for global."
            )

        template, attributes = self._output_metadata()
        showery_points = WeatherSymbols(
            wxtree=self.tree, ignore_day_night=True
        ).process(self.cubes)

        result = create_new_diagnostic_cube(
            "precipitation_is_showery",
            "1",
            template,
            mandatory_attributes=attributes,
            data=showery_points.data.astype(FLOAT_DTYPE),
        )

        return result
