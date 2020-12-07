# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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

import iris
import numpy as np

from improver import BasePlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.probabilistic import (
    extract_diagnostic_name,
    find_threshold_coordinate,
    probability_is_above_or_below,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)


class ShowerCondition(BasePlugin):
    """Plugin to calculate whether precipitation is showery based on input
    cloud, texture and / or convective ratio probability fields"""

    def __init__(self):
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
                "diagnostic_threshold": 0.05,
                "probability_threshold": 0.5,
                "operator": "above",
            },
        }
        self.conditions_global = {
            "low_and_medium_type_cloud_area_fraction": {
                "diagnostic_threshold": 0.8125,
                "probability_threshold": 0.5,
                "operator": "below",
            },
            "convective_ratio": {
                "diagnostic_threshold": 0.8,
                "probability_threshold": 0.5,
                "operator": "above",
            },
        }
        self.relative_to_threshold = "above"
        self.cubes = None
        self.tree = None

    def _calculate_shower_condition(self, shape):
        """Calculate deterministic "precipitation is showery" field"""
        showery_points = np.ones(shape, dtype=FLOAT_DTYPE)
        for cube in self.cubes:
            name = extract_diagnostic_name(cube.name())
            if probability_is_above_or_below(cube) != self.relative_to_threshold:
                msg = "Expected cube of {} {} threshold"
                raise ValueError(msg.format(name, self.relative_to_threshold))

            slice_constraint = iris.Constraint(
                coord_values={
                    name: lambda cell: np.isclose(
                        cell.point, self.tree[name]["diagnostic_threshold"]
                    )
                }
            )
            threshold_slice = cube.extract(slice_constraint)
            if threshold_slice is None:
                msg = "Cube {} does not contain required threshold {}"
                raise ValueError(
                    msg.format(cube.name(), self.tree[name]["diagnostic_threshold"])
                )

            prob = self.tree[name]["probability_threshold"]
            if self.tree[name]["operator"] == "above":
                condition_met = np.where(threshold_slice.data >= prob, 1, 0)
            else:
                condition_met = np.where(threshold_slice.data < prob, 1, 0)
            showery_points = np.multiply(showery_points, condition_met)
        return showery_points.astype(FLOAT_DTYPE)

    def _output_metadata(self):
        """Returns template cube and mandatory attributes for result"""
        template = next(
            self.cubes[0].slices_over(find_threshold_coordinate(self.cubes[0]))
        )
        template.remove_coord(find_threshold_coordinate(self.cubes[0]))
        attributes = generate_mandatory_attributes(self.cubes)
        return template, attributes

    def process(self, cloud=None, cloud_texture=None, conv_ratio=None):
        """
        Determine the shower condition from global or UK data depending
        on input fields

        Args:
            cloud (iris.cube.Cube or None):
                Probability of total cloud amount above threshold
            cloud_texture (iris.cube.Cube or None):
                Probability of texture of total cloud amount above threshold
            conv_ratio (iris.cube.Cube or None):
                Probability of convective ratio above threshold

        Returns:
            iris.cube.Cube:
                Binary (0/1) "precipitation is showery"

        Raises:
            ValueError: if inputs are incomplete
        """
        if cloud_texture is None:
            if cloud is None or conv_ratio is None:
                raise ValueError("Incomplete inputs")
            self.cubes = [cloud, conv_ratio]
            self.tree = self.conditions_global
        else:
            self.cubes = [cloud_texture]
            self.tree = self.conditions_uk

        template, attributes = self._output_metadata()
        showery_points = self._calculate_shower_condition(template.shape)

        result = create_new_diagnostic_cube(
            "precipitation_is_showery",
            "1",
            template,
            mandatory_attributes=attributes,
            data=showery_points,
        )

        return result
