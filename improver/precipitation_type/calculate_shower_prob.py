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
"""Plugin to calculate probability of showery precipitation"""

import iris
import numpy as np

from improver import BasePlugin
from improver.metadata.probabilistic import (
    extract_diagnostic_name,
    find_threshold_coordinate,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)


class ShowerProbability(BasePlugin):
    """Plugin to calculate probability of showery precipitation"""

    def __init__(self):
        """Set up fixed thresholds from which to diagnose showers from different
        input fields"""
        self.conditions_uk = {
            "texture_of_low_and_medium_cloud_area_fraction": {
                "diagnostic_threshold": 0.05,
                "probability_threshold": 0.5,
                "operator": "above",
            },
        }
        self.conditions_global = {
             "low_and_medium_cloud_area_fraction": {
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
        self.cubes = None
        self.tree = None

    def _calculate_shower_probability(self):
        """Calculate deterministic "shower probability" field"""
        shower_probability = np.ones_like(self.cubes[0].data)
        for cube in self.cubes:
            name = extract_diagnostic_name(cube.name())
            threshold_slice = cube.extract(
                iris.Constraint(name=self.tree[name]["diagnostic_threshold"])
            )
            prob = self.tree[name]["probability_threshold"]
            if tree[name]["operator"] == "above":
                condition_met = np.where(threshold_slice.data >= prob, 1, 0)
            else:
                condition_met = np.where(threshold_slice.data < prob, 1, 0)
            shower_probability = np.multiply(shower_probability, condition_met)
        return shower_probability

    def _output_metadata(self):
        """Returns template cube and mandatory attributes for result"""
        template = next(self.cubes[0].slices_over(find_threshold_coordinate(self.cubes[0])))
        template.remove_coord(find_threshold_coord(self.cubes[0]))
        attributes = generate_mandatory_attributes(self.cubes)
        return template, attributes

    def process(self, cloud=None, cloud_texture=None, conv_ratio=None):
        """
        Calculate shower probability from global or UK decision tree depending
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
                Binary (0/1) "probability of showery precipitation"

                TODO does this need a "threshold" coordinate (for technical reasons)?
                By metadata standard it is analogous to "probability of snow at
                surface", so should not have a "threshold" coordinate.  But it may not
                be possible to adapt weather symbols code in the time available.

        Raises:
            ValueError: if inputs are incorrect or incomplete
        """
        if cloud_texture is None:
            if cloud is None or conv_ratio is None:
                raise ValueError("Incomplete inputs")
            self.cubes = [cloud, conv_ratio]
            self.tree = self.conditions_global
        else:
            self.cubes = [cloud_texture]
            self.tree = self.conditions_uk

        shower_probability = self.calculate_shower_probability()
        template, attributes = self._output_metadata()

        result = create_new_diagnostic_cube(
            "showery_precipitation",
            "1",
            template,
            mandatory_attributes=attributes,
            data=shower_probability,
        )
            
        return result
