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
"""Plugin to construct a shower conditions probability"""

from typing import Dict, List, Tuple

import numpy as np
from iris.cube import Cube

from improver import BasePlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.threshold import BasicThreshold
from improver.blending.calculate_weights_and_blend import WeightAndBlend


class ShowerCondition(BasePlugin):
    """Plugin to calculate whether precipitation is showery based on input
    cloud amounts and the convective ratio."""

    def __init__(
        self, cloud_threshold: float = 0.5, convection_threshold: float = 0.5
    ) -> None:
        """
        Args:
            cloud_threshold:
                The fractional cloud coverage value at which to threshold the
                cloud data; default 0.5.
            convection_threshold:
                The convective ratio value at which to threshold the convective
                ratio data; default 0.5.
        """
        self.cloud_threshold = cloud_threshold
        self.convection_threshold = convection_threshold

    def _output_metadata(self, cube: Cube) -> Tuple[Cube, Dict]:
        """
        Returns template cube and mandatory attributes for result. The template
        cube is modified to introduce an implied shower conditions threshold.

        Args:
            cube:
                The cube to use as a template, and from which to extract
                attributes for use in the new diagnostic cube.
        Returns:
            A tuple containing the template cube and attributes.
        """
        template = cube.copy()
        shower_threshold = find_threshold_coordinate(template)

        # We introduce an implied threshold of shower conditions.
        # Above 50% conditions are showery.
        template.coord(shower_threshold).rename("shower_condition")
        template.coord("shower_condition").var_name = "threshold"
        template.coord("shower_condition").points = 0.5

        attributes = generate_mandatory_attributes([cube])
        return template, attributes

    def process(self, cloud: Cube, convection: Cube) -> Cube:
        """
        Create a shower condition probability from cloud fraction and convective
        ratio fields. This plugin thresholds the two input diagnostics,
        creates a hybrid probability field from the resulting binary fields,
        and then collapses the realizations to give a non-binary probability
        field that represents the likelihood of conditions being showery.

        Args:
            cloud:
                A cube of cloud fraction.
            convection:
                A cube of convective ratio.

        Returns:
            Probability of showery conditions.

        Raises:
            ValueError: If inputs are not those expected.
        """
        if ("cloud_area_fraction" not in cloud.name() or
                "convective_ratio" not in convection.name()):
            msg = ("A cloud area fraction and convective ratio are required, "
                   f"but the inputs were: {cloud.name()}, {convection.name()}")
            raise ValueError(msg)

        cloud_thresholded = BasicThreshold(
            self.cloud_threshold, comparison_operator="<="
        ).process(cloud)
        convection_thresholded = BasicThreshold(self.convection_threshold).process(
            convection
        )

        # Fill any missing data in the convective ratio field with zeroes.
        convection_thresholded.data = convection_thresholded.data.filled(0)
        # Create a combined field taking the maximum of each input
        shower_probability = np.maximum(
            cloud_thresholded.data, convection_thresholded.data
        ).astype(FLOAT_DTYPE)

        template, attributes = self._output_metadata(convection_thresholded)
        result = create_new_diagnostic_cube(
            "probability_of_shower_condition_above_threshold",
            "1",
            template,
            mandatory_attributes=attributes,
            data=shower_probability,
        )

        # Perform a realization collapse
        return WeightAndBlend("realization", "linear", y0val=0.5, ynval=0.5).process(
            result
        )
