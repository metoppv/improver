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
"""Module for utilities that manipulate probabilities."""

import operator
from typing import Any, Dict

from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError


def comparison_operator_dict() -> Dict[str, Any]:
    """Generate dictionary linking string comparison operators to functions.
    Each key contains a dict of:
    - 'function': The operator function for this comparison_operator,
    - 'spp_string': Comparison_Operator string for use in CF-convention metadata
    """
    comparison_operator_dict = {}
    comparison_operator_dict.update(
        dict.fromkeys(
            ["ge", "GE", ">="],
            {
                "function": operator.ge,
                "spp_string": "greater_than_or_equal_to",
                "inverse": "lt",
            },
        )
    )
    comparison_operator_dict.update(
        dict.fromkeys(
            ["gt", "GT", ">"],
            {"function": operator.gt, "spp_string": "greater_than", "inverse": "le"},
        )
    )
    comparison_operator_dict.update(
        dict.fromkeys(
            ["le", "LE", "<="],
            {
                "function": operator.le,
                "spp_string": "less_than_or_equal_to",
                "inverse": "gt",
            },
        )
    )
    comparison_operator_dict.update(
        dict.fromkeys(
            ["lt", "LT", "<"],
            {"function": operator.lt, "spp_string": "less_than", "inverse": "ge"},
        )
    )
    return comparison_operator_dict


def invert_probabilities(cube: Cube) -> Cube:
    """Given a cube with a probability threshold, invert the probabilities
    relative to the existing thresholding inequality. Update the coordinate
    metadata to indicate the new threshold inequality.

    Args:
        cube:
            A probability cube with a threshold coordinate.

    Returns:
        Cube with the probabilities inverted relative to the input thresholding
        inequality.

    Raises:
        ValueError: If no threshold coordinate is found.
    """
    try:
        threshold = cube.coord(var_name="threshold")
    except CoordinateNotFoundError:
        raise ValueError(
            "Cube does not have a threshold coordinate, probabilities "
            "cannot be inverted if present."
        )

    comparison_operator_lookup = comparison_operator_dict()
    inequality = threshold.attributes["spp__relative_to_threshold"]
    (inverse,) = set(
        [
            value["inverse"]
            for key, value in comparison_operator_lookup.items()
            if value["spp_string"] == inequality
        ]
    )
    new_inequality = comparison_operator_lookup[inverse]["spp_string"]
    inverted_probabilities = cube.copy(data=(1.0 - cube.data))
    inverted_probabilities.coord(threshold).attributes[
        "spp__relative_to_threshold"
    ] = new_inequality

    new_name = (
        cube.name().replace("above", "below")
        if "above" in cube.name()
        else cube.name().replace("below", "above")
    )
    inverted_probabilities.rename(new_name)

    return inverted_probabilities
