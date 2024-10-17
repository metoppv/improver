# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module for utilities that manipulate probabilities."""

import operator
from collections import namedtuple
from typing import Dict

from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError


def comparison_operator_dict() -> Dict[str, namedtuple]:
    """Generate dictionary linking string comparison operators to functions.
    Each key contains a dict of:
    - 'function': The operator function for this comparison_operator,
    - 'spp_string': Comparison_Operator string for use in CF-convention metadata
    - 'inverse': The inverse operator, i.e. ge has an inverse of lt.
    """

    inequality = namedtuple("inequality", "function, spp_string, inverse")

    comparison_operator_dict = {}
    comparison_operator_dict.update(
        dict.fromkeys(
            ["ge", "GE", ">="],
            inequality(
                function=operator.ge,
                spp_string="greater_than_or_equal_to",
                inverse="lt",
            ),
        )
    )
    comparison_operator_dict.update(
        dict.fromkeys(
            ["gt", "GT", ">"],
            inequality(function=operator.gt, spp_string="greater_than", inverse="le",),
        )
    )
    comparison_operator_dict.update(
        dict.fromkeys(
            ["le", "LE", "<="],
            inequality(
                function=operator.le, spp_string="less_than_or_equal_to", inverse="gt",
            ),
        )
    )
    comparison_operator_dict.update(
        dict.fromkeys(
            ["lt", "LT", "<"],
            inequality(function=operator.lt, spp_string="less_than", inverse="ge",),
        )
    )
    return comparison_operator_dict


def to_threshold_inequality(cube: Cube, above: bool = True) -> Cube:
    """Takes a cube and a target relative to threshold inequality; above or not
    above. The function returns probabilities in relation to the threshold values
    with the target inequality.

    The threshold inequality is limited to being above (above=True) or below
    (above=False) a threshold, rather than more specific targets such as
    "greater_than_or_equal_to". It is not possible to flip probabilities from
    e.g. "less_than_or_equal_to" to "greater_than_or_equal_to", only to
    "greater_than". As such the operation will use the valid inversion that
    achieves the broader target inequality.

    Args:
        cube:
            A probability cube with a threshold coordinate.
        above:
            Targets an above (gt, ge) threshold inequality if True, otherwise
            targets a below (lt, le) threshold inequality if False.

    Returns:
        A cube with the probabilities relative to the threshold values with
        the target inequality.

    Raised:
        ValueError: If the input cube has no threshold coordinate.
    """
    try:
        threshold = cube.coord(var_name="threshold")
    except CoordinateNotFoundError:
        raise ValueError(
            "Cube does not have a threshold coordinate, probabilities "
            "cannot be inverted if present."
        )

    inequality = threshold.attributes["spp__relative_to_threshold"]
    spp_lookup = comparison_operator_dict()
    above_attr = [spp_lookup[ineq].spp_string for ineq in ["ge", "gt"]]
    below_attr = [spp_lookup[ineq].spp_string for ineq in ["le", "lt"]]

    if (inequality in below_attr and above) or (inequality in above_attr and not above):
        return invert_probabilities(cube)
    return cube


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
            value.inverse
            for key, value in comparison_operator_lookup.items()
            if value.spp_string == inequality
        ]
    )
    new_inequality = comparison_operator_lookup[inverse].spp_string
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
