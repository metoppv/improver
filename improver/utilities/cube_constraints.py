#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module for helping to create Iris constraints."""

from typing import List

import iris
from iris import Constraint


def create_sorted_lambda_constraint(
    coord_name: str, values: List[float], tolerance: float = 1.0e-7
) -> Constraint:
    """
    Create a lambda constraint for a range. This formulation of specifying
    a lambda constraint has the benefit of not needing to hardcode the name
    for the coordinate, so that this can be determined at runtime.

    The created function uses float values. As a result, a small tolerance is
    used to spread the ends of the ranges to help with float equality
    matching. Note that the relative tolerance will not affect values of zero.
    Adding/subtracting an absolute tolerance is not done due to the
    difficulty of selecting an appropriate value given the very small values
    of precipitation rates expressed in m s-1.

    Args:
        coord_name:
            Name of the coordinate.
        values:
            A list of two values that represent the inclusive end points
            of a range.
        tolerance:
            A relative tolerance value to ensure equivalence matching when
            using float32 values. Values of zero will be unchanged.

    Returns:
        Constraint representative of a range of values.
    """
    values = [float(i) for i in values]
    values = sorted(values)
    values[0] = (1.0 - tolerance) * values[0]
    values[1] = (1.0 + tolerance) * values[1]
    constr = iris.Constraint(
        coord_values={coord_name: lambda cell: values[0] <= cell.point <= values[1]}
    )
    return constr
