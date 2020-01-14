#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Module for helping to create Iris constraints."""

import iris


def create_sorted_lambda_constraint(coord_name, values, tolerance=1.0E-7):
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
        coord_name (str):
            Name of the coordinate.
        values (list):
            A list of two values that represent the inclusive end points
            of a range.
        tolerance (float):
            A relative tolerance value to ensure equivalence matching when
            using float32 values. Values of zero will be unchanged.

    Returns:
        iris.Constraint:
            Constraint representative of a range of values.

    """
    values = [float(i) for i in values]
    values = sorted(values)
    values[0] = (1. - tolerance) * values[0]
    values[1] = (1. + tolerance) * values[1]
    constr = (
        iris.Constraint(
            coord_values={
                coord_name: lambda cell: values[0] <= cell <= values[1]}))
    return constr
