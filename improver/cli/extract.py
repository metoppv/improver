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
"""Script to extract a subset of input file data, given constraints."""

from improver import cli
from improver.cli import parameters


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            *,
            constraints: parameters.multi(min=1),
            units: cli.comma_separated_list = None,
            ignore_failure=False):
    """ Extract a subset of a single cube.

    Extracts subset of data from a single cube, subject to equality-based
    constraints.
    Using a set of constraints, extract a sub-cube from the provided cube if it
    is available.

    Args:
        cube (iris.cube.Cube):
            The Cube from which a sub-cube is extracted
        constraints (list):
            The constraint(s) to be applied. These must be of the form
            "key=value", eg "threshold=1". Multiple constraints can be provided
            by repeating this keyword before each. Scalars, boolean and string
            values are supported. Lists of values can be provided
            e.g. key=[value1, value2, value3]. Alternatively, ranges can also
            be specified e.g. key=[value1:value3].
            When a range is specified, this is inclusive of the endpoints of
            the range.
        units (list):
            List of units as strings corresponding to each coordinate in the
            list of constraints. One or more "units" may be None and units may
            only be associated with coordinate constraints. The list should be
            entered as a comma separated list without spaces, e.g. mm/hr,K.
        ignore_failure (bool):
            Option to ignore constraint match failure and return the input
            cube.

    Returns:
        iris.cube.Cube:
            A single cube matching the input constraints or None. If no
            sub-cube is found within the cube that matches the constraints.
    """
    from improver.utilities.cube_extraction import extract_subcube

    result = extract_subcube(cube, constraints, units)

    if result is None and ignore_failure:
        return cube
    if result is None:
        msg = "Constraint(s) could not be matched in input cube"
        raise ValueError(msg)
    return result
