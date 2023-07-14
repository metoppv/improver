#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""CLI to enforce the sum total of a set of forecasts to be equal to a reference."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcubelist,
    reference_name: str,
    return_name: str,
    ignore_zero_total: bool = False,
):
    """Module to enforce that the sum of data in a list of cubes is equal to the
    corresponding data in a reference cube. Only one of the updated cubes is returned.

    The data are updated as follows, if there are 2 cubes to be normalised containing
    data points a and b respectively with corresponding reference r, then:

    a_new = r * (a / (a + b))
    b_new = r * (b / (a + b))

    which ensures that r = a_new + b_new.

    Args:
        cubes (List of iris.cube.Cube): A list of cubes containing both the cubes to be
            updated and the reference cube. The reference cube will be identified by
            matching the cube name to reference_name.
        reference_name (str): The name of the reference cube, this must match exactly
            one cube in cubes.
        return_name (str): The name of the cube to be returned. this must match exactly
            one cube in cubes.
        ignore_zero_total (bool): Determines whether an error will be raised in the
            instance where the total of non-reference cubes is zero but the
            corresponding value in reference is non-zero. If True this error will not be
            raised - instead the data in the updated cubes will remain as zero values,
            if False an error will be raised.

    Returns:
        iris.cube.Cube:
            The cube in cubes with name matching return_name. The data in the cube will
            have been updated, but the metadata will be identical.

    """
    from improver.utilities.flatten import flatten
    from improver.utilities.forecast_reference_enforcement import (
        normalise_to_reference,
        split_cubes_by_name,
    )

    cubes = flatten(cubes)

    reference_cube, input_cubes = split_cubes_by_name(cubes, reference_name)

    if len(reference_cube) == 1:
        reference_cube = reference_cube[0]
    else:
        msg = (
            f"Exactly one cube with a name matching reference_name is required, but "
            f"{len(reference_cube)} were found."
        )
        raise ValueError(msg)

    output = normalise_to_reference(input_cubes, reference_cube, ignore_zero_total)

    output = output.extract_cube(return_name)

    return output
