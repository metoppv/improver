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
"""Script to combine netcdf data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube,
            operation='+',
            new_name=None,
            use_midpoint=False,
            check_metadata=False):
    r"""Combine input cubes.

    Combine the input cubes into a single cube using the requested operation.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            An iris CubeList to be combined.
        operation (str):
            An operation to use in combining input cubes. One of:
            +, -, \*, add, subtract, multiply, min, max, mean
        new_name (str):
            New name for the resulting dataset.
        use_midpoint (bool):
            If False (not set), uses the upper bound as the new coordinate
            point for expanded coordinates (eg time for accumulations / max in
            period).  If True, uses the mid-point.
        check_metadata (bool):
            If True, warn on metadata mismatch between inputs.

    Returns:
        result (iris.cube.Cube):
            Returns a cube with the combined data.
    """
    from improver.cube_combiner import CubeCombiner
    from iris.cube import CubeList
    if not cubes:
        raise TypeError("A cube is needed to be combined.")
    if new_name is None:
        new_name = cubes[0].name()
    result = CubeCombiner(operation, warnings_on=check_metadata).process(
        CubeList(cubes), new_name, use_midpoint=use_midpoint)

    return result
