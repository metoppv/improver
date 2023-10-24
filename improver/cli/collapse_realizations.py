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
"""Script to collapse the realizations dimension of a cube."""


from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube, *, method: str = "mean", new_name: str = None,
):
    """Collapse the realization dimension of a cube.

    Args:
        cube (iris.cube.Cube):
            Cube to be collapsed.
        method (str):
            One of "sum", "mean", "median", "std_dev", "min", "max".
        new_name (str):
            New name for output cube; if None use iris default.

    Returns:
        iris.cube.Cube:
            Collapsed cube. Dimensions are the same as input cube,
            without realization dimension.

    Raises:
        ValueError: if realization is not a dimension coordinate.
    """

    from improver.utilities.cube_manipulation import collapse_realizations

    if not (cube.coords("realization", dim_coords=True)):
        raise ValueError("realization must be a dimension coordinate.")

    collapsed_cube = collapse_realizations(cube, method=method)
    if new_name:
        collapsed_cube.rename(new_name)

    return collapsed_cube
