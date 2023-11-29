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
"""Script to calculate the maximum over the height coordinate"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    *,
    lower_height_bound: float = None,
    upper_height_bound: float = None,
    new_name: str =None,
):
    """Calculate the maximum value over the height coordinate of a cube. If height bounds are
    specified then the maximum value between these height levels is calculated.

    Args:
        cube (iris.cube.Cube):
            A cube with a height coordinate.
        lower_height_bound (float):
            The lower bound for the height coordinate. This is either a float or None if no lower
            bound is desired. Any specified bounds should have the same units as the height
            coordinate of cube.
        upper_height_bound (float):
            The upper bound for the height coordinate. This is either a float or None if no upper
            bound is desired. Any specified bounds should have the same units as the height
            coordinate of cube.
        new_name (str):
            The new name to be assigned to the output cube. If unspecified the name of the original
            cube is used.
    Returns:
        A cube of the maximum value over the height coordinate or maximum value between the provided
        height bounds."""

    from improver.utilities.cube_manipulation import maximum_in_height

    return maximum_in_height(
        cube,
        lower_height_bound=lower_height_bound,
        upper_height_bound=upper_height_bound,
        new_name=new_name,
    )
