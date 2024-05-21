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
from typing import Union

from iris.cube import Cube, CubeList

from improver.utilities.flatten import flatten


def as_cubelist(*cubes: Union[Cube, CubeList]):
    """
    Standardise input handling of cube/cubelist arguments.

    The role of this function is to flatten the provided inputs and thereby
    return a single CubeList object.

    Args:
        cubes:
            Input data provided in the form of one or more cubes or cubelists (or mixture thereof).

    Returns:
        CubeList:
            A CubeList containing all the cubes provided as input.
    """
    if not cubes or all([not cube for cube in cubes]):
        raise ValueError("One or more cubes should be provided.")
    cubes = CubeList(flatten(cubes))
    # Remove CubeList verification for iris >=3.3.0
    for cube in cubes:
        if not hasattr(cube, "add_aux_coord"):
            raise TypeError("CubeList contains a non iris Cube object.")
    return cubes


def as_cube(cube: Union[Cube, CubeList]):
    """
    Standardise input handling of cube arguments.

    The role of this function is to return a single cube object.

    Args:
        cube:
            Input data provided in the form of a cube or cubelists.

    Returns:
        Cube:
            A single cube.
    """
    if not cube:
        raise ValueError("A cube should be provided.")
    if isinstance(cube, CubeList):
        if len(cube) > 1:
            raise ValueError("A single cube should be provided.")
        cube = cube[0]
    if not isinstance(cube, Cube):
        raise TypeError("A cube should be provided.")
    return cube
