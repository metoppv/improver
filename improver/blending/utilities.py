# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Utilities to support weighted blending"""


def find_blend_dim_coord(cube, blend_coord):
    """
    Find the name of the dimension coordinate across which to perform the blend,
    since the input "blend_coord" may be an auxiliary coordinate.

    Args:
        cube (iris.cube.Cube):
            Cube to be blended
        blend_coord (str):
            Name of coordinate to blend over

    Returns:
        str:
            Name of dimension coordinate associated with blend dimension

    Raises:
        ValueError:
            If blend coordinate is associated with more or less than one dimension
    """
    blend_dim = cube.coord_dims(blend_coord)
    if len(blend_dim) != 1:
        if len(blend_dim) < 1:
            msg = f"Blend coordinate {blend_coord} has no associated dimension"
        else:
            msg = (
                "Blend coordinate must only be across one dimension. Coordinate "
                f"{blend_coord} is associated with dimensions {blend_dim}."
            )
        raise ValueError(msg)

    return cube.coord(dimensions=blend_dim[0], dim_coords=True).name()
