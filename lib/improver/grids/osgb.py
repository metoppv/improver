# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Module providing the OSGB Ordance Survey UK National Grid."""

import iris.coord_systems
import iris.coords
import iris.cube
import numpy as np


def _make_osgb_grid():
    """
    Create a two-dimensional Cube that represents the standard UK grid with
    a given spacing (~2km).

    Returns
    -------
    iris.cube.Cube instance
        Cube mapped to the standard UK PP grid.

    """
    # Grid resolution
    nx, ny = 548, 704

    # Grid extents / m
    north, south = 1223000, -185000
    east, west = 857000, -239000

    data = np.zeros([ny, nx])

    cs = iris.coord_systems.OSGB()
    x_coord = iris.coords.DimCoord(np.linspace(west, east, nx),
                                   'projection_x_coordinate',
                                   units='m', coord_system=cs)
    y_coord = iris.coords.DimCoord(np.linspace(south, north, ny),
                                   'projection_y_coordinate',
                                   units='m', coord_system=cs)
    x_coord.guess_bounds()
    y_coord.guess_bounds()
    cube = iris.cube.Cube(data)
    cube.add_dim_coord(y_coord, 0)
    cube.add_dim_coord(x_coord, 1)
    return cube


# OSGB UK grid
OSGBGRID = _make_osgb_grid()
