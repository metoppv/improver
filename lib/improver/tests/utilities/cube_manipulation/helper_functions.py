# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""
Frequently used helper functions for unittests for the  "cube_manipulation"
module.
"""

from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
import numpy as np


def set_up_percentile_cube(data, phenomenon_standard_name, phenomenon_units,
                           percentiles=np.array([10, 50, 90]), timesteps=1,
                           y_dimension_length=3, x_dimension_length=3):
    """
    Create a cube containing multiple percentile values
    for the coordinate.
    """
    cube = Cube(data, standard_name=phenomenon_standard_name,
                units=phenomenon_units)
    coord_long_name = "percentile_over_realization"
    cube.add_dim_coord(
        DimCoord(percentiles,
                 long_name=coord_long_name,
                 units='%'), 0)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord(np.linspace(402192.5, 402292.5, timesteps),
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, y_dimension_length),
                                "latitude", units="degrees"), 2)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, x_dimension_length),
                                "longitude", units="degrees"), 3)
    return cube


def set_up_percentile_temperature_cube():
    """ Create a cube with metadata and values suitable for air temperature."""
    data = np.array([[[[0.1, 0.1, 0.1],
                       [0.2, 0.2, 0.2],
                       [0.5, 0.5, 0.5]]],
                     [[[1.0, 1.0, 1.0],
                       [0.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5]]],
                     [[[2.0, 3.0, 4.0],
                       [0.8, 1.2, 1.6],
                       [1.5, 2.0, 3.0]]]])
    return (
        set_up_percentile_cube(data, "air_temperature", "K"))


def check_coord_type(cube, coord):
    '''Function to test whether coord is classified
       as scalar or auxiliary coordinate.

       Args:
           cube (iris.cube.Cube):
               Iris cube containing coordinates to be checked
           coord (iris.coords.DimCoord or iris.coords.AuxCoord):
               Cube coordinate to check
    '''
    coord_scalar = True
    coord_aux = False
    cube_summary = cube.summary()
    aux_ind = cube_summary.find("Auxiliary")
    if coord in cube_summary[aux_ind:]:
        coord_scalar = False
        coord_aux = True
    return coord_scalar, coord_aux
