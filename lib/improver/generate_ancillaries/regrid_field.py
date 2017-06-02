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
"""Module containing temporary regridding utility for improver
   ancillary generation module"""

import iris
import numpy as np

from improver.grids.laea import UK_LAEA_GRID
from improver.grids.latlon import GLOBAL_LATLON_GRID


def regrid_field(field, grid):
    '''
    Regrids fields onto the standard grid

    Inputs
    -------
    field : cube
        cube to be regridded onto Standard_Grid
    grid : string
        the grid we wish to interpolate to

    Exceptions
    -----------
    - Raises a ValueError if NaNs are found in the field following regridding
        (this would indicate the input field domain was smaller than the
         standard grid) UNLESS: grid is global field.
    '''
    if grid == 'glm':
        field = field.regrid(STANDARD_GRIDS[grid],
                             iris.analysis.Linear())
    else:
        field = field.regrid(
            UK_LAEA_GRID, iris.analysis.Linear(extrapolation_mode='nan'))
    if np.any(np.isnan(field.data)):
        msg = 'Model domain must be larger than Standard grid domain'
        raise ValueError(msg)
    return field
