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
"""Functions to set up cubes for use in weighting-related unit tests."""

from cf_units import Unit

import iris
from iris.coords import DimCoord
from iris.cube import Cube

import numpy as np

def set_up_cube():
    """A helper function to set up input cubes for unit tests.
       The cube has latitude, longitude and time dimensions

    Returns:
        cube : iris.cube.Cube
                dummy cube for testing

    """
    data = np.zeros((2, 2, 2))

    orig_cube = Cube(data, units="m",
                     standard_name="lwe_thickness_of_precipitation_amount")
    orig_cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2),
                                     'latitude', units='degrees'), 1)
    orig_cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                     units='degrees'), 2)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    orig_cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                     "time", units=tunit), 0)
    orig_cube.add_aux_coord(DimCoord([0, 1],
                                     "forecast_period", units="hours"), 0)
    return orig_cube


def set_up_precipitation_cube():
    """Set up a precipitation cube."""
    cube = set_up_cube()
    data = np.zeros((2, 2, 2))
    data[0][:][:] = 1.0
    data[1][:][:] = 2.0
    cube.data = data
    return cube


def cubes_for_tests():
    """Set up cubes for unit tests."""
    cube = set_up_weights_cube()
    forecast_period = 0
    constr = iris.Constraint(forecast_period=forecast_period)
    central_cube = cube.extract(constr)
    return cube, central_cube, forecast_period
