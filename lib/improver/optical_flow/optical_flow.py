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
This module defines optical flow velocity calculation and extrapolation
classes for advection nowcasting of precipitation fields.
"""

import iris
import numpy as np
from improver.utilities.cube_checker import find_dimension_coordinate_mismatch


class AdvectField(object):
    """
    Class to advect a 2D spatial field given velocities along the two vector
    dimensions
    """

    def __init__(self, vel_x, vel_y):

        """
        Initialises the plugin TODO velocity units in "time steps"

        Args:
            vel_x (iris.cube.Cube):
                Cube containing a 2D array of velocities along the x 
                coordinate axis
            vel_y (numpy.ndarray):
                Cube containing a 2D array of velocities along the y
                coordinate axis
        """

        # check input velocity cubes have the same coordinates TODO error type?
        if find_dimension_coordinate_mismatch(vel_x, vel_y):
            raise ValueError('Velocity cubes on unmatched grids')

        self.vel_x = vel_x
        self.vel_y = vel_y

        self.x_coord = vel_x.coord("x")
        self.y_coord = vel_x.coord("y")


    def process(self, cube, dt=1, bgd=0.0):

        """
        Extrapolates spatial data from an input cube using advection
        velocities.  Performs "backwards" advection TODO update with
        details.

        Args:
            cube (iris.cube.Cube):
                The cube containing data to be advected
            dt (int):
                Advection time step TODO units
            bgd (float):
                ??? TODO find out!

        Returns:
            advected_cube (iris.cube.Cube):
                New cube with updated time and extrapolated data
        """

        # check coordinates TODO error type?
        if cube.coord("x") != self.x_coord or cube.coord("y") != self.y_coord:
            raise ValueError('Input data grid does not match advection '
                             'velocities')

        adv_field = np.zeros(cube.data.shape)

        # copied from Martina's code
        # TODO basic unit tests, then refactor with eg sensible treatment of time coordinates
        ydim, xdim = cube.data.shape
        (ygrid, xgrid) = np.meshgrid(np.arange(xdim),
                                    np.arange(ydim))
        oldx_frac = -self.vel_x.data * dt + xgrid.astype(float)
        oldy_frac = -self.vel_y.data * dt + ygrid.astype(float)
        adv_field = np.full(cube.data.shape, bgd)
        cond1 = (oldx_frac >= 0.) & (oldy_frac >= 0.) & (
                oldx_frac < ydim) & (oldy_frac < xdim)
        adv_field[cond1] = 0
        oldx_l = oldx_frac.astype(int)
        oldx_r = oldx_l + 1
        x_frac_r = oldx_frac - oldx_l.astype(float)
        oldy_u = oldy_frac.astype(int)
        oldy_d = oldy_u + 1
        y_frac_d = oldy_frac - oldy_u.astype(float)
        cond2 = ((oldx_l >= 0) & (oldy_u >= 0) & (oldx_l < ydim) &
                (oldy_u < xdim) & cond1)
        cond3 = ((oldx_r >= 0) & (oldy_u >= 0) & (oldx_r < ydim) &
                (oldy_u < xdim) & cond1)
        cond4 = ((oldx_l >= 0) & (oldy_d >= 0) & (oldx_l < ydim) &
                (oldy_d < xdim) & cond1)
        cond5 = ((oldx_r >= 0) & (oldy_d >= 0) & (oldx_r < ydim) &
                (oldy_d < xdim) & cond1)
        for ii, cond in enumerate([cond2, cond3, cond4, cond5], 2):
            xorig = xgrid[cond]
            yorig = ygrid[cond]
            if ii == 2:
                xfr = 1.-x_frac_r
                yfr = 1.-y_frac_d
                xc = oldx_l[cond]
                yc = oldy_u[cond]
            elif ii == 3:
                xfr = x_frac_r
                yfr = 1. - y_frac_d
                xc = oldx_r[cond]
                yc = oldy_u[cond]
            elif ii == 4:
                xfr = 1.-x_frac_r
                yfr = y_frac_d
                xc = oldx_l[cond]
                yc = oldy_d[cond]
            elif ii == 5:
                xfr = x_frac_r
                yfr = y_frac_d
                xc = oldx_r[cond]
                yc = oldy_d[cond]
            adv_field[xorig, yorig] = (
                adv_field[xorig, yorig] + cube.data[xc, yc] *
                xfr[xorig, yorig]*yfr[xorig, yorig])

        adv_cube = iris.cube.Cube(data=adv_field)
        
        # TODO update time coordinate on advected cube

        return adv_cube
