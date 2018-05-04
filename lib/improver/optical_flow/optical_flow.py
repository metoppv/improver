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
This module defines the optical flow velocity calculation and extrapolation
classes for advection nowcasting of precipitation fields.
"""
import numpy as np
from iris.exceptions import InvalidCubeError
from iris.exceptions import CoordinateNotFoundError

from improver.utilities.cube_checker import check_for_x_and_y_axes


class AdvectField(object):
    """
    Class to advect a 2D spatial field given velocities along the two vector
    dimensions
    """

    def __init__(self, vel_x, vel_y):
        """
        Initialises the plugin.  Velocities are expected to be on a regular
        grid (such that grid spacing in metres is the same at all points in
        the domain).

        Args:
            vel_x (iris.cube.Cube):
                Cube containing a 2D array of velocities along the x
                coordinate axis
            vel_y (iris.cube.Cube):
                Cube containing a 2D array of velocities along the y
                coordinate axis
        """

        # check each input velocity cube has precisely two non-scalar
        # dimension coordinates (spatial x/y)
        self._check_input_coords(vel_x)
        self._check_input_coords(vel_y)

        # check input velocity cubes have the same spatial coordinates
        if (vel_x.coord(axis="x") != vel_y.coord(axis="x") or
                vel_x.coord(axis="y") != vel_y.coord(axis="y")):
            raise InvalidCubeError("Velocity cubes on unmatched grids")

        vel_x.convert_units('m s-1')
        vel_y.convert_units('m s-1')

        self.vel_x = vel_x
        self.vel_y = vel_y

        self.x_coord = vel_x.coord(axis="x")
        self.y_coord = vel_x.coord(axis="y")

    @staticmethod
    def _check_input_coords(cube, require_time=None):
        """
        Checks an input cube has precisely two non-scalar dimension coordinates
        (spatial x/y), or raises an error.  If "require_time" is set to True,
        raises an error if no scalar time coordinate is present.

        Args:
            cube (iris.cube.Cube):
                Cube to be checked
            require_time (bool):
                Flag to check for a scalar time coordinate

        Raises:
            InvalidCubeError if coordinate requirements are not met
        """
        # check that cube has both x and y axes
        try:
            check_for_x_and_y_axes(cube)
        except ValueError as msg:
            raise InvalidCubeError(msg)

        # check that cube data has only two non-scalar dimensions
        data_shape = np.array(cube.shape)
        non_scalar_coords = np.sum(np.where(data_shape > 1, 1, 0))
        if non_scalar_coords > 2:
            raise InvalidCubeError('Cube has {:d} (more than 2) non-scalar '
                                   'coordinates'.format(non_scalar_coords))

        if require_time:
            try:
                _ = cube.coord("time")
            except CoordinateNotFoundError:
                raise InvalidCubeError('Input cube has no time coordinate')

    @staticmethod
    def _increment_output_array(indata, outdata, cond, xdest_grid, ydest_grid,
                                xsrc_grid, ysrc_grid, x_weight, y_weight):
        """
        Calculate and add contribution to the advected array from one source
        grid point, for all points where boolean condition "cond" is valid.

        Args:
            indata (numpy.ndarray):
                2D numpy array of source data to be advected
            outdata (numpy.ndarray):
                2D numpy array for advected output, modified in place by
                this method (is both input and output).
            cond (numpy.ndarray):
                2D boolean mask of points to be processed
            xdest_grid (numpy.ndarray):
                Integer x-coordinates of all points on destination grid
            ydest_grid (numpy.ndarray):
                Integer y-coordinates of all points on destination grid
            xsrc_grid (numpy.ndarray):
                Integer x-coordinates of all points on source grid
            ysrc_grid (numpy.ndarray):
                Integer y-coordinates of all points on source grid
            x_weight (numpy.ndarray):
                Fractional contribution to destination grid of source data
                advected along the x-axis.  Positive definite.
            y_weight (numpy.ndarray):
                Fractional contribution to destination grid of source data
                advected along the y-axis.  Positive definite.
        """
        xdest = xdest_grid[cond]
        ydest = ydest_grid[cond]
        xsrc = xsrc_grid[cond]
        ysrc = ysrc_grid[cond]
        outdata[ydest, xdest] += (
            indata[ysrc, xsrc]*x_weight[ydest, xdest]*y_weight[ydest, xdest])

    def _advect_field(self, data, grid_vel_x, grid_vel_y, timestep, bgd):
        """
        Performs a dimensionless grid-based extrapolation of spatial data
        using advection velocities via a backwards method.

        Args:
            data (numpy.ndarray):
                2D numpy data array to be advected
            grid_vel_x (numpy.ndarray):
                Velocity in the x direction (in grid points per second)
            grid_vel_y (numpy.ndarray):
                Velocity in the y direction (in grid points per second)
            timestep (int):
                Advection time step in seconds
            bgd (float):
                Default output value for spatial points where data cannot be
                extrapolated (source is out of bounds)

        Returns:
            adv_field (numpy.ndarray):
                2D float array of advected data values
        """

        # Initialise advected field with "background" default value
        adv_field = np.full(data.shape, bgd)

        # Set up grids of data coordinates (meshgrid inverts coordinate order)
        ydim, xdim = data.shape
        (xgrid, ygrid) = np.meshgrid(np.arange(xdim),
                                     np.arange(ydim))

        # For each grid point on the output field, trace its (x,y) "source"
        # location backwards using advection velocities.  The source location
        # is generally fractional: eg with advection velocities of 0.5 grid
        # squares per second, the value at [2, 2] is represented by the value
        # that was at [1.5, 1.5] 1 second ago.
        xsrc_point_frac = -grid_vel_x * timestep + xgrid.astype(float)
        ysrc_point_frac = -grid_vel_y * timestep + ygrid.astype(float)

        # For all the points where fractional source coordinates are within
        # the bounds of the field, set the output field to 0
        def point_in_bounds(x, y, nx, ny):
            """Check point (y, x) lies within defined bounds"""
            return (x >= 0.) & (x < nx) & (y >= 0.) & (y < ny)

        cond_pt = point_in_bounds(xsrc_point_frac, ysrc_point_frac, xdim, ydim)
        adv_field[cond_pt] = 0

        # Find the integer points surrounding the fractional source coordinates
        xsrc_point_lower = xsrc_point_frac.astype(int)
        ysrc_point_lower = ysrc_point_frac.astype(int)
        x_points = [xsrc_point_lower, xsrc_point_lower + 1]
        y_points = [ysrc_point_lower, ysrc_point_lower + 1]

        # Calculate the distance-weighted fractional contribution of points
        # surrounding the source coordinates
        x_weight_upper = xsrc_point_frac - xsrc_point_lower.astype(float)
        y_weight_upper = ysrc_point_frac - ysrc_point_lower.astype(float)
        x_weights = [1. - x_weight_upper, x_weight_upper]
        y_weights = [1. - y_weight_upper, y_weight_upper]

        # Advect data from each of the four source points onto the output grid
        for xpt, xwt in zip(x_points, x_weights):
            for ypt, ywt in zip(y_points, y_weights):
                cond = point_in_bounds(xpt, ypt, xdim, ydim) & cond_pt
                self._increment_output_array(data, adv_field, cond, xgrid,
                                             ygrid, xpt, ypt, xwt, ywt)

        return adv_field

    def process(self, cube, timestep, bgd=0.0):
        """
        Extrapolates input cube data and updates validity time.  The input
        cube should have precisely two non-scalar dimension coordinates
        (spatial x/y), and is expected to be in a projection such that grid
        spacing is the same (or very close) at all points within the spatial
        domain.  The input cube should also have a "time" coordinate.

        Args:
            cube (iris.cube.Cube):
                The 2D cube containing data to be advected
            timestep (datetime.timedelta):
                Advection time step
            bgd (float):
                Default output value for spatial points where data cannot be
                extrapolated (source is out of bounds)

        Returns:
            advected_cube (iris.cube.Cube):
                New cube with updated time and extrapolated data
        """
        # check that the input cube has precisely two non-scalar dimension
        # coordinates (spatial x/y) and a scalar time coordinate
        self._check_input_coords(cube, require_time=True)

        # check spatial coordinates match those of plugin velocities
        if (cube.coord(axis="x") != self.x_coord or
                cube.coord(axis="y") != self.y_coord):
            raise InvalidCubeError("Input data grid does not match advection "
                                   "velocities")

        # derive velocities in "grid squares per second"
        def grid_spacing(coord):
            """Calculate grid spacing along a given spatial axis"""
            new_coord = coord.copy()
            new_coord.convert_units('m')
            return float(np.diff((new_coord).points)[0])

        grid_vel_x = self.vel_x.data / grid_spacing(cube.coord(axis="x"))
        grid_vel_y = self.vel_y.data / grid_spacing(cube.coord(axis="y"))

        # perform advection and create output cube
        advected_data = self._advect_field(cube.data, grid_vel_x, grid_vel_y,
                                           timestep.total_seconds(), bgd)
        advected_cube = cube.copy(data=advected_data)

        # increment output cube time
        original_datetime, = \
            (cube.coord("time").units).num2date(cube.coord("time").points)
        new_datetime = original_datetime + timestep
        new_time = (cube.coord("time").units).date2num(new_datetime)
        advected_cube.coord("time").points = new_time

        return advected_cube
