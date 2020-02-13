# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
This module defines plugins used to create nowcast extrapolation forecasts.
"""
import datetime
import warnings

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.exceptions import CoordinateNotFoundError, InvalidCubeError

from improver import BasePlugin
from improver.metadata.amend import amend_attributes, set_history_attribute
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.utilities import (
    create_new_diagnostic_cube, generate_mandatory_attributes)
from improver.nowcasting.optical_flow import check_input_coords
from improver.nowcasting.utilities import ApplyOrographicEnhancement


class AdvectField(BasePlugin):
    """
    Class to advect a 2D spatial field given velocities along the two vector
    dimensions
    """

    def __init__(self, vel_x, vel_y, attributes_dict=None):
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
            attributes_dict (dict):
                Dictionary containing information for amending the attributes
                of the output cube.
        """

        # check each input velocity cube has precisely two non-scalar
        # dimension coordinates (spatial x/y)
        check_input_coords(vel_x)
        check_input_coords(vel_y)

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

        # Initialise metadata dictionary.
        if attributes_dict is None:
            attributes_dict = {}
        self.attributes_dict = attributes_dict

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = ('<AdvectField: vel_x={}, vel_y={}, '
                  'attributes_dict={}>'.format(
                      repr(self.vel_x), repr(self.vel_y),
                      self.attributes_dict))
        return result

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

    def _advect_field(self, data, grid_vel_x, grid_vel_y, timestep):
        """
        Performs a dimensionless grid-based extrapolation of spatial data
        using advection velocities via a backwards method.  Points where data
        cannot be extrapolated (ie the source is out of bounds) are given a
        fill value of np.nan and masked.

        Args:
            data (numpy.ndarray or numpy.ma.MaskedArray):
                2D numpy data array to be advected
            grid_vel_x (numpy.ndarray):
                Velocity in the x direction (in grid points per second)
            grid_vel_y (numpy.ndarray):
                Velocity in the y direction (in grid points per second)
            timestep (int):
                Advection time step in seconds

        Returns:
            numpy.ma.MaskedArray:
                2D float array of advected data values with masked "no data"
                regions
        """
        # Cater for special case where timestep (int) is 0
        if timestep == 0:
            return data

        # Initialise advected field with np.nan
        adv_field = np.full(data.shape, np.nan, dtype=np.float32)

        # Set up grids of data coordinates (meshgrid inverts coordinate order)
        ydim, xdim = data.shape
        (xgrid, ygrid) = np.meshgrid(np.arange(xdim),
                                     np.arange(ydim))

        # For each grid point on the output field, trace its (x,y) "source"
        # location backwards using advection velocities.  The source location
        # is generally fractional: eg with advection velocities of 0.5 grid
        # squares per second, the value at [2, 2] is represented by the value
        # that was at [1.5, 1.5] 1 second ago.
        xsrc_point_frac = -grid_vel_x * timestep + xgrid.astype(np.float32)
        ysrc_point_frac = -grid_vel_y * timestep + ygrid.astype(np.float32)

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
        x_weights = np.array([1. - x_weight_upper, x_weight_upper],
                             dtype=np.float32)
        y_weights = np.array([1. - y_weight_upper, y_weight_upper],
                             dtype=np.float32)

        # Check whether the input data is masked - if so substitute NaNs for
        # the masked data.  Note there is an implicit type conversion here: if
        # data is of integer type this unmasking will convert it to float.
        if isinstance(data, np.ma.MaskedArray):
            data = np.where(data.mask, np.nan, data.data)

        # Advect data from each of the four source points onto the output grid
        for xpt, xwt in zip(x_points, x_weights):
            for ypt, ywt in zip(y_points, y_weights):
                cond = point_in_bounds(xpt, ypt, xdim, ydim) & cond_pt
                self._increment_output_array(
                    data, adv_field, cond, xgrid, ygrid, xpt, ypt, xwt, ywt)

        # Replace NaNs with a mask
        adv_field = np.ma.masked_where(~np.isfinite(adv_field), adv_field)

        return adv_field

    @staticmethod
    def _update_time(input_time, advected_cube, timestep):
        """Increment validity time on the advected cube

        Args:
            input_time (iris.coords.Coord):
                Time coordinate from source cube
            advected_cube (iris.cube.Cube):
                Cube containing advected data (modified in place)
            timestep (datetime.timedelta)
                Time difference between the advected output and the source
        """
        original_datetime = next(input_time.cells())[0]
        new_datetime = original_datetime + timestep
        new_time = input_time.units.date2num(new_datetime)
        time_coord_name = "time"
        time_coord_spec = TIME_COORDS[time_coord_name]
        time_coord = advected_cube.coord(time_coord_name)
        time_coord.points = new_time
        time_coord.convert_units(time_coord_spec.units)
        time_coord.points = np.around(time_coord.points).astype(
            time_coord_spec.dtype)

    @staticmethod
    def _add_forecast_reference_time(input_time, advected_cube):
        """Add or replace a forecast reference time on the advected cube"""
        try:
            advected_cube.remove_coord("forecast_reference_time")
        except CoordinateNotFoundError:
            pass
        frt_coord_name = "forecast_reference_time"
        frt_coord_spec = TIME_COORDS[frt_coord_name]
        frt_coord = input_time.copy()
        frt_coord.rename(frt_coord_name)
        frt_coord.convert_units(frt_coord_spec.units)
        frt_coord.points = np.around(frt_coord.points).astype(
            frt_coord_spec.dtype)
        advected_cube.add_aux_coord(frt_coord)

    @staticmethod
    def _add_forecast_period(advected_cube, timestep):
        """Add or replace a forecast period on the advected cube"""
        try:
            advected_cube.remove_coord("forecast_period")
        except CoordinateNotFoundError:
            pass

        forecast_period_seconds = np.int32(timestep.total_seconds())
        forecast_period_coord = AuxCoord(forecast_period_seconds,
                                         standard_name="forecast_period",
                                         units="seconds")
        advected_cube.add_aux_coord(forecast_period_coord)

    def _create_output_cube(self, cube, advected_data, timestep):
        """
        Create a cube and appropriate metadata to contain the advected forecast

        Args:
            cube (iris.cube.Cube):
                Source cube (before advection)
            advected_data (numpy.ndarray):
                Advected data
            timestep (datetime.timedelta):
                Time difference between the advected output and the source

        Returns:
            iris.cube.Cube
        """
        attributes = generate_mandatory_attributes([cube])
        if "institution" in cube.attributes.keys():
            attributes["source"] = "{} Nowcast".format(
                attributes["institution"])
        else:
            attributes["source"] = "Nowcast"
        advected_cube = create_new_diagnostic_cube(
            cube.name(), cube.units, cube, attributes, data=advected_data)
        amend_attributes(advected_cube, self.attributes_dict)
        set_history_attribute(advected_cube, "Nowcast")

        self._update_time(
            cube.coord("time").copy(), advected_cube, timestep)
        self._add_forecast_reference_time(
            cube.coord("time").copy(), advected_cube)
        self._add_forecast_period(advected_cube, timestep)

        return advected_cube

    def process(self, cube, timestep):
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

        Returns:
            iris.cube.Cube:
                New cube with updated time and extrapolated data.  New data
                are filled with np.nan and masked where source data were
                out of bounds (ie where data could not be advected from outside
                the cube domain).

        """
        # check that the input cube has precisely two non-scalar dimension
        # coordinates (spatial x/y) and a scalar time coordinate
        check_input_coords(cube, require_time=True)

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
            return np.float32(np.diff((new_coord).points)[0])

        grid_vel_x = self.vel_x.data / grid_spacing(cube.coord(axis="x"))
        grid_vel_y = self.vel_y.data / grid_spacing(cube.coord(axis="y"))

        # raise a warning if data contains unmasked NaNs
        nan_count = np.count_nonzero(~np.isfinite(cube.data))
        if nan_count > 0:
            warnings.warn("input data contains unmasked NaNs")

        # perform advection and create output cube
        advected_data = self._advect_field(cube.data, grid_vel_x, grid_vel_y,
                                           round(timestep.total_seconds()))
        advected_cube = self._create_output_cube(cube, advected_data, timestep)
        return advected_cube


class CreateExtrapolationForecast(BasePlugin):
    """
    Class to create a nowcast extrapolation forecast using advection.
    For precipitation rate forecasts, orographic enhancement must be used.
    """

    def __init__(self, input_cube, vel_x, vel_y,
                 orographic_enhancement_cube=None, attributes_dict=None):
        """
        Initialises the object.
        This includes checking if orographic enhancement is provided and
        removing the orographic enhancement from the input file ready for
        extrapolation.
        An error is raised if the input cube is precipitation rate but no
        orographic enhancement cube is provided.

        Args:
            input_cube (iris.cube.Cube):
                A 2D cube containing data to be advected.
            vel_x (iris.cube.Cube):
                Cube containing a 2D array of velocities along the x
                coordinate axis
            vel_y (iris.cube.Cube):
                Cube containing a 2D array of velocities along the y
                coordinate axis
            orographic_enhancement_cube (iris.cube.Cube):
                Cube containing the orographic enhancement fields. May have
                data for multiple times in the cube. The orographic enhancement
                is removed from the input_cube before advecting, and added
                back on after advection.
            attributes_dict (dict):
                Dictionary containing information for amending the attributes
                of the output cube.
        """
        if not (vel_x and vel_y):
            raise TypeError("Neither x velocity or y velocity can be None")

        self.orographic_enhancement_cube = orographic_enhancement_cube
        if self.orographic_enhancement_cube:
            input_cube, = ApplyOrographicEnhancement("subtract").process(
                input_cube, self.orographic_enhancement_cube)
        elif ("precipitation_rate" in input_cube.name()
              or "rainfall_rate" in input_cube.name()):
            msg = ("For precipitation or rainfall fields, orographic "
                   "enhancement cube must be supplied.")
            raise ValueError(msg)
        self.input_cube = input_cube
        self.advection_plugin = AdvectField(
            vel_x, vel_y, attributes_dict=attributes_dict)

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = ('<CreateExtrapolationForecast: input_cube = {}, '
                  'orographic_enhancement_cube = {}, '
                  'advection_plugin = {}>'.format(
                      repr(self.input_cube),
                      repr(self.orographic_enhancement_cube),
                      repr(self.advection_plugin)))
        return result

    def extrapolate(self, leadtime_minutes):
        """
        Produce a new forecast cube for the supplied lead time. Creates a new
        advected forecast and then reapplies the orographic enhancement if it
        is supplied.

        Args:
            leadtime_minutes (float):
                The forecast leadtime we want to generate a forecast for
                in minutes.

        Returns:
            iris.cube.Cube:
                New cube with updated time and extrapolated data.  New data
                are filled with np.nan and masked where source data were
                out of bounds (ie where data could not be advected from outside
                the cube domain).

        Raises:
            ValueError: If no leadtime_minutes are provided.
        """
        # cast to float as datetime.timedelta cannot accept np.int
        timestep = datetime.timedelta(minutes=float(leadtime_minutes))
        forecast_cube = self.advection_plugin.process(
            self.input_cube, timestep)
        if self.orographic_enhancement_cube:
            # Add orographic enhancement.
            forecast_cube, = ApplyOrographicEnhancement("add").process(
                forecast_cube, self.orographic_enhancement_cube)

        return forecast_cube

    def process(self, interval, max_lead_time):
        """
        Generate nowcasts at required intervals up to the maximum lead time

        Args:
            interval (int):
                Lead time interval, in minutes
            max_lead_time (int):
                Maximum lead time required, in minutes

        Returns:
            iris.cube.CubeList:
                List of forecast cubes at the required lead times
        """
        lead_times = np.arange(0, max_lead_time + 1, interval)
        forecast_cubes = iris.cube.CubeList()
        for lead_time in lead_times:
            forecast_cubes.append(self.extrapolate(lead_time))
        return forecast_cubes
