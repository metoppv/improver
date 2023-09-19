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
"""Class for Temporal Interpolation calculations."""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.constants.time_types import TIME_COORDS
from improver.utilities.cube_manipulation import MergeCubes
from improver.utilities.round import round_close
from improver.utilities.solar import DayNightMask, calc_solar_elevation
from improver.utilities.spatial import lat_lon_determine, transform_grid_to_lat_lon
from improver.utilities.temporal import iris_time_to_datetime
from improver.wind_calculations.wind_direction import WindDirection


class TemporalInterpolation(BasePlugin):

    """
    Interpolate data to intermediate times between the validity times of two
    cubes. This can be used to fill in missing data (e.g. for radar fields) or
    to ensure data is available at the required intervals when model data is
    not available at these times.
    """

    def __init__(
        self,
        interval_in_minutes: Optional[int] = None,
        times: Optional[List[datetime]] = None,
        interpolation_method: str = "linear",
    ) -> None:
        """
        Initialise class.

        Args:
            interval_in_minutes:
                Specifies the interval in minutes at which to interpolate
                between the two input cubes. A number of minutes which does not
                divide up the interval equally will raise an exception.

                   | e.g. cube_t0 valid at 03Z, cube_t1 valid at 06Z,
                   | interval_in_minutes = 60 --> interpolate to 04Z and 05Z.

            times:
                A list of datetime objects specifying the times to which to
                interpolate.
            interpolation_method:
                Method of interpolation to use. Default is linear.
                Only methods in known_interpolation_methods can be used.

        Raises:
            ValueError: If neither interval_in_minutes nor times are set.
            ValueError: If interpolation method not in known list.
        """
        if interval_in_minutes is None and times is None:
            raise ValueError(
                "TemporalInterpolation: One of "
                "'interval_in_minutes' or 'times' must be set. "
                "Currently both are none."
            )
        if interval_in_minutes is not None and times is not None:
            raise ValueError(
                "TemporalInterpolation: Only one of "
                "'interval_in_minutes' or 'times' must be set. "
                "Currently both are set."
            )
        self.interval_in_minutes = interval_in_minutes
        self.times = times
        known_interpolation_methods = ["linear", "solar", "daynight"]
        if interpolation_method not in known_interpolation_methods:
            raise ValueError(
                "TemporalInterpolation: Unknown interpolation "
                "method {}. ".format(interpolation_method)
            )
        self.interpolation_method = interpolation_method

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = (
            "<TemporalInterpolation: interval_in_minutes: {}, " "times: {}, method: {}>"
        )
        return result.format(
            self.interval_in_minutes, self.times, self.interpolation_method
        )

    def construct_time_list(
        self, initial_time: datetime, final_time: datetime
    ) -> List[Tuple[str, List[datetime]]]:
        """
        A function to construct a list of datetime objects formatted
        appropriately for use by iris' interpolation method.

        Args:
            initial_time:
                The start of the period over which a time list is to be
                constructed.
            final_time:
                The end of the period over which a time list is to be
                constructed.

        Returns:
            A list containing a tuple that specifies the coordinate and a
            list of points along that coordinate to which to interpolate,
            as required by the iris interpolation method, e.g.::

                    [('time', [<datetime object 0>,
                               <datetime object 1>])]

        Raises:
            ValueError: If list of times provided falls outside the range
                        specified by the initial and final times.
            ValueError: If the interval_in_minutes does not divide the time
                        range up equally.
        """
        time_list = []
        if self.times is not None:
            self.times = sorted(self.times)
            if self.times[0] < initial_time or self.times[-1] > final_time:
                raise ValueError(
                    "List of times falls outside the range given by "
                    "initial_time and final_time. "
                )
            time_list = self.times
        elif self.interval_in_minutes is not None:
            if (final_time - initial_time).seconds % (
                60 * self.interval_in_minutes
            ) != 0:
                msg = (
                    "interval_in_minutes of {} does not"
                    " divide into the interval of"
                    " {} mins equally.".format(
                        self.interval_in_minutes,
                        int((final_time - initial_time).seconds / 60),
                    )
                )
                raise ValueError(msg)

            time_entry = initial_time
            while True:
                time_entry = time_entry + timedelta(minutes=self.interval_in_minutes)
                if time_entry >= final_time:
                    break
                time_list.append(time_entry)

        return [("time", time_list)]

    @staticmethod
    def enforce_time_coords_dtype(cube: Cube) -> Cube:
        """
        Enforce the data type of the time, forecast_reference_time and
        forecast_period within the cube, so that time coordinates do not
        become mis-represented. The units of the time and
        forecast_reference_time are enforced to be
        "seconds since 1970-01-01 00:00:00" with a datatype of int64.
        The units of forecast_period are enforced to be seconds with a datatype
        of int32. This functions modifies the cube in-place.

        Args:
            cube:
                The cube that will have the datatype and units for the
                time, forecast_reference_time and forecast_period coordinates
                enforced.

        Returns:
            Cube where the datatype and units for the
            time, forecast_reference_time and forecast_period coordinates
            have been enforced.
        """
        for coord_name in ["time", "forecast_reference_time", "forecast_period"]:
            coord_spec = TIME_COORDS[coord_name]
            if cube.coords(coord_name):
                coord = cube.coord(coord_name)
                coord.convert_units(coord_spec.units)
                coord.points = round_close(coord.points, dtype=coord_spec.dtype)
                if hasattr(coord, "bounds") and coord.bounds is not None:
                    coord.bounds = round_close(coord.bounds, dtype=coord_spec.dtype)
        return cube

    @staticmethod
    def calc_sin_phi(dtval: datetime, lats: ndarray, lons: ndarray) -> ndarray:
        """
        Calculate sin of solar elevation

        Args:
            dtval:
                Date and time.
            lats:
                Array 2d of latitudes for each point
            lons:
                Array 2d of longitudes for each point

        Returns:
            Array of sine of solar elevation at each point
        """
        day_of_year = (dtval - datetime(dtval.year, 1, 1)).days
        utc_hour = (dtval.hour * 60.0 + dtval.minute) / 60.0
        sin_phi = calc_solar_elevation(
            lats, lons, day_of_year, utc_hour, return_sine=True
        )
        return sin_phi

    @staticmethod
    def calc_lats_lons(cube: Cube) -> Tuple[ndarray, ndarray]:
        """
        Calculate the lats and lons of each point from a non-latlon cube,
        or output a 2d array of lats and lons, if the input cube has latitude
        and longitude coordinates.

        Args:
            cube:
                cube containing x and y axis

        Returns:
            - 2d Array of latitudes for each point.
            - 2d Array of longitudes for each point.
        """
        trg_crs = lat_lon_determine(cube)
        if trg_crs is not None:
            xycube = next(cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]))
            lats, lons = transform_grid_to_lat_lon(xycube)
        else:
            lats_row = cube.coord("latitude").points
            lons_col = cube.coord("longitude").points
            lats = np.repeat(lats_row[:, np.newaxis], len(lons_col), axis=1)
            lons = np.repeat(lons_col[np.newaxis, :], len(lats_row), axis=0)
        return lats, lons

    def solar_interpolate(self, diag_cube: Cube, interpolated_cube: Cube) -> CubeList:
        """
        Temporal Interpolation code using solar elevation for
        parameters (e.g. solar radiation parameters like
        Downward Shortwave (SW) radiation or UV index)
        which are zero if the sun is below the horizon and
        scaled by the sine of the solar elevation angle if the sun is above the
        horizon.

        Args:
            diag_cube:
                cube containing diagnostic data valid at the beginning
                of the period and at the end of the period.
            interpolated_cube:
                cube containing Linear interpolation of
                diag_cube at interpolation times in time_list.

        Returns:
            A list of cubes interpolated to the desired times.
        """

        interpolated_cubes = iris.cube.CubeList()
        (lats, lons) = self.calc_lats_lons(diag_cube)
        prev_data = diag_cube[0].data
        next_data = diag_cube[1].data
        dtvals = iris_time_to_datetime(diag_cube.coord("time"))
        # Calculate sine of solar elevation for cube valid at the
        # beginning of the period.
        dtval_prev = dtvals[0]
        sin_phi_prev = self.calc_sin_phi(dtval_prev, lats, lons)
        # Calculate sine of solar elevation for cube valid at the
        # end of the period.
        dtval_next = dtvals[1]
        sin_phi_next = self.calc_sin_phi(dtval_next, lats, lons)
        # Length of time between beginning and end in seconds
        diff_step = (dtval_next - dtval_prev).seconds

        for single_time in interpolated_cube.slices_over("time"):
            # Calculate sine of solar elevation for cube at this
            # interpolated time.
            dtval_interp = iris_time_to_datetime(single_time.coord("time"))[0]
            sin_phi_interp = self.calc_sin_phi(dtval_interp, lats, lons)
            # Length of time between beginning and interpolated time in seconds
            diff_interp = (dtval_interp - dtval_prev).seconds
            # Set all values to 0.0, to be replaced
            # with values calculated through this solar method.
            single_time.data[:] = 0.0
            sun_up = np.where(sin_phi_interp > 0.0)
            # Solar value is calculated only for points where the sun is up
            # and is a weighted combination of the data using the sine of
            # solar elevation and the data in the diag_cube valid
            # at the beginning and end.

            # If the diag_cube containing data valid at the
            # beginning of the period and at the end of the period
            # has more than x and y coordinates
            # the calculation needs to adapted to accommodate this.
            if len(single_time.shape) > 2:
                prevv = prev_data[..., sun_up[0], sun_up[1]] / sin_phi_prev[sun_up]
                nextv = next_data[..., sun_up[0], sun_up[1]] / sin_phi_next[sun_up]
                single_time.data[..., sun_up[0], sun_up[1]] = sin_phi_interp[sun_up] * (
                    prevv + (nextv - prevv) * (diff_interp / diff_step)
                )
            else:
                prevv = prev_data[sun_up] / sin_phi_prev[sun_up]
                nextv = next_data[sun_up] / sin_phi_next[sun_up]
                single_time.data[sun_up] = sin_phi_interp[sun_up] * (
                    prevv + (nextv - prevv) * (diff_interp / diff_step)
                )
            # cube with new data added to interpolated_cubes cube List.
            interpolated_cubes.append(single_time)
        return interpolated_cubes

    @staticmethod
    def daynight_interpolate(interpolated_cube: Cube) -> CubeList:
        """
        Set linearly interpolated data to zero for parameters
        (e.g. solar radiation parameters) which are zero if the
        sun is below the horizon.

        Args:
            interpolated_cube:
                cube containing Linear interpolation of
                cube at interpolation times in time_list.

        Returns:
            A list of cubes interpolated to the desired times.
        """
        daynightplugin = DayNightMask()
        daynight_mask = daynightplugin(interpolated_cube)
        index = daynight_mask.data == daynightplugin.night
        interpolated_cube.data[..., index] = 0.0
        return iris.cube.CubeList(list(interpolated_cube.slices_over("time")))

    def process(self, cube_t0: Cube, cube_t1: Cube) -> CubeList:
        """
        Interpolate data to intermediate times between validity times of
        cube_t0 and cube_t1.

        Args:
            cube_t0:
                A diagnostic cube valid at the beginning of the period within
                which interpolation is to be permitted.
            cube_t1:
                A diagnostic cube valid at the end of the period within which
                interpolation is to be permitted.

        Returns:
            A list of cubes interpolated to the desired times.

        Raises:
            TypeError: If cube_t0 and cube_t1 are not of type iris.cube.Cube.
            CoordinateNotFoundError: The input cubes contain no time
                                     coordinate.
            ValueError: Cubes contain multiple validity times.
            ValueError: The input cubes are ordered such that the initial time
                        cube has a later validity time than the final cube.
        """
        if not isinstance(cube_t0, iris.cube.Cube) or not isinstance(
            cube_t1, iris.cube.Cube
        ):
            msg = (
                "Inputs to TemporalInterpolation are not of type "
                "iris.cube.Cube, first input is type "
                "{}, second input is type {}".format(type(cube_t0), type(cube_t1))
            )
            raise TypeError(msg)

        try:
            (initial_time,) = iris_time_to_datetime(cube_t0.coord("time"))
            (final_time,) = iris_time_to_datetime(cube_t1.coord("time"))
        except CoordinateNotFoundError:
            msg = (
                "Cube provided to TemporalInterpolation contains no time " "coordinate."
            )
            raise CoordinateNotFoundError(msg)
        except ValueError:
            msg = (
                "Cube provided to TemporalInterpolation contains multiple "
                "validity times, only one expected."
            )
            raise ValueError(msg)

        if initial_time > final_time:
            raise ValueError(
                "TemporalInterpolation input cubes "
                "ordered incorrectly"
                ", with the final time being before the initial "
                "time."
            )

        time_list = self.construct_time_list(initial_time, final_time)

        # If the units of the two cubes are degrees, assume we are dealing with
        # directions. Convert the directions to complex numbers so
        # interpolations (esp. the 0/360 wraparound) are handled in a sane
        # fashion.
        if cube_t0.units == "degrees" and cube_t1.units == "degrees":
            cube_t0.data = WindDirection.deg_to_complex(cube_t0.data)
            cube_t1.data = WindDirection.deg_to_complex(cube_t1.data)

        cubes = iris.cube.CubeList([cube_t0, cube_t1])
        cube = MergeCubes()(cubes)

        interpolated_cube = cube.interpolate(time_list, iris.analysis.Linear())
        if cube_t0.units == "degrees" and cube_t1.units == "degrees":
            interpolated_cube.data = WindDirection.complex_to_deg(
                interpolated_cube.data
            )

        self.enforce_time_coords_dtype(interpolated_cube)
        interpolated_cubes = iris.cube.CubeList()
        if self.interpolation_method == "solar":
            interpolated_cubes = self.solar_interpolate(cube, interpolated_cube)
        elif self.interpolation_method == "daynight":
            interpolated_cubes = self.daynight_interpolate(interpolated_cube)
        else:
            for single_time in interpolated_cube.slices_over("time"):
                interpolated_cubes.append(single_time)


        return interpolated_cubes
