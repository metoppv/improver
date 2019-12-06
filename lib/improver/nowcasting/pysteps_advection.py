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
"""Semi-Lagrangian backward advection plugin using pysteps"""

import numpy as np
from datetime import timedelta

from iris.coords import AuxCoord

from improver.metadata.amend import (amend_attributes, set_history_attribute)
from improver.utilities.spatial import (
    check_if_grid_is_equal_area, calculate_grid_spacing)
from improver.utilities.temporal import (
    iris_time_to_datetime, datetime_to_iris_time)
from improver.nowcasting.utilities import ApplyOrographicEnhancement
from improver.utilities.stdout_trap import Capture_Stdout

# PySteps prints a message on import to stdout - trap this
# This should be removed for PySteps v1.1.0 which has a configuration setting
# for this
with Capture_Stdout() as _:
    from pysteps.extrapolation.semilagrangian import extrapolate


class PystepsExtrapolate(object):
    """Wrapper for the pysteps semi-Lagrangian extrapolation method

    Reference:
        https://pysteps.readthedocs.io/en/latest/generated/
        pysteps.extrapolation.semilagrangian.extrapolate.html
    """
    def __init__(self, interval, max_lead_time):
        """
        Initialise the plugin

        Args:
            interval (int):
                Lead time interval, in minutes
            max_lead_time (int):
                Maximum lead time required, in minutes
        """
        self.interval = interval
        self.num_timesteps = max_lead_time // interval

    def _get_precip_rate(self):
        """
        From the initial cube, generate a precipitation rate array in mm h-1
        with orographic enhancement subtracted, as required for advection

        Returns:
            np.ndarray:
                2D precipitation rate array in mm h-1
        """
        if self.orogenh:
            self.analysis_cube, = ApplyOrographicEnhancement(
                "subtract").process(self.analysis_cube, self.orogenh)
        elif "precipitation_rate" in self.analysis_cube.name():
            msg = ("For precipitation fields, orographic enhancement "
                   "cube must be supplied.")
            raise ValueError(msg)
        self.analysis_cube.convert_units('mm h-1')
        return np.ma.filled(self.analysis_cube.data, np.nan)

    def _generate_displacement_array(self, ucube, vcube):
        """
        Create displacement array of shape (2 x m x n) required by pysteps
        algorithm

        Args:
            ucube (iris.cube.Cube):
                Cube of x-advection velocities
            vcube (iris.cube.Cube):
                Cube of y-advection velocities

        Returns:
            displacement (np.ndarray):
                Array of shape (2, m, n) containing the x- and y-components
                of the m*n displacement field (format required for pysteps
                extrapolation algorithm)
        """
        def _calculate_displacement(cube, interval, gridlength):
            """
            Calculate displacement for each time step using velocity cube and
            time interval

            Args:
                cube (iris.cube.Cube):
                    Cube of velocities in the x or y direction
                interval (int):
                    Lead time interval, in minutes
                gridlength (float):
                    Size of grid square, in metres

            Returns:
                np.ndarray:
                    Array of displacements in grid squares per time step
            """
            cube_ms = cube.copy()
            cube_ms.convert_units('m s-1')
            displacement = cube_ms.data*interval*60. / gridlength
            return np.ma.filled(displacement, np.nan)

        gridlength = calculate_grid_spacing(self.analysis_cube, 'metres')
        udisp = _calculate_displacement(ucube, self.interval, gridlength)
        vdisp = _calculate_displacement(vcube, self.interval, gridlength)
        displacement = np.array([udisp, vdisp])
        return displacement

    def _reformat_analysis_cube(self):
        """
        Add forecast reference time and forecast period coordinates (if they do
        not already exist) and nowcast attributes to analysis cube
        """
        coords = [coord.name() for coord in self.analysis_cube.coords()]
        if "forecast_reference_time" not in coords:
            frt_coord = self.analysis_cube.coord('time').copy()
            frt_coord.rename('forecast_reference_time')
            self.analysis_cube.add_aux_coord(frt_coord)
        if "forecast_period" not in coords:
            self.analysis_cube.add_aux_coord(
                AuxCoord(np.array([0], dtype=np.int32),
                         'forecast_period', 'seconds'))
        # set nowcast attributes
        self.analysis_cube.attributes['source'] = 'MONOW'
        self.analysis_cube.attributes['title'] = (
            'MONOW Extrapolation Nowcast on UK 2 km Standard Grid')

    def _set_up_output_cubes(self, all_forecasts):
        """
        Convert 3D numpy array into list of cubes with correct time metadata.
        All other metadata are inherited from self.analysis_cube.

        Args:
            all_forecasts (np.ndarray):
                Array of 2D forecast fields returned by extrapolation function

        Returns:
            forecast_cubes (iris.cube.CubeList):
                List of extrapolated cubes with correct time coordinates
        """
        current_datetime = iris_time_to_datetime(
            self.analysis_cube.coord('time'))[0]
        forecast_cubes = [self.analysis_cube.copy()]
        for i in range(len(all_forecasts)):
            # copy forecast data into template cube
            new_cube = self.analysis_cube.copy(
                data=all_forecasts[i, :, :].astype(np.float32))
            # update time and forecast period coordinates
            current_datetime += timedelta(seconds=self.interval*60)
            current_time = datetime_to_iris_time(current_datetime)
            new_cube.coord('time').points = np.array(
                [current_time], dtype=np.int64)
            new_cube.coord('forecast_period').points = np.array(
                [(i+1)*self.interval*60], dtype=np.int32)
            forecast_cubes.append(new_cube)
        return forecast_cubes

    def _generate_forecast_cubes(self, all_forecasts, attributes_dict):
        """
        Convert forecast arrays into IMPROVER output cubes with re-added
        orographic enhancement

        Args:
            all_forecasts (np.ndarray):
                Array of 2D forecast fields returned by extrapolation function
            attributes_dict (dict or None):
                Dictionary containing information for amending the attributes
                of the output cube.

        Returns:
            forecast_cubes (list):
                List of iris.cube.Cube instances containing forecasts at all
                required lead times, and conforming to the IMPROVER metadata
                standard.
        """
        if not attributes_dict:
            attributes_dict = {}

        # re-mask forecast data
        all_forecasts = np.ma.masked_invalid(all_forecasts)

        # generate list of forecast cubes
        self._reformat_analysis_cube()
        timestamped_cubes = self._set_up_output_cubes(all_forecasts)

        # re-convert units and re-add orographic enhancement
        forecast_cubes = []
        for cube in timestamped_cubes:
            cube.convert_units(self.required_units)
            if self.orogenh:
                cube, = ApplyOrographicEnhancement("add").process(
                    cube, self.orogenh)

            # Update meta-data
            amend_attributes(cube, attributes_dict)
            set_history_attribute(cube, "Nowcast")
            forecast_cubes.append(cube)
        return forecast_cubes

    def process(self, initial_cube, ucube, vcube, orographic_enhancement,
                attributes_dict=None):
        """
        Extrapolate the initial precipitation field using the velocities
        provided to the required forecast lead times

        Args:
            initial_cube (iris.cube.Cube):
                Cube of precipitation at initial time
            ucube (iris.cube.Cube):
                x-advection velocities
            vcube (iris.cube.Cube):
                y-advection velocities
            orographic_enhancement (iris.cube.Cube):
                Cube containing orographic enhancement fields at all required
                lead times
            attributes_dict (dict or None):
                Dictionary containing information for amending the attributes
                of the output cube.

        Returns:
            forecast_cubes (list):
                List of extrapolated iris.cube.Cube instances at the required
                lead times (including T+0 / analysis time)
        """
        # ensure input cube is suitable for advection
        if 'rate' not in initial_cube.name():
            msg = '{} is not a precipitation rate cube'
            raise ValueError(msg.format(initial_cube.name()))
        check_if_grid_is_equal_area(initial_cube)

        self.analysis_cube = initial_cube.copy()
        self.required_units = initial_cube.units
        self.orogenh = orographic_enhancement

        # get unmasked precipitation rate array to input into advection
        precip_rate = self._get_precip_rate()

        # calculate displacement in grid squares per time step
        displacement = self._generate_displacement_array(ucube, vcube)

        # call pysteps extrapolation method
        all_forecasts = extrapolate(
            precip_rate, displacement, self.num_timesteps,
            allow_nonfinite_values=True)

        # repackage data as IMPROVER masked cubes
        forecast_cubes = self._generate_forecast_cubes(
            all_forecasts, attributes_dict)

        return forecast_cubes
