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
"""Semi-Lagrangian backward advection plugin"""

import sys
import numpy as np
from datetime import timedelta

from iris.coords import AuxCoord

# path to temporary pysteps installation
sys.path.append("/home/h02/bayliffe/.local/lib/python3.6/site-packages/")
from pysteps.extrapolation.semilagrangian import extrapolate

from improver.utilities.temporal import (
    iris_time_to_datetime, datetime_to_iris_time)
from improver.nowcasting.utilities import ApplyOrographicEnhancement


class PystepsExtrapolate(object):
    """Wrapper for the pysteps semi-Lagrangian extrapolation method

    Reference:
        https://pysteps.readthedocs.io/en/latest/generated/
        pysteps.extrapolation.semilagrangian.extrapolate.html
    """

    @staticmethod
    def _get_displacement(cube, interval):
        """
        Generate displacement field for each time step using velocity cube
        and interval

        Args:
            cube (iris.cube.Cube):
                Cube of velocities in the x or y direction
            interval (int):
                Lead time interval, in minutes     
      
        Returns:
            displacement (np.ndarray):
                2D array of displacements to be applied to each time step
        """
        cube_ms = ucube.copy()
        cube_ms.convert_units('m s-1')
        displacement = cube_ms.data*interval*60.
        return np.ma.filled(displacement, np.nan)

    def _generate_forecast_cubes(self, all_forecasts, interval):
        """
        Convert numpy array into list of IMPROVER cubes.  Assumes forecast cube
        output starts at T+1.

        Args:
            all_forecasts (np.ndarray):
                Array of 2D forecast fields returned by extrapolation function
            interval (int):
                Time interval between forecasts, in minutes

        Returns:
            forecast_cubes (iris.cube.CubeList):
                List of extrapolated cubes with correct time coordinates
        """
        current_datetime = iris_time_to_datetime(
            self.cube.coord('time').points[0])
        frt_coord = self.cube.coord('time').copy()
        frt_coord.rename('forecast_reference_time')
        self.cube.add_aux_coord(frt_coord)
        self.cube.add_aux_coord(
            AuxCoord([0], 'forecast_period', 'seconds'))

        forecast_cubes = [self.cube.copy()]
        for i in len(all_forecasts):
            # copy forecast data into template cube
            new_cube = self.cube.copy(forecast[i, :, :])
            # calculate new validity time
            current_datetime += timedelta(seconds=interval*60)
            current_time = datetime_to_iris_time(
                current_datetime, time_units='seconds')
            new_cube.coord('time').points = [current_time]
            # add a forecast period
            new_cube.add_aux_coord(
                AuxCoord([interval*60], 'forecast_period', 'seconds'))
            forecast_cubes.append(new_cube)
        return forecast_cubes

    def process(self, initial_cube, ucube, vcube, interval, max_lead_time,
                orographic_enhancement):
        """
        Extrapolate the initial precipitation field using the velocities
        provided to

        Args:
            initial_cube (iris.cube.Cube):
                Cube of precipitation at initial time
            ucube (iris.cube.Cube):
                x-advection velocities
            vcube (iris.cube.Cube):
                y-advection velocities
            interval (int):
                Lead time interval, in minutes
            max_lead_time (int):
                Maximum lead time required, in minutes
            orographic_enhancement (iris.cube.Cube):
                Cube containing orographic enhancement fields at all required
                lead times

        Returns:
            forecast_cubes (iris.cube.CubeList):
                List of extrapolated cubes at the required lead times
                (including analysis)
        """
        # subtract orographic enhancement
        self.cube = initial_cube.copy()
        self.cube, = ApplyOrographicEnhancement("subtract").process(
            self.cube, orographic_enhancement)

        # get precipitation rate data in pysteps-acceptable format
        self.cube.convert_units('mm h-1')
        precip_rate = np.ma.filled(self.cube.data, np.nan)

        # establish timesteps required TODO excludes T+0 for now - check
        num_timesteps = max_lead_time // interval

        # generate displacement array in metres for each time step
        udisp = self._get_displacement(ucube, interval)
        vdisp = self._get_displacement(vcube, interval)
        displacement = np.concatenate(udisp, vdisp)
        
        # call pysteps extrapolation method
        all_forecasts = extrapolate(precip_rate, displacement, num_timesteps)
        # TODO does this include T+0 field, or must that be appended?  Assume
        # no T+0 field for now

        # repackage data as IMPROVER cubes
        forecast_cubes = self._generate_forecast_cubes(all_forecasts, interval)

        # re-convert units and re-add orographic enhancement
        for cube in forecast_cubes:
            cube.convert_units(initial_cube.units)
            cube, = ApplyOrographicEnhancement("add").process(
                cube, orographic_enhancement)

        return forecast_cubes
