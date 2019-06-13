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
This module defines the accumulation class for calculating precipitation
accumulations from advected radar fields.
"""
import numpy as np

import iris
from iris.exceptions import CoordinateNotFoundError, InvalidCubeError
from improver.utilities.cube_units import (enforce_coordinate_units_and_dtypes,
                                           enforce_diagnostic_units_and_dtypes)


class Accumulation:
    """
    Class to calculate precipitation accumulations from radar rates fields
    provided at discrete time intervals. The plugin will return one fewer
    cubes of accumulation than the number of input precipitation rate cubes.
    The time intervals between the precipitation rate cubes determine the
    accumulation periods.
    """
    def __init__(self, accumulation_units='m'):
        """
        Initialise the plugin.
        """
        self.accumulation_units = accumulation_units

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = '<Accumulation>'
        return result

    @staticmethod
    def sort_cubes_by_time(cubes):
        """
        Sort cubes in time ascending order (from earliest time to latest time).

        Args:
            cubes (iris.cube.CubeList):
                A cubelist containing input precipitation rate cubes.
        Returns:
            cubes (iris.cube.CubeList):
                The cubelist in ascending time order.
            times (list):
                A list of the validity times of the precipitation rate cubes in
                integer seconds since 1970-01-01 00:00:00.
        """
        times = np.array([cube.coord('time').points[0] for cube in cubes])
        time_sorted = np.argsort(times)
        times = times[time_sorted]
        cubes = list(np.array(cubes)[time_sorted])
        return cubes, times

    @staticmethod
    def create_accumulation_cube(cube, time_interval):
        """
        Create a new cube to contain the calculated accumulation data.

        Args:
            cube (iris.cube.Cube):
                The precipitation rate cube that is valid at the end of the
                accumulation period.
            time_interval (int):
                The number of seconds between the validity times of the
                precipitation rate cubes that bookend the accumulation period.
        Returns:
            accumulation_cube (iris.cube.Cube):
                A new cube based upon the precipitation rate cubes. The name
                and time coordinates are modified to reflect the accumulation
                data that will be contained within the cube.
        """
        validity_time, = cube.coord('time').points
        time_coordinate = cube.coord('time').copy(
            points=validity_time,
            bounds=(validity_time - time_interval, validity_time))
        fp, = cube.coord('forecast_period').points
        fp_coordinate = cube.coord('forecast_period').copy(
            points=validity_time,
            bounds=(fp - time_interval, fp))

        accumulation_cube = cube.copy()
        accumulation_cube.rename('lwe_thickness_of_precipitation_amount')
        accumulation_cube.replace_coord(time_coordinate)
        accumulation_cube.replace_coord(fp_coordinate)
        accumulation_cube.units = 'm'
        return accumulation_cube

    def process(self, cubes):
        """
        Calculate period precipitation accumulations based upon precipitation
        rate fields.

        Args:
            cubes (iris.cube.CubeList):
                A cubelist containing input precipitation rate cubes.
        Returns:
            accumulation_cubes (iris.cube.CubeList):
                A cubelist containing precipitation accumulation cubes where
                the accumulation periods are determined by the intervals
                between the precipitation rate cubes.
        """
        # Standardise inputs to expected units.
        cubes = enforce_coordinate_units_and_dtypes(
            cubes, ['time', 'forecast_reference_time', 'forecast_period'],
            inplace=False)
        enforce_diagnostic_units_and_dtypes(cubes)

        # Sort cubes into time order and calculate intervals.
        cubes, times = self.sort_cubes_by_time(cubes)
        time_intervals = np.diff(times, axis=0)
        accumulation_cubes = iris.cube.CubeList()

        iterator = zip(cubes[:-1], cubes[1:], time_intervals)

        # Calculate accumulations and convert to desired units.
        for start_cube, end_cube, time_interval in iterator:
            accumulation_cube = self.create_accumulation_cube(
                end_cube, time_interval)
            accumulation_data = start_cube.data * time_interval
            accumulation_cube.data = accumulation_data
            accumulation_cube.convert_units(self.accumulation_units)
            accumulation_cubes.append(accumulation_cube)

        return accumulation_cubes
