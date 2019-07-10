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
This module defines the Accumulation class for calculating precipitation
accumulations from advected radar fields. It is also possible to create longer
accumulations from shorter intervals.
"""
import warnings
import numpy as np

import iris
from improver.utilities.cube_manipulation import expand_bounds
from improver.utilities.cube_units import (enforce_coordinate_units_and_dtypes,
                                           enforce_diagnostic_units_and_dtypes)


class Accumulation:
    """
    Class to calculate precipitation accumulations from radar rates fields
    provided at discrete time intervals. The plugin will calculate
    accumulations between each pair of rates fields provided. These will be
    used to construct the accumulation period requested when possible, and
    cubes of this desired period are then returned.
    """
    def __init__(self, accumulation_units='m', accumulation_period=None):
        """
        Initialise the plugin.

        Args:
            accumulation_units (str):
                The physical units in which the accumulation should be
                returned. The default is metres.
            accumulation_period (int):
                The desired accumulation period in seconds. This period
                must be evenly divisible by the time intervals of the input
                cubes. The default is None, in which case an accumulation is
                calculated across the span of time covered by the input rates
                cubes.
        """
        self.accumulation_units = accumulation_units
        self.accumulation_period = accumulation_period

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = ('<Accumulation: accumulation_units={}, '
                  'accumulation_period={}>')
        return result.format(self.accumulation_units, self.accumulation_period)

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
        cubes = iris.cube.CubeList(np.array(cubes)[time_sorted])
        return cubes, times

    def get_period_sets(self, time_interval, cubes):
        """
        Return sub-sets of the input cube list that will be used to construct
        the accumulation period specified by self.accumulation_period.

        Args:
            time_interval (int):
                The time interval between the input precipitation rate cubes in
                seconds.
            cubes (list):
                The input precitation rate cubes.
        Returns:
            cube_subsets (list of lists):
                A list containing lists which comprise the rates cubes required
                to construct a sequence of accumulation cubes that match the
                period given by self.accumulation_period. If
                self.accumulation_period is None then the whole list of cubes
                is returned to give one accumulation across the whole set.
        Raises:
            ValueError: Accumulation period cannot be created from time
                        intervals of input cubes.
        """
        # If no accumulation period is provided, calculate accumulation across
        # all provided cubes.
        if self.accumulation_period is None:
            return [cubes]

        fraction, cube_subset = np.modf(self.accumulation_period /
                                        time_interval)

        if fraction != 0:
            msg = ("The specified accumulation period ({}) is not divisible "
                   "by the time intervals between rates cubes ({}). As "
                   "a result it is not possible to calculate the desired "
                   "total accumulation period.".format(
                       self.accumulation_period, time_interval))
            raise ValueError(msg)

        cube_subset = int(cube_subset)

        # We expect 1 more rates cubes than accumulations cubes, if we have
        # more or less, a targetted accumulation_period will be incomplete.
        unused_cubes = int(len(cubes) % cube_subset)
        if unused_cubes != 1 and cube_subset != 1:
            msg = ("The provided cubes result in a partial period given the "
                   "specified accumulation_period, i.e. the number of cubes "
                   "is insufficient to give a set of complete periods. Only "
                   "complete periods will be returned.")
            warnings.warn(msg)

        cube_subsets = []
        for index in range(0, len(cubes) - unused_cubes, cube_subset):
            cube_subsets.append(cubes[index:index + cube_subset + 1])
        return cube_subsets

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
                the accumulation periods are determined by plugin argument
                accumulation_period.
        """
        # Standardise inputs to expected units.
        cubes = enforce_coordinate_units_and_dtypes(
            cubes, ['time', 'forecast_reference_time', 'forecast_period'],
            inplace=False)
        enforce_diagnostic_units_and_dtypes(cubes)

        # Sort cubes into time order and calculate intervals.
        cubes, times = self.sort_cubes_by_time(cubes)
        try:
            time_interval, = np.unique(np.diff(times, axis=0))
        except ValueError:
            msg = ("Accumulation is designed to work with "
                   "rates cubes at regualar time intervals. Cubes "
                   "provided are unevenly spaced in time; time intervals are "
                   "{}.".format(np.diff(times, axis=0)))
            raise ValueError(msg)

        cube_subsets = self.get_period_sets(time_interval, cubes)

        accumulation_cubes = iris.cube.CubeList()
        for cube_subset in cube_subsets:
            # Handle accumulation periods equal to the rates cube time interval
            # in which case the last subset does not get used.
            if len(cube_subset) == 1:
                continue

            accumulation = 0.
            # Accumulations are calculated using the mean precipitation rate
            # calculated from the rates cubes that bookend the desired period.
            iterator = zip(cube_subset[0:-1], cube_subset[1:])
            for start_cube, end_cube in iterator:
                accumulation += ((start_cube.data + end_cube.data) *
                                 time_interval * 0.5)

            cube_name = 'lwe_thickness_of_precipitation_amount'
            accumulation_cube = expand_bounds(
                cube_subset[0],
                iris.cube.CubeList(cube_subset),
                expanded_coords={'time': 'upper', 'forecast_period': 'upper'})
            accumulation_cube.rename(cube_name)
            accumulation_cube.units = 'm'

            # Calculate new data and insert into cube.
            accumulation_cube.data = accumulation
            accumulation_cube.convert_units(self.accumulation_units)
            accumulation_cubes.append(accumulation_cube)

        return accumulation_cubes
