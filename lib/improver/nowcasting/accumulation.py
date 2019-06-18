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
accumulations from advected radar fields. It also includes the
AccumulationAggregator class for combining these into longer periods.
"""
import warnings
import numpy as np

import iris
from iris.exceptions import CoordinateNotFoundError, InvalidCubeError
from improver.cube_combiner import CubeCombiner
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

        Args:
            accumulation_units (str):
                The physical units in which the accumulation should be
                returned. The default is metres.
        """
        self.accumulation_units = accumulation_units

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = '<Accumulation: accumulation_units={}>'
        return result.format(self.accumulation_units)

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

        # Assumes rates cubes bookend the accumulations, e.g. there is one
        # more rates cube than expected accumulation cubes.
        iterator = zip(cubes[:-1], cubes[1:], time_intervals)

        # Calculate accumulations and convert to desired units.
        for start_cube, end_cube, time_interval in iterator:

            # Use CubeCombiner to expand coordinate bounds.
            cube_name = 'lwe_thickness_of_precipitation_amount'
            accumulation_cube = CubeCombiner('add').process(
                iris.cube.CubeList([start_cube, end_cube]), cube_name,
                expanded_coord={'time': 'upper', 'forecast_period': 'upper'})
            accumulation_cube.units = 'm'

            # Calculate new data and insert into cube.
            accumulation_data = start_cube.data * time_interval
            accumulation_cube.data = accumulation_data
            accumulation_cube.convert_units(self.accumulation_units)
            accumulation_cubes.append(accumulation_cube)

        return accumulation_cubes


class AccumulationAggregator:

    """Aggregate accumulations over short periods into longer periods."""

    def __init__(self, accumulation_period=None):
        """
        Initialise the plugin.

        Args:
            accumulation_period (int):
                The desired accumulation period in seconds. This period
                must be evenly divisible by the time intervals of the input
                cubes. The default is None, in which case an aggregation is
                made across all input cubes.
        """
        self.accumulation_period = accumulation_period

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = '<AccumulationAggregator: accumulation_period={}>'
        return result.format(self.accumulation_period)

    def check_accumulation_period(self, time_interval, ncubes):
        """
        Check to ensure the requested accumulation period can be created from
        the input cubes.

        Args:
            time_interval (int):
                The time interval between the input cubes in seconds.
            ncubes (int):
                The number of input cubes.
        Returns:
            cube_subset (int) or ncubes (int):
                The integer number of cubes that must be combined to create the
                desired total accumulation period, as given by
                self.accumulation_period. Is self.accumulation_period is None
                then ncubes is returned as the whole list will be aggregated.
        Raises:
            ValueError: Accumulation period cannot be created from time
                        intervals of input cubes.
        """
        if self.accumulation_period is None:
            return ncubes

        fraction, cube_subset = np.modf(self.accumulation_period /
                                        time_interval)
        if fraction != 0:
            msg = ("The specified accumulation period ({}) is not divisible "
                   "by the time intervals between accumulation cubes ({}). As "
                   "a result it is not possible to calculate the desired "
                   "accumulation.".format(
                       self.accumulation_period, time_interval))
            raise ValueError(msg)

        if ncubes % cube_subset != 0:
            msg = ("The provided cubes result in a partial period given the "
                   "specified accumulation_period, i.e. the number of cubes "
                   "is insufficient to give a set of complete periods. Only "
                   "complete periods will be returned.")
            warnings.warn(msg)

        return int(cube_subset)

    @staticmethod
    def chunk_list(cubes, cube_subset):
        """
        Break the CubeList into a set of CubeLists each containint the
        required number of cubes to be combined to create the desired
        accumulation_period.

        Args:
            cubes (iris.cube.CubeList):
                The input cubes.
            cube_subset (int):
                The integer number of cubes that must be combined to create the
                desired total accumulation period, as given by
                self.accumulation_period.
        Returns:
            generator:
                A generator that produces the short sub-lists required to
                create new accumulation periods.
        """
        for index in range(0, len(cubes), cube_subset):
            yield iris.cube.CubeList(cubes[index:index + cube_subset])

    def process(self, cubes):
        """
        Calculate aggregated accumulations over the desired accumulation period
        after checking this is possible given the input cubes.

        Args:
            cubes (iris.cube.CubeList):
                A cubelist containing input precipitation accumulation cubes.
        Returns:
            accumulation_cubes (iris.cube.CubeList):
                A cubelist containing precipitation accumulation cubes where
                the accumulation periods are determined by the given
                accumulation_period unless this is None, in which case a single
                cube will be returned covering the entire period of the input
                cubes.
        """
        # Sort cubes into time order and calculate intervals.
        cubes, times = Accumulation.sort_cubes_by_time(cubes)
        ncubes = len(cubes)

        # Find the common time interval between accumulation cubes.
        try:
            time_interval, = np.unique(np.diff(times, axis=0))
        except ValueError:
            msg = ("AccumulationAggregator is designed to work with "
                   "accumulation cubes at regualar time intervals. Cubes "
                   "provided are unevenly spaced in time; time intervals are "
                   "{}.".format(np.diff(times, axis=0)))
            raise ValueError(msg)

        cube_subset_no = self.check_accumulation_period(time_interval, ncubes)
        cubes_to_aggregate = self.chunk_list(cubes, cube_subset_no)

        aggregated_cubes = iris.cube.CubeList()
        for cube_subset in cubes_to_aggregate:

            # Don't produce an accumulation cube for any incomplete periods.
            if len(cube_subset) != cube_subset_no:
                break

            # Aggregate shorter period accumulations and expand coord bounds.
            aggregated_cube = CubeCombiner('add').process(
                cube_subset, cube_subset[0].name(),
                expanded_coord={'time': 'upper', 'forecast_period': 'upper'})
            aggregated_cubes.append(aggregated_cube)

        return aggregated_cubes
