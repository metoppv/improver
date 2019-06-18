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
import warnings
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
            points=fp, bounds=(fp - time_interval, fp))

        accumulation_cube = cube.copy()
        accumulation_cube.rename('lwe_thickness_of_precipitation_amount')
        accumulation_cube.replace_coord(time_coordinate)
        accumulation_cube.replace_coord(fp_coordinate)
        accumulation_cube.units = 'm'
        return accumulation_cube

    @staticmethod
    def determine_masking(cube):
        """
        Determine if the input cube contains a mask and if so what that mask
        is. This is used to mask the accumulation data that is returned. If the
        input data is not masked, the resulting accumulation data will not be
        masked either.

        Args:
            cube (iris.cube.Cube):
                The cube for which the masking is being checked.
        Returns:
            array_type (numpy class instance):
                The array class that should be used to store the accumulation
                data.
            mask (dict):
                A kwargs dictionary that includes the mask from the input cube.
                This is returned as a dict to simplify the invocation of the
                numpy class where this can be provided as a kwarg.
        """
        try:
            input_mask = cube.data.mask
        except AttributeError:
            array_type = np.array
            mask = {}
        else:
            array_type = np.ma.MaskedArray
            mask = {'mask': input_mask}

        return array_type, mask


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
            accumulation_cube = self.create_accumulation_cube(
                end_cube, time_interval)

#            cube_name = 'lwe_thickness_of_precipitation_amount'
#            accumulation_cube = CubeCombiner('add').process(
#                [start_cube, end_cube], cube_name,
#                expanded_coord={'time':'upper', 'forecast_period':'upper')
#            accumulation_cube.units = 'm'

            array_type, mask = self.determine_masking(start_cube)
            accumulation_data = start_cube.data * time_interval
            accumulation_cube.data = array_type(accumulation_data, **mask)
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
        """
        fraction, integer = np.modf(self.accumulation_period / time_interval)
        if fraction != 0:
            msg = ("The specified accumulation period ({}) is not divisible "
                   "by the time intervals between accumulation cubes ({}). As "
                   "a result it is not possible to calculate the desired "
                   "accumulation.".format(
                       self.accumulation_period, time_interval))
            raise ValueError(msg)

        if ncubes % integer != 0:
            msg = ("The provided cubes result in a partial period given the "
                   "specified accumulation_period, i.e. the number of cubes "
                   "is insufficient to give a set of complete periods. Only "
                   "complete periods will be returned.")
            warnings.warn(msg)

        return int(integer)

    @staticmethod
    def chunk_list(cubes, cube_subset):
        for index in range(0, len(cubes), cube_subset):
            yield cubes[index:index + cube_subset]

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
        # Standardise inputs to expected units.
        cubes = enforce_coordinate_units_and_dtypes(
            cubes, ['time', 'forecast_reference_time', 'forecast_period'],
            inplace=False)
        enforce_diagnostic_units_and_dtypes(cubes)

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
        for cube_subset in cubes_to_aggregate:

            # Don't produce an accumulation cube for any incomplete periods.
            if len(cube_subset) != cube_subset_no:
                break

            aggregated_cube = cube_subset.merge_cube()
            aggregated_cube = aggregated_cube.collapsed('time',
                                                        iris.analysis.SUM)
            print(aggregated_cube)
            print('-'*50)
