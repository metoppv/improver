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
import iris
import numpy as np

from improver import BasePlugin
from improver.metadata.check_datatypes import check_mandatory_standards
from improver.utilities.cube_manipulation import expand_bounds


class Accumulation(BasePlugin):
    """
    Class to calculate precipitation accumulations from radar rates fields
    provided at discrete time intervals. The plugin will calculate
    accumulations between each pair of rates fields provided. These will be
    used to construct the accumulation period requested when possible, and
    cubes of this desired period are then returned.
    """
    def __init__(self, accumulation_units='m', accumulation_period=None,
                 forecast_periods=None):
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
            forecast_periods (iterable):
                The forecast periods in seconds that define the end of an
                accumulation period.

        """
        self.accumulation_units = accumulation_units
        self.accumulation_period = accumulation_period
        self.forecast_periods = forecast_periods

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = ('<Accumulation: accumulation_units={}, '
                  'accumulation_period={}s, '
                  'forecast_periods={}s>')
        return result.format(self.accumulation_units, self.accumulation_period,
                             self.forecast_periods)

    @staticmethod
    def sort_cubes_by_time(cubes):
        """
        Sort cubes in time ascending order (from earliest time to latest time).

        Args:
            cubes (iris.cube.CubeList):
                A cubelist containing input precipitation rate cubes.
        Returns:
            tuple: tuple containing:
                **cubes** (iris.cube.CubeList):
                    The cubelist in ascending time order.
                **times** (list):
                    A list of the validity times of the precipitation rate
                    cubes in integer seconds since 1970-01-01 00:00:00.
        """
        times = np.array([cube.coord('time').points[0] for cube in cubes])
        time_sorted = np.argsort(times)
        times = times[time_sorted]
        cubes = iris.cube.CubeList(np.array(cubes)[time_sorted])
        return cubes, times

    def _check_inputs(self, cubes):
        """Check the inputs prior to calculating the accumulations.

        Args:
            cubes: iris.cube.CubeList
                Cube list of precipitation rates that will be checked for their
                appropriateness in calculating the requested accumulations.
                The timesteps between the cubes in this cubelist are
                expected to be regular.

        Returns:
            (tuple): tuple containing:
                **cubes** (iris.cube.CubeList):
                    Modified version of the input cube list of precipitation
                    rates that have had the units of the coordinates and
                    cube data enforced. The cube list has also been sorted by
                    time.

                **time_interval** (float):
                    Interval between the timesteps from the input cubelist.

        Raises:
            ValueError: The input rates cubes must be at regularly spaced
                time intervals.
            ValueError: The accumulation period is less than the time interval
                between the rates cubes.
            ValueError: The specified accumulation period is not cleanly
                divisible by the time interval.

        """
        # Standardise inputs to expected units
        standardised_cubes = []
        for cube in cubes:
            check_mandatory_standards(cube)
            new_cube = cube.copy()
            new_cube.convert_units('m s-1')
            standardised_cubes.append(new_cube)
        cubes = standardised_cubes

        # Sort cubes into time order and calculate intervals.
        cubes, times = self.sort_cubes_by_time(cubes)

        try:
            time_interval, = np.unique(np.diff(times, axis=0)).astype(
                np.int32)
        except ValueError:
            msg = ("Accumulation is designed to work with "
                   "rates cubes at regular time intervals. Cubes "
                   "provided are unevenly spaced in time; time intervals are "
                   "{}.".format(np.diff(times, axis=0)))
            raise ValueError(msg)

        if self.accumulation_period is None:
            # If no accumulation period is specified, assume that the input
            # cubes will be used to construct a single accumulation period.
            self.accumulation_period, = (
                cubes[-1].coord("forecast_period").points)

        # Ensure that the accumulation period is int32.
        self.accumulation_period = np.int32(self.accumulation_period)

        fraction, integral = np.modf(self.accumulation_period / time_interval)

        # Check whether the accumulation period is less than the time_interval
        # i.e. the integral is equal to zero. In this case, the rates cubes
        # are too widely spaced to compute the requested accumulation period.
        if integral == 0:
            msg = (
                "The accumulation_period is less than the time interval "
                "between the rates cubes. The rates cubes provided are "
                "therefore insufficient for computing the accumulation period "
                "requested. accumulation period specified: {}, "
                "time interval specified: {}".format(
                    self.accumulation_period, time_interval))
            raise ValueError(msg)

        # Ensure the accumulation period is cleanly divisible by the time
        # interval.
        if fraction != 0:
            msg = ("The specified accumulation period ({}) is not divisible "
                   "by the time intervals between rates cubes ({}). As "
                   "a result it is not possible to calculate the desired "
                   "total accumulation period.".format(
                       self.accumulation_period, time_interval))
            raise ValueError(msg)

        if self.forecast_periods is None:
            # If no forecast periods are specified, then the accumulation
            # periods calculated will end at the forecast period from
            # each of the input cubes.
            self.forecast_periods = [
                cube.coord("forecast_period").points for cube in cubes
                if cube.coord("forecast_period").points >= time_interval]

        # Check whether any forecast periods are less than the accumulation
        # period. This is expected if the accumulation period is e.g. 1 hour,
        # however, the forecast periods are e.g. [15, 30, 45] minutes.
        # In this case, the forecast periods are filtered, so that only
        # complete accumulation periods will be calculated.
        if any(self.forecast_periods < self.accumulation_period):
            forecast_periods = [fp for fp in self.forecast_periods
                                if fp >= self.accumulation_period]
            self.forecast_periods = forecast_periods

        return cubes, time_interval

    def _get_cube_subsets(self, cubes, forecast_period):
        """Finding the subset of cubes from the input cubelist that are
        within the accumulation period, based on the required forecast period
        that defines the upper bound of the accumulation period and the length
        of the accumulation period.

        Args:
            cubes (iris.cube.CubeList):
                Cubelist containing all the rates cubes that are available
                to be used to calculate accumulations.
            forecast_period (int or numpy.ndarray):
                Forecast period in seconds matching the upper bound of the
                accumulation period.

        Returns:
            iris.cube.CubeList:
                Cubelist that defines the cubes used to calculate
                the accumulations.

        """
        # If the input is a numpy array, get the integer value from the array
        # for use in the constraint.
        if isinstance(forecast_period, np.ndarray):
            forecast_period, = forecast_period
        start_point = forecast_period - self.accumulation_period

        constr = iris.Constraint(
            forecast_period=lambda fp: start_point <= fp <= forecast_period)

        return cubes.extract(constr)

    @staticmethod
    def _calculate_accumulation(cube_subset, time_interval):
        """Calculate the accumulation for the requested accumulation period
        by finding the mean rate between each adjacent pair of cubes within
        the cube_subset and multiplying this mean rate by the time_interval,
        in order to compute an accumulation. The accumulation between each
        pair of cubes is summed, in order to generate a total accumulation
        using all of the cubes within the cube_subset.

        Args:
            cube_subset (iris.cube.CubeList):
                Cubelist containing all the rates cubes that will be used
                to calculate the accumulation.
            time_interval (float):
                Interval between the timesteps from the input cubelist.

        Returns:
            accumulation (numpy.ndarray) or None:
                If either the forecast period given by the input cube is not
                a requested forecast_period at which to calculate the
                accumulations, or the number of input cubelist is only
                sufficient to partially cover the desired accumulation, then
                None is returned.
                If an accumulation can be successfully computed, then a
                numpy array is returned.

        """
        accumulation = 0.
        # Accumulations are calculated using the mean precipitation rate
        # calculated from the rates cubes that bookend the desired period.
        iterator = zip(cube_subset[0:-1], cube_subset[1:])
        for start_cube, end_cube in iterator:
            accumulation += ((start_cube.data + end_cube.data) *
                             time_interval * 0.5)
        return accumulation

    @staticmethod
    def _set_metadata(cube_subset):
        """Set the metadata on the accumulation cube. This includes
        expanding the bounds to cover the accumulation period with the
        point within the time and forecast_period coordinates recorded as the
        upper bound of the accumulation period.

        Args:
            cube_subset(iris.cube.CubeList):
                Cubelist containing the subset of cubes used to calculate
                the accumulations. The bounds from these cubes will be used
                to set the metadata on the output accumulation cube.

        Returns:
            iris.cube.Cube:
                Accumulation cube with the desired metadata.

        """
        cube_name = 'lwe_thickness_of_precipitation_amount'
        accumulation_cube = expand_bounds(
            cube_subset[0].copy(),
            iris.cube.CubeList(cube_subset),
            ["time", "forecast_period"])
        accumulation_cube.rename(cube_name)
        accumulation_cube.units = 'm'
        return accumulation_cube

    def process(self, cubes):
        """
        Calculate period precipitation accumulations based upon precipitation
        rate fields. All calculations are performed in SI units, so
        precipitation rates are converted to "m/s" and times into seconds
        before calculations are performed. The output units of accumulation
        are set by the plugin keyword argument accumulation_units.

        Args:
            cubes (iris.cube.CubeList):
                A cubelist containing input precipitation rate cubes.
        Returns:
            iris.cube.CubeList:
                A cubelist containing precipitation accumulation cubes where
                the accumulation periods are determined by plugin argument
                accumulation_period.
        """
        cubes, time_interval = self._check_inputs(cubes)

        accumulation_cubes = iris.cube.CubeList()

        for forecast_period in self.forecast_periods:
            cube_subset = self._get_cube_subsets(cubes, forecast_period)
            accumulation = self._calculate_accumulation(
                cube_subset, time_interval)
            accumulation_cube = self._set_metadata(cube_subset)

            # Calculate new data and insert into cube.
            accumulation_cube.data = accumulation
            accumulation_cube.convert_units(self.accumulation_units)
            accumulation_cubes.append(accumulation_cube)

        return accumulation_cubes
