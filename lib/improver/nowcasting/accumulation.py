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
import datetime
import warnings
import numpy as np
from datetime import timezone

import iris
from iris.exceptions import CoordinateNotFoundError, InvalidCubeError
from improver.utilities.temporal import (iris_time_to_datetime,
                                         datetime_to_iris_time)


class Accumulation:
    """
    Class to calculate precipitation accumulations from radar rates fields
    provided at discrete time intervals. The plugin will return one fewer
    cubes of accumulation than the number of input precipitation rate cubes.
    """
    def __init__(self):
        """
        Init
        """

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = ('<OpticalFlow: data_smoothing_radius_km: {}, '
                  'data_smoothing_method: {}, iterations: {}, '
                  'point_weight: {}, metadata_dict: {}>')
        return result.format(
            self.data_smoothing_radius_km, self.data_smoothing_method,
            self.iterations, self.point_weight, self.metadata_dict)

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
            times (list of datetime objects):
                A list of the validity times of the precipitation rate cubes as
                datetime objects in matching order to the returned cubelist.
        """
        times = np.array([
            iris_time_to_datetime(cube.coord('time'))[0] for cube in cubes])
        time_sorted = np.argsort(times)
        times = times[time_sorted]
        cubes = list(np.array(cubes)[time_sorted])
        return cubes, times

    @staticmethod
    def calculate_intervals(times):
        """
        Calculate the time intervals between the precipitation rate cubes and
        return this as a list.

        Args:
            times (list of datetime objects):
                A list of the validity times of the precipitation rate cubes as
                datetime objects in ascending order.
        Returns:
            time_intervals (list of datetime objects):
                A list of time intervals between precipitation rate cubes.
        """
        time_intervals = np.diff(times, axis=0)
        return time_intervals

    @staticmethod
    def calculate_accumulation_times(times, time_intervals):
        """
        Calculate the validity times for the precipitation accumulation cubes
        where these fall at the end of the accumulation period. Return these
        in integer seconds since 1970-01-01 00:00:00.

        Args:
            times (list of datetime objects):
                A list of the validity times of the precipitation rate cubes as
                datetime objects in ascending order.
            time_intervals (list of datetime objects):
                A list of time intervals between precipitation rate cubes.
        Returns:
            accumulation_times (list):
                A list of validity times for the accumulation cubes in integer
                seconds since 1970-01-01 00:00:00.
        """
        accumulation_times = times + time_intervals
        accumulation_times = np.array(
            [time.replace(tzinfo=timezone.utc).timestamp()
             for time in accumulation_times], dtype=np.int64)

        return accumulation_times

    @staticmethod
    def create_accumulation_cube(cube, time_interval,
                                 accumulation_time):
        """
        Create a new cube to contain the calculated accumulation data.

        Args:
            cube (iris.cube.Cube):
                The precipitation rate cube that is valid at the beginning of
                the accumulation period.
        """
        time_coordinate = cube.coord('time').copy(
            points=accumulation_time,
            bounds=(accumulation_time - time_interval.seconds, accumulation_time))
        fp, = cube.coord('forecast_period').points
        fp_coordinate = cube.coord('forecast_period').copy(
            points=fp + time_interval.seconds,
            bounds=(fp, fp + time_interval.seconds))

        accumulation_cube = cube.copy()
        accumulation_cube.rename('lwe_thickness_of_precipitation_amount')
        accumulation_cube.replace_coord(time_coordinate)
        accumulation_cube.replace_coord(fp_coordinate)
        return accumulation_cube

    def integrate_rates(self, cubes, time_intervals,
                        accumulation_times):
        accumulation_cubes = []
        iterator = zip(cubes, time_intervals, accumulation_times)
        for cube, time_interval, accumulation_time in iterator:
            accumulation_cube = self.create_accumulation_cube(
                cube, time_interval, accumulation_time)
            accumulation_data = cube.data * time_interval.seconds
            accumulation_cube.data = accumulation_data
            accumulation_cubes.append(accumulation_cube)

        return accumulation_cubes

    def process(self, cubes):
        """
        """
        # TODO Check units are in seconds, or be adaptable to not seconds.
        cubes, times = self.sort_cubes_by_time(cubes)
        time_intervals = self.calculate_intervals(times)
        accumulation_times = self.calculate_accumulation_times(
            times[:-1], time_intervals)
        accumulation_cubes = self.integrate_rates(
            cubes[:-1], time_intervals, accumulation_times)

        return accumulation_cubes
