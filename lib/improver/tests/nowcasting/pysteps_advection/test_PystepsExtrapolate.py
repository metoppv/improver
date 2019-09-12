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
"""Unit tests for PystepsExtrapolate plugin"""

import datetime
import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.nowcasting.pysteps_advection import PystepsExtrapolate
from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, add_coordinate)


def _make_initial_rain_cube(time_now):
    """Construct an 8x8 masked cube of rainfall rate for testing"""

    rain_data = np.array(
        [[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
         [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0],
         [0.1, 0.3, 0.5, 0.6, 0.4, 0.2, 0.1, 0.0],
         [0.2, 0.6, 1.0, 1.3, 1.1, 0.5, 0.3, 0.1],
         [0.1, 0.2, 0.6, 1.0, 0.7, 0.4, 0.1, 0.0],
         [0.0, 0.1, 0.2, 0.5, 0.4, 0.1, 0.0, 0.0],
         [0.0, 0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    # create an irregular mask
    rain_mask = np.array(
        [[True, True, False, False, False, True, True, True],
         [True, False, False, False, False, False, True, True],
         [False, False, False, False, False, False, False, True],
         [False, False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False, True],
         [True, False, False, False, False, False, True, True],
         [True, True, False, False, False, True, True, True]])

    rain_data = np.ma.MaskedArray(rain_data, mask=rain_mask)
    rain_cube = set_up_variable_cube(
        rain_data, name='rainfall_rate', units='mm/h',
        spatial_grid='equalarea', time=time_now, frt=time_now)
    rain_cube.remove_coord('forecast_period')
    rain_cube.remove_coord('forecast_reference_time')

    return rain_cube


def _make_orogenh_cube(time_now, interval, max_lead_time):
    """Construct an orographic enhancement cube with data valid for
    every lead time"""
    orogenh_data = 0.05*np.ones((8, 8), dtype=np.float32)
    orogenh_cube = set_up_variable_cube(
        orogenh_data, name='orographic_enhancement', units='mm/h',
        spatial_grid='equalarea', time=time_now, frt=time_now)

    time_points = [time_now]
    lead_time = 0
    while lead_time <= max_lead_time:
        lead_time += interval
        new_point = (
            time_points[-1] + datetime.timedelta(seconds=60*interval))
        time_points.append(new_point)

    orogenh_cube = add_coordinate(
        orogenh_cube, time_points, "time", is_datetime=True)
    return orogenh_cube


class Test_process(IrisTest):
    """Test wrapper for pysteps semi-Lagrangian extrapolation"""

    def setUp(self):
        """Set up test velocity and rainfall cubes"""
        time_now = datetime.datetime(2019, 9, 10, 15)
        wind_data = np.ones((8, 8), dtype=np.float32)
        self.ucube = set_up_variable_cube(
            wind_data, name='precipitation_advection_x_velocity',
            units='m/s', spatial_grid='equalarea', time=time_now,
            frt=time_now)
        self.vcube = set_up_variable_cube(
            wind_data, name='precipitation_advection_y_velocity',
            units='m/s', spatial_grid='equalarea', time=time_now,
            frt=time_now)
        self.rain_cube = _make_initial_rain_cube(time_now)

        self.interval = 15
        self.max_lead_time = 120
        self.orogenh_cube = _make_orogenh_cube(
            time_now, self.interval, self.max_lead_time)

        # set up all grids with 3.6 km spacing (1 m/s = 3.6 km/h)
        xmin = 0
        ymin = 200000
        step = 3600
        xpoints = np.arange(xmin, xmin+8*step, step).astype(np.float32)
        ypoints = np.arange(ymin, ymin+8*step, step).astype(np.float32)
        for cube in [
                self.ucube, self.vcube, self.rain_cube, self.orogenh_cube]:
            cube.coord(axis='x').points = xpoints
            cube.coord(axis='y').points = ypoints

    def test_basic(self):
        """Test output is a list of cubes with expected contents"""
        result = PystepsExtrapolate().process(
            self.rain_cube, self.ucube, self.vcube, self.interval,
            self.max_lead_time, self.orogenh_cube)
        self.assertIsInstance(result, list)
        # check result is a list including a cube at the analysis time
        self.assertEqual(len(result), 9)
        self.assertIsInstance(result[0], iris.cube.Cube)

        # now we have some problems with this data...
        print(result[1])
        print(result[1].data)
        # TODO all data points are nans





if __name__ == '__main__':
    unittest.main()
