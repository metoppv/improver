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
""" Unit tests for DayNightMask class """

import unittest
from datetime import datetime, timedelta

import cf_units as unit
import iris
import numpy as np
import pytz
from iris.tests import IrisTest

from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    construct_scalar_time_coords,
    set_up_variable_cube,
)
from improver.utilities.solar import DayNightMask

ATTRIBUTES = {"institution": "Met Office", "title": "A model field"}


def create_spot_cube(xrange=175, yrange=85):
    """Create a test spot-cube."""
    n_sites = 99
    latitudes = np.tile(np.linspace(-yrange, yrange, 11), 9)
    longitudes = np.repeat(np.linspace(-xrange, xrange, 9), 11)
    altitudes = np.zeros(n_sites, dtype=np.float32)
    wmo_ids = np.arange(1000, 1000 + n_sites, 1)
    data = np.zeros(n_sites)
    args = (altitudes, latitudes, longitudes, wmo_ids)

    time = datetime(2023, 6, 6, 15)
    frt = datetime(2023, 6, 6, 12)
    time_bounds = None
    time_coords = construct_scalar_time_coords(time, time_bounds, frt)
    time_coords = [item[0] for item in time_coords]

    spot_cube = build_spotdata_cube(
        data, "generic_spot_cube", "1", *args, scalar_coords=time_coords,
    )
    spot_cube.attributes = ATTRIBUTES
    return spot_cube


class Test__init__(IrisTest):
    """ Test initialisation of the DayNightMask class """

    def test_basic_init(self):
        """ Test Initiation of DayNightMask Object"""
        plugin = DayNightMask()
        self.assertEqual(plugin.day, 1)
        self.assertEqual(plugin.night, 0)
        self.assertEqual(plugin.irregular, False)


class Test__repr__(IrisTest):
    """ Test string representation """

    def test_basic_repr(self):
        """ Test Representation string of DayNightMask Object"""
        expected = "<DayNightMask : Day = 1, Night = 0>"
        result = str(DayNightMask())
        self.assertEqual(result, expected)


class Test__create_daynight_mask(IrisTest):
    """ Test string representation """

    def setUp(self):
        """Set up the cube for testing."""
        data = np.ones((1, 16, 16), dtype=np.float32)
        data[:, 7, 7] = 0.0

        self.cube = set_up_variable_cube(
            data, "precipitation_amount", "kg m^-2", "equalarea", attributes=ATTRIBUTES
        )
        self.spot_cube = create_spot_cube()

    def test_basic_daynight_mask(self):
        """ Test this creates a blank mask cube for gridded data"""
        plugin = DayNightMask()
        result = plugin._create_daynight_mask(self.cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.long_name, "day_night_mask")
        self.assertEqual(result.units, unit.Unit("1"))
        self.assertEqual(result.data.min(), DayNightMask().night)
        self.assertEqual(result.data.max(), DayNightMask().night)
        self.assertEqual(result.attributes["title"], "Day-Night mask")
        self.assertEqual(result.attributes["institution"], "Met Office")
        self.assertEqual(result.dtype, np.int32)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(plugin.irregular, False)

    def test_basic_daynight_mask_spot(self):
        """ Test this creates a blank mask cube for spot data"""
        plugin = DayNightMask()
        result = plugin._create_daynight_mask(self.spot_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.long_name, "day_night_mask")
        self.assertEqual(result.units, unit.Unit("1"))
        self.assertEqual(result.data.min(), DayNightMask().night)
        self.assertEqual(result.data.max(), DayNightMask().night)
        self.assertEqual(result.attributes["title"], "Day-Night mask")
        self.assertEqual(result.attributes["institution"], "Met Office")
        self.assertEqual(result.dtype, np.int32)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(plugin.irregular, True)


class Test__daynight_lat_lon_cube(IrisTest):
    """ Test string representation """

    def setUp(self):
        """Set up the cube for testing."""
        data = np.ones((16, 16), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data,
            "precipitation_amount",
            "kg m^-2",
            grid_spacing=1,
            domain_corner=(49, -8),
        )
        self.cube_360 = set_up_variable_cube(
            data,
            "precipitation_amount",
            "kg m^-2",
            grid_spacing=1,
            domain_corner=(49, 345),
        )
        self.spot_cube = create_spot_cube(xrange=10, yrange=10)

    def test_basic_lat_lon_cube_gridded(self):
        """ Test this create a blank gridded mask cube"""
        day_of_year = 10
        utc_hour = 12.0
        expected_result = np.ones((16, 16))
        plugin = DayNightMask()
        mask_cube = plugin._create_daynight_mask(self.cube)

        result = plugin._daynight_lat_lon_cube(mask_cube, day_of_year, utc_hour)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.data, expected_result)

    def test_basic_lat_lon_cube_360(self):
        """ Test this still works with 360 data"""
        day_of_year = 10
        utc_hour = 0.0
        expected_result = np.zeros((16, 16))
        plugin = DayNightMask()
        mask_cube_360 = plugin._create_daynight_mask(self.cube_360)

        result = plugin._daynight_lat_lon_cube(mask_cube_360, day_of_year, utc_hour)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.data, expected_result)

    def test_basic_lat_lon_cube_spot(self):
        """ Test this create a blank spot mask cube"""
        day_of_year = 10
        utc_hour = 12.0
        expected_result = np.ones((99))
        plugin = DayNightMask()
        spot_mask_cube = plugin._create_daynight_mask(self.spot_cube)

        result = plugin._daynight_lat_lon_cube(spot_mask_cube, day_of_year, utc_hour)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.data, expected_result)


class Test_process(IrisTest):

    """Test DayNight Mask."""

    def setUp(self):
        """Set up the cubes for testing."""
        data = np.ones((16, 16), dtype=np.float32)
        data[7, 7] = 0.0
        vt = datetime(2015, 11, 20, 8, 0)
        self.cube = set_up_variable_cube(
            data,
            "precipitation_amount",
            "kg m^-2",
            "equalarea",
            grid_spacing=2000,
            domain_corner=(0, -30000),
            time=vt,
            frt=vt,
        )

        # Cube with time coordinate with bounds.  A very small bounding period
        # is used to capture day-night variation across the small domain.
        bounds = timedelta(minutes=4)
        self.cube_time_bounds = set_up_variable_cube(
            data,
            "precipitation_amount",
            "kg m^-2",
            "equalarea",
            grid_spacing=2000,
            domain_corner=(0, -30000),
            time=vt + bounds,
            time_bounds=(vt, vt + bounds),
            frt=vt,
        )

        # Lat lon cubes
        self.cube_lat_lon = set_up_variable_cube(
            data,
            "precipitation_amount",
            "kg m^-2",
            grid_spacing=1,
            domain_corner=(49, -8),
            time=vt,
            frt=vt,
        )
        self.cube_lat_lon_360 = set_up_variable_cube(
            data,
            "precipitation_amount",
            "kg m^-2",
            grid_spacing=1,
            domain_corner=(49, 345),
            time=vt,
            frt=vt,
        )
        self.spot_cube = create_spot_cube()

    def test_basic_standard_grid_ccrs(self):
        """Test day_night mask with standard_grid_ccrs projection."""
        result = DayNightMask().process(self.cube)
        expected_result = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        self.assertArrayEqual(result.data, expected_result)

    def test_time_as_dimension(self):
        """Test day_night mask for a cube with multiple times."""
        datetime_points = [datetime(2015, 11, 20, 8, 0), datetime(2015, 11, 20, 14, 0)]
        cube = add_coordinate(self.cube, datetime_points, "time", is_datetime=True)

        result = DayNightMask().process(cube)
        expected_result = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                np.ones((16, 16)),
            ]
        )
        self.assertArrayEqual(result.data, expected_result)
        self.assertEqual(result.shape, cube.shape)

    def test_basic_lat_lon(self):
        """Test day_night mask with lat lon data."""
        result = DayNightMask().process(self.cube_lat_lon)
        expected_result = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        self.assertArrayEqual(result.data, expected_result)

    def test_basic_lat_lon_360(self):
        """Test day_night mask with lat lon data 360 data."""
        result = DayNightMask().process(self.cube_lat_lon_360)
        expected_result = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        self.assertArrayEqual(result.data, expected_result)

    def test_time_bounds_standard_grid_ccrs(self):
        """Test day_night mask with standard_grid_ccrs projection for a cube
        with a time coordinate with bounds.

        This test compares against a reference case in which a cube without
        time bounds is adjusted such that its time falls at the mid-point of
        the time bounds being tested. This should return the same result as
        the cube with time bounds. The result is also compared against an
        array of expected values."""

        vt = datetime(2015, 11, 20, 8, 2, tzinfo=pytz.UTC)
        ref = self.cube.copy()
        ref.coord("time").points = [vt.timestamp()]
        ref_result = DayNightMask().process(ref)

        result = DayNightMask().process(self.cube_time_bounds)
        expected_result = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        self.assertArrayEqual(result.data, expected_result)
        self.assertArrayEqual(result.data, ref_result.data)

    def test_spot_mask(self):
        """Test day_night mask with lat lon spot data. Note that the
        expected data is shown as a grid for neatness, but is flattened
        in the comparison as the returned spot mask is one dimensional."""
        result = DayNightMask().process(self.spot_cube)
        expected_result = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            ]
        )
        self.assertArrayEqual(result.data, expected_result.flatten())


if __name__ == "__main__":
    unittest.main()
