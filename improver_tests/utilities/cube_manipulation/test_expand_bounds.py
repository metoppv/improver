# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function "cube_manipulation.expand_bounds".
"""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
from cf_units import date2num

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import expand_bounds

TIME_UNIT = "seconds since 1970-01-01 00:00:00"
CALENDAR = "gregorian"


class Test_expand_bounds(unittest.TestCase):
    """Test expand_bounds function"""

    def setUp(self):
        """Set up a cubelist for testing"""

        data = 275.5 * np.ones((3, 3), dtype=np.float32)
        frt = dt(2015, 11, 19, 0)
        time_points = [dt(2015, 11, 19, 1), dt(2015, 11, 19, 3)]
        time_bounds = [
            [dt(2015, 11, 19, 0), dt(2015, 11, 19, 2)],
            [dt(2015, 11, 19, 1), dt(2015, 11, 19, 3)],
        ]

        self.cubelist = iris.cube.CubeList([])
        for tpoint, tbounds in zip(time_points, time_bounds):
            cube = set_up_variable_cube(data, frt=frt, time=tpoint, time_bounds=tbounds)
            self.cubelist.append(cube)

        self.expected_bounds_seconds = [
            np.int64(date2num(dt(2015, 11, 19, 0), TIME_UNIT, CALENDAR)),
            np.int64(date2num(dt(2015, 11, 19, 3), TIME_UNIT, CALENDAR)),
        ]

        self.expected_bounds_hours = [
            date2num(dt(2015, 11, 19, 0), "hours since 1970-01-01 00:00:00", CALENDAR),
            date2num(dt(2015, 11, 19, 3), "hours since 1970-01-01 00:00:00", CALENDAR),
        ]

    def test_basic_time(self):
        """Test that expand_bound produces sensible bounds."""
        time_point = np.around(
            date2num(dt(2015, 11, 19, 3), TIME_UNIT, CALENDAR)
        ).astype(np.int64)
        expected_result = iris.coords.DimCoord(
            [time_point],
            bounds=self.expected_bounds_seconds,
            standard_name="time",
            units=TIME_UNIT,
        )
        result = expand_bounds(self.cubelist[0], self.cubelist, ["time"])
        self.assertEqual(result.coord("time"), expected_result)

    def test_multiple_coordinate_expanded(self):
        """Test that expand_bound produces sensible bounds when more than one
        coordinate is operated on, in this case expanding both the time and
        forecast period coordinates."""
        time_point = np.around(
            date2num(dt(2015, 11, 19, 3), TIME_UNIT, CALENDAR)
        ).astype(np.int64)
        expected_result_time = iris.coords.DimCoord(
            [time_point],
            bounds=self.expected_bounds_seconds,
            standard_name="time",
            units=TIME_UNIT,
        )
        expected_result_fp = iris.coords.DimCoord(
            [10800], bounds=[0, 10800], standard_name="forecast_period", units="seconds"
        )

        result = expand_bounds(
            self.cubelist[0], self.cubelist, ["time", "forecast_period"]
        )
        self.assertEqual(result.coord("time"), expected_result_time)
        self.assertEqual(result.coord("forecast_period"), expected_result_fp)
        self.assertEqual(result.coord("time").dtype, np.int64)

    def test_basic_no_time_bounds(self):
        """Test that it creates appropriate bounds if there are no time bounds
        on the input cubes."""
        for cube in self.cubelist:
            cube.coord("time").bounds = None

        time_point = np.around(
            date2num(dt(2015, 11, 19, 3), TIME_UNIT, CALENDAR)
        ).astype(np.int64)
        time_bounds = [
            np.around(date2num(dt(2015, 11, 19, 1), TIME_UNIT, CALENDAR)).astype(
                np.int64
            ),
            np.around(date2num(dt(2015, 11, 19, 3), TIME_UNIT, CALENDAR)).astype(
                np.int64
            ),
        ]
        expected_result = iris.coords.DimCoord(
            time_point, bounds=time_bounds, standard_name="time", units=TIME_UNIT
        )

        result = expand_bounds(self.cubelist[0], self.cubelist, ["time"])
        self.assertEqual(result.coord("time"), expected_result)

    def test_fails_with_multi_point_coord(self):
        """Test that if an error is raised if a coordinate with more than
        one point is given"""
        emsg = "the expand bounds function should only be used on a"
        with self.assertRaisesRegex(ValueError, emsg):
            expand_bounds(self.cubelist[0], self.cubelist, ["latitude"])

    def test_error_remove_bounds(self):
        """Test the expand_bounds function fails if its effect would be
        to remove bounds from a bounded coordinate, i.e. if a mixture of
        bounded and unbounded coordinates are input"""
        self.cubelist[1].coord("time").bounds = None
        msg = "cannot expand bounds for a mixture of bounded / unbounded"
        with self.assertRaisesRegex(ValueError, msg):
            expand_bounds(self.cubelist[0], self.cubelist, ["time"])


if __name__ == "__main__":
    unittest.main()
