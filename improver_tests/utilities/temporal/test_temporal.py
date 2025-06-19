# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for temporal utilities."""

import unittest
from datetime import datetime, timedelta

import cftime
import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import CellMethod
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
from iris.time import PartialDateTime

from improver.metadata.constants.time_types import TIME_COORDS
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.temporal import (
    cycletime_to_datetime,
    cycletime_to_number,
    datetime_constraint,
    datetime_to_cycletime,
    datetime_to_iris_time,
    extract_cube_at_time,
    extract_nearest_time_point,
    integrate_time,
    iris_time_to_datetime,
    relabel_to_period,
)


class Test_cycletime_to_datetime(IrisTest):
    """Test that a cycletime of a format such as YYYYMMDDTHHMMZ is converted
    into a datetime object."""

    def test_basic(self):
        """Test that a datetime object is returned of the expected value."""
        cycletime = "20171122T0100Z"
        dt = datetime(2017, 11, 22, 1, 0)
        result = cycletime_to_datetime(cycletime)
        self.assertIsInstance(result, datetime)
        self.assertEqual(result, dt)

    def test_define_cycletime_format(self):
        """Test when a cycletime is defined."""
        cycletime = "201711220100"
        dt = datetime(2017, 11, 22, 1, 0)
        result = cycletime_to_datetime(cycletime, cycletime_format="%Y%m%d%H%M")
        self.assertEqual(result, dt)


class Test_datetime_to_cycletime(IrisTest):
    """Test that a datetime object can be converted into a cycletime
    of a format such as YYYYMMDDTHHMMZ."""

    def test_basic(self):
        """Test that a datetime object is returned of the expected value."""
        dt = datetime(2017, 11, 22, 1, 0)
        cycletime = "20171122T0100Z"
        result = datetime_to_cycletime(dt)
        self.assertIsInstance(result, str)
        self.assertEqual(result, cycletime)

    def test_define_cycletime_format(self):
        """Test when a cycletime is defined."""
        dt = datetime(2017, 11, 22, 1, 0)
        cycletime = "201711220100"
        result = datetime_to_cycletime(dt, cycletime_format="%Y%m%d%H%M")
        self.assertEqual(result, cycletime)

    def test_define_cycletime_format_with_seconds(self):
        """Test when a cycletime is defined with seconds."""
        dt = datetime(2017, 11, 22, 1, 0)
        cycletime = "20171122010000"
        result = datetime_to_cycletime(dt, cycletime_format="%Y%m%d%H%M%S")
        self.assertEqual(result, cycletime)


class Test_cycletime_to_number(IrisTest):
    """Test that a cycletime of a format such as YYYYMMDDTHHMMZ is converted
    into a numeric time value."""

    def test_basic(self):
        """Test that a number is returned of the expected value."""
        cycletime = "20171122T0000Z"
        dt = 419808
        result = cycletime_to_number(cycletime)
        self.assertIsInstance(result, np.int64)
        self.assertAlmostEqual(result, dt)

    def test_cycletime_format_defined(self):
        """Test when a cycletime is defined."""
        cycletime = "201711220000"
        dt = 419808.0
        result = cycletime_to_number(cycletime, cycletime_format="%Y%m%d%H%M")
        self.assertAlmostEqual(result, dt)

    def test_alternative_units_defined(self):
        """Test when alternative units are defined. The result is cast as
        an integer as seconds should be of this type and compared as such.
        There are small precision errors in the 7th decimal place of the
        returned float."""
        cycletime = "20171122T0000Z"
        dt = 1511308800
        result = cycletime_to_number(
            cycletime, time_unit="seconds since 1970-01-01 00:00:00"
        )
        self.assertEqual(int(np.round(result)), dt)

    def test_alternative_calendar_defined(self):
        """Test when an alternative calendar is defined."""
        cycletime = "20171122T0000Z"
        dt = 419520.0
        result = cycletime_to_number(cycletime, calendar="365_day")
        self.assertAlmostEqual(result, dt)


class Test_iris_time_to_datetime(IrisTest):
    """Test iris_time_to_datetime"""

    def setUp(self):
        """Set up an input cube"""
        self.cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            time=datetime(2017, 2, 17, 6, 0),
            frt=datetime(2017, 2, 17, 3, 0),
        )

    def test_basic(self):
        """Test iris_time_to_datetime returns list of datetime"""
        result = iris_time_to_datetime(self.cube.coord("time"))
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, cftime.DatetimeGregorian)
        self.assertEqual(result[0], datetime(2017, 2, 17, 6, 0))

    def test_bounds(self):
        """Test iris_time_to_datetime returns list of datetimes calculated
        from the coordinate bounds."""
        # Assign time bounds equivalent to [
        # datetime(2017, 2, 17, 5, 0),
        # datetime(2017, 2, 17, 6, 0)]
        self.cube.coord("time").bounds = [1487307600, 1487311200]

        result = iris_time_to_datetime(self.cube.coord("time"), point_or_bound="bound")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)
        for item in result[0]:
            self.assertIsInstance(item, cftime.DatetimeGregorian)
        self.assertEqual(result[0][0], datetime(2017, 2, 17, 5, 0))
        self.assertEqual(result[0][1], datetime(2017, 2, 17, 6, 0))

    def test_input_cube_unmodified(self):
        """Test that an input cube with unexpected coordinate units is not
        modified"""
        self.cube.coord("time").convert_units("hours since 1970-01-01 00:00:00")
        self.cube.coord("time").points = self.cube.coord("time").points.astype(np.int64)
        reference_coord = self.cube.coord("time").copy()
        iris_time_to_datetime(self.cube.coord("time"))
        self.assertArrayEqual(self.cube.coord("time").points, reference_coord.points)
        self.assertArrayEqual(self.cube.coord("time").units, reference_coord.units)
        self.assertEqual(self.cube.coord("time").dtype, np.int64)


class Test_datetime_to_iris_time(IrisTest):
    """Test the datetime_to_iris_time function."""

    def setUp(self):
        """Define datetime for use in tests."""
        self.dt_in = datetime(2017, 2, 17, 6, 0)
        self.cftime_in = cftime.DatetimeGregorian(2017, 2, 17, hour=6, minute=0)
        self.expected = 1487311200.0

    def test_seconds(self):
        """Test datetime_to_iris_time returns float with expected value
        in seconds"""
        result = datetime_to_iris_time(self.dt_in)
        self.assertIsInstance(result, np.int64)
        self.assertEqual(result, self.expected)

    def test_cftime(self):
        """Test datetime_to_iris_time returns float with expected value
        in seconds when a cftime.DatetimeGregorian object is provided."""
        result = datetime_to_iris_time(self.cftime_in)
        self.assertIsInstance(result, np.int64)
        self.assertEqual(result, self.expected)


class Test_datetime_constraint(IrisTest):
    """
    Test construction of an iris.Constraint from a python datetime object.
    """

    def setUp(self):
        """Set up test cubes"""
        cube = set_up_variable_cube(
            np.ones((12, 12), dtype=np.float32),
            time=datetime(2017, 2, 17, 6, 0),
            frt=datetime(2017, 2, 17, 6, 0),
        )
        cube.remove_coord("forecast_period")
        self.time_points = np.arange(1487311200, 1487354400, 3600).astype(np.int64)
        self.cube = add_coordinate(
            cube,
            self.time_points,
            "time",
            dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00",
        )

    def test_constraint_list_equality(self):
        """Check a list of constraints is as expected."""
        plugin = datetime_constraint
        time_start = datetime(2017, 2, 17, 6, 0)
        time_limit = datetime(2017, 2, 17, 18, 0)
        dt_constraint = plugin(time_start, time_max=time_limit)
        result = self.cube.extract(dt_constraint)
        self.assertEqual(result.shape, (12, 12, 12))
        self.assertArrayEqual(result.coord("time").points, self.time_points)

    def test_constraint_type(self):
        """Check type is iris.Constraint."""
        plugin = datetime_constraint
        dt_constraint = plugin(datetime(2017, 2, 17, 6, 0))
        self.assertIsInstance(dt_constraint, iris.Constraint)

    def test_valid_constraint(self):
        """Test use of constraint at a time valid within the cube."""
        plugin = datetime_constraint
        dt_constraint = plugin(datetime(2017, 2, 17, 6, 0))
        result = self.cube.extract(dt_constraint)
        self.assertIsInstance(result, Cube)

    def test_invalid_constraint(self):
        """Test use of constraint at a time invalid within the cube."""
        plugin = datetime_constraint
        dt_constraint = plugin(datetime(2017, 2, 17, 18, 0))
        result = self.cube.extract(dt_constraint)
        self.assertNotIsInstance(result, Cube)


class Test_extract_cube_at_time(IrisTest):
    """
    Test wrapper for iris cube extraction at desired times.
    """

    def setUp(self):
        """Set up a test cube with several time points"""
        cube = set_up_variable_cube(
            np.ones((12, 12), dtype=np.float32),
            time=datetime(2017, 2, 17, 6, 0),
            frt=datetime(2017, 2, 17, 6, 0),
        )
        cube.remove_coord("forecast_period")
        self.time_points = np.arange(1487311200, 1487354400, 3600).astype(np.int64)
        self.cube = add_coordinate(
            cube,
            self.time_points,
            "time",
            dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00",
        )
        self.time_dt = datetime(2017, 2, 17, 6, 0)
        self.time_constraint = iris.Constraint(
            time=lambda cell: cell.point
            == PartialDateTime(
                self.time_dt.year,
                self.time_dt.month,
                self.time_dt.day,
                self.time_dt.hour,
            )
        )

    def test_valid_time(self):
        """Case for a time that is available within the diagnostic cube."""
        plugin = extract_cube_at_time
        cubes = CubeList([self.cube])
        result = plugin(cubes, self.time_dt, self.time_constraint)
        self.assertIsInstance(result, Cube)

    def test_valid_time_for_coord_with_bounds(self):
        """Case for a time that is available within the diagnostic cube.
        Test it still works for coordinates with bounds."""
        plugin = extract_cube_at_time
        self.cube.coord("time").guess_bounds()
        cubes = CubeList([self.cube])
        result = plugin(cubes, self.time_dt, self.time_constraint)
        self.assertIsInstance(result, Cube)

    def test_invalid_time(self):
        """Case for a time that is unavailable within the diagnostic cube."""
        plugin = extract_cube_at_time
        time_dt = datetime(2017, 2, 18, 6, 0)
        time_constraint = iris.Constraint(
            time=PartialDateTime(time_dt.year, time_dt.month, time_dt.day, time_dt.hour)
        )
        cubes = CubeList([self.cube])
        warning_msg = "Forecast time"

        with pytest.warns(UserWarning, match=warning_msg):
            plugin(cubes, time_dt, time_constraint)


class Test_extract_nearest_time_point(IrisTest):
    """Test the extract_nearest_time_point function."""

    def setUp(self):
        """Set up a cube for the tests."""
        cube = set_up_variable_cube(
            np.ones((1, 7, 7), dtype=np.float32),
            time=datetime(2015, 11, 23, 7, 0),
            frt=datetime(2015, 11, 23, 3, 0),
        )
        cube.remove_coord("forecast_period")
        time_points = [1448262000, 1448265600]
        self.cube = add_coordinate(
            cube,
            time_points,
            "time",
            dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00",
            order=[1, 0, 2, 3],
        )

    def test_time_coord(self):
        """Test that the nearest time point within the time coordinate is
        extracted."""
        expected = self.cube[:, 0, :, :]
        time_point = datetime(2015, 11, 23, 6, 31)
        result = extract_nearest_time_point(
            self.cube, time_point, allowed_dt_difference=1800
        )
        self.assertEqual(result, expected)

    def test_time_coord_lower_case(self):
        """Test that the nearest time point within the time coordinate is
        extracted, when a time of 07:30 is requested."""
        expected = self.cube[:, 0, :, :]
        time_point = datetime(2015, 11, 23, 7, 30)
        result = extract_nearest_time_point(
            self.cube, time_point, allowed_dt_difference=1800
        )
        self.assertEqual(result, expected)

    def test_time_coord_upper_case(self):
        """Test that the nearest time point within the time coordinate is
        extracted, when a time of 07:31 is requested."""
        expected = self.cube[:, 1, :, :]
        time_point = datetime(2015, 11, 23, 7, 31)
        result = extract_nearest_time_point(
            self.cube, time_point, allowed_dt_difference=1800
        )
        self.assertEqual(result, expected)

    def test_forecast_reference_time_coord(self):
        """Test that the nearest time point within the forecast_reference_time
        coordinate is extracted."""
        later_frt = self.cube.copy()
        later_frt.coord("forecast_reference_time").points = (
            later_frt.coord("forecast_reference_time").points + 3600
        )
        cubes = iris.cube.CubeList([self.cube, later_frt])
        cube = cubes.merge_cube()
        expected = self.cube
        time_point = datetime(2015, 11, 23, 3, 29)
        result = extract_nearest_time_point(
            cube,
            time_point,
            time_name="forecast_reference_time",
            allowed_dt_difference=1800,
        )
        self.assertEqual(result, expected)

    def test_exception_using_allowed_dt_difference(self):
        """Test that an exception is raised, if the time point is outside of
        the allowed difference specified in seconds."""
        time_point = datetime(2017, 11, 23, 6, 0)
        msg = "is not available within the input cube"
        with self.assertRaisesRegex(ValueError, msg):
            extract_nearest_time_point(
                self.cube, time_point, allowed_dt_difference=3600
            )

    def test_time_name_exception(self):
        """Test that an exception is raised, if an invalid time name
        is specified."""
        time_point = datetime(2017, 11, 23, 6, 0)
        msg = "The time_name must be either 'time' or 'forecast_reference_time'"
        with self.assertRaisesRegex(ValueError, msg):
            extract_nearest_time_point(
                self.cube, time_point, time_name="forecast_period"
            )


class Test_relabel_to_period(unittest.TestCase):
    """Test relabel_to_period function."""

    def setUp(self):
        """Set up cubes for testing."""
        self.cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            time=datetime(2017, 2, 17, 6, 0),
            frt=datetime(2017, 2, 17, 3, 0),
        )
        self.cube_with_bounds = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            time=datetime(2017, 2, 17, 6, 0),
            time_bounds=(datetime(2017, 2, 17, 5, 0), datetime(2017, 2, 17, 6, 0)),
            frt=datetime(2017, 2, 17, 3, 0),
        )

    def test_basic(self):
        """Test correct bounds present on time and forecast_period coordinates
        for instantaneous input."""
        expected_time = self.cube.coord("time").copy()
        expected_time.bounds = np.array(
            [
                datetime_to_iris_time(datetime(2017, 2, 17, 5, 0)),
                datetime_to_iris_time(datetime(2017, 2, 17, 6, 0)),
            ],
            dtype=TIME_COORDS["time"].dtype,
        )
        expected_fp = self.cube.coord("forecast_period").copy()
        expected_fp.bounds = np.array(
            [2 * 3600, 3 * 3600], TIME_COORDS["forecast_period"].dtype
        )
        result = relabel_to_period(self.cube, 1)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.coord("time"), expected_time)
        self.assertEqual(result.coord("forecast_period"), expected_fp)

    def test_input_period_diagnostic(self):
        """Test correct bounds present on time and forecast_period coordinates
        for an input period diagnostic."""
        expected_time = self.cube.coord("time").copy()
        expected_time.bounds = np.array(
            [
                datetime_to_iris_time(datetime(2017, 2, 17, 3, 0)),
                datetime_to_iris_time(datetime(2017, 2, 17, 6, 0)),
            ],
            dtype=TIME_COORDS["time"].dtype,
        )
        expected_fp = self.cube.coord("forecast_period").copy()
        expected_fp.bounds = np.array(
            [0, 3 * 3600], dtype=TIME_COORDS["forecast_period"].dtype
        )
        result = relabel_to_period(self.cube_with_bounds, 3)
        self.assertEqual(result.coord("time"), expected_time)
        self.assertEqual(result.coord("forecast_period"), expected_fp)

    def test_no_period(self):
        """Test error raised when no period supplied."""
        msg = "A period must be specified when relabelling a diagnostic"
        with self.assertRaisesRegex(ValueError, msg):
            relabel_to_period(self.cube)

    def test_zero_period(self):
        """Test error raised when an invalid value for the period is supplied."""
        msg = "Only periods of one hour or greater are supported"
        with self.assertRaisesRegex(ValueError, msg):
            relabel_to_period(self.cube, period=0)


@pytest.fixture
def period_cube(data, period_lengths):
    """
    Generates a cube of average lightning flash rate within a period.

    Args:
        data:
            Data with a leading dimension of the same length as the
            period_lengths list.
        period_lengths:
            A list of period lengths in seconds. If multiple period lengths
            are provided the returned cube will have a leading time dimension.
    Returns:
        A period cube with units of m-2 s-1 for use in testing time
        integration using the period defined by the time bounds.
    """

    frt = datetime(2024, 6, 11, 12)
    # Enable consistent slicing even if 1 period and a 2D data array are passed in
    if len(period_lengths) == 1:
        data = np.expand_dims(data, 0)

    cubes = CubeList()
    for ii, dslice in enumerate(data):
        # Sum periods to get time so that they don't overlap.
        time_offset = np.sum(period_lengths[: ii + 1]).astype(np.float64)
        time = frt + timedelta(seconds=time_offset)
        cubes.append(
            set_up_variable_cube(
                dslice.astype(np.float32),
                name="frequency_of_lightning_flashes_per_unit_area",
                units="m-2 s-1",
                time=time,
                frt=frt,
                time_bounds=((time - timedelta(seconds=period_lengths[ii])), time),
            )
        )

    cube = cubes.merge_cube()
    cube.cell_methods = [
        CellMethod("point", coords=["latitude", "longitude"]),
        CellMethod("max", coords=["time"]),
    ]
    return cube


@pytest.mark.parametrize(
    "kwargs,data,period_lengths,expected",
    [
        # Array with a rate of 1 per second, with a period of 3-hours.
        # The resulting data are a count of 10800 (3-hours in seconds * rate)
        ({}, np.ones((5, 5)), [10800], np.full((5, 5), 10800)),
        # Array with two rates, 1/200 per second and 3/400 per second. Two
        # periods of 3 and 2 hours leading to counts of 54 in each period;
        # (10800 * 0.005) and (7200 * 0.0075).
        (
            {},
            np.full((2, 5, 5), [[[0.005]], [[0.0075]]]),
            [10800, 7200],
            np.full((2, 5, 5), 54),
        ),
        # Array with rates of 1/200 per second. Two periods of 3 and 2 hours.
        # The resulting data are a count of 54 (10800 * 0.005), and 36
        # (7200 * 0.005).
        (
            {},
            np.full((2, 5, 5), 0.005),
            [10800, 7200],
            np.full((2, 5, 5), [[[54]], [[36]]]),
        ),
        # Duplicates the first test but sets a new name for the resulting
        # diagnostic and tests that it is used.
        (
            {"new_name": "number_of_lightning_flashes_per_unit_area"},
            np.ones((5, 5)),
            [10800],
            np.full((5, 5), 10800),
        ),
    ],
)
def test_integrate_time(period_cube, kwargs, expected):
    """Tests for the integrate_time function. Checks that the data is as
    expected following multiplication by the time period. Checks the units
    have been updated, a suitable cell method has been added, the time
    coordinate is unchanged, still describing a period, and that if a new
    diagnostic name is specified, this as been applied. Also checks that
    existing cell_methods related to time are removed, and those not related
    to time are preserved."""

    result = integrate_time(period_cube.copy(), **kwargs)

    np.testing.assert_array_equal(result.data, expected)
    assert result.units == Unit("m-2")

    cm_names = [name for cm in result.cell_methods for name in cm.coord_names]
    cm_methods = [cm.method for cm in result.cell_methods]

    assert "sum" in cm_methods
    assert "point" in cm_methods
    assert "max" not in cm_methods
    assert "time" in cm_names
    assert "latitude" in cm_names
    assert "longitude" in cm_names

    assert result.coord("time") == period_cube.coord("time")
    if kwargs:
        assert result.name() == kwargs["new_name"]


@pytest.mark.parametrize("data,period_lengths", [(np.ones((5, 5)), [3600])])
def test_integrate_non_second_units(period_cube):
    """Test that input data with a rate expressed in a different time unit,
    e.g. per minute, is returned with units that describe the data
    correctly. The data itself will be expressed as an integral over seconds
    but the cube units will include the necessary factor to account for the
    differing input units."""

    expected = np.full(period_cube.shape, 3600)
    expected_units = Unit(f"{1./60} m-2")
    period_cube.units = Unit("m-2 minute-1")

    result = integrate_time(period_cube.copy())

    np.testing.assert_array_equal(result.data, expected)
    assert result.units == expected_units


@pytest.mark.parametrize("data,period_lengths", [(np.ones((5, 5)), [10800])])
def test_integrate_time_exception(period_cube):
    """Tests that the integrate_time function raises an exception if the cube
    passed in does not have bounds on the time coordinate."""

    period_cube.coord("time").bounds = None
    with pytest.raises(ValueError, match="time coordinate must have bounds"):
        integrate_time(period_cube.copy())


if __name__ == "__main__":
    unittest.main()
