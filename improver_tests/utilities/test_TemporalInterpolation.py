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
"""Unit tests for temporal utilities."""

import datetime
from datetime import datetime as dt
import unittest

from typing import List, Tuple, Optional

import iris
import numpy as np
import pytest

from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.temporal_interpolation import TemporalInterpolation


def _grid_params(spatial_grid: str, npoints: int) -> Tuple[Tuple[float, float], float]:
    """Set domain corner and grid spacing for lat-lon or equal area
    projections."""

    domain_corner = None
    grid_spacing = None
    if spatial_grid == "latlon":
        domain_corner = (40, -20)
        grid_spacing = 40 / (npoints - 1)
    elif spatial_grid == "equalarea":
        domain_corner = (-100000, -400000)
        grid_spacing = np.around(1000000.0 / npoints)
    return domain_corner, grid_spacing


def diagnostic_cube(time: dt, frt: dt, data: np.ndarray, spatial_grid: str, realizations: Optional[List] = None) -> Cube:
    """Return a diagnostic cube containing the provided data.

    Args:
        time:
            A datetime object that gives the validity time of the cube.
        frt:
            The forecast reference time for the cube.
        data:
            The data to be contained in the cube.
        spatial_grid:
            Whether this is a lat-lon or equal areas projection.
    Returns:
        A diagnostic cube for use in testing.
    """
    npoints = data.shape[0]
    domain_corner, grid_spacing = _grid_params(spatial_grid, npoints)

    if realizations:
        data = np.stack([data] * len(realizations))

    return set_up_variable_cube(
        data,
        time=time,
        frt=frt,
        spatial_grid=spatial_grid,
        domain_corner=domain_corner,
        grid_spacing=grid_spacing,
        realizations=realizations,
    )


def multi_time_cube(times: List, data: np.ndarray, spatial_grid: str, bounds: bool = False, realizations: Optional[List] = None) -> Cube:
    """Return a multi-time diagnostic cube containing the provided data.

    Args:
        times:
            A list of datetime objects that gives the validity times for
            the cube.
        data:
            The data to be contained in the cube. If the cube is 3-D the
            leading dimension should be the same size and the list of times
            and will be sliced to associate each slice with each time.
        spatial_grid:
            Whether this is a lat-lon or equal areas projection.
        bounds:
            If True return time coordinates with time bounds.
    Returns:
        A diagnostic cube for use in testing.
    """
    cubes = CubeList()
    if data.ndim == 2:
        data = np.stack([data] * len(times))

    frt = sorted(times)[0] - (times[1] - times[0])  # Such that guess bounds is +ve
    for time, data_slice in zip(times, data):
        cubes.append(diagnostic_cube(time, frt, data_slice, spatial_grid, realizations=realizations))

    cube = cubes.merge_cube()

    if bounds:
        for crd in ["time", "forecast_period"]:
            cube.coord(crd).guess_bounds(bound_position=1.0)
    return cube


def non_standard_times(times: List, data: np.ndarray, spatial_grid: str, bounds: bool = False) -> Cube:
    """Return a multi-time diagnostic cube containing the provided data.
    The units of the time dimensions are made non-standards compliant.

    Args:
        times:
            A list of datetime objects that gives the validity times for
            the cube.
        data:
            The data to be contained in the cube. If the cube is 3-D the
            leading dimension should be the same size and the list of times
            and will be sliced to associate each slice with each time.
        spatial_grid:
            Whether this is a lat-lon or equal areas projection.
        bounds:
            If True return time coordinates with time bounds.
    Returns:
        A diagnostic cube for use in testing.
    """
    cube = multi_time_cube(times, data, spatial_grid, bounds=bounds)

    epoch = "hours since 1970-01-01 00:00:00"
    for crd in ["time", "forecast_reference_time"]:
        cube.coord(crd).convert_units(epoch)
        cube.coord(crd).points = cube.coord(crd).points.astype(np.int32)

    cube.coord("forecast_period").convert_units("hours")
    cube.coord("forecast_period").points.astype(np.float32)

    return cube


@pytest.fixture
def solar_expected():
    """Return the expected values for the solar interpolation tests."""
    return np.array(
        [
            [0.02358028, 0.15887623, 0.2501732, 0.32049885, 0.3806127],
            [0.0, 0.09494493, 0.21051247, 0.2947393, 0.36431003],
            [0.0, 0.0, 0.11747278, 0.23689085, 0.32841164],
            [0.0, 0.0, 0.0, 0.0, 0.15872595],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )


@pytest.fixture
def daynight_mask():
    """The mask matching the terminator position for the daynight
    interpolation tests."""
    return np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({}, "TemporalInterpolation: One of"),  # No target times defined
        ({"interval_in_minutes":60, "times":[datetime.datetime(2017, 11, 1, 9)]}, "TemporalInterpolation: Only one of"),  # Two methods of defining targets used
        ({"interval_in_minutes":60, "interpolation_method":"invalid"}, "TemporalInterpolation: Unknown interpolation method"),  # Invalid interpolation method requested
    ]
)
def test__init__(kwargs, exception):
    """Test exceptions raised by the __init__ method."""
    with pytest.raises(ValueError, match=exception):
        TemporalInterpolation(**kwargs)


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"interval_in_minutes": 60}, None),  # Generate times between bounds using interval
        ({"times": None}, None),  # Use the expected times as the input
        ({"times": datetime.datetime(2017, 11, 1, 10)}, "List of times falls outside the range given by"),  # Use the expected times, plus another outside the range as the input
        ({"interval_in_minutes":61}, "interval_in_minutes of"),  # Use an invalid interval
    ]
)
def test_construct_time_list(kwargs, exception):
    """Test construction of target times using various inputs and testing
    exceptions that can be raised."""
    time_0 = datetime.datetime(2017, 11, 1, 3)
    time_1 = datetime.datetime(2017, 11, 1, 9)

    # Expected times are all those interpolated to and time_1.
    times = []
    for i in range(4, 10):
        times.append(datetime.datetime(2017, 11, 1, i))
    expected = [("time", list(times))]

    # If a times kwarg is supplied, populate the value with the default
    # times plus any others specified in the kwarg.
    try:
        target_times = times.copy()
        target_times.append(kwargs["times"]) if kwargs["times"] is not None else target_times
    except KeyError:
        pass
    else:
        kwargs["times"] = target_times

    plugin = TemporalInterpolation(**kwargs)

    # If an exception is provided as a kwarg test for this.
    if exception is not None:
        with pytest.raises(ValueError, match=exception):
            plugin.construct_time_list(time_0, time_1)
    else:
        result = plugin.construct_time_list(time_0, time_1)
        assert isinstance(result, list)
        assert expected == result


@pytest.mark.parametrize("bounds", (False, True))
def test_enforce_time_coords_dtype(bounds):
    """Test that the datatypes and units of the time, forecast_reference_time
    and forecast_period coordinates have been enforced."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 6, 9]]
    data = np.ones((10, 10), dtype=np.float32)
    cube = non_standard_times(times, data, "latlon", bounds=bounds)

    expected_coord_dtypes = {
        "time": np.int64,
        "forecast_reference_time": np.int64,
        "forecast_period": np.int32,
    }
    expected_coord_units = {
        "time": "seconds since 1970-01-01 00:00:00",
        "forecast_reference_time": "seconds since 1970-01-01 00:00:00",
        "forecast_period": "seconds",
    }

    plugin = TemporalInterpolation(interval_in_minutes=60)
    result = plugin.enforce_time_coords_dtype(cube.copy())

    assert isinstance(result, Cube)

    for crd, expected in expected_coord_dtypes.items():
        assert result.coord(crd).points.dtype == expected
        try:
            assert result.coord(crd).bounds.dtype == expected
        except AttributeError:
            pass
    for crd, expected in expected_coord_units.items():
        assert result.coord(crd).units == expected


def test_sin_phi():
    """Test that the function returns the values expected."""
    latitudes = np.array([50.0, 50.0, 50.0])
    longitudes = np.array([-5.0, 0.0, 5.0])
    dtval = datetime.datetime(2017, 1, 11, 8)
    expected_array = np.array([-0.05481607, -0.00803911, 0.03659632])
    plugin = TemporalInterpolation(
        interval_in_minutes=60, interpolation_method="solar"
    )
    result = plugin.calc_sin_phi(dtval, latitudes, longitudes)
    assert isinstance(result, np.ndarray)
    np.testing.assert_almost_equal(result, expected_array)


@pytest.mark.parametrize("spatial_grid,expected_lats,expected_lons",
    [
        (
            "latlon",
            np.array([[40.0, 40.0, 40.0], [60.0, 60.0, 60.0], [80.0, 80.0, 80.0]]),
            np.array([[-20.0, 0.0, 20.0], [-20.0, 0.0, 20.0], [-20.0, 0.0, 20.0]]),
        ),
        (
            "equalarea",
            np.array(
                [
                    [53.84618597, 53.99730779, 53.93247526],
                    [56.82670954, 56.99111356, 56.9205672],
                    [59.8045105, 59.98499383, 59.90752513],
                ]
            ),
            np.array(
                [
                    [-8.58580705, -3.51660018, 1.56242662],
                    [-9.06131306, -3.59656346, 1.88105082],
                    [-9.63368459, -3.69298822, 2.26497216],
                ]
            )
        ),
    ]
)
def test_calc_lats_lons(spatial_grid, expected_lats, expected_lons):
    """Test that the function returns the lats and lons expected for a native
    lat-lon projection and for an equal areas projection."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 6, 9]]
    data = np.ones((3, 3), dtype=np.float32)
    cube = multi_time_cube(times, data, spatial_grid)

    plugin = TemporalInterpolation(
        interval_in_minutes=60, interpolation_method="solar"
    )
    result_lats, result_lons = plugin.calc_lats_lons(cube)
    assert isinstance(result_lats, np.ndarray)
    assert result_lats.shape == (3, 3)
    assert isinstance(result_lons, np.ndarray)
    assert result_lons.shape == (3, 3)
    np.testing.assert_almost_equal(result_lats, expected_lats)
    np.testing.assert_almost_equal(result_lons, expected_lons)


@pytest.mark.parametrize("realizations", (None, [0, 1, 2]))
def test_solar_interpolation(solar_expected, realizations):
    """Test interpolation using the solar method. Apply it to deterministic
    and ensemble data."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [6, 8, 10]]
    npoints = 5
    data = np.stack(
        [
            np.zeros((npoints, npoints), dtype=np.float32),
            np.ones((npoints, npoints), dtype=np.float32),
            np.ones((npoints, npoints), dtype=np.float32),
        ]
    )
    cube = multi_time_cube(times, data, "latlon", realizations=realizations)
    interpolated_cube = cube[1].copy()
    interpolated_cube = iris.util.new_axis(interpolated_cube, "time")
    cube = cube[0::2]

    plugin = TemporalInterpolation(interpolation_method="solar", times=[times[1]])

    result = plugin.solar_interpolate(cube, interpolated_cube)
    assert isinstance(result, CubeList)
    result, = result
    assert result.coord("time").points == 1509523200
    assert result.coord("forecast_period").points[0] == 3600 * 4
    if result.ndim == 2:
        np.testing.assert_almost_equal(result.data, solar_expected)
    else:
        for dslice in result.data:
            np.testing.assert_almost_equal(dslice, solar_expected)


@pytest.mark.parametrize("realizations", (None, [0, 1, 2]))
def test_daynight_interpolation(daynight_mask, realizations):
    """Test daynight function applies a suitable mask to interpolated
    data. In this test the day-night terminator crosses the domain to
    ensure the impact is captured. A deterministic and ensemble input
    are tested."""

    frt = datetime.datetime(2017, 11, 1, 6)
    time = datetime.datetime(2017, 11, 1, 8)
    data = np.ones((10, 10), dtype=np.float32) * 4
    interpolated_cube = diagnostic_cube(time, frt, data, "latlon", realizations=realizations)

    expected = np.where(daynight_mask == 0, 0, data)

    plugin = TemporalInterpolation(interpolation_method="daynight", times=[time])
    result = plugin.daynight_interpolate(interpolated_cube)
    assert isinstance(result, CubeList)

    result, = result
    assert result.coord("time").points == 1509523200
    assert result.coord("forecast_period").points[0] == 7200

    if result.ndim == 2:
        np.testing.assert_almost_equal(result.data, expected)
    else:
        for dslice in result.data:
            np.testing.assert_almost_equal(dslice, expected)


class Test_process(IrisTest):

    """Test interpolation of cubes to intermediate times using the plugin."""

    def setUp(self):
        """Set up the test inputs."""
        self.time_0 = datetime.datetime(2017, 11, 1, 3)
        self.time_extra = datetime.datetime(2017, 11, 1, 6)
        self.time_1 = datetime.datetime(2017, 11, 1, 9)
        self.npoints = 10

        domain_corner, grid_spacing = _grid_params("latlon", self.npoints)

        data_time_0 = np.ones((self.npoints, self.npoints), dtype=np.float32)
        data_time_1 = np.ones((self.npoints, self.npoints), dtype=np.float32) * 7
        self.cube_time_0 = set_up_variable_cube(
            data_time_0,
            time=self.time_0,
            frt=self.time_0,
            domain_corner=domain_corner,
            grid_spacing=grid_spacing,
        )
        self.cube_time_1 = set_up_variable_cube(
            data_time_1,
            time=self.time_1,
            frt=self.time_0,
            domain_corner=domain_corner,
            grid_spacing=grid_spacing,
        )

    def test_return_type(self):
        """Test that an iris cubelist is returned."""

        result = TemporalInterpolation(interval_in_minutes=180).process(
            self.cube_time_0, self.cube_time_1
        )
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_wind_direction_interpolation_over_north(self):
        """Test that wind directions are interpolated properly at the 0/360
        circular cross-over."""

        data_time_0 = np.ones((self.npoints, self.npoints), dtype=np.float32) * 350
        data_time_1 = np.ones((self.npoints, self.npoints), dtype=np.float32) * 20
        domain_corner, grid_spacing = _grid_params("latlon", self.npoints)
        cube_time_0 = set_up_variable_cube(
            data_time_0,
            name="wind_from_direction",
            units="degrees",
            time=self.time_0,
            frt=self.time_0,
            domain_corner=domain_corner,
            grid_spacing=grid_spacing,
        )
        cube_time_1 = set_up_variable_cube(
            data_time_1,
            name="wind_from_direction",
            units="degrees",
            time=self.time_1,
            frt=self.time_1,
            domain_corner=domain_corner,
            grid_spacing=grid_spacing,
        )
        expected_data = np.full((self.npoints, self.npoints), 5, dtype=np.float32)
        (result,) = TemporalInterpolation(interval_in_minutes=180).process(
            cube_time_0, cube_time_1
        )

        self.assertArrayAlmostEqual(expected_data, result.data, decimal=4)

    def test_wind_direction_interpolation(self):
        """Test that wind directions are interpolated properly when the interpolation
        doesn't cross the 0/360 boundary."""

        data_time_0 = np.ones((self.npoints, self.npoints), dtype=np.float32) * 40
        data_time_1 = np.ones((self.npoints, self.npoints), dtype=np.float32) * 60
        domain_corner, grid_spacing = _grid_params("latlon", self.npoints)
        cube_time_0 = set_up_variable_cube(
            data_time_0,
            units="degrees",
            time=self.time_0,
            frt=self.time_0,
            domain_corner=domain_corner,
            grid_spacing=grid_spacing,
        )
        cube_time_1 = set_up_variable_cube(
            data_time_1,
            units="degrees",
            time=self.time_1,
            frt=self.time_1,
            domain_corner=domain_corner,
            grid_spacing=grid_spacing,
        )
        expected_data = expected_data = np.full(
            (self.npoints, self.npoints), 50, dtype=np.float32
        )
        (result,) = TemporalInterpolation(interval_in_minutes=180).process(
            cube_time_0, cube_time_1
        )

        self.assertArrayAlmostEqual(expected_data, result.data, decimal=4)

    def test_valid_single_interpolation(self):
        """Test interpolating to the mid point of the time range. Expect the
        data to be half way between, and the time coordinate should be at
        06Z November 11th 2017."""

        expected_data = np.ones((self.npoints, self.npoints)) * 4
        expected_time = [1509516000]
        expected_fp = 3 * 3600
        (result,) = TemporalInterpolation(interval_in_minutes=180).process(
            self.cube_time_0, self.cube_time_1
        )

        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertArrayAlmostEqual(result.coord("time").points, expected_time)
        self.assertAlmostEqual(result.coord("forecast_period").points[0], expected_fp)

    def test_valid_multiple_interpolations(self):
        """Test interpolating to every hour between the two input cubes.
        Check the data increments as expected and the time coordinates are also
        set correctly.

        NB Interpolation in iris is prone to float precision errors of order
        10E-6, hence the need to use AlmostEqual below."""

        result = TemporalInterpolation(interval_in_minutes=60).process(
            self.cube_time_0, self.cube_time_1
        )
        for i, cube in enumerate(result):
            expected_data = np.ones((self.npoints, self.npoints)) * i + 2
            expected_time = [1509508800 + i * 3600]

            self.assertArrayAlmostEqual(expected_data, cube.data)
            self.assertArrayAlmostEqual(
                cube.coord("time").points, expected_time, decimal=5
            )
            self.assertAlmostEqual(
                cube.coord("forecast_period").points[0], (i + 1) * 3600
            )

    def test_valid_interpolation_from_given_list(self):
        """Test interpolating to a point defined in a list between the two
        input cube validity times. Check the data increments as expected and
        the time coordinates are also set correctly.

        NB Interpolation in iris is prone to float precision errors of order
        10E-6, hence the need to use AlmostEqual below."""

        (result,) = TemporalInterpolation(times=[self.time_extra]).process(
            self.cube_time_0, self.cube_time_1
        )
        expected_data = np.ones((self.npoints, self.npoints)) * 4
        expected_time = [1509516000]
        expected_fp = 3 * 3600

        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertArrayAlmostEqual(
            result.coord("time").points, expected_time, decimal=5
        )
        self.assertAlmostEqual(result.coord("forecast_period").points[0], expected_fp)
        self.assertEqual(result.data.dtype, np.float32)
        self.assertEqual(str(result.coord("time").points.dtype), "int64")
        self.assertEqual(str(result.coord("forecast_period").points.dtype), "int32")

    def test_solar_interpolation_from_given_list(self):
        """Test solar interpolating to a point defined in a list
        between the two input cube validity times.
        Check the data increments as expected and
        the time coordinates are also set correctly."""

        plugin = TemporalInterpolation(
            times=[self.time_extra], interpolation_method="solar"
        )
        (result,) = plugin.process(self.cube_time_0, self.cube_time_1)
        expected_time = [1509516000]
        expected_fp = 3 * 3600

        self.assertArrayAlmostEqual(
            result.coord("time").points, expected_time, decimal=5
        )
        self.assertAlmostEqual(result.coord("forecast_period").points[0], expected_fp)
        self.assertEqual(result.data.dtype, np.float32)
        self.assertEqual(str(result.coord("time").points.dtype), "int64")
        self.assertEqual(str(result.coord("forecast_period").points.dtype), "int32")

    def test_daynight_interpolation_from_given_list(self):
        """Test daynight interpolating to a point defined in a list
        between the two input cube validity times.
        Check the data increments as expected and
        the time coordinates are also set correctly."""

        plugin = TemporalInterpolation(
            times=[self.time_extra], interpolation_method="daynight"
        )
        (result,) = plugin.process(self.cube_time_0, self.cube_time_1)
        expected_data = np.zeros((self.npoints, self.npoints))
        expected_data[:2, 7:] = 4.0
        expected_data[2, 8:] = 4.0
        expected_data[3, 9] = 4.0
        expected_time = [1509516000]
        expected_fp = 3 * 3600

        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertArrayAlmostEqual(
            result.coord("time").points, expected_time, decimal=5
        )
        self.assertAlmostEqual(result.coord("forecast_period").points[0], expected_fp)
        self.assertEqual(result.data.dtype, np.float32)
        self.assertEqual(str(result.coord("time").points.dtype), "int64")
        self.assertEqual(str(result.coord("forecast_period").points.dtype), "int32")

    def test_input_cube_without_time_coordinate(self):
        """Test that an exception is raised if a cube is provided without a
        time coordiate."""

        self.cube_time_0.remove_coord("time")

        msg = "Cube provided to TemporalInterpolation " "contains no time coordinate"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            TemporalInterpolation(interval_in_minutes=180).process(
                self.cube_time_0, self.cube_time_1
            )

    def test_input_cubes_in_incorrect_time_order(self):
        """Test that an exception is raised if the cube representing the
        initial time has a validity time that is after the cube representing
        the final time."""

        msg = "TemporalInterpolation input cubes ordered incorrectly"
        with self.assertRaisesRegex(ValueError, msg):
            TemporalInterpolation(interval_in_minutes=180).process(
                self.cube_time_1, self.cube_time_0
            )

    def test_input_cube_with_multiple_times(self):
        """Test that an exception is raised if a cube is provided that has
        multiple validity times, e.g. a multi-entried time dimension."""

        second_time = self.cube_time_0.copy()
        second_time.coord("time").points = self.time_extra.timestamp()
        cube = iris.cube.CubeList([self.cube_time_0, second_time])
        cube = cube.merge_cube()

        msg = "Cube provided to TemporalInterpolation contains multiple"
        with self.assertRaisesRegex(ValueError, msg):
            TemporalInterpolation(interval_in_minutes=180).process(
                cube, self.cube_time_1
            )

    def test_input_cubelists_raises_exception(self):
        """Test that providing cubelists instead of cubes raises an
        exception."""

        cubes = iris.cube.CubeList([self.cube_time_1])

        msg = "Inputs to TemporalInterpolation are not of type "
        with self.assertRaisesRegex(TypeError, msg):
            TemporalInterpolation(interval_in_minutes=180).process(
                cubes, self.cube_time_0
            )


if __name__ == "__main__":
    unittest.main()
