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
from typing import List, Optional, Tuple

import iris
import numpy as np
import pytest
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.temporal_interpolation import TemporalInterpolation


def _grid_params(spatial_grid: str, npoints: int) -> Tuple[Tuple[float, float], float]:
    """Set domain corner and grid spacing for lat-lon or equal area
    projections.

    Args:
        spatial_grid:
            "latlon" or "equalarea" to determine the type of projection.
        npoints:
            The number of grid points to use in both x and y.
    Returns:
        A tuple containing a further tuple that includes the grid corner
        coordinates, and a single value specifying the grid spacing.
    """

    domain_corner = None
    grid_spacing = None
    if spatial_grid == "latlon":
        domain_corner = (40, -20)
        grid_spacing = 40 / (npoints - 1)
    elif spatial_grid == "equalarea":
        domain_corner = (-100000, -400000)
        grid_spacing = np.around(1000000.0 / npoints)
    return domain_corner, grid_spacing


def diagnostic_cube(
    time: dt,
    frt: dt,
    data: np.ndarray,
    spatial_grid: str,
    realizations: Optional[List] = None,
) -> Cube:
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
        realizations:
            An optional list of realizations identifiers. The length of this
            list will determine how many realizations are created.
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


def multi_time_cube(
    times: List,
    data: np.ndarray,
    spatial_grid: str,
    bounds: bool = False,
    realizations: Optional[List] = None,
) -> Cube:
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
        realizations:
            An optional list of realizations identifiers. The length of this
            list will determine how many realizations are created.
    Returns:
        A diagnostic cube for use in testing.
    """
    cubes = CubeList()
    if data.ndim == 2:
        data = np.stack([data] * len(times))

    frt = sorted(times)[0] - (times[1] - times[0])  # Such that guess bounds are +ve
    for time, data_slice in zip(times, data):
        cubes.append(
            diagnostic_cube(
                time, frt, data_slice, spatial_grid, realizations=realizations
            )
        )

    cube = cubes.merge_cube()

    if bounds:
        for crd in ["time", "forecast_period"]:
            cube.coord(crd).guess_bounds(bound_position=1.0)
    return cube


def non_standard_times(
    times: List, data: np.ndarray, spatial_grid: str, bounds: bool = False
) -> Cube:
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


def mask_values():
    """The mask matching the terminator position for the daynight
    interpolation tests."""
    return np.array(
        [
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
            ],
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )


@pytest.fixture
def daynight_mask():
    """Fixture to return mask values."""
    return mask_values()


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({}, "TemporalInterpolation: One of"),  # No target times defined
        (
            {"interval_in_minutes": 60, "times": [datetime.datetime(2017, 11, 1, 9)]},
            "TemporalInterpolation: Only one of",
        ),  # Two methods of defining targets used
        (
            {"interval_in_minutes": 60, "interpolation_method": "invalid"},
            "TemporalInterpolation: Unknown interpolation method",
        ),  # Invalid interpolation method requested
        (
            {"interval_in_minutes": 60, "maximum": True, "minimum": True},
            "Only one type of period diagnostics may be specified:",
        ),  # Invalid interpolation method requested
    ],
)
def test__init__(kwargs, exception):
    """Test exceptions raised by the __init__ method."""
    with pytest.raises(ValueError, match=exception):
        TemporalInterpolation(**kwargs)


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        (
            {"interval_in_minutes": 60},
            None,
        ),  # Generate times between bounds using interval
        ({"times": None}, None),  # Use the expected times as the input
        (
            {"times": datetime.datetime(2017, 11, 1, 10)},
            "List of times falls outside the range given by",
        ),  # Use the expected times, plus another outside the range as the input
        (
            {"interval_in_minutes": 61},
            "interval_in_minutes of",
        ),  # Use an invalid interval
    ],
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
        target_times.append(kwargs["times"]) if kwargs[
            "times"
        ] is not None else target_times
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
    """Test that the function returns the values expected for solar
    elevation."""

    latitudes = np.array([50.0, 50.0, 50.0])
    longitudes = np.array([-5.0, 0.0, 5.0])
    dtval = datetime.datetime(2017, 1, 11, 8)
    expected_array = np.array([-0.05481607, -0.00803911, 0.03659632])
    plugin = TemporalInterpolation(interval_in_minutes=60, interpolation_method="solar")
    result = plugin.calc_sin_phi(dtval, latitudes, longitudes)
    assert isinstance(result, np.ndarray)
    np.testing.assert_almost_equal(result, expected_array)


@pytest.mark.parametrize(
    "spatial_grid,expected_lats,expected_lons",
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
            ),
        ),
    ],
)
def test_calc_lats_lons(spatial_grid, expected_lats, expected_lons):
    """Test that the function returns the lats and lons expected for a native
    lat-lon projection and for an equal areas projection."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 6, 9]]
    data = np.ones((3, 3), dtype=np.float32)
    cube = multi_time_cube(times, data, spatial_grid)

    plugin = TemporalInterpolation(interval_in_minutes=60, interpolation_method="solar")
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
    (result,) = result
    assert result.coord("time").points == 1509523200
    assert result.coord("forecast_period").points[0] == 3600 * 4
    if result.ndim == 2:
        np.testing.assert_almost_equal(result.data, solar_expected)
    else:
        for dslice in result.data:
            np.testing.assert_almost_equal(dslice, solar_expected)


@pytest.mark.parametrize("realizations", (None, [0, 1, 2]))
@pytest.mark.parametrize(
    "interp_times,expected_times,expected_fps",
    [
        ([8], [1509523200], [7200]),  # Interpolate to a single time.
        ([8, 10], [1509523200, 1509530400], [7200, 14400]),  # Interpolate to two times.
    ],
)
def test_daynight_interpolation(
    daynight_mask, realizations, interp_times, expected_times, expected_fps
):
    """Test daynight function applies a suitable mask to interpolated
    data. In this test the day-night terminator crosses the domain to
    ensure the impact is captured. A deterministic and ensemble input
    are tested."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in interp_times]
    data = np.ones((10, 10), dtype=np.float32) * 4
    if len(times) > 1:
        interpolated_cube = multi_time_cube(
            times, data, "latlon", realizations=realizations
        )
    else:
        frt = datetime.datetime(2017, 11, 1, 6)
        interpolated_cube = diagnostic_cube(
            times, frt, data, "latlon", realizations=realizations
        )

    plugin = TemporalInterpolation(interpolation_method="daynight", times=[times])
    result = plugin.daynight_interpolate(interpolated_cube)

    assert isinstance(result, CubeList)
    for index, cube in enumerate(result):
        expected = np.where(daynight_mask[index] == 0, 0, data)

        assert cube.coord("time").points == expected_times[index]
        assert cube.coord("forecast_period").points[0] == expected_fps[index]

        if cube.coords("realization"):
            cslices = cube.slices_over("realization")
        else:
            cslices = [cube]
        for cslice in cslices:
            np.testing.assert_almost_equal(cslice.data, expected)


@pytest.mark.parametrize("bearings,expected_value", [([350, 20], 5), ([40, 60], 50)])
def test_process_wind_direction(bearings, expected_value):
    """Test that wind directions are interpolated properly at the 0/360
    circular cross-over and away from this cross-over. The interpolated
    values are returned, along with the cube corresponding to the later
    input time. This later cube should be the same as the input."""
    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    npoints = 10
    data = np.stack(
        [
            np.ones((npoints, npoints), dtype=np.float32) * bearings[0],
            np.ones((npoints, npoints), dtype=np.float32) * bearings[1],
        ]
    )
    cube = multi_time_cube(times, data, "latlon")
    cube.rename("wind_from_direction")
    cube.units = "degrees"

    expected = np.full((npoints, npoints), expected_value, dtype=np.float32)
    result = TemporalInterpolation(interval_in_minutes=180).process(cube[0], cube[1])

    assert isinstance(result, CubeList)
    np.testing.assert_almost_equal(result[0].data, expected, decimal=4)
    # Ideally the data at the later of the input cube times would be
    # completely unchanged by the interpolation, but there are some small
    # precision level changes.
    np.testing.assert_almost_equal(result[1].data, cube[1].data, decimal=5)


@pytest.mark.parametrize(
    "kwargs,offsets,expected",
    [
        ({"interval_in_minutes": 180}, [3, 6], [4, 7]),
        ({"interval_in_minutes": 60}, [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]),
        ({"times": [datetime.datetime(2017, 11, 1, 6)]}, [3, 6], [4, 7]),
        (
            {
                "times": [datetime.datetime(2017, 11, 1, 8)],
                "interpolation_method": "daynight",
            },
            [5],
            [6 * mask_values()[0]],
        ),
        (
            {
                "times": [datetime.datetime(2017, 11, 1, 8)],
                "interpolation_method": "solar",
            },
            [5],
            [None],
        ),
    ],
)
def test_process_interpolation(kwargs, offsets, expected):
    """Test the process method with a variety of kwargs, selecting different
    interpolation methods and output times. Check that the returned times and
    data are as expected."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    npoints = 10
    data = np.stack(
        [
            np.ones((npoints, npoints), dtype=np.float32),
            np.ones((npoints, npoints), dtype=np.float32) * 7,
        ]
    )
    cube = multi_time_cube(times, data, "latlon")

    result = TemporalInterpolation(**kwargs).process(cube[0], cube[1])

    for i, (offset, value) in enumerate(zip(offsets, expected)):
        expected_data = np.full((npoints, npoints), value)
        expected_time = 1509505200 + (offset * 3600)
        expected_fp = (6 + offset) * 3600

        assert result[i].coord("time").points[0] == expected_time
        assert result[i].coord("forecast_period").points[0] == expected_fp
        assert result[i].coord("time").points.dtype == "int64"
        assert result[i].coord("forecast_period").points.dtype == "int32"
        if value is not None:
            np.testing.assert_almost_equal(result[i].data, expected_data)


def test_input_cube_without_time_coordinate():
    """Test that an exception is raised if a cube is provided without a
    time coordiate."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    data = np.ones((5, 5), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon")
    cube_0 = cube[0]
    cube_1 = cube[1]
    cube_1.remove_coord("time")

    msg = "Cube provided to TemporalInterpolation contains no time coordinate"
    with pytest.raises(CoordinateNotFoundError, match=msg):
        TemporalInterpolation(interval_in_minutes=180).process(cube_0, cube_1)


def test_input_cubes_in_incorrect_time_order():
    """Test that an exception is raised if the cube representing the
    initial time has a validity time that is after the cube representing
    the final time."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    data = np.ones((5, 5), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon")
    cube_0 = cube[0]
    cube_1 = cube[1]

    msg = "TemporalInterpolation input cubes ordered incorrectly"
    with pytest.raises(ValueError, match=msg):
        TemporalInterpolation(interval_in_minutes=180).process(cube_1, cube_0)


def test_input_cube_with_multiple_times():
    """Test that an exception is raised if a cube is provided that has
    multiple validity times, e.g. a multi-entried time dimension."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 6, 9]]
    data = np.ones((5, 5), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon")
    cube_0 = cube[0:2]
    cube_1 = cube[2]

    msg = "Cube provided to TemporalInterpolation contains multiple"
    with pytest.raises(ValueError, match=msg):
        TemporalInterpolation(interval_in_minutes=60).process(cube_0, cube_1)


def test_input_cubelists_raises_exception():
    """Test that providing cubelists instead of cubes raises an
    exception."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    data = np.ones((5, 5), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon")
    cubes = CubeList([cube[0], cube[1]])

    msg = "Inputs to TemporalInterpolation are not of type "
    with pytest.raises(TypeError, match=msg):
        TemporalInterpolation(interval_in_minutes=180).process(cubes, cube[1])


def test_invalid_method_for_period_exception():
    """Test that providing a period diagnostic and attempting to apply an
    unsuitable interpolation method raises an exception."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    data = np.ones((5, 5), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon", bounds=True)

    msg = "Period diagnostics can only be temporally interpolated"
    with pytest.raises(ValueError, match=msg):
        TemporalInterpolation(
            interval_in_minutes=180, interpolation_method="solar"
        ).process(cube[0], cube[1])


def test_period_without_chosen_type_exception():
    """Test that providing a period diagnostic but not specifying a type from
    the available minimum, maximum, or accumulation raises an exception."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    data = np.ones((5, 5), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon", bounds=True)

    msg = "A type of period must be specified when interpolating"
    with pytest.raises(ValueError, match=msg):
        TemporalInterpolation(interval_in_minutes=180).process(cube[0], cube[1])


def test_period_unequal_to_interval_exception():
    """Test that providing a period diagnostic where the represented
    periods overlap raises an exception."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    data = np.ones((5, 5), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon", bounds=True)
    for crd in ["time", "forecast_period"]:
        bounds = cube.coord(crd).bounds
        bounds = [[lower, upper + 3600] for lower, upper in bounds]
        cube.coord(crd).bounds = bounds

    msg = "The diagnostic provided represents the period"
    with pytest.raises(ValueError, match=msg):
        TemporalInterpolation(interval_in_minutes=180, accumulation=True).process(
            cube[0], cube[1]
        )


@pytest.mark.parametrize(
    "input_times,expected_time_bounds,expected_fp_bounds",
    [
        (
            [3, 6, 9],  # Forecast reference time ends up as 0 AM.
            [[1509505200, 1509516000], [1509516000, 1509526800]],  # 3-6, 6-9 AM
            [[10800, 21600], [21600, 32400]],  # T+3 - T+6, T+6 - T+9
        ),
        (
            [3, 4, 6, 9],  # Forecast reference time ends up as 2 AM.
            [
                [1509505200, 1509508800],
                [1509508800, 1509516000],
                [1509516000, 1509526800],
            ],  # 3-4, 4-6, 6-9 AM
            [
                [3600, 7200],
                [7200, 14400],
                [14400, 25200],
            ],  # T+1 - T+2, T+2 - T+4, T+4 - T+7
        ),
        (
            [3, 4, 9],  # Forecast reference time ends up as 2 AM.
            [[1509505200, 1509508800], [1509508800, 1509526800]],  # 3-4, 4-9 AM
            [[3600, 7200], [7200, 25200]],  # T+1 - T+2, T+2 - T+7
        ),
    ],
)
def test_add_bounds(input_times, expected_time_bounds, expected_fp_bounds):
    """Test the add bounds method creates the expected bounds for interpolated
    data with different interpolated time intervals."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in input_times]

    data = np.ones((5, 5), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon")
    # The first of the input times is used to represent the earlier of the
    # input cubes. The other input time represent the interpolated times.
    cube_t0 = cube[0]
    interpolated_cube = cube[1:].copy()

    # Note the interval_in_minutes defined here is not used but required.
    TemporalInterpolation(interval_in_minutes=60).add_bounds(cube_t0, interpolated_cube)

    assert (interpolated_cube.coord("time").bounds == expected_time_bounds).all()
    assert (
        interpolated_cube.coord("forecast_period").bounds == expected_fp_bounds
    ).all()


@pytest.mark.parametrize(
    "kwargs,values,offsets,expected",
    [
        # Equal adjacent accumulations, divided into equal shorter periods.
        (
            {"interval_in_minutes": 180, "accumulation": True},
            [5, 5],
            [3, 6],
            [2.5, 2.5],
        ),
        # Equal adjacent e.g. period maxes, shorter periods have the same max.
        ({"interval_in_minutes": 180, "maximum": True}, [5, 5], [3, 6], [5, 5],),
        # Equal adjacent e.g. period minimums, shorter periods have the same
        # min.
        ({"interval_in_minutes": 180, "minimum": True}, [5, 5], [3, 6], [5, 5],),
        # Trend of increasing accumulations with time, which is reflected
        # in the shorter periods generated.
        (
            {"interval_in_minutes": 180, "accumulation": True},
            [3, 9],
            [3, 6],
            [3.375, 5.625],
        ),
        # Trend of increasing maxes with time, which is reflected in the
        # shorter periods generated.
        ({"interval_in_minutes": 180, "maximum": True}, [3, 9], [3, 6], [6, 9],),
        # Later input period minimum is 9, expect all new periods to be >= 9
        ({"interval_in_minutes": 180, "minimum": True}, [3, 9], [3, 6], [9, 9],),
        # Trend of increasing accumulations with time, which is reflected
        # in the shorter periods generated.
        (
            {"interval_in_minutes": 120, "accumulation": True},
            [0, 9],
            [2, 4, 6],
            [1, 3, 5],
        ),
        # Trend of increasing maxes with time, which is reflected in the
        # shorter periods generated.
        ({"interval_in_minutes": 120, "maximum": True}, [0, 9], [2, 4, 6], [3, 6, 9],),
        # Trend of increasing maxes with time, which is reflected in the
        # shorter periods generated.
        ({"interval_in_minutes": 120, "minimum": True}, [0, 9], [2, 4, 6], [9, 9, 9],),
        # Later input period is 0, expect all new periods to be 0
        (
            {"interval_in_minutes": 120, "accumulation": True},
            [9, 0],
            [2, 4, 6],
            [0, 0, 0],
        ),
        # Later input period max is 0, expect all new periods to be 0
        ({"interval_in_minutes": 120, "maximum": True}, [9, 0], [2, 4, 6], [0, 0, 0],),
        # Later input period minimum is 0, expect all new periods to be >= 0
        ({"interval_in_minutes": 120, "minimum": True}, [9, 0], [2, 4, 6], [6, 3, 0],),
        # Equal adjacent accumulations, divided into unequal shorter periods.
        (
            {"times": [datetime.datetime(2017, 11, 1, 4)], "accumulation": True},
            [6, 6],
            [1, 6],
            [1, 5],
        ),
        # Equal adjacent e.g. period maxes, unequal shorter periods have the
        # same max.
        (
            {"times": [datetime.datetime(2017, 11, 1, 4)], "maximum": True},
            [6, 6],
            [1, 6],
            [6, 6],
        ),
        # Equal adjacent e.g. period minimums, unequal shorter periods have
        # the same min.
        (
            {"times": [datetime.datetime(2017, 11, 1, 4)], "minimum": True},
            [6, 6],
            [1, 6],
            [6, 6],
        ),
        # Trend of increasing accumulations with time, which is reflected
        # in the unequal shorter periods generated.
        (
            {"times": [datetime.datetime(2017, 11, 1, 4)], "accumulation": True},
            [0, 9],
            [1, 6],
            [0.25, 8.75],
        ),
        # Trend of decreasing accumulations with time, which is reflected
        # in the unequal shorter periods generated.
        (
            {"times": [datetime.datetime(2017, 11, 1, 4)], "accumulation": True},
            [12, 3],
            [1, 6],
            [0.75, 2.25],
        ),
    ],
)
def test_process_periods(kwargs, values, offsets, expected):
    """Test the process method when applied to period diagnostics, some
    accumlations and some not."""

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    npoints = 10
    data = np.stack(
        [
            np.full((npoints, npoints), values[0], dtype=np.float32),
            np.full((npoints, npoints), values[1], dtype=np.float32),
        ]
    )
    cube = multi_time_cube(times, data, "latlon", bounds=True)

    result = TemporalInterpolation(**kwargs).process(cube[0], cube[1])

    for i, (offset, value) in enumerate(zip(offsets, expected)):
        expected_data = np.full((npoints, npoints), value)
        expected_time = 1509505200 + (offset * 3600)
        expected_fp = (6 + offset) * 3600

        assert result[i].coord("time").points[0] == expected_time
        assert result[i].coord("forecast_period").points[0] == expected_fp
        assert result[i].coord("time").points.dtype == "int64"
        assert result[i].coord("forecast_period").points.dtype == "int32"
        if value is not None:
            np.testing.assert_almost_equal(result[i].data, expected_data)
