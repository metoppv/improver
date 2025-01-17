# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the PrecipitationDuration plugin.
"""

from datetime import datetime, timedelta

import iris
import numpy as np
import pytest
from iris.cube import CubeList
from numpy.testing import assert_array_equal, assert_array_almost_equal

from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube
from improver.precipitation_type.precipitation_duration import PrecipitationDuration
from improver.metadata.probabilistic import get_threshold_coord_name_from_probability_name
from improver.utilities.cube_manipulation import collapse_time

DEFAULT_ACC_NAME = "probability_of_lwe_thickness_of_precipitation_amount_above_threshold"
DEFAULT_ACC_THRESH_NAME = "lwe_thickness_of_precipitation_amount"
DEFAULT_RATE_NAME = "probability_of_lwe_precipitation_rate_above_threshold"
DEFAULT_RATE_THRESH_NAME = "lwe_precipitation_rate"


def data_times(start_time, end_time, period):
    """Define the times for the input cubes."""
    frt = start_time - 2 * period
    times = []
    bounds = []
    time = start_time + period
    while time <= end_time:
        times.append(time)
        bounds.append([time - period, time])
        time += period
    return frt, times, bounds


def multi_time_cube(frt, times, bounds, data, thresh, diagnostic_name, units):
    """Create diagnostic cubes describing period data for each input time."""
    cubes = CubeList()
    for time, time_bounds, diagnostic_data in zip(times, bounds, data):
        cubes.append(
            set_up_probability_cube(
                diagnostic_data.astype(np.float32),
                thresh,
                time=time,
                time_bounds=time_bounds,
                frt=frt,
                variable_name=diagnostic_name,
                threshold_units=units,
            )
        )
    return cubes


@pytest.fixture
def precip_cubes(start_time, end_time, period):
    """Create precipitation rate and accumulation cubes valid at a range of
    times. These cubes have default threshold and data values for tests
    where setting these is of no interest."""

    frt, times, bounds = data_times(start_time, end_time, period)
    acc_thresh=[0.1 / 1000.]
    rate_thresh=[4.0 / (3600. * 1000.)]
    acc_data = np.ones((len(times), 1, 3, 4))
    rate_data = np.ones((len(times), 1, 3, 4))

    cubes = CubeList()
    cubes.extend(multi_time_cube(frt, times, bounds, acc_data, acc_thresh, DEFAULT_ACC_THRESH_NAME, "m"))
    cubes.extend(multi_time_cube(frt, times, bounds, rate_data, rate_thresh, DEFAULT_RATE_THRESH_NAME, "m/s"))
    return cubes


@pytest.fixture
def precip_cubes_custom(start_time, end_time, period, acc_data, rate_data, acc_thresh, rate_thresh):
    """Create precipitation rate and accumulation cubes valid at a range of
    times. The thresholds and data must be provided. Thresholds are expected
    in units of mm for accumulations and mm/hr for rates; these are converted
    to SI units when creating the cubes."""

    frt, times, bounds = data_times(start_time, end_time, period)
    acc_thresh = [item / 1000. for item in acc_thresh]
    rate_thresh = [item / (3600. * 1000.) for item in rate_thresh]
    cubes = CubeList()
    cubes.extend(multi_time_cube(frt, times, bounds, acc_data, acc_thresh, DEFAULT_ACC_THRESH_NAME, "m"))
    cubes.extend(multi_time_cube(frt, times, bounds, rate_data, rate_thresh, DEFAULT_RATE_THRESH_NAME, "m/s"))
    return cubes


@pytest.mark.parametrize(
    "start_time,end_time,period",
    [
        # Period of 3-hours
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 6),
            timedelta(hours=3),
        ),
        # Period of 9-hours
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 9),
            timedelta(hours=9),
        ),
    ],
)
def test__period_in_hours(start_time, end_time, period, precip_cubes):
    """Test that the period is calculated correctly from the input cubes."""

    plugin = PrecipitationDuration(0, 0, 0)
    plugin._period_in_hours(precip_cubes)

    assert plugin.period == period.total_seconds() / 3600


def test__period_in_hours_exception():
    """Test that an exception is raised if the input cubes have different
    periods."""

    frt = datetime(2025, 1, 15, 0)
    times = [frt + timedelta(hours=3), frt + timedelta(hours=7)]
    bounds = [[frt, frt + timedelta(hours=3)], [frt + timedelta(hours=3), frt + timedelta(hours=7)]]
    data = np.ones((2, 1, 3, 4))

    precip_cubes = multi_time_cube(frt, times, bounds, data, [4.0], DEFAULT_RATE_NAME, "m/s")

    with pytest.raises(ValueError, match="Cubes with inconsistent periods"):
        plugin = PrecipitationDuration(0, 0, 0)
        plugin._period_in_hours(precip_cubes)


@pytest.mark.parametrize(
    "acc_thresh,rate_thresh,period,expected_acc,expected_rate",
    [
        (0.1, 0.2, timedelta(hours=1), [0.0001], [5.56E-8]),  # 1-hour period so acc thresh is just converted to SI
        (0.1, 0.2, timedelta(hours=3), [0.0003], [5.56E-8]),  # 3-hour period so acc thresh is multiplied by 3 and converted to SI
        (1, 1, timedelta(hours=6), [0.006], [2.778E-7]),  # 6-hour period so acc thresh is multiplied by 6 and converted to SI
    ],
)
def test__construct_thresholds(acc_thresh, rate_thresh, period, expected_acc, expected_rate):
    """Test that the thresholds are constructed correctly. Inputs are in units
    involving mm, but all outputs are in SI units. Accumulation thresholds,
    which are provided as the accumulation per hour, are multiplied up by the
    period to reflect the expected accumulation for a given period input."""

    plugin = PrecipitationDuration(acc_thresh, rate_thresh, 24)
    plugin.period = period.total_seconds() / 3600

    acc_thresh, rate_thresh, = plugin._construct_thresholds()

    assert_array_almost_equal(acc_thresh, expected_acc)
    assert_array_almost_equal(rate_thresh, expected_rate)


@pytest.mark.parametrize(
    "acc_thresh,rate_thresh,acc_name,rate_name",
    [
        (0.1, 0.1, DEFAULT_ACC_THRESH_NAME, DEFAULT_RATE_THRESH_NAME),
        (0.1, 0.1, "kittens", "puppies"),
    ],
)
def test__construct_constraints(acc_thresh, rate_thresh, acc_name, rate_name):
    """Test that iris constraints for the given thresholds are constructed and
    returned correctly."""

    acc_thresh_name = f"probability_of_{acc_name}_above_threshold"
    rate_thresh_name = f"probability_of_{rate_name}_above_threshold"

    plugin = PrecipitationDuration(
        acc_thresh,
        rate_thresh,
        24,
        accumulation_diagnostic=acc_thresh_name,
        rate_diagnostic=rate_thresh_name,
    )
    accumulation_constraint, rate_constraint = plugin._construct_constraints(acc_thresh, rate_thresh)

    assert isinstance(accumulation_constraint, iris.Constraint)
    assert isinstance(rate_constraint, iris.Constraint)
    assert accumulation_constraint._name == acc_thresh_name
    assert rate_constraint._name == rate_thresh_name
    assert acc_name in accumulation_constraint._coord_values.keys()
    assert rate_name in rate_constraint._coord_values.keys()


@pytest.mark.parametrize(
    "start_time,end_time,period,acc_data,rate_data,acc_thresh,rate_thresh,expected",
    [
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 2),
            timedelta(hours=1),
            np.ones((2, 1, 3, 4)),
            np.ones((2, 1, 3, 4)),
            [0.1],
            [4],
            np.ones((3, 4))
        ),
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 9),
            timedelta(hours=3),
            np.ones((3, 1, 3, 4)),
            np.ones((3, 1, 3, 4)),
            [0.1],
            [4],
            np.ones((3, 4))
        ),
    ]
)
def test_process(start_time, end_time, period, acc_data, rate_data, acc_thresh, rate_thresh, expected, precip_cubes_custom):
    """Test the plugin produces the expected output. The creation of the
    output cube is also tested here."""

    total_period = (end_time - start_time).total_seconds() / 3600
    period_hours = period.total_seconds() / 3600
    acc_thresh = [item/period_hours for item in acc_thresh]

    plugin = PrecipitationDuration(acc_thresh, rate_thresh, total_period)

    result = plugin.process(precip_cubes_custom)
    assert_array_equal(result.data, expected)
    print(result)
