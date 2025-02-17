# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the PrecipitationDuration plugin.
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Union

import iris
import iris.util
import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from numpy import ndarray
from numpy.testing import assert_array_almost_equal, assert_array_equal

from improver.precipitation.precipitation_duration import PrecipitationDuration
from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    set_up_spot_probability_cube,
)

DEFAULT_ACC_NAME = (
    "probability_of_lwe_thickness_of_precipitation_amount_above_threshold"
)
DEFAULT_ACC_THRESH_NAME = "lwe_thickness_of_precipitation_amount"
DEFAULT_RATE_NAME = "probability_of_lwe_precipitation_rate_above_threshold"
DEFAULT_RATE_THRESH_NAME = "lwe_precipitation_rate"
DEFAULT_PERCENTILES = [10, 50, 90]


def data_times(
    start_time: datetime, end_time: datetime, period: timedelta
) -> Tuple[datetime, List[datetime], List[List[datetime]]]:
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


def multi_time_cube(
    frt: datetime,
    times: Tuple[List[datetime]],
    bounds: Tuple[List[List[datetime]]],
    data: ndarray,
    thresh: List[float],
    diagnostic_name: str,
    units: str,
    function=set_up_probability_cube,
) -> CubeList:
    """Create diagnostic cubes describing period data for each input time.
    If the input data has 5 dimensions the first is assumed to be time and
    the second is assumed to be a realization dimension."""

    cubes = CubeList()
    for time, time_bounds, diagnostic_data in zip(times, bounds, data):
        realization_cubes = CubeList()
        for index, realization_data in enumerate(diagnostic_data):
            cube = function(
                realization_data.astype(np.float32),
                thresh,
                time=time,
                time_bounds=time_bounds,
                frt=frt,
                variable_name=diagnostic_name,
                threshold_units=units,
            )
            realization_coord = AuxCoord([index], standard_name="realization", units=1)
            cube.add_aux_coord(realization_coord)
            realization_cubes.append(cube)
        cubes.append(realization_cubes.merge_cube())
    return cubes


@pytest.fixture(params=[set_up_probability_cube, set_up_spot_probability_cube])
def precip_cubes(
    request,
    start_time: datetime,
    end_time: datetime,
    period: timedelta,
    acc_data: ndarray,
    rate_data: ndarray,
    acc_thresh: List[float],
    rate_thresh: List[float],
) -> CubeList:
    """Create precipitation rate and accumulation cubes valid at a range of
    times. The thresholds and data must be provided. Thresholds are expected
    in units of mm for accumulations and mm/hr for rates; these are converted
    to SI units when creating the cubes. The accumulation threshold is
    multiplied up by the period. This means the accumulation threshold
    argument represents the accumulation per hour, which is what the user will
    specify when using the plugin. Multi-realization cubes will be returned.

    Args:
        start_time: The start time of the input cubes (the lower bound of the
                    first time coordinate).
        end_time: The end time of the input cubes (the upper bound of the last
                  time coordinate).
        period: The period of the input cubes.
        acc_data: The accumulation probabilities for the input cubes.
        rate_data: The rate probabilities for the input cubes.
        acc_thresh: The accumulation threshold for the input cubes.
        rate_thresh: The rate threshold for the input cubes.
    Returns:
        CubeList: A list of cubes representing the input data with bespoke
                  periods, times, thresholds, and data.
    """

    function = request.param

    if function == set_up_spot_probability_cube:
        acc_sites = np.product(acc_data.shape[-2:])
        acc_data = acc_data.reshape(acc_data.shape[:-2] + (acc_sites,))
        rate_sites = np.product(rate_data.shape[-2:])
        rate_data = rate_data.reshape(rate_data.shape[:-2] + (rate_sites,))

    frt, times, bounds = data_times(start_time, end_time, period)
    period_hours = period.total_seconds() / 3600
    acc_thresh = [period_hours * item / 1000.0 for item in acc_thresh]
    rate_thresh = [item / (3600.0 * 1000.0) for item in rate_thresh]
    cubes = CubeList()
    cubes.extend(
        multi_time_cube(
            frt,
            times,
            bounds,
            acc_data,
            acc_thresh,
            DEFAULT_ACC_THRESH_NAME,
            "m",
            function=function,
        )
    )
    cubes.extend(
        multi_time_cube(
            frt,
            times,
            bounds,
            rate_data,
            rate_thresh,
            DEFAULT_RATE_THRESH_NAME,
            "m/s",
            function=function,
        )
    )
    return cubes


@pytest.mark.parametrize(
    "start_time,end_time,period,acc_data,rate_data,acc_thresh,rate_thresh",
    [
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 6),
            timedelta(hours=3),
            np.ones((2, 1, 1, 3, 3)),
            np.ones((2, 1, 1, 3, 3)),
            [0.1],
            [4],
        ),  # Period of 3-hours
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 9),
            timedelta(hours=9),
            np.ones((1, 1, 1, 3, 3)),
            np.ones((1, 1, 1, 3, 3)),
            [0.1],
            [4],
        ),  # Period of 9-hours
    ],
)
def test__period_in_hours(
    start_time: datetime,
    end_time: datetime,
    period: timedelta,
    acc_data: ndarray,
    rate_data: ndarray,
    acc_thresh: List[float],
    rate_thresh: List[float],
    precip_cubes: CubeList,
):
    """Test that the period is calculated correctly from the input cubes."""

    plugin = PrecipitationDuration(0, 0, 0, DEFAULT_PERCENTILES)
    plugin._period_in_hours(precip_cubes)

    assert plugin.period == period.total_seconds() / 3600


def test__period_in_hours_exception():
    """Test that an exception is raised if the input cubes have different
    periods."""

    frt = datetime(2025, 1, 15, 0)
    times = [frt + timedelta(hours=3), frt + timedelta(hours=7)]
    bounds = [
        [frt, frt + timedelta(hours=3)],
        [frt + timedelta(hours=3), frt + timedelta(hours=7)],
    ]
    data = np.ones((2, 1, 1, 3, 4))

    precip_cubes = multi_time_cube(
        frt, times, bounds, data, [4.0], DEFAULT_RATE_NAME, "m/s"
    )

    with pytest.raises(ValueError, match="Cubes with inconsistent periods"):
        plugin = PrecipitationDuration(0, 0, 0, DEFAULT_PERCENTILES)
        plugin._period_in_hours(precip_cubes)


@pytest.mark.parametrize(
    "acc_thresh,rate_thresh,period,expected_acc,expected_rate",
    [
        (
            0.1,
            0.2,
            timedelta(hours=1),
            [0.0001],
            [5.56e-8],
        ),  # 1-hour period so acc thresh is just converted to SI
        (
            0.1,
            0.2,
            timedelta(hours=3),
            [0.0003],
            [5.56e-8],
        ),  # 3-hour period so acc thresh is multiplied by 3 and converted to SI
        (
            1,
            1,
            timedelta(hours=6),
            [0.006],
            [2.778e-7],
        ),  # 6-hour period so acc thresh is multiplied by 6 and converted to SI
        (
            [0.1, 1],
            [0.2, 0.4],
            timedelta(hours=1),
            [0.0001, 0.001],
            [5.56e-8, 1.112e-7],
        ),  # Check mulitple thresholds can be accepted and converted.
        (
            ["0.1", "1"],
            ["0.2", "0.4"],
            timedelta(hours=1),
            [0.0001, 0.001],
            [5.56e-8, 1.112e-7],
        ),  # Check thresholds given as strings.
    ],
)
def test__construct_thresholds(
    acc_thresh: Union[float, str, List],
    rate_thresh: Union[float, str, List],
    period: timedelta,
    expected_acc: List[float],
    expected_rate: List[float],
):
    """Test that the thresholds are constructed correctly. Inputs are in units
    involving mm, but all outputs are in SI units. Accumulation thresholds,
    which are provided as the accumulation per hour, are multiplied up by the
    period to reflect the expected accumulation for a given period input."""

    plugin = PrecipitationDuration(acc_thresh, rate_thresh, 24, DEFAULT_PERCENTILES)
    plugin.period = period.total_seconds() / 3600

    (
        acc_thresh,
        rate_thresh,
    ) = plugin._construct_thresholds()

    assert_array_almost_equal(acc_thresh, expected_acc)
    assert_array_almost_equal(rate_thresh, expected_rate)


DEFAULT_ACC_NAME, DEFAULT_ACC_THRESH_NAME, DEFAULT_RATE_NAME, DEFAULT_RATE_THRESH_NAME


@pytest.mark.parametrize(
    "diagnostic,threshold,threshold_name",
    [
        (DEFAULT_ACC_NAME, 0.1, DEFAULT_ACC_THRESH_NAME),
        (DEFAULT_RATE_NAME, 4, DEFAULT_RATE_THRESH_NAME),
    ],
)
def test__construct_constraint(diagnostic: str, threshold: float, threshold_name: str):
    """Test that iris constraints for the given thresholds are constructed and
    returned correctly."""

    plugin = PrecipitationDuration(
        threshold,
        threshold,
        24,
        DEFAULT_PERCENTILES,
        accumulation_diagnostic=diagnostic,
        rate_diagnostic=diagnostic,
    )
    constraint = plugin._construct_constraint(diagnostic, threshold, threshold_name)

    assert isinstance(constraint, iris.Constraint)
    assert constraint._name == diagnostic
    assert threshold in constraint._coord_values.keys()


@pytest.mark.parametrize(
    "start_time,end_time,period,acc_data,rate_data,acc_thresh,rate_thresh,diagnostic,threshold_name,threshold_values",
    [
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 6),
            timedelta(hours=3),
            np.ones((2, 2, 2, 3, 3)),
            np.ones((2, 1, 1, 3, 3)),
            [0.1, 0.2],
            [4],
            DEFAULT_RATE_NAME,
            DEFAULT_RATE_THRESH_NAME,
            np.array([4.0 / (3600.0 * 1000.0)], dtype=np.float32),
        ),  # Rate cube extraction.
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 2),
            timedelta(hours=1),
            np.ones((2, 2, 2, 3, 3)),
            np.ones((2, 1, 1, 3, 3)),
            [0.1, 0.2],
            [4],
            DEFAULT_ACC_NAME,
            DEFAULT_ACC_THRESH_NAME,
            np.array([0.0001], dtype=np.float32),
        ),  # 1-hour period accumulation extraction.
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 6),
            timedelta(hours=3),
            np.ones((2, 2, 2, 3, 3)),
            np.ones((2, 1, 1, 3, 3)),
            [0.1, 0.2],
            [4],
            DEFAULT_ACC_NAME,
            DEFAULT_ACC_THRESH_NAME,
            np.array([0.0003], dtype=np.float32),
        ),  # 3-hour period accumulation extraction with a suitably scaled
        # threshold value.
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 2),
            timedelta(hours=1),
            np.ones((2, 2, 2, 3, 3)),
            np.ones((2, 1, 1, 3, 3)),
            [0.1, 0.2],
            [4],
            DEFAULT_ACC_NAME,
            DEFAULT_ACC_THRESH_NAME,
            np.array([0.0001, 0.0002], dtype=np.float32),
        ),  # 1-hour period accumulation extraction multiple thresholds.
    ],
)
def test__extract_cubes(
    start_time: datetime,
    end_time: datetime,
    period: timedelta,
    diagnostic: str,
    threshold_name: str,
    threshold_values: ndarray,
    precip_cubes: CubeList,
):
    """Test the extraction of target diagnostics and thresholds and the
    merging of resulting outputs to form a single cube."""
    plugin = PrecipitationDuration(1.0, 7.0, 2, DEFAULT_PERCENTILES)
    result = plugin._extract_cubes(
        precip_cubes,
        diagnostic,
        threshold_name,
        threshold_values,
    )
    assert isinstance(result, Cube)
    assert result.name() == diagnostic
    assert result.coord("time").shape[0] == ((end_time - start_time) / period)
    assert result.coord(var_name="threshold").name() == threshold_name
    assert_array_equal(result.coord(var_name="threshold").points, threshold_values)


@pytest.mark.parametrize(
    "diagnostic,threshold_name,threshold_values",
    [
        ("kittens", DEFAULT_ACC_THRESH_NAME, np.array([2], dtype=np.float32)),
        (DEFAULT_ACC_NAME, "kittens", np.array([2], dtype=np.float32)),
        (DEFAULT_ACC_NAME, DEFAULT_ACC_THRESH_NAME, np.array([4], dtype=np.float32)),
    ],
)
def test__extract_cubes_exception_thresholds(
    diagnostic: str, threshold_name: str, threshold_values: ndarray
):
    """Test an exception is raised if the input cubes do not contain the
    required thresholds."""

    time_args = data_times(
        datetime(2025, 1, 15, 0), datetime(2025, 1, 15, 2), timedelta(hours=1)
    )
    data = np.ones((2, 2, 1, 3, 4))
    cubes = multi_time_cube(*time_args, data, [2.0], DEFAULT_ACC_THRESH_NAME, "m")

    plugin = PrecipitationDuration(1.0, 7.0, 2, DEFAULT_PERCENTILES)
    msg = "The requested diagnostic or threshold is not available."

    with pytest.raises(ValueError, match=msg):
        plugin._extract_cubes(
            cubes,
            diagnostic,
            threshold_name,
            threshold_values,
        )


@pytest.mark.parametrize(
    "start_time,end_time,period,acc_data,rate_data,acc_thresh,rate_thresh",
    [
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 3),
            timedelta(hours=3),
            np.ones((1, 2, 1, 3, 3)),  # 1 time, 2 realizations, 1 threshold, y, x.
            np.ones((1, 2, 1, 3, 3)),  # Rate cube not used.
            [0.1],  # 1 accumulation threshold.
            [4],  # Rate cube not used.
        ),  # Time and threshold scalars to be promoted
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 6),
            timedelta(hours=3),
            np.ones((2, 2, 1, 3, 3)),  # 2 times, 2 realizations, 1 threshold, y, x.
            np.ones((2, 2, 1, 3, 3)),  # Rate cube not used.
            [0.1],  # 1 accumulation threshold.
            [4],  # Rate cube not used.
        ),  # Threshold scalar to be promoted
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 6),
            timedelta(hours=3),
            np.ones((1, 2, 2, 3, 3)),  # 1 times, 2 realizations, 2 thresholds, y, x.
            np.ones((1, 2, 1, 3, 3)),  # Rate cube not used.
            [0.1, 0.2],  # 2 accumulation thresholds.
            [4],  # Rate cube not used.
        ),  # Time scalar to be promoted
    ],
)
def test__structure_inputs(
    start_time: datetime,
    end_time: datetime,
    period: timedelta,
    precip_cubes: CubeList,
):
    """Test that all expected dim coords are in place after this method has
    been applied and ordered as expected. The threshold coordinate should also
    be returned."""

    # Squeeze length 1 dim coords to scalars so they can be restored.
    cube = iris.util.squeeze(precip_cubes[0])

    expected_dimcrd_order = [
        "realization",
        "lwe_thickness_of_precipitation_amount",
        "time",
    ]

    plugin = PrecipitationDuration(1.0, 7.0, 2, DEFAULT_PERCENTILES)
    result_cube, result_thresh = plugin._structure_inputs(cube)

    dimcrds = [crd.name() for crd in result_cube.coords(dim_coords=True)]
    assert dimcrds[0:3] == expected_dimcrd_order
    assert result_thresh.name() == "lwe_thickness_of_precipitation_amount"


@pytest.mark.parametrize(
    "start_time,end_time,period,acc_data,rate_data,acc_thresh,rate_thresh,expected",
    [
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 2),
            timedelta(hours=1),
            np.ones((2, 2, 1, 3, 4)),  # 2 times, 2 realizations, 1 threshold, y, x
            np.ones((2, 2, 1, 3, 4)),  # 2 times, 2 realizations, 1 threshold, y, x
            [0.1],
            [4],
            np.ones((3, 3, 4)),  # 3 percentiles, y, x
        ),  # Accumulation and rate probabilities are 1 for both input hours
        # and all realizations. The resulting percentiles are all 1 for the
        # combined period at all points.
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 9),
            timedelta(hours=3),
            np.ones((3, 2, 1, 3, 4)),  # 3 times, 2 realizations, 1 threshold, y, x
            np.ones((3, 2, 1, 3, 4)),  # 3 times, 2 realizations, 1 threshold, y, x
            [0.1],
            [4],
            np.ones((3, 3, 4)),  # 3 percentiles, y, x
        ),  # As above but for 3 hour input periods, with 3 of them comprising
        # the total period.
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 2),
            timedelta(hours=1),
            np.stack(
                [
                    np.stack([[np.ones((3, 4))], [np.zeros((3, 4))]]),
                    np.stack([[np.ones((3, 4))], [np.zeros((3, 4))]]),
                ]
            ),  # 2 times, 2 realizations, 1 threshold, y, x
            np.ones((2, 2, 1, 3, 4)),  # 2 times, 2 realizations, 1 threshold, y, x
            [0.1],
            [4],
            np.stack(
                [
                    np.full((3, 4), 0.1),
                    np.full((3, 4), 0.5),
                    np.full((3, 4), 0.9),
                ]
            ).astype(np.float32),  # 3 percentiles, y, x
        ),  # Accumulation probabilities are 1 in the first realization and 0
        # in the second for both input hours. Half of the ensemble is
        # classified as satisfying the rate and accumulation thresholds
        # for both input periods. Realization 0 = 100% of total period
        # classified as wet. Realization 1 = 0% of total period classified
        # as wet. This results in the 50th percentile being a fraction of
        # 0.5. Likewise the 10th and 90th percentiles are 0.1 and 0.9
        # respectively.
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 2),
            timedelta(hours=1),
            np.stack(
                [
                    np.stack([[np.ones((3, 4))], [np.ones((3, 4))]]),
                    np.stack([[np.zeros((3, 4))], [np.zeros((3, 4))]]),
                ]
            ),  # 2 times, 2 realizations, 1 threshold, y, x
            np.ones((2, 2, 1, 3, 4)),  # 2 times, 2 realizations, 1 threshold, y, x
            [0.1],
            [4],
            np.full((3, 3, 4), 0.5, dtype=np.float32),  # 3 percentiles, y, x
        ),  # Accumulation probabilities are 1 in both ensemble members for the
        # first time and both 0 for the second time. This means that the
        # fraction of the target period classified as wet by each ensemble
        # member is 0.5. Given that percentiles are generated from the
        # possible values within the ensemble the only value that can be
        # returned for all the requested percentiles is 0.5.
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 9),
            timedelta(hours=3),
            np.ones((3, 2, 2, 3, 4)),  # 3 times, 2 realizations, 2 thresholds, y, x
            np.ones((3, 2, 2, 3, 4)),  # 3 times, 2 realizations, 2 thresholds, y, x
            [0.1, 1.0],
            [4, 8],
            np.ones(
                (3, 2, 2, 3, 4)
            ),  # 3 percentiles, 2 acc thresholds, 2 rate thresholds, y, x
        ),  # Multiple thresholds for both the accumulation and maximum rate.
        # Again the total period fraction percentiles are 1 in this case as all
        # input probabilities are 1 and all periods are classified as wet.
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 12),
            timedelta(hours=3),
            np.ones((4, 3, 2, 3, 4)),  # 4 times, 3 realizations, 2 thresholds, y, x
            np.r_[[[1] * 12, [0] * 12] * 12]
            .reshape(4, 3, 2, 3, 4)  # 4 times, 3 realizations, 2 thresholds, y, x
            .astype(np.float32),  # 4 times, 3 realizations, 2 thresholds, y, x
            [0.1, 1.0],
            [2, 4],
            np.r_[[[1] * 12, [0] * 12] * 6]
            .reshape(3, 2, 2, 3, 4)
            .astype(
                np.float32
            ),  # 3 percentiles, 2 acc thresholds, 2 rate thresholds, y, x
        ),  # Maximum rate in period probabilities are 1 for the lower
        # threshold and 0 for the higher threshold for every time and realization.
        # The resulting percentiles are all 0 or 1 as a fraction of the total
        # period. So the 10th, 50th, and 90th percentiles are all 1 for
        # combinations that include the lower rate threshold, and are all
        # 0 for combinations that include the upper rate threshold.
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 3),
            timedelta(hours=3),
            np.r_[[1, 0] * 12]
            .reshape(1, 2, 1, 3, 4)
            .astype(np.float32),  # 1 time, 2 realizations, 1 threshold, y, x
            np.ones((1, 2, 1, 3, 4)),  # 1 time, 2 realizations, 1 threshold, y, x
            [0.1],
            [4],
            np.r_[[1, 0] * 18]
            .reshape(3, 3, 4)
            .astype(np.float32),  # 3 percentiles, y, x
        ),  # A single threshold, 2 realizations, and a single time input.
        # The accumulation input binary probabilities are set in stripes of
        # 1 and 0. This resulting percentiles are either 0 or 1; either
        # none of the period or the whole period are classified as wet.
        (
            datetime(2025, 1, 15, 0),
            datetime(2025, 1, 15, 5),
            timedelta(hours=1),
            np.ones((5, 2, 1, 5, 5)),  # 5 times, 2 realizations, 1 threshold, y, x
            np.repeat(
                [
                    np.r_[[1] * 5 * j, [0] * 5 * (5 - j)].reshape(5, 5)
                    for j in range(5, 0, -1)
                ],
                2,
                axis=0,
            )
            .astype(np.float32)
            .reshape(5, 2, 1, 5, 5),  # 5 times, 2 realizations, 1 threshold, y, x
            [0.1],
            [4],
            np.repeat(
                np.stack(
                    [np.r_[np.full((5), i)] for i in np.arange(1, 0, -0.2)]
                ).reshape(1, 5, 5),
                3,
                axis=0,
            ).astype(np.float32),  # 3 percentiles, y, x
        ),  # Accumulation probabilities vary by row with time such that the
        # total period fraction for the top row is 1, the second row is
        # 0.8, etc. down to 0.2 for the bottom row. Duplicated across realizations
        # to avoid any further complication. Each location ends up with only a
        # single fraction of the total period across the ensemble, which again
        # means that the percentiles returned are all the same for the 10th, 50th
        # and 90th percentile for a given location, e.g 1 for the top row
        # and 0.2 for the bottom row.
    ],
)
def test_process(
    start_time: datetime,
    end_time: datetime,
    period: timedelta,
    acc_data: ndarray,
    rate_data: ndarray,
    acc_thresh: List[float],
    rate_thresh: List[float],
    expected: ndarray,
    precip_cubes: CubeList,
):
    """Test the plugin produces the expected output. The calculation method
    _calculate_fractions is tested here as is the creation of the
    output cube."""

    total_period = (end_time - start_time).total_seconds()
    period_hours = period.total_seconds() / 3600

    plugin = PrecipitationDuration(
        acc_thresh, rate_thresh, total_period / 3600, DEFAULT_PERCENTILES
    )
    result = plugin.process(precip_cubes)

    if result.coords("wmo_id"):
        n_sites = np.product(acc_data.shape[-2:])
        expected = expected.reshape(expected.shape[:-2] + (n_sites,))

    assert_array_equal(result.data, expected)
    assert_array_almost_equal(
        result.coord("lwe_thickness_of_precipitation_amount").points,
        period.total_seconds() / 3600 * np.array(acc_thresh) / 1000,
    )
    assert_array_almost_equal(
        result.coord("lwe_precipitation_rate").points,
        np.array(rate_thresh) / (3600 * 1000),
    )
    assert result.attributes["precipitation_sampling_period_in_hours"] == period_hours
    assert np.diff(result.coord("time").bounds) == total_period
    assert result.coord(
        long_name="precipitation_accumulation_threshold_for_wet"
    ) is result.coord("lwe_thickness_of_precipitation_amount")
    assert result.coord(
        long_name="precipitation_rate_threshold_for_wet"
    ) is result.coord("lwe_precipitation_rate")


def test_process_exception_differing_time():
    """Test an exception is raised if the input cubes have differing time
    dimensions, meaning they cannot be used together."""

    cubes = CubeList()
    time_args = data_times(
        datetime(2025, 1, 15, 0), datetime(2025, 1, 15, 2), timedelta(hours=1)
    )
    data = np.ones((2, 2, 1, 3, 4))
    cubes.extend(
        multi_time_cube(*time_args, data, [1.0 / 1000], DEFAULT_ACC_THRESH_NAME, "m")
    )

    time_args = data_times(
        datetime(2025, 1, 15, 0), datetime(2025, 1, 15, 3), timedelta(hours=1)
    )
    data = np.ones((3, 2, 1, 3, 4))
    cubes.extend(
        multi_time_cube(
            *time_args, data, [7.0 / (3600 * 1000)], DEFAULT_RATE_THRESH_NAME, "m/s"
        )
    )

    plugin = PrecipitationDuration(1.0, 7.0, 2, DEFAULT_PERCENTILES)
    msg = (
        "Precipitation accumulation and maximum rate in period cubes "
        "have differing time coordinates and cannot be used together."
    )
    with pytest.raises(ValueError, match=msg):
        plugin.process(cubes)


def test_process_exception_total_period():
    """Test an exception is raised if the input cubes do not combine to cover
    the specified target total period."""

    time_args = data_times(
        datetime(2025, 1, 15, 0), datetime(2025, 1, 15, 2), timedelta(hours=1)
    )
    data = np.ones((2, 2, 1, 3, 4))

    cubes = CubeList()
    cubes.extend(
        multi_time_cube(*time_args, data, [1.0 / 1000], DEFAULT_ACC_THRESH_NAME, "m")
    )
    cubes.extend(
        multi_time_cube(
            *time_args, data, [7.0 / (3600 * 1000)], DEFAULT_RATE_THRESH_NAME, "m/s"
        )
    )

    target_period = 24
    plugin = PrecipitationDuration(1.0, 7.0, target_period, DEFAULT_PERCENTILES)
    msg = (
        "Input cubes do not combine to create the expected target "
        "period. The period covered by the cubes passed in is: "
        f"2.0 hours. Target is {target_period} hours."
    )
    with pytest.raises(ValueError, match=msg):
        plugin.process(cubes)


def test_process_exception_no_realization():
    """Test an exception is raised if the input cubes do not have multi-valued
    realization coordinates."""

    time_args = data_times(
        datetime(2025, 1, 15, 0), datetime(2025, 1, 15, 2), timedelta(hours=1)
    )
    data = np.ones((2, 1, 1, 3, 4))

    cubes = CubeList()
    cubes.extend(
        multi_time_cube(*time_args, data, [1.0 / 1000], DEFAULT_ACC_THRESH_NAME, "m")
    )
    cubes.extend(
        multi_time_cube(
            *time_args, data, [7.0 / (3600 * 1000)], DEFAULT_RATE_THRESH_NAME, "m/s"
        )
    )

    plugin = PrecipitationDuration(1.0, 7.0, 2, DEFAULT_PERCENTILES)
    msg = "This plugin requires input data from multiple realizations."
    with pytest.raises(ValueError, match=msg):
        plugin.process(cubes)


def test_process_exception_mismatched_realization():
    """Test an exception is raised if the input cubes do not have matching
    realization coordinates."""

    time_args = data_times(
        datetime(2025, 1, 15, 0), datetime(2025, 1, 15, 2), timedelta(hours=1)
    )
    data = np.ones((2, 2, 1, 3, 4))

    cubes = CubeList()
    cubes.extend(
        multi_time_cube(*time_args, data, [1.0 / 1000], DEFAULT_ACC_THRESH_NAME, "m")
    )
    for cube in cubes:
        cube.coord("realization").points = [1, 2]
    cubes.extend(
        multi_time_cube(
            *time_args, data, [7.0 / (3600 * 1000)], DEFAULT_RATE_THRESH_NAME, "m/s"
        )
    )

    plugin = PrecipitationDuration(1.0, 7.0, 2, DEFAULT_PERCENTILES)
    msg = (
        "Mismatched realization coordinates between accumulation and "
        "max rate inputs. These must be the same."
    )
    with pytest.raises(ValueError, match=msg):
        plugin.process(cubes)


def test__calculate_fractions_exception_masked_data():
    """Test an exception is raised if the input cubes contain masked data."""

    time_args = data_times(
        datetime(2025, 1, 15, 0), datetime(2025, 1, 15, 2), timedelta(hours=1)
    )
    data = np.ones((2, 2, 1, 3, 4))
    data = np.ma.masked_where(data == 1, data)
    data.mask[0, 0, 0, 0] = False

    cubes = CubeList()
    cubes.extend(
        multi_time_cube(*time_args, data, [1.0 / 1000], DEFAULT_ACC_THRESH_NAME, "m")
    )
    cubes.extend(
        multi_time_cube(
            *time_args, data, [7.0 / (3600 * 1000)], DEFAULT_RATE_THRESH_NAME, "m/s"
        )
    )

    plugin = PrecipitationDuration(1.0, 7.0, 2, DEFAULT_PERCENTILES)
    msg = "Precipitation duration plugin cannot handle masked data."
    with pytest.raises(ValueError, match=msg):
        plugin.process(cubes)
