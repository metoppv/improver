# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function collapse_time.
"""

from datetime import datetime, timedelta

import iris
import numpy as np
import pytest
from iris.cube import CubeList
from numpy.testing import assert_array_equal

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import collapse_time


@pytest.fixture
def data_times():
    frt = datetime(2025, 1, 15, 3, 0)
    times = []
    for hour in range(3, 9 + 1, 3):
        time = frt + timedelta(hours=hour)
        time_bounds = [time - timedelta(hours=3), time]
        times.append((frt, time, time_bounds))
    return times


@pytest.fixture
def multi_time_cube(data_times):
    data = 281 * np.ones((3, 3)).astype(np.float32)
    cubes = CubeList()
    for frt, time, time_bounds in data_times:
        cubes.append(
            set_up_variable_cube(data, time=time, time_bounds=time_bounds, frt=frt)
        )

    return cubes.merge_cube()


@pytest.mark.parametrize("collapse_crd", ["time", "forecast_period"])
def test_basic(multi_time_cube, data_times, collapse_crd):
    """Test that a collapsed cube is returned with the expected
    time coordiantes. The point should be at the end of the bounds
    and the bounds should span the original bounds."""

    expected_time = data_times[-1][1]
    expected_time_bounds = (data_times[0][-1][0], data_times[-1][-1][-1])
    expected_fp = (data_times[-1][-1][-1] - data_times[0][2][0]).total_seconds()
    expected_fp_bounds = [
        (data_times[0][2][0] - data_times[0][0]).total_seconds(),
        expected_fp,
    ]

    result = collapse_time(multi_time_cube, collapse_crd, iris.analysis.SUM)

    assert result.coord("time").cell(0).point == expected_time
    assert_array_equal(result.coord("time").cell(0).bound, expected_time_bounds)
    assert result.coord("forecast_period").points[0] == expected_fp
    assert_array_equal(result.coord("forecast_period").bounds[0], expected_fp_bounds)


@pytest.mark.parametrize(
    "collapse_crd", ["times_tables", "forecast_periodicity", "kittens"]
)
def test_exception(multi_time_cube, data_times, collapse_crd):
    """Test that an exception is raised when attempting to collapse a
    coordinate that is not time or forecast_period.."""

    msg = (
        "The collapse_time wrapper should only be used for collapsing "
        "the time or forecast_period coordinates."
    )

    with pytest.raises(ValueError, match=msg):
        collapse_time(multi_time_cube, collapse_crd, iris.analysis.SUM)
