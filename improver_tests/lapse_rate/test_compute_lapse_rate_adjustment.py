# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the compute_lapse_rate_adjustment function."""

import unittest

import numpy as np
import pytest

from improver.constants import ELR
from improver.lapse_rate import compute_lapse_rate_adjustment
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


def set_up_cubes(lapse_rate):
    """Set up cubes.

    Args:
        lapse_rate: Value to use as the lapse rate within a cube.

    Returns:
        Cube of the orographic difference and a cube of the lapse rate.
    """
    orog_diff = np.array(
        [
            [10, 20, -10, -20],
            [200.0, -200, 10, 0],
            [30, 40, 50, -50],
            [5, 0.5, 40, 60],
        ],
        dtype=np.float32,
    )
    orog_diff_cube = set_up_variable_cube(
        orog_diff,
        name="surface_altitude_difference",
        units="m",
        spatial_grid="equalarea",
    )
    lapse_rate_cube = set_up_variable_cube(
        np.full((4, 4), lapse_rate, dtype=np.float32),
        name="lapse_rate",
        units="K m-1",
        spatial_grid="equalarea",
    )
    return orog_diff_cube, lapse_rate_cube


@pytest.mark.parametrize("max_orog_diff_limit", (50, 70,))
@pytest.mark.parametrize("lapse_rate", (-ELR, 0, ELR))
def test_compute_lapse_rate_adjustment(lapse_rate, max_orog_diff_limit):
    """Test the computation of the lapse rate adjustment."""
    orog_diff_cube, lapse_rate_cube = set_up_cubes(lapse_rate)
    result = compute_lapse_rate_adjustment(
        lapse_rate_cube.data,
        orog_diff_cube.data,
        max_orog_diff_limit=max_orog_diff_limit,
    )

    expected_data = lapse_rate_cube.data * orog_diff_cube.data
    if np.isclose(lapse_rate, -ELR) and max_orog_diff_limit == 50:
        expected_data[1, 0] = -0.65
        expected_data[1, 1] = 0.65
        expected_data[3, 3] = 0.26
    elif np.isclose(lapse_rate, -ELR) and max_orog_diff_limit == 70:
        expected_data[1, 0] = -0.39
        expected_data[1, 1] = 0.39
    elif np.isclose(lapse_rate, 0) and max_orog_diff_limit == 50:
        expected_data[1, 0] = -0.975
        expected_data[1, 1] = 0.975
        expected_data[3, 3] = -0.065
    elif np.isclose(lapse_rate, 0) and max_orog_diff_limit == 70:
        expected_data[1, 0] = -0.845
        expected_data[1, 1] = 0.845

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected_data, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
