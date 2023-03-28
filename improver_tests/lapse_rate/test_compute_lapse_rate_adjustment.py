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
