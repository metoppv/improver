# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Unit tests for the GenerateTimezoneMask plugin."""

import pytest
from datetime import datetime
import numpy as np
from numpy.testing import assert_array_almost_equal

from iris.cube import Cube

from improver.generate_ancillaries.generate_timezone_mask import GenerateTimezoneMask
from ..set_up_test_cubes import set_up_variable_cube


@pytest.fixture
def global_grid() -> Cube:
    """Global grid template"""

    data = np.zeros((19, 37), dtype=np.float32)
    cube = set_up_variable_cube(
        data, name="template", grid_spacing=10, domain_corner=(-90, -180)
    )
    return cube


@pytest.fixture
def uk_grid() -> Cube:
    """UK grid template"""

    data = np.zeros((21, 22), dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="template",
        spatial_grid="equalarea",
        grid_spacing=96900.0,
        domain_corner=(-1036000.0, -1158000.0),
    )
    return cube


def test__set_time(uk_grid):
    """Test time is set correctly from either the cube or user."""

    # Set by the cube time coordinate
    expected = datetime(2017, 11, 10, 4)
    plugin = GenerateTimezoneMask()
    plugin(uk_grid)
    assert plugin.time == expected

    # Set by the user provided argument
    expected = datetime(2020, 7, 16, 15)
    plugin = GenerateTimezoneMask(time="20200716T1500Z")
    plugin(uk_grid)
    assert plugin.time == expected


@pytest.mark.parametrize("grid_fixture", ["global_grid", "uk_grid"])
def test__get_coordinate_pairs(request, grid_fixture):
    """Test that a selection of elements are as expected in what is returned
    by the _get_coordinate_pairs function. Tests are for both a native lat-long
    grid and for an equal areas grid that must be transformed."""

    sample_points = [0, 10, -1]
    expected = {
        "global_grid": [[-90.0, -180.0], [-90.0, -80.0], [90.0, 180.0]],
        "uk_grid": [[44.517, -17.117], [45.548, -4.913], [62.026, 14.410]],
    }

    grid = request.getfixturevalue(grid_fixture)
    result = GenerateTimezoneMask()._get_coordinate_pairs(grid)
    for i, ii in enumerate(sample_points):
        assert_array_almost_equal(result[:, ii], expected[grid_fixture][i], decimal=3)


# _calculate_tz_offsets
# _get_timezone
# _calculate_offset
# _group_timezones
# process
