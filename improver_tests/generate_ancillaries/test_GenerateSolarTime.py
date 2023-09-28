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
"""Unit tests for the GenerateSolarTime plugin."""

from datetime import datetime, timezone

import numpy as np
import pytest
from iris.cube import Cube

from improver.generate_ancillaries.generate_derived_solar_fields import (
    SOLAR_TIME_CF_NAME,
    GenerateSolarTime,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

ATTRIBUTES = {
    "source": "IMRPOVER tests",
    "institution": "Australian Bureau of Meteorology",
    "title": "Test data on sample grid",
}


@pytest.fixture
def target_grid() -> Cube:
    return set_up_variable_cube(
        data=np.ones((10, 8), dtype=np.float32),
        name="template",
        attributes=ATTRIBUTES,
    )


@pytest.fixture
def target_grid_equal_area() -> Cube:
    return set_up_variable_cube(
        data=np.ones((10, 8), dtype=np.float32),
        name="template",
        spatial_grid="equalarea",
        attributes=ATTRIBUTES,
    )


@pytest.mark.parametrize("new_title", (None, "IMPROVER ancillary on sample grid"))
def test__create_solar_time_cube(target_grid, new_title):

    solar_time_data = np.zeros_like(target_grid.data)
    time = datetime(2022, 1, 1, 0, 0)

    result = GenerateSolarTime()._create_solar_time_cube(
        solar_time_data, target_grid, time, new_title
    )

    # Check time value match inputs
    assert (
        result.coord("time").points[0] == time.replace(tzinfo=timezone.utc).timestamp()
    )
    # Check that the dim coords are the spatial coords only, matching those from target_grid
    assert result.coords(dim_coords=True) == [
        target_grid.coord(axis="Y"),
        target_grid.coord(axis="X"),
    ]
    # Check variable attributes
    assert result.name() == SOLAR_TIME_CF_NAME
    assert result.units == "hours"

    assert result.attributes["source"] == "IMPROVER"
    assert result.attributes.get("title") == new_title
    assert result.attributes["institution"] == target_grid.attributes["institution"]


def test_process_lat_lon(target_grid):
    time = datetime(2022, 1, 1, 0, 0)
    result = GenerateSolarTime().process(target_grid, time)
    # Check cube has same spatial coords as target_grid
    assert result.coords(dim_coords=True) == target_grid.coords(dim_coords=True)
    # Check data is sensible
    assert result.dtype == np.float32
    assert np.all(np.logical_and((result.data >= 0.0), (result.data < 24.0)))


def test_process_equal_area(target_grid_equal_area):
    time = datetime(2022, 1, 1, 0, 0)
    result = GenerateSolarTime().process(target_grid_equal_area, time)
    # Check cube has same spatial coords as target_grid
    assert result.coords(dim_coords=True) == target_grid_equal_area.coords(
        dim_coords=True
    )
    # Check data is sensible
    assert result.dtype == np.float32
    assert np.all(np.logical_and((result.data >= 0.0), (result.data < 24.0)))
