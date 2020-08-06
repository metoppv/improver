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
"""Tests for the ConvectionRatioFromComponents plugin."""

import iris
import numpy as np
import pytest
from iris.cube import Cube, CubeList
from numpy.testing import assert_allclose, assert_equal, assert_raises_regex

from improver.convection import ConvectionRatioFromComponents
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

GLOBAL_ATTRIBUTES = {
    "title": "MOGREPS-G Model Forecast on Global 20 km Standard Grid",
    "source": "Met Office Unified Model",
    "institution": "Met Office",
}

UK_ATTRIBUTES = {
    "title": "MOGREPS-UK Model Forecast on UK 2 km Standard Grid",
    "source": "Met Office Unified Model",
    "institution": "Met Office",
}


@pytest.fixture(name="global_grid")
def global_grid_fixture() -> Cube:
    """Global grid template"""

    data = np.zeros((19, 37), dtype=np.float32)
    cubes = CubeList([])
    cubes.append(
        set_up_variable_cube(
            data.copy(),
            name="convective_precipitation_rate",
            units="m s-1",
            grid_spacing=10,
            domain_corner=(-90, -180),
            attributes=GLOBAL_ATTRIBUTES,
        )
    )
    cubes.append(
        set_up_variable_cube(
            data.copy(),
            name="dynamic_precipitation_rate",
            units="m s-1",
            grid_spacing=10,
            domain_corner=(-90, -180),
            attributes=GLOBAL_ATTRIBUTES,
        )
    )
    return cubes


@pytest.fixture(name="uk_grid")
def uk_grid_fixture() -> Cube:
    """UK grid template"""

    data = np.zeros((21, 22), dtype=np.float32)
    cubes = CubeList([])
    cubes.append(
        set_up_variable_cube(
            data.copy(),
            name="convective_precipitation_rate",
            units="m s-1",
            spatial_grid="equalarea",
            grid_spacing=96900.0,
            domain_corner=(-1036000.0, -1158000.0),
            attributes=UK_ATTRIBUTES,
        )
    )
    cubes.append(
        set_up_variable_cube(
            data.copy(),
            name="dynamic_precipitation_rate",
            units="m s-1",
            spatial_grid="equalarea",
            grid_spacing=96900.0,
            domain_corner=(-1036000.0, -1158000.0),
            attributes=UK_ATTRIBUTES,
        )
    )
    return cubes


@pytest.mark.parametrize("grid_fixture", ["global_grid", "uk_grid"])
def test_basic(request, grid_fixture):
    """Ensure Plugin returns object of correct type and meta-data and that
    no precip => masked array"""
    grid = request.getfixturevalue(grid_fixture)
    expected_array = np.ma.masked_all_like(grid[0].data)
    result = ConvectionRatioFromComponents()(grid.copy())
    assert isinstance(result, iris.cube.Cube)
    assert_allclose(result.data, expected_array)
    assert result.attributes == grid[0].attributes
    assert result.long_name == "convective_ratio"
    assert result.units == "1"


# These tuples represent one data point. The first two values are the convective and
# dynamic precipitation rate respectively. The last value is the expected result.
@pytest.mark.parametrize(
    "data_con_dyn_out",
    [
        (1.0, 0.0, 1.0),
        (1.0, 1, 0.5),
        (0.0, 1.0, 0.0),
        (0.9e-9, 0.0, np.inf),
        (1.1e-9, 0.0, 1.0),
        (0.9e-9, 0.9e-9, 0.5),
    ],
)
@pytest.mark.parametrize("grid_fixture", ["global_grid", "uk_grid"])
def test_data(request, grid_fixture, data_con_dyn_out):
    """Test that the data are calculated as expected for a selection of values
    including either side of the minimum precipitation rate tolerance 1e-9 m s-1"""
    grid = request.getfixturevalue(grid_fixture)
    for i in range(2):
        grid[i].data[0, 0] = data_con_dyn_out[i]
    expected_array = np.ma.masked_all_like(grid[0].data)
    if np.isfinite(data_con_dyn_out[2]):
        expected_array[0, 0] = data_con_dyn_out[2]
    result = ConvectionRatioFromComponents()(grid)
    assert_allclose(result.data, expected_array)
    # assert_allclose doesn't check masks appropriately, so check separately
    # pylint: disable=no-member
    assert_equal(result.data.mask, expected_array.mask)


def test_bad_name(request):
    """Test we get a useful error if one of the input cubes is incorrectly named."""
    grid = request.getfixturevalue("uk_grid")
    grid[0].rename("kittens")
    with assert_raises_regex(
        ValueError, "Cannot find a cube named 'convective_precipitation_rate' in "
    ):
        ConvectionRatioFromComponents()(grid)


def test_bad_units(request):
    """Test we get a useful error if the input cubes have non-SI units."""
    grid = request.getfixturevalue("uk_grid")
    _ = [c.convert_units("mm h-1") for c in grid]
    with assert_raises_regex(AssertionError, "Units of 'm s-1' required, not "):
        ConvectionRatioFromComponents()(grid)
