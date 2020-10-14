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
""" Tests of SnowFraction plugin"""

from datetime import datetime

import iris
import numpy as np
import pytest
from cf_units import Unit

from improver.snow_fraction import SnowFraction
from improver.synthetic_data.set_up_test_cubes import (
    construct_scalar_time_coords,
    set_up_variable_cube,
)

COMMON_ATTRS = {
    "source": "Unit test",
    "institution": "Met Office",
    "title": "Post-Processed IMPROVER unit test",
}
RAIN_DATA = np.array([[0, 0.0], [0.5, 1]], dtype=np.float32)
SNOW_DATA = np.array([[0, 0.5], [1, 0.0]], dtype=np.float32)


def setup_cubes(rain_data=RAIN_DATA, snow_data=SNOW_DATA, name="{phase}rate"):
    """Make CF-compliant rain and snow cubes from supplied arrays"""
    if "rate" in name:
        units = "m s-1"
        time_bounds = None
    else:
        units = "m"
        time_bounds = (datetime(2017, 11, 10, 3, 0), datetime(2017, 11, 10, 4, 0))
    rain = set_up_variable_cube(
        rain_data,
        name.format(phase="rain"),
        units,
        time_bounds=time_bounds,
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
    )
    snow = set_up_variable_cube(
        snow_data,
        name.format(phase="lwe_snow"),
        units,
        time_bounds=time_bounds,
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
    )
    return rain, snow


@pytest.mark.parametrize(
    "cube_name", ("{phase}rate", "thickness_of_{phase}fall_amount")
)
def test_basic(cube_name):
    """Run a test with four values, including one that will trigger divide-by-zero.
    Check data and metadata of result."""
    rain, snow = setup_cubes(name=cube_name)

    expected_data = np.array([[1.0, 1.0], [2.0 / 3.0, 0.0]], dtype=np.float32)
    result = SnowFraction()(iris.cube.CubeList([rain, snow]))
    assert isinstance(result, iris.cube.Cube)
    assert str(result.units) == "1"
    assert result.name() == "snow_fraction"
    assert result.attributes == COMMON_ATTRS
    assert np.allclose(result.data, expected_data)


def test_acclen_mismatch_error():
    """Test the process function with mismatched accumulation lengths"""
    rain, snow = setup_cubes(name="thickness_of_{phase}fall_amount")
    time_coords = construct_scalar_time_coords(
        [c.point for c in snow.coord("time").cells()],
        (datetime(2017, 11, 10, 1, 0), datetime(2017, 11, 10, 4, 0)),
        snow.coord("forecast_reference_time").cell(0).point,
    )
    _ = [snow.replace_coord(coord) for coord, _ in time_coords]
    with pytest.raises(
        ValueError, match="Rain and snow cubes do not have the same time coord"
    ):
        SnowFraction()(iris.cube.CubeList([rain, snow]))


def test_dims_mismatch_error():
    """Test the process function with mismatched dimensions"""
    rain, snow = setup_cubes(snow_data=np.array([[0, 0.5]], dtype=np.float32))
    with pytest.raises(
        ValueError, match="Rain and snow cubes are not on the same grid"
    ):
        SnowFraction()(iris.cube.CubeList([rain, snow]))


def test_missing_cube_error():
    """Test the process function with one cube missing"""
    rain, _ = setup_cubes()
    with pytest.raises(ValueError, match="Expected exactly 2 input cubes, found 1"):
        SnowFraction()(iris.cube.CubeList([rain]))


def test_wrong_input_names_error():
    """Test the process function with incorrect input cubes"""
    rain, snow = setup_cubes()
    rain.rename("kittens")
    snow.rename("puppies")
    with pytest.raises(
        ValueError,
        match="Could not find both rain and snow in \['kittens', 'puppies'\]",
    ):
        SnowFraction()(iris.cube.CubeList([rain, snow]))


def test_input_name_matches_both_phases_error():
    """Test the process function with an input cube that has both rain and snow in its name"""
    rain, snow = setup_cubes()
    rain.rename("its_raining_snowy_kittens")
    snow.rename("puppies")
    with pytest.raises(
        ValueError,
        match="Failed to find unique rain and snow cubes from \['its_raining_snowy_kittens', 'puppies'\]",
    ):
        SnowFraction()(iris.cube.CubeList([rain, snow]))


def test_non_coercing_units_error():
    """Test the process function with input cubes of incompatible units"""
    rain, snow = setup_cubes()
    rain.units = Unit("K")
    with pytest.raises(
        ValueError, match="Unable to convert from 'Unit\('m s-1'\)' to 'Unit\('K'\)'."
    ):
        SnowFraction()(iris.cube.CubeList([rain, snow]))
