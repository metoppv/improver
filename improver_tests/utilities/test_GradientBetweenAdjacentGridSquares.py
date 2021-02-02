# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
""" Tests of GradientBetweenAdjacentGridSquares plugin."""

import numpy as np
import pytest
from iris.cube import Cube

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import GradientBetweenAdjacentGridSquares


@pytest.fixture(name="wind_speed")
def wind_speed_fixture() -> Cube:
    """Wind speed in m/s"""
    data = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.float32)
    cube = set_up_variable_cube(
        data, name="wind_speed", units="m s^-1", spatial_grid="equalarea"
    )
    for axis in ["x", "y"]:
        cube.coord(axis=axis).points = np.array([0, 1, 2], dtype=np.float32)
    return cube


@pytest.fixture(name="make_expected")
def make_expected_fixture() -> callable:
    """Factory as fixture for generating a cube of varying size."""

    def _make_expected(shape, value) -> Cube:
        """Create a cube filled with data of a specific shape and value."""
        data = np.full(shape, value, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="gradient_of_wind_speed",
            units="s^-1",
            spatial_grid="equalarea",
            attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        )
        for index, axis in enumerate(["y", "x"]):
            cube.coord(axis=axis).points = np.array(
                np.arange(shape[index]), dtype=np.float32
            )
        return cube

    return _make_expected


@pytest.mark.parametrize(
    "grid",
    (
        {"regrid": False, "xshape": (3, 2), "yshape": (2, 3)},
        {"regrid": True, "xshape": (3, 3), "yshape": (3, 3)},
    ),
)
def test_gradient(wind_speed, make_expected, grid):
    """Check calculating the gradient with and without regridding"""
    expected_x = make_expected(grid["xshape"], 1)
    expected_y = make_expected(grid["yshape"], 2)
    gradient_x, gradient_y = GradientBetweenAdjacentGridSquares(regrid=grid["regrid"])(
        wind_speed
    )
    for result, expected in zip((gradient_x, gradient_y), (expected_x, expected_y)):
        assert result.name() == expected.name()
        assert result.attributes == expected.attributes
        assert result.units == expected.units
        np.testing.assert_allclose(result.data, expected.data, rtol=1e-5, atol=1e-8)
