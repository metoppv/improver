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
""" Tests of GradientBetweenAdjacentGridSquares plugin."""

import unittest

import numpy as np
import pytest
from iris.cube import Cube

from improver.utilities.spatial import GradientBetweenAdjacentGridSquares

from ..set_up_test_cubes import set_up_variable_cube


@pytest.fixture(name="orography")
def orography_fixture() -> Cube:
    """Orography in m"""
    data = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.float32)
    cube = set_up_variable_cube(
        data, name="surface_altitude", units="m", spatial_grid="equalarea"
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
            name="gradient_of_surface_altitude",
            units="m",
            spatial_grid="equalarea",
        )
        for index, axis in enumerate(["y", "x"]):
            cube.coord(axis=axis).points = np.array(
                np.arange(shape[index]), dtype=np.float32
            )
        return cube

    return _make_expected


def check_assertions(result, expected):
    """Compare results of test functions"""
    assert isinstance(result, tuple)
    assert result[0].name() == expected[0].name()
    assert result[1].name() == expected[1].name()
    assert result[0].attributes == expected[0].attributes
    assert result[1].attributes == expected[1].attributes
    np.testing.assert_allclose(result[0].data, expected[0].data, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(result[1].data, expected[1].data, rtol=1e-5, atol=1e-8)


def test_with_regrid(orography, make_expected):
    """Check calculating the gradient with regridding enabled."""
    x_cube = make_expected((3, 3), 1)
    y_cube = make_expected((3, 3), 2)
    result = GradientBetweenAdjacentGridSquares(regrid=True)(orography)
    check_assertions(result, (x_cube, y_cube))


def test_without_regrid(orography, make_expected):
    """Check calculating the gradient with regridding disabled."""
    x_cube = make_expected((3, 2), 1)
    y_cube = make_expected((2, 3), 2)
    result = GradientBetweenAdjacentGridSquares()(orography)
    check_assertions(result, (x_cube, y_cube))


if __name__ == "__main__":
    unittest.main()
