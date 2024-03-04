# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""" Tests of GradientBetweenAdjacentGridSquares plugin."""

import numpy as np
import pytest
from iris.cube import Cube

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import GradientBetweenAdjacentGridSquares


EQUAL_AREA_GRID_SPACING = 1000  # Meters


@pytest.fixture(name="make_input")
def make_wind_speed_fixture() -> callable:
    """Factory as fixture for generating a wind speed cube as test input."""

    def _make_input() -> Cube:
        """Wind speed in m/s"""
        data = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.float32)
        cube = set_up_variable_cube(
            data, name="wind_speed", units="m s^-1", spatial_grid="equalarea", grid_spacing=EQUAL_AREA_GRID_SPACING,
        )
        # for axis in ["x", "y"]:
        #     print(cube.coord(axis=axis).points)
        return cube

    return _make_input


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
            grid_spacing=EQUAL_AREA_GRID_SPACING,
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
def test_gradient(make_input, make_expected, grid):
    """Check calculating the gradient with and without regridding"""
    # print(wind_speed)
    # print("\n\n\n")
    wind_speed = make_input()
    expected_x = make_expected(grid["xshape"], 1 / EQUAL_AREA_GRID_SPACING)
    expected_y = make_expected(grid["yshape"], 2 / EQUAL_AREA_GRID_SPACING)
    gradient_x, gradient_y = GradientBetweenAdjacentGridSquares(regrid=grid["regrid"])(
        wind_speed
    )
    for result, expected in zip((gradient_x, gradient_y), (expected_x, expected_y)):
        assert result.name() == expected.name()
        assert result.attributes == expected.attributes
        assert result.units == expected.units
        np.testing.assert_allclose(result.data, expected.data, rtol=1e-5, atol=1e-8)

# TODO: needs to also work with lat/long grids.