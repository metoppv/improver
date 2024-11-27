# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests of GradientBetweenAdjacentGridSquares plugin."""

import numpy as np
import pytest

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import GradientBetweenAdjacentGridSquares

EQUAL_AREA_GRID_SPACING = 2000  # Metres
# Distances covered when travelling degrees north-south or east-west:
DISTANCE_PER_DEGREE_AT_EQUATOR = 111198.9234485458
DISTANCE_PER_DEGREE_AT_10_DEGREES_NORTH = 109509.56193873892

# Values cycle from east to west (sine wave). Values also decrease from equator to pole.
INPUT_DATA = (
    np.array(
        [[0, 100, 0, -100], [0, 200, 0, -200], [0, 100, 0, -100]], dtype=np.float32
    ),
)
# The standard calculations return these values (last column of x omitted for non-circular data).
# These are not a cosine wave because we don't have sufficient data points.
EXPECTED_DATA = (
    np.array(
        [
            [0.05, -0.05, -0.05, 0.05],
            [0.1, -0.1, -0.1, 0.1],
            [0.05, -0.05, -0.05, 0.05],
        ],
        dtype=np.float32,
    ),
    np.array([[0.0, 0.05, 0.0, -0.05], [0.0, -0.05, 0.0, 0.05]], dtype=np.float32),
)
# These data are expected when the above are regridded back to the source grid and the source
# data are not circular.
REGRIDDED_DATA = (
    np.array(
        [[0.1, 0, -0.05, -0.05], [0.2, 0, -0.1, -0.1], [0.1, 0, -0.05, -0.05]],
        dtype=np.float32,
    ),
    np.array(
        [[0.0, 0.1, 0.0, -0.1], [0.0, 0.0, 0.0, 0.0], [0.0, -0.1, 0.0, 0.1]],
        dtype=np.float32,
    ),
)
# These data are expected when the EXPECTED_DATA are regridded back to the source grid and the
# source data are circular, which gives us a cosine wave.
CIRCULAR_DATA = (
    np.array(
        [[0.05, 0, -0.05, 0.0], [0.1, 0, -0.1, 0.0], [0.05, 0, -0.05, 0.0]],
        dtype=np.float32,
    ),
    np.array(
        [[0.0, 0.1, 0.0, -0.1], [0.0, 0.0, 0.0, 0.0], [0.0, -0.1, 0.0, 0.1]],
        dtype=np.float32,
    ),
)


@pytest.mark.parametrize(
    "projected, circular, regrid, example_x, example_y",
    (
        (False, False, False, EXPECTED_DATA[0], EXPECTED_DATA[1]),
        (False, False, True, REGRIDDED_DATA[0], REGRIDDED_DATA[1]),
        (False, True, False, EXPECTED_DATA[0], EXPECTED_DATA[1]),
        (False, True, True, CIRCULAR_DATA[0], CIRCULAR_DATA[1]),
        (True, False, False, EXPECTED_DATA[0], EXPECTED_DATA[1]),
        (True, False, True, REGRIDDED_DATA[0], REGRIDDED_DATA[1]),
    ),
)
def test_data(projected, circular, regrid, example_x, example_y, data=INPUT_DATA[0]):
    """Tests that the plugin produces the expected data for valid projections"""
    x_grid_spacing = EQUAL_AREA_GRID_SPACING if projected else 90
    cube = set_up_variable_cube(
        data,
        spatial_grid="equalarea" if projected else "latlon",
        x_grid_spacing=x_grid_spacing,
    )
    expected_x = example_x.copy()
    expected_y = example_y.copy()
    if circular:
        cube.coord(axis="x").circular = True
    elif not regrid:  # Drop final column
        expected_x = expected_x[..., :-1]
    if not projected:  # Adjust expected values to represent 10 degrees rather than 2km
        expected_x[0::2] /= (
            x_grid_spacing
            * DISTANCE_PER_DEGREE_AT_10_DEGREES_NORTH
            / EQUAL_AREA_GRID_SPACING
        )
        expected_x[1] /= (
            x_grid_spacing * DISTANCE_PER_DEGREE_AT_EQUATOR / EQUAL_AREA_GRID_SPACING
        )
        expected_y /= 10 * DISTANCE_PER_DEGREE_AT_EQUATOR / EQUAL_AREA_GRID_SPACING
    plugin = GradientBetweenAdjacentGridSquares(regrid=regrid)
    result_x, result_y = plugin(cube)
    assert np.allclose(expected_x, result_x.data)
    assert np.allclose(expected_y, result_y.data)


# By default, projected cubes have a spacing of 2000, and 10 for latlon.
# Coords are centred on zero.
projected_y_coord_points = [-1000, 1000]
projected_x_coord_points = [-2000, 0, 2000]
latlon_y_coord_points = [-5, 5]
latlon_x_coord_points = [-10, 0, 10, 180]


@pytest.mark.parametrize(
    "projected, circular, expected_x_points, expected_y_points",
    (
        (False, False, latlon_x_coord_points[:-1], latlon_y_coord_points),
        (False, True, latlon_x_coord_points, latlon_y_coord_points),
        (True, False, projected_x_coord_points, projected_y_coord_points),
    ),
)
@pytest.mark.parametrize("data", (INPUT_DATA[0],))
@pytest.mark.parametrize("regrid", (True, False))
def test_metadata(
    data, projected, circular, expected_x_points, expected_y_points, regrid
):
    """Tests that the plugin produces cubes with the right metadata"""
    cube = set_up_variable_cube(
        data,
        spatial_grid="equalarea" if projected else "latlon",
        standard_grid_metadata="gl_det",
        attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
    )
    if circular:
        cube.coord(axis="x").circular = True
    expected_x_coord = cube.coord(axis="x").copy(expected_x_points)
    expected_y_coord = cube.coord(axis="y").copy(expected_y_points)
    plugin = GradientBetweenAdjacentGridSquares(regrid=regrid)
    result_x, result_y = plugin(cube)
    for result, name in [(result_x, "x"), (result_y, "y")]:
        assert result.name() == f"gradient_of_air_temperature_wrt_{name}"
        assert result.units == "K m-1"
        assert result.attributes == cube.attributes
        assert result.data.dtype == np.float32
    if regrid:
        # In regrid mode, we expect the original spatial coords
        for axis in "xy":
            assert result_x.coord(axis=axis) == cube.coord(axis=axis)
            assert result_y.coord(axis=axis) == cube.coord(axis=axis)
    else:
        # Regrid=False => expected coords apply to one coord of one result
        # (the one that the gradient has been calculated along)
        assert result_x.coord(axis="y") == cube.coord(axis="y")
        assert result_y.coord(axis="x") == cube.coord(axis="x")
        assert result_x.coord(axis="x") == expected_x_coord
        assert result_y.coord(axis="y") == expected_y_coord


@pytest.mark.parametrize("regrid", (True, False))
def test_error(regrid, data=INPUT_DATA[0], projected=True, circular=True):
    """Tests that an error is raised if a projected cube has a circular x coordinate.
    The content of the error is checked in the tests for the class that raises it."""
    cube = set_up_variable_cube(
        data, spatial_grid="equalarea" if projected else "latlon"
    )
    if circular:
        cube.coord(axis="x").circular = True
    plugin = GradientBetweenAdjacentGridSquares(regrid=regrid)
    with pytest.raises(NotImplementedError):
        plugin(cube)
