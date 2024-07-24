# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""" Tests of GradientBetweenAdjacentGridSquares plugin."""

import numpy as np
import pytest

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import GradientBetweenAdjacentGridSquares

EQUAL_AREA_GRID_SPACING = 2000  # Metres
# Distances covered when travelling degrees north-south or east-west:
DISTANCE_PER_DEGREE_AT_EQUATOR = 111198.9234485458
DISTANCE_PER_DEGREE_AT_5_DEGREES_NORTH = 110775.77797295121
DISTANCE_PER_DEGREE_AT_10_DEGREES_NORTH = 109509.56193873892

# Values increase from east to west, and therefore decrease across the meridian.
# Values also increase from south to north.
EXAMPLE_DATA = (
    np.array([[0, 100, 200], [300, 400, 500], [400, 500, 600]], dtype=np.float32),
    np.array(
        [
            [0.05, 0.05, -0.00294436568],
            [0.05, 0.05, -0.00294436568],
            [0.05, 0.05, -0.00294436568],
        ],
        dtype=np.float32,
    ),
    np.array([[0.15, 0.15, 0.15], [0.05, 0.05, 0.05]], dtype=np.float32),
)


@pytest.mark.parametrize(
    "projected, circular", ((False, False), (False, True), (True, False))
)
@pytest.mark.parametrize("data, example_x, example_y", (EXAMPLE_DATA,))
def test_data(data, example_x, example_y, projected, circular):
    """Tests that the plugin produces the expected data when regrid mode is off"""
    cube = set_up_variable_cube(
        data, spatial_grid="equalarea" if projected else "latlon"
    )
    expected_x = example_x.copy()
    expected_y = example_y.copy()
    if circular:
        cube.coord(axis="x").circular = True
    else:  # Drop final column
        expected_x = expected_x[..., :-1]
    if not projected:  # Adjust expected values to represent 10 degrees rather than 2km
        expected_x[0::2] /= (
            10 * DISTANCE_PER_DEGREE_AT_10_DEGREES_NORTH / EQUAL_AREA_GRID_SPACING
        )
        expected_x[1] /= 10 * DISTANCE_PER_DEGREE_AT_EQUATOR / EQUAL_AREA_GRID_SPACING
        expected_y /= 10 * DISTANCE_PER_DEGREE_AT_EQUATOR / EQUAL_AREA_GRID_SPACING
    plugin = GradientBetweenAdjacentGridSquares(regrid=False)
    result_x, result_y = plugin(cube)
    assert np.allclose(expected_x, result_x.data)
    assert np.allclose(expected_y, result_y.data)


projected_y_coord_points = [-1000, 1000]
projected_x_coord_points = [-1000, 1000]
latlon_y_coord_points = [-5, 5]
latlon_x_coord_points = [-5, 5, 180]


@pytest.mark.parametrize(
    "projected, circular, expected_x_points, expected_y_points",
    (
        (False, False, latlon_x_coord_points[:-1], latlon_y_coord_points),
        (False, True, latlon_x_coord_points, latlon_y_coord_points),
        (True, False, projected_x_coord_points, projected_y_coord_points),
    ),
)
@pytest.mark.parametrize("data", (EXAMPLE_DATA[0],))
def test_metadata(data, projected, circular, expected_x_points, expected_y_points):
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
    plugin = GradientBetweenAdjacentGridSquares(regrid=False)
    result_x, result_y = plugin(cube)
    for result in (result_x, result_y):
        assert result.name() == "gradient_of_air_temperature"
        assert result.units == "K m-1"
        assert result.attributes == cube.attributes
    assert result_x.coord(axis="y") == cube.coord(axis="y")
    assert result_y.coord(axis="x") == cube.coord(axis="x")
    assert result_x.coord(axis="x") == expected_x_coord
    assert result_y.coord(axis="y") == expected_y_coord


@pytest.mark.parametrize("projected, circular", ((True, True),))
@pytest.mark.parametrize("data", (EXAMPLE_DATA[0],))
def test_error(data, projected, circular):
    """Tests that an error is raised if a projected cube has a circular x coordinate"""
    cube = set_up_variable_cube(
        data, spatial_grid="equalarea" if projected else "latlon"
    )
    if circular:
        cube.coord(axis="x").circular = True
    plugin = GradientBetweenAdjacentGridSquares(regrid=False)
    with pytest.raises(NotImplementedError):
        plugin(cube)
