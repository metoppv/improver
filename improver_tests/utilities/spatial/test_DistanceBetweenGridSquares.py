# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""" Tests of DifferenceBetweenAdjacentGridSquares plugin, which encompasses all paths through
LatLonCubeDistanceCalculator, ProjectionCubeDistanceCalculator and BaseDistanceCalculator."""
from typing import Tuple
import pytest

import numpy as np
from iris.cube import Cube
from iris.coords import DimCoord
from iris.coord_systems import CoordSystem, GeogCS, TransverseMercator
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import DistanceBetweenGridSquares


EARTH_RADIUS = 6371229.0  # metres

TEST_LATITUDES = np.array([0, 10, 20])
# Distance covered when travelling 10 degrees north/south:
Y_GRID_SPACING = 1111949  # Metres

DISTANCE_PER_DEGREE_AT_EQUATOR = 111319.49079327357
DISTANCE_PER_DEGREE_AT_10_DEGREES_NORTH = 109639.32210546243
DISTANCE_PER_DEGREE_AT_20_DEGREES_NORTH = 104646.93093328059

ONE_DEGREE_DISTANCE_AT_TEST_LATITUDES = np.array(
    [
        DISTANCE_PER_DEGREE_AT_EQUATOR,
        DISTANCE_PER_DEGREE_AT_10_DEGREES_NORTH,
        DISTANCE_PER_DEGREE_AT_20_DEGREES_NORTH,
    ]
).reshape((3, 1))

TRANSVERSE_MERCATOR_GRID_SPACING = 2000.0  # Metres


def make_equalarea_test_cube(shape, grid_spacing, units="metres"):
    """Creates a cube using the Lambert Azimuthal Equal Area projection for testing"""
    data = np.ones(shape, dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        spatial_grid="equalarea",
        x_grid_spacing=grid_spacing,
        y_grid_spacing=grid_spacing,
    )
    cube.coord("projection_x_coordinate").convert_units(units)
    cube.coord("projection_y_coordinate").convert_units(units)
    return cube


def make_test_cube(
    shape: Tuple[int, int],
    coordinate_system: CoordSystem,
    x_axis_values: np.ndarray,
    y_axis_values: np.ndarray,
) -> Cube:
    """Creates an example cube for use as test input that can have unequal spatial coordinates."""
    example_data = np.ones(shape, dtype=np.float32)
    cube = set_up_variable_cube(
        example_data,
        spatial_grid="latlon" if type(coordinate_system) == GeogCS else "equalarea",
    )
    cube.replace_coord(cube.coord(axis="x").copy(x_axis_values))
    cube.replace_coord(cube.coord(axis="y").copy(y_axis_values))
    return cube


def make_transverse_mercator_test_cube(shape: Tuple[int, int]) -> Cube:
    """
    Data are on a 2 km Transverse Mercator grid with an inverted y-axis,
    located in the UK.
    """
    # UKPP projection
    transvers_mercator_coord_system = TransverseMercator(
        latitude_of_projection_origin=49.0,
        longitude_of_central_meridian=-2.0,
        false_easting=400000.0,
        false_northing=-100000.0,
        scale_factor_at_central_meridian=0.9996013045310974,
        ellipsoid=GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.91),
    )
    xo = 400000.0
    yo = 0.0
    y_points = TRANSVERSE_MERCATOR_GRID_SPACING * (shape[0] - np.arange(shape[0])) + yo
    x_points = TRANSVERSE_MERCATOR_GRID_SPACING * np.arange(shape[1]) + xo
    return make_test_cube(shape, transvers_mercator_coord_system, x_points, y_points)


def make_latlon_test_cube(
    shape: Tuple[int, int], latitudes: np.ndarray, longitudes: np.ndarray
) -> Cube:
    """Creates a cube using the Geographic coordinate system with its origin at the
    intersecton of the equator and the prime meridian."""
    return make_test_cube(shape, GeogCS(EARTH_RADIUS), longitudes, latitudes)


@pytest.mark.parametrize(
    "longitudes, is_circular",
    (
        ([0, 10, 20], False),
        ([0, 5, 10], False),
        ([0, 11, 22], False),
        ([0, 60, 120], False),
        ([0, 120, 240], False),
        ([0, 120, 240], True),
        ([0, 60, 120, 180, 240, 300], False),
        ([0, 60, 120, 180, 240, 300], True),
        ([0, 10], False),
        ([0, 20], False),
        ([-20, 20], False),
        ([0, 30, 60, 90, 120], False),
        ([0, 120, 180, 300], False),
        ([0, 120, 180, 300], True),
    ),
)
def test_latlon_cube(longitudes, is_circular):
    """Basic test for a cube using a geographic coordinate system."""
    input_cube = make_latlon_test_cube(
        (len(TEST_LATITUDES), len(longitudes)), TEST_LATITUDES, longitudes
    )
    expected_y_distances = np.full(
        (len(TEST_LATITUDES) - 1, len(longitudes)), Y_GRID_SPACING
    )
    expected_longitudes = longitudes.copy()
    if is_circular:
        input_cube.coord(axis="x").circular = True
        expected_longitudes.append(360 + longitudes[0])
    expected_x_distances = (
        np.diff(expected_longitudes) * ONE_DEGREE_DISTANCE_AT_TEST_LATITUDES
    )
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares(input_cube)()
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "m"
        np.testing.assert_allclose(
            result.data, expected.data, rtol=33e-4, atol=0
        )  # Allowing 0.33% error for spherical earth approximation.


def test_equalarea_cube():
    """Basic test for a non-circular cube using a Lambert Azumutal Equal Area projection"""
    spacing = 1000
    input_cube = make_equalarea_test_cube((3, 3), grid_spacing=spacing)
    expected_x_distances_cube_shape = (3, 2)

    expected_x_distances = np.full(expected_x_distances_cube_shape, spacing)
    expected_y_distances = np.full((2, 3), spacing)
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares(input_cube)()
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "m"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-5, atol=0)


def test_equalarea_circular_cube_error():
    """Test for error with circular cube using a Lambert Azumutal Equal Area projection"""
    input_cube = make_equalarea_test_cube((3, 3), grid_spacing=1000)
    input_cube.coord(axis="x").circular = True
    with pytest.raises(
        NotImplementedError, match="Cannot calculate distances between bounding points"
    ):
        DistanceBetweenGridSquares(input_cube)()


def test_equalarea_cube_nonstandard_units():
    """
    Test for a cube using a Lambert Azumutal Equal Area projection with units of
    kilometers for its x and y axes.
    """
    input_cube = make_equalarea_test_cube((3, 3), grid_spacing=10, units="km")
    expected_x_distances = np.full((3, 2), 10)
    expected_y_distances = np.full((2, 3), 10)
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares(input_cube)()
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "m"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-5, atol=0)


def test_transverse_mercator_cube():
    """Test for a cube using a Transverse Mercator projection"""
    input_cube = make_transverse_mercator_test_cube((3, 2))
    expected_x_distances = np.array(
        [
            [TRANSVERSE_MERCATOR_GRID_SPACING],
            [TRANSVERSE_MERCATOR_GRID_SPACING],
            [TRANSVERSE_MERCATOR_GRID_SPACING],
        ]
    )
    expected_y_distances = np.full((2, 2), TRANSVERSE_MERCATOR_GRID_SPACING)
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares(input_cube)()
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "m"
        np.testing.assert_allclose(
            result.data, expected.data, rtol=2e-3, atol=0
        )  # Allowing 0.2% error for spherical earth approximation.


def test_distance_cube_with_no_coordinate_system():
    """
    Test for a cube with no specified coordinate system but known distances between
    adjacent grid points
    """
    data = np.ones((3, 3))
    x_coord = DimCoord(np.arange(3), "projection_x_coordinate", units="km")
    y_coord = DimCoord(np.arange(3), "projection_y_coordinate", units="km")
    input_cube = Cube(
        data,
        long_name="topography",
        units="m",
        dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
    )
    expected_x_distances = np.full((3, 2), 1000)
    expected_y_distances = np.full((2, 3), 1000)
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares(input_cube)()
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "m"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-5, atol=0)


def test_degrees_cube_with_no_coordinate_system_information():
    """
    Tests that a cube which does not contain enough information to determine distances
    between grid points is handled appropriately.
    """
    input_cube = make_test_cube(
        shape=(3, 3),
        coordinate_system=GeogCS(EARTH_RADIUS),
        x_axis_values=np.arange(3),
        y_axis_values=np.arange(3),
    )
    input_cube.coord(axis="x").coord_system = None
    input_cube.coord(axis="y").coord_system = None
    with IrisTest().assertRaisesRegex(
        expected_exception=ValueError,
        expected_regex="Unsupported cube coordinate system.*",
    ):
        DistanceBetweenGridSquares(input_cube)()
