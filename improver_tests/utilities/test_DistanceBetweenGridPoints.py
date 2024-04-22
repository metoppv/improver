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

""" Tests of DifferenceBetweenAdjacentGridSquares plugin."""
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

TEST_LATITUDES = [0, 10, 20]
# Distance covered when travelling 10 degrees north/south:
Y_GRID_SPACING = 1111949  # Metres

DISTANCE_PER_DEGREE_AT_EQUATOR = 111319.49079327357
DISTANCE_PER_DEGREE_AT_10_DEGREES_NORTH = 109639.32210546243
DISTANCE_PER_DEGREE_AT_20_DEGREES_NORTH = 104646.93093328059

ONE_DEGREE_DISTANCE_AT_TEST_LATITUDES = np.array(
    [
        DISTANCE_PER_DEGREE_AT_EQUATOR,
        DISTANCE_PER_DEGREE_AT_10_DEGREES_NORTH,
        DISTANCE_PER_DEGREE_AT_20_DEGREES_NORTH
    ]
).reshape((3, 1))

TRANSVERSE_MERCATOR_GRID_SPACING = 2000.0  # Metres

# Todo: change all references to 'full haversine'

def make_equalarea_test_cube(shape, grid_spacing, units="metres"):
    """Creates a cube using the Lambert Azimuthal Equal Area projection for testing"""
    data = np.ones(shape, dtype=np.float32)
    cube = set_up_variable_cube(
        data, spatial_grid="equalarea", x_grid_spacing=grid_spacing, y_grid_spacing=grid_spacing
    )
    cube.coord("projection_x_coordinate").convert_units(units)
    cube.coord("projection_y_coordinate").convert_units(units)
    return cube

#TODO: do I still need this now that set_up_variable_cube supports different x and y grid spacing?
def make_test_cube(
    shape: Tuple[int, int],
    coordinate_system: CoordSystem,
    x_axis_name: str,
    x_axis_values: np.ndarray,
    y_axis_name: str,
    y_axis_values: np.ndarray,
    xy_axis_units: str,
) -> Cube:
    """Creates an example cube for use as test input."""
    example_data = np.ones(shape, dtype=np.float32)
    dimcoords = [
        (
            DimCoord(
                y_axis_values,
                standard_name=y_axis_name,
                units=xy_axis_units,
                coord_system=coordinate_system,
            ),
            0,
        ),
        (
            DimCoord(
                x_axis_values,
                standard_name=x_axis_name,
                units=xy_axis_units,
                coord_system=coordinate_system,
            ),
            1,
        ),
    ]
    cube = Cube(
        example_data,
        standard_name="land_ice_basal_temperature",
        units="kelvin",
        dim_coords_and_dims=dimcoords,
    )
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
    return make_test_cube(
        shape,
        transvers_mercator_coord_system,
        "projection_x_coordinate",
        x_points,
        "projection_y_coordinate",
        y_points,
        "metres",
    )


def make_latlon_test_cube(
    shape: Tuple[int, int], latitudes: np.ndarray, longitudes: np.ndarray
) -> Cube:
    """Creates a cube using the Geographic coordinate system with its origin at the
    intersecton of the equator and the prime meridian."""
    return make_test_cube(
        shape,
        GeogCS(EARTH_RADIUS),
        "longitude",
        longitudes,
        "latitude",
        latitudes,
        "degrees",
    )


@pytest.mark.parametrize(
    "longs",
    (
        #Longitudes, Cube_is_circular
        ([0, 10, 20], False),
        ([0, 5, 10], False),
        ([0, 11, 22], False),
        ([0, 60, 120], False),
        ([-60, 0, 60], False),
        ([0, 120, 240], False),
        ([0, 120, 240], True),
        ([-120, -60, 0, 60, 120, 180], False),
        ([-120, -60, 0, 60, 120, 180], True),
        ([0, 10], False),
        ([0, 20], False),
        ([-20, 20], False),
        ([-60, -30, 0, 30, 60], False) # Todo: dim coords are not correct. Think it's a floating point error? Probably need a test for this.
    )
)  # Todo: I think I can specify whether or not a cube axis is circular (cube.coord(axis='x').circular). From this, I can work out whether I'm expecting 2 or 3 distances along the x axis. Might need to check that latitude isn't always considered cirular. Seems unlikely, UKV is done in lat/longs.
# tODO: Should I test for non-uniform grids here too? Probably.
def test_latlon_cube(longs):
    """Basic test for a cube using a geographic coordinate system."""
    longitudes, is_circular = longs
    input_cube = make_latlon_test_cube(
        (len(TEST_LATITUDES), len(longitudes)), TEST_LATITUDES, longitudes
    )
    expected_y_distances = np.full((len(TEST_LATITUDES) - 1, len(longitudes)), Y_GRID_SPACING)
    if is_circular:
        input_cube.coord(axis="x").circular = True
        longitudes.append(360 + longitudes[0])  # TODO: I feel like this could be clearer.
    expected_x_distances = np.diff(longitudes) * ONE_DEGREE_DISTANCE_AT_TEST_LATITUDES
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares(input_cube)()
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "metres"
        np.testing.assert_allclose(
            result.data, expected.data, rtol=33e-4, atol=0
        )  # Allowing 0.33% error for spherical earth approximation.


@pytest.mark.parametrize(
    "test_case",
    (
        # Distances, Cube_is_circular
        (1000, False),
        (13358333, True),  # 13358333 ~= 1/3 of the Earths' circumference
    )
)
def test_equalarea_cube(test_case):
    """Basic test for a cube using a Lambert Azumutal Equal Area projection"""
    spacing, circular = test_case
    input_cube = make_equalarea_test_cube((3, 3), grid_spacing=spacing)
    if circular:
        input_cube.coord(axis="x").circular = True
        expected_x_distances_cube_shape = (3, 3)
    else:
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
        assert result.units == "metres"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-5, atol=0)


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
        assert result.units == "metres"
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
        assert result.units == "metres"
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
        assert result.units == "metres"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-5, atol=0)


def test_degrees_cube_with_no_coordinate_system_information():
    """
    Tests that a cube which does not contain enough information to determine distances
    between grid points is handled appropriately.
    """
    input_cube = make_test_cube(
        shape=(3, 3),
        coordinate_system=None,
        x_axis_name="projection_x_coordinate",
        x_axis_values=np.arange(3),
        y_axis_name="projection_y_coordinate",
        y_axis_values=np.arange(3),
        xy_axis_units="degrees",
    )
    with IrisTest().assertRaisesRegex(
        expected_exception=ValueError,
        expected_regex="Unsupported cube coordinate system.*",
    ):
        _, _ = DistanceBetweenGridSquares(input_cube)()
