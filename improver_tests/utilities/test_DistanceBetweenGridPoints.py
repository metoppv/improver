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

import numpy as np
from iris.cube import Cube
from iris.coords import DimCoord
from iris.coord_systems import CoordSystem, GeogCS, TransverseMercator
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import DistanceBetweenGridSquares


EARTH_RADIUS = 6.6371e3  # meters

# Distances covered when travelling 10 degrees east/west at different latitudes:
X_GRID_SPACING_AT_EQUATOR = 1111949  # Metres
X_GRID_SPACING_AT_10_DEGREES_NORTH = 1095014  # Metres
X_GRID_SPACING_AT_20_DEGREES_NORTH = 1044735  # Metres
# Distance covered when travelling 10 degrees north/south:
Y_GRID_SPACING = 1111949  # Metres

TRANSVERSE_MERCATOR_GRID_SPACING = 2000.0  # Metres


def make_equalarea_test_cube(shape, grid_spacing, units="meters"):
    """Creates a cube using the Lambert Azimuthal Equal Area projection for testing"""
    data = np.ones(shape, dtype=np.float32)
    cube = set_up_variable_cube(
        data, spatial_grid="equalarea", grid_spacing=grid_spacing
    )
    cube.coord("projection_x_coordinate").convert_units(units)
    cube.coord("projection_y_coordinate").convert_units(units)
    return cube


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
    TMercCS = TransverseMercator(
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
        TMercCS,
        "projection_x_coordinate",
        x_points,
        "projection_y_coordinate",
        y_points,
        "meters",
    )


def make_latlon_test_cube(
    shape: Tuple[int, int], latitudes: np.ndarray, longitudes: np.ndarray
) -> Cube:
    """Creates a cube using the Geographic projection for testing"""
    return make_test_cube(
        shape,
        GeogCS(EARTH_RADIUS),
        "longitude",
        longitudes,
        "latitude",
        latitudes,
        "degrees",
    )


def test_latlon_cube_nonuniform_spacing():
    input_cube = make_latlon_test_cube(
        (2, 3), latitudes=[0, 20], longitudes=[0, 10, 20]
    )
    expected_x_distances = np.array(
        [
            [X_GRID_SPACING_AT_EQUATOR, X_GRID_SPACING_AT_EQUATOR],
            [X_GRID_SPACING_AT_20_DEGREES_NORTH, X_GRID_SPACING_AT_20_DEGREES_NORTH],
        ]
    )
    expected_y_distances = np.full((1, 3), 2 * Y_GRID_SPACING)
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "meters"
        np.testing.assert_allclose(
            result.data, expected.data, rtol=2e-3, atol=0
        )  # Allowing 0.2% error for spherical earth approximation.


def test_latlon_cube_unequal_xy_dims():
    input_cube = make_latlon_test_cube(
        (3, 2), latitudes=[0, 10, 20], longitudes=[0, 10]
    )
    expected_x_distances = np.array(
        [
            [X_GRID_SPACING_AT_EQUATOR],
            [X_GRID_SPACING_AT_10_DEGREES_NORTH],
            [X_GRID_SPACING_AT_20_DEGREES_NORTH],
        ]
    )
    expected_y_distances = np.full((2, 2), Y_GRID_SPACING)
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "meters"
        np.testing.assert_allclose(
            result.data, expected.data, rtol=2e-3, atol=0
        )  # Allowing 0.2% error for spherical earth approximation.


def test_latlon_cube():
    input_cube = make_latlon_test_cube(
        (3, 3), latitudes=[0, 10, 20], longitudes=[0, 10, 20]
    )
    expected_x_distances = np.array(
        [
            [X_GRID_SPACING_AT_EQUATOR, X_GRID_SPACING_AT_EQUATOR],
            [X_GRID_SPACING_AT_10_DEGREES_NORTH, X_GRID_SPACING_AT_10_DEGREES_NORTH],
            [X_GRID_SPACING_AT_20_DEGREES_NORTH, X_GRID_SPACING_AT_20_DEGREES_NORTH],
        ]
    )
    expected_y_distances = np.full((2, 3), Y_GRID_SPACING)
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "meters"
        np.testing.assert_allclose(
            result.data, expected.data, rtol=2e-3, atol=0
        )  # Allowing 0.2% error for spherical earth approximation.


def test_equalarea_cube():
    input_cube = make_equalarea_test_cube((3, 3), grid_spacing=1000)
    expected_x_distances = np.full((3, 2), 1000)
    expected_y_distances = np.full((2, 3), 1000)
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "meters"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-5, atol=0)


def test_equalarea_cube_nonstandard_units():
    input_cube = make_equalarea_test_cube((3, 3), grid_spacing=10, units="km")
    expected_x_distances = np.full((3, 2), 10)
    expected_y_distances = np.full((2, 3), 10)
    (
        calculated_x_distances_cube,
        calculated_y_distances_cube,
    ) = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "meters"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-5, atol=0)


def test_transverse_mercator_cube():
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
    ) = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "meters"
        np.testing.assert_allclose(
            result.data, expected.data, rtol=2e-3, atol=0
        )  # Allowing 0.2% error for spherical earth approximation.


def test_distance_cube_with_no_coordinate_system():
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
    ) = DistanceBetweenGridSquares()(input_cube)
    for result, expected in zip(
        (calculated_x_distances_cube, calculated_y_distances_cube),
        (expected_x_distances, expected_y_distances),
    ):
        assert result.units == "meters"
        np.testing.assert_allclose(result.data, expected.data, rtol=2e-5, atol=0)


def test_degrees_cube_with_no_coordinate_system_information():
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
        _, _ = DistanceBetweenGridSquares()(input_cube)
