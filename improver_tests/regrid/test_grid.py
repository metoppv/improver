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
"""Unit tests for the functions from grid.py."""

import itertools
from collections import namedtuple
from datetime import datetime

import numpy as np
import pytest
from iris.tests import IrisTest

from improver.regrid.grid import (
    calculate_input_grid_spacing,
    create_regrid_cube,
    ensure_ascending_coord,
    flatten_spatial_dimensions,
    get_cube_coord_names,
    latlon_from_cube,
    latlon_names,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LAT = "latitude"
LON = "longitude"
PROJX = "projection_x_coordinate"
PROJY = "projection_y_coordinate"
R9N = "realization"


@pytest.fixture(autouse=True)
def latlon_single():
    return set_up_variable_cube(
        np.reshape(np.linspace(1, 9, 9, dtype=np.float32), (3, 3)),
        spatial_grid="latlon",
    )


@pytest.fixture(autouse=True)
def latlon_ens():
    return set_up_variable_cube(
        np.reshape(np.linspace(1, 60, 60, dtype=np.float32), (3, 4, 5)),
        spatial_grid="latlon",
    )


@pytest.fixture(autouse=True)
def equal_single():
    return set_up_variable_cube(
        np.reshape(np.linspace(1, 9, 9, dtype=np.float32), (3, 3)),
        spatial_grid="equalarea",
    )


@pytest.fixture(autouse=True)
def equal_ens():
    return set_up_variable_cube(
        np.reshape(np.linspace(1, 60, 60, dtype=np.float32), (3, 4, 5)),
        spatial_grid="equalarea",
    )


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("latlon_single", [LAT, LON]),
        ("equal_single", [PROJY, PROJX]),
        ("latlon_ens", [R9N, LAT, LON]),
        ("equal_ens", [R9N, PROJY, PROJX]),
    ],
)
def test_get_cube_coord_names(request, fixture_name, expected):
    """Test the get_cube_coord_names function"""
    cube = request.getfixturevalue(fixture_name)
    names = get_cube_coord_names(cube)
    assert names == expected


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("latlon_single", (LAT, LON)),
        ("equal_single", (PROJY, PROJX)),
        ("latlon_ens", (LAT, LON)),
        ("equal_ens", (PROJY, PROJX)),
    ],
)
def test_latlon_names(request, fixture_name, expected):
    """Test the latlon_names function"""
    cube = request.getfixturevalue(fixture_name)
    names = latlon_names(cube)
    assert names == expected


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("latlon_single", np.array(list(itertools.product([-10, 0, 10], repeat=2)))),
        ("equal_single", np.array(list(itertools.product([-2000, 0, 2000], repeat=2)))),
    ],
)
def test_latlon_from_cube(request, fixture_name, expected):
    """Test the latlon_from_cube function"""
    cube = request.getfixturevalue(fixture_name)
    latlons = latlon_from_cube(cube)
    np.testing.assert_equal(latlons, expected)


@pytest.mark.parametrize(
    "fixture_name", ["latlon_single", "equal_single", "latlon_ens", "equal_ens"],
)
def test_flatten_spatial_dimensions(request, fixture_name):
    """Test the flatten_spatial_dimensions function"""
    cube = request.getfixturevalue(fixture_name)
    flat, lat_index, lon_index = flatten_spatial_dimensions(cube)
    if cube.data.ndim == 2:
        np.testing.assert_equal(flat, cube.data.flatten())
    else:
        assert flat.shape == (20, 3)
        np.testing.assert_equal(flat[0:2, :], [[1, 21, 41], [2, 22, 42]])


@pytest.mark.parametrize("flip", (True, False))
def test_ensure_ascending_coord(flip):
    """Test the ensure_ascending_coord function"""

    """Set up a lat/lon cube"""
    lat_lon_cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32))
    lon_coord = lat_lon_cube.coord("longitude").points
    lat_coord = lat_lon_cube.coord("latitude").points
    if flip:
        lat_lon_cube.coord("longitude").points = lon_coord[::-1]
        lat_lon_cube.coord("latitude").points = lat_coord[::-1]
    lat_lon_cube = ensure_ascending_coord(lat_lon_cube)

    np.testing.assert_allclose(lat_lon_cube.coord("latitude").points, lat_coord)
    np.testing.assert_allclose(lat_lon_cube.coord("longitude").points, lon_coord)


class Test_calculate_input_grid_spacing(IrisTest):
    """Test the calculate_input_grid_spacing function"""

    def setUp(self):
        """Set up a lat/lon cube"""
        self.unit = "degrees"
        self.lat_lon_cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32))
        self.equal_area_cube = set_up_variable_cube(
            np.ones((5, 5), dtype=np.float32), spatial_grid="equalarea"
        )

    def test_incorrect_projection(self):
        """Test ValueError for incorrect projections"""
        msg = "Input grid is not on a latitude/longitude system"
        with self.assertRaisesRegex(ValueError, msg):
            calculate_input_grid_spacing(self.equal_area_cube)

    def test_descending_lat_lon_coordinates(self):
        """Test ValueError for descending coordinates"""
        self.lat_lon_cube.coord("longitude").points = [20.0, 10.0, 0.0, -10.0, -20.0]
        msg = "Input grid coordinates are not ascending"
        with self.assertRaisesRegex(ValueError, msg):
            calculate_input_grid_spacing(self.lat_lon_cube)

    def test_lat_lon_equal_spacing(self):
        """Test grid spacing outputs with lat-lon grid in degrees"""
        result = calculate_input_grid_spacing(self.lat_lon_cube)
        self.assertAlmostEqual(result, (10.0, 10.0))


Names = namedtuple(
    "Names", ["setup_name", "setup_var_name", "standard_name", "long_name", "var_name"]
)


@pytest.mark.parametrize(
    "cube_names",
    [
        Names("air_temperature", None, "air_temperature", None, None),
        Names("not_a_standard_name", None, None, "not_a_standard_name", None),
        Names(
            "air_temperature", "some_var_name", "air_temperature", None, "some_var_name"
        ),
    ],
)
def test_create_regrid_cube(cube_names):
    """Test the create_regrid_cube function, including standard, long and var names."""

    source_cube_latlon = set_up_variable_cube(
        np.ones((2, 5, 5), dtype=np.float32),
        name=cube_names.setup_name,
        time=datetime(2018, 11, 10, 8, 0),
        frt=datetime(2018, 11, 10, 0, 0),
    )
    if cube_names.setup_var_name:
        source_cube_latlon.var_name = cube_names.setup_var_name
    target_cube_equalarea = set_up_variable_cube(
        np.ones((10, 10), dtype=np.float32), spatial_grid="equalarea"
    )
    data = np.repeat(1.0, 200).reshape(2, 10, 10)
    cube_v = create_regrid_cube(data, source_cube_latlon, target_cube_equalarea)
    assert cube_v.shape == (2, 10, 10)
    assert cube_v.coord(axis="x").standard_name == "projection_x_coordinate"
    assert cube_v.coord(axis="y").standard_name == "projection_y_coordinate"
    assert cube_v.coord("forecast_reference_time") == source_cube_latlon.coord(
        "forecast_reference_time"
    )
    assert cube_v.standard_name == cube_names.standard_name
    assert cube_v.long_name == cube_names.long_name
    assert cube_v.var_name == cube_names.var_name
