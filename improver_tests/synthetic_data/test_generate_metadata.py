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
"""Tests for metadata cube generation."""

from datetime import datetime

import iris
import numpy as np
import pytest

from improver.synthetic_data.generate_metadata import generate_metadata
from improver.utilities.temporal import iris_time_to_datetime


@pytest.fixture(name="default_data")
def default_data_fixture(npoints=10000):
    return np.zeros((8, npoints, npoints), dtype=int)


@pytest.fixture(name="default_resolution_latlon")
def default_resolution_latlon_fixture():
    return 0.02


@pytest.fixture(name="default_resolution_equalarea")
def default_resolution_equalarea_fixture():
    return 2000


@pytest.fixture(name="default_npoints")
def default_npoints_fixture():
    return 71


@pytest.fixture(name="default_ensemble_members")
def default_ensemble_members_fixture():
    return 8


def test_default_cube(
    default_resolution_latlon, default_npoints, default_ensemble_members,
):
    cube = generate_metadata("air_pressure")

    assert isinstance(cube, iris.cube.Cube)
    assert cube.standard_name == "air_pressure"
    assert cube.name() == "air_pressure"
    assert cube.units == "Pa"

    assert cube.ndim == 3
    assert cube.shape == (default_ensemble_members, default_npoints, default_npoints)

    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == "latitude"
    assert cube.coords()[2].name() == "longitude"

    assert iris_time_to_datetime(cube.coord("time"))[0] == datetime(2017, 11, 10, 4, 0)
    assert iris_time_to_datetime(cube.coord("forecast_reference_time"))[0] == datetime(
        2017, 11, 10, 4, 0
    )

    assert np.diff(cube.coord(axis="x").points)[0] == pytest.approx(
        default_resolution_latlon
    )


def test_unknown_output_variable():
    name = "humidity"

    with pytest.raises(KeyError):
        generate_metadata(name)


def test_set_npoints(default_ensemble_members,):
    npoints = 500

    cube = generate_metadata("air_pressure", npoints=npoints)

    assert isinstance(cube, iris.cube.Cube)
    assert cube.standard_name == "air_pressure"
    assert cube.name() == "air_pressure"
    assert cube.units == "Pa"

    assert cube.ndim == 3
    assert cube.shape == (default_ensemble_members, npoints, npoints)

    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == "latitude"
    assert cube.coords()[2].name() == "longitude"

    assert iris_time_to_datetime(cube.coord("time"))[0] == datetime(2017, 11, 10, 4, 0)
    assert iris_time_to_datetime(cube.coord("forecast_reference_time"))[0] == datetime(
        2017, 11, 10, 4, 0
    )


def test_set_time(
    default_resolution_latlon, default_npoints, default_ensemble_members,
):
    cube = generate_metadata("air_pressure", time="20200101T0000Z")

    assert isinstance(cube, iris.cube.Cube)
    assert cube.standard_name == "air_pressure"
    assert cube.name() == "air_pressure"
    assert cube.units == "Pa"

    assert cube.ndim == 3
    assert cube.shape == (default_ensemble_members, default_npoints, default_npoints)

    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == "latitude"
    assert cube.coords()[2].name() == "longitude"

    assert iris_time_to_datetime(cube.coord("time"))[0] == datetime(2020, 1, 1, 0, 0)
    assert iris_time_to_datetime(cube.coord("forecast_reference_time"))[0] == datetime(
        2017, 11, 10, 4, 0
    )

    assert np.diff(cube.coord(axis="x").points)[0] == pytest.approx(
        default_resolution_latlon
    )


def test_set_frt(
    default_resolution_latlon, default_npoints, default_ensemble_members,
):
    cube = generate_metadata("air_pressure", frt="20170101T0000Z")

    assert isinstance(cube, iris.cube.Cube)
    assert cube.standard_name == "air_pressure"
    assert cube.name() == "air_pressure"
    assert cube.units == "Pa"

    assert cube.ndim == 3
    assert cube.shape == (default_ensemble_members, default_npoints, default_npoints)

    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == "latitude"
    assert cube.coords()[2].name() == "longitude"

    assert iris_time_to_datetime(cube.coord("time"))[0] == datetime(2017, 11, 10, 4, 0)
    assert iris_time_to_datetime(cube.coord("forecast_reference_time"))[0] == datetime(
        2017, 1, 1, 0, 0
    )

    assert np.diff(cube.coord(axis="x").points)[0] == pytest.approx(
        default_resolution_latlon
    )


def test_set_resolution(
    default_npoints, default_ensemble_members,
):
    resolution = 5
    cube = generate_metadata("air_pressure", resolution=resolution)

    assert isinstance(cube, iris.cube.Cube)
    assert cube.standard_name == "air_pressure"
    assert cube.name() == "air_pressure"
    assert cube.units == "Pa"

    assert cube.ndim == 3
    assert cube.shape == (default_ensemble_members, default_npoints, default_npoints)

    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == "latitude"
    assert cube.coords()[2].name() == "longitude"

    assert np.diff(cube.coord(axis="x").points)[0] == resolution

    assert iris_time_to_datetime(cube.coord("time"))[0] == datetime(2017, 11, 10, 4, 0)


def test_set_ensemble_members(default_npoints,):
    ensemble_members = 4

    cube = generate_metadata("air_pressure", ensemble_members=ensemble_members)

    assert isinstance(cube, iris.cube.Cube)
    assert cube.standard_name == "air_pressure"
    assert cube.name() == "air_pressure"
    assert cube.units == "Pa"

    assert cube.ndim == 3
    assert cube.shape == (ensemble_members, default_npoints, default_npoints)

    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == "latitude"
    assert cube.coords()[2].name() == "longitude"

    assert iris_time_to_datetime(cube.coord("time"))[0] == datetime(2017, 11, 10, 4, 0)


def test_disable_ensemble(default_npoints,):
    cube = generate_metadata("air_pressure", ensemble_members=0)

    assert isinstance(cube, iris.cube.Cube)
    assert cube.standard_name == "air_pressure"
    assert cube.name() == "air_pressure"
    assert cube.units == "Pa"

    assert cube.ndim == 2
    assert cube.shape == (default_npoints, default_npoints)

    assert cube.coords()[0].name() == "latitude"
    assert cube.coords()[1].name() == "longitude"

    assert iris_time_to_datetime(cube.coord("time"))[0] == datetime(2017, 11, 10, 4, 0)


def test_set_spatial_grid(
    default_npoints, default_ensemble_members,
):
    spatial_grid = "equalarea"

    cube = generate_metadata("air_pressure", spatial_grid=spatial_grid)

    assert isinstance(cube, iris.cube.Cube)
    assert cube.standard_name == "air_pressure"
    assert cube.name() == "air_pressure"
    assert cube.units == "Pa"

    assert cube.ndim == 3
    assert cube.shape == (default_ensemble_members, default_npoints, default_npoints)

    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == "projection_y_coordinate"
    assert cube.coords()[2].name() == "projection_x_coordinate"

    assert iris_time_to_datetime(cube.coord("time"))[0] == datetime(2017, 11, 10, 4, 0)


@pytest.mark.xfail
def test_set_altitude_levels(
    default_npoints, default_ensemble_members,
):
    height_levels = 3

    cube = generate_metadata("air_pressure", height_levels=height_levels)

    assert isinstance(cube, iris.cube.Cube)
    assert cube.standard_name == "air_pressure"
    assert cube.name() == "air_pressure"
    assert cube.units == "Pa"

    assert cube.ndim == 4
    assert cube.shape == (
        default_ensemble_members,
        height_levels,
        default_npoints,
        default_npoints,
    )

    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == "height"
    assert cube.coords()[2].name() == "projection_y_coordinate"
    assert cube.coords()[3].name() == "projection_x_coordinate"

    assert iris_time_to_datetime(cube.coord("time"))[0] == datetime(2017, 11, 10, 4, 0)
