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

from improver.grids import GLOBAL_GRID_CCRS, STANDARD_GRID_CCRS
from improver.synthetic_data.generate_metadata import generate_metadata
from improver.utilities.temporal import datetime_to_iris_time, iris_time_to_datetime

NAME_DEFAULT = "air_temperature"
UNITS_DEFAULT = "K"
SPATIAL_GRID_DEFAULT = "latlon"
NPOINTS_DEFAULT = 71
ENSEMBLE_MEMBERS_DEFAULT = 8
TIME_DEFAULT = datetime(2017, 11, 10, 4, 0)
FRT_DEFAULT = datetime(2017, 11, 10, 4, 0)
FORECAST_PERIOD_DEFAULT = [0]
NDIMS_DEFAULT = 3
ATTRIBUTES_DEFAULT = {}


def _spatial_grid_defaults(spatial_grid):
    """ Returns dictionary of coord y,x names, default resolution, units and coord_system for requested spatial grid.

    Returns:
        Dict:
            Dictionary containing spatial grid attributes
    """
    coord_dict = {
        "latlon": {
            "y": "latitude",
            "x": "longitude",
            "resolution": 0.02,
            "units": "degrees",
            "coord_system": GLOBAL_GRID_CCRS,
        },
        "equalarea": {
            "y": "projection_y_coordinate",
            "x": "projection_x_coordinate",
            "resolution": 2000,
            "units": "metres",
            "coord_system": STANDARD_GRID_CCRS,
        },
    }
    return coord_dict[spatial_grid]


def _check_cubes_are_same(cube_result, cube_expected):
    """ Asserts that cubes are equal """
    assert cube_result == cube_expected


def _check_cube_shape_different(cube):
    """ Asserts that cube shape has been changed from default but name, units and attributes are unchanged """
    default_cube = generate_metadata()
    assert cube.shape != default_cube.shape
    assert iris.util.describe_diff(cube, default_cube) is None


def test_default():
    """ Tests default metadata cube generated """
    cube = generate_metadata()

    assert cube.standard_name == NAME_DEFAULT
    assert cube.name() == NAME_DEFAULT
    assert cube.units == UNITS_DEFAULT

    assert cube.ndim == NDIMS_DEFAULT
    assert cube.shape == (ENSEMBLE_MEMBERS_DEFAULT, NPOINTS_DEFAULT, NPOINTS_DEFAULT)

    spatial_grid_values = _spatial_grid_defaults(SPATIAL_GRID_DEFAULT)
    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == spatial_grid_values["y"]
    assert cube.coords()[2].name() == spatial_grid_values["x"]

    for axis in ("y", "x"):
        assert cube.coord(axis=axis).units == spatial_grid_values["units"]
        assert cube.coord(axis=axis).coord_system == spatial_grid_values["coord_system"]
        assert np.diff(cube.coord(axis=axis).points)[0] == pytest.approx(
            spatial_grid_values["resolution"]
        )

    assert np.count_nonzero(cube.data) == 0

    assert iris_time_to_datetime(cube.coord("time"))[0] == TIME_DEFAULT
    assert (
        iris_time_to_datetime(cube.coord("forecast_reference_time"))[0] == FRT_DEFAULT
    )
    assert cube.coord("forecast_period").points == FORECAST_PERIOD_DEFAULT


def test_set_name_no_units():
    """ Tests cube generated with specified name, automatically setting units, and the rest of the values set as default values """
    name = "air_pressure"
    cube = generate_metadata(name=name)

    assert cube.standard_name == name
    assert cube.name() == name
    assert cube.units == "Pa"

    # Assert that no other values have unexpectedly changed by returning changed values to defaults and comparing against default cube
    default_cube = generate_metadata()
    cube.standard_name = NAME_DEFAULT
    cube.name = NAME_DEFAULT
    cube.units = UNITS_DEFAULT

    _check_cubes_are_same(cube, default_cube)


def test_set_name_units():
    """ Tests cube generated with specified name and units, and the rest of the values set as default values"""
    name = "air_pressure"
    units = "pascal"
    cube = generate_metadata(name=name, units=units)

    assert cube.standard_name == name
    assert cube.name() == name
    assert cube.units == units

    # Assert that no other values have unexpectedly changed by returning changed values to defaults and comparing against default cube
    default_cube = generate_metadata()
    cube.standard_name = NAME_DEFAULT
    cube.name = NAME_DEFAULT
    cube.units = UNITS_DEFAULT

    _check_cubes_are_same(cube, default_cube)


def test_name_unknown_no_units():
    """ Tests error raised if output variable name not in iris.std_names.STD_NAMES and no unit provided """
    name = "temperature"
    msg = "Units of {} are not known.".format(name)

    with pytest.raises(ValueError, match=msg):
        generate_metadata(name)


def test_name_unknown_with_units():
    """ Tests cube generated with specified name which isn't a CF standard name, specified units, and the rest of the values set as default values"""
    name = "lapse_rate"
    units = "K m-1"
    cube = generate_metadata(name=name, units=units)

    # "lapse_rate" not CF standard so standard_name expected None
    assert cube.standard_name is None
    assert cube.name() == name
    assert cube.units == units

    # Assert that no other values have unexpectedly changed by returning changed values to defaults and comparing against default cube
    default_cube = generate_metadata()
    # Iris.cube.Cube.rename() assigns standard_name if valid
    cube.rename(NAME_DEFAULT)
    cube.units = UNITS_DEFAULT

    _check_cubes_are_same(cube, default_cube)


@pytest.mark.parametrize("spatial_grid", ["latlon", "equalarea"])
def test_set_spatial_grid(spatial_grid):
    """ Tests different spatial grids used generates cubes with default values for that spatial grid """
    cube = generate_metadata(spatial_grid=spatial_grid)

    expected_spatial_grid_values = _spatial_grid_defaults(spatial_grid)

    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == expected_spatial_grid_values["y"]
    assert cube.coords()[2].name() == expected_spatial_grid_values["x"]

    for axis in ("y", "x"):
        assert cube.coord(axis=axis).units == expected_spatial_grid_values["units"]
        assert (
            cube.coord(axis=axis).coord_system
            == expected_spatial_grid_values["coord_system"]
        )
        assert np.diff(cube.coord(axis=axis).points)[0] == pytest.approx(
            expected_spatial_grid_values["resolution"]
        )

    # Assert that no other values have unexpectedly changed by returning changed values to defaults and comparing against default cube
    default_cube = generate_metadata()

    if spatial_grid != SPATIAL_GRID_DEFAULT:
        default_spatial_grid_values = _spatial_grid_defaults(SPATIAL_GRID_DEFAULT)
        cube.coords()[1].rename(default_spatial_grid_values["y"])
        cube.coords()[2].rename(default_spatial_grid_values["x"])

        for axis in ("y", "x"):
            cube.coord(axis=axis).points = default_cube.coord(axis=axis).points
            cube.coord(axis=axis).units = default_spatial_grid_values["units"]
            cube.coord(axis=axis).coord_system = default_spatial_grid_values[
                "coord_system"
            ]

        _check_cubes_are_same(cube, default_cube)
    else:
        _check_cubes_are_same(cube, default_cube)


def test_spatial_grid_not_supported():
    """ Tests error raised if spatial grid not supported """
    spatial_grid = "other"
    msg = "Spatial grid {} not supported. Choose either latlon or equalarea.".format(
        spatial_grid
    )

    with pytest.raises(ValueError, match=msg):
        generate_metadata(spatial_grid=spatial_grid)


def test_set_time():
    """ Tests cube generated with specified time and the rest of the values set as default values """
    time = datetime(2020, 1, 1, 0, 0)
    cube = generate_metadata(time=time)

    assert iris_time_to_datetime(cube.coord("time"))[0] == time
    assert cube.coord("forecast_period").points > [0]

    # Assert that no other values have unexpectedly changed by returning changed values to defaults and comparing against default cube
    default_cube = generate_metadata()
    cube.coord("time").points = datetime_to_iris_time(TIME_DEFAULT)
    cube.coord("forecast_period").points = FORECAST_PERIOD_DEFAULT

    _check_cubes_are_same(cube, default_cube)


def test_set_frt():
    """ Tests cube generated with specified forecast reference time and the rest of the values set as default values """
    frt = datetime(2017, 1, 1, 0, 0)
    cube = generate_metadata(frt=frt)

    assert iris_time_to_datetime(cube.coord("forecast_reference_time"))[0] == frt
    assert cube.coord("forecast_period").points > [0]

    # Assert that no other values have unexpectedly changed by returning changed values to defaults and comparing against default cube
    default_cube = generate_metadata()

    cube.coord("forecast_reference_time").points = datetime_to_iris_time(FRT_DEFAULT)
    cube.coord("forecast_period").points = FORECAST_PERIOD_DEFAULT

    _check_cubes_are_same(cube, default_cube)


def test_set_frt_after_time():
    """ Tests error raised when forecast reference time after time """
    frt = datetime(2020, 1, 1, 0, 0)
    msg = "Cannot set up cube with negative forecast period"
    with pytest.raises(ValueError, match=msg):
        generate_metadata(frt=frt)


def test_set_resolution():
    """ Tests cube generated with specified resolution and the rest of the values set as default values """
    resolution = 5
    cube = generate_metadata(resolution=resolution)

    assert np.diff(cube.coord(axis="x").points)[0] == resolution

    # Assert that no other values have unexpectedly changed by returning changed values to defaults and comparing against default cube
    default_cube = generate_metadata()
    for axis in ("y", "x"):
        cube.coord(axis=axis).points = default_cube.coord(axis=axis).points

    _check_cubes_are_same(cube, default_cube)


def test_set_attributes():
    """ Tests cube generated with specified attributes and the rest of the values set as default values """
    attributes = {"source": "IMPROVER"}
    cube = generate_metadata(attributes=attributes)

    assert cube.attributes == attributes

    # Assert that no other values have unexpectedly changed by returning changed values to defaults and comparing against default cube
    default_cube = generate_metadata()
    cube.attributes = ATTRIBUTES_DEFAULT

    _check_cubes_are_same(cube, default_cube)


def test_set_domain_corner():
    """ Tests cube generated with specified domain corner and the rest of the values set as default values """
    domain_corner = (0, 0)
    cube = generate_metadata(domain_corner=domain_corner)

    assert cube.coord(axis="y").points[0] == 0
    assert cube.coord(axis="x").points[0] == 0

    # Assert that no other values have unexpectedly changed by returning changed values to defaults and comparing against default cube
    default_cube = generate_metadata()
    cube.coord(axis="y").points = default_cube.coord(axis="y").points
    cube.coord(axis="x").points = default_cube.coord(axis="x").points
    _check_cubes_are_same(cube, default_cube)


def test_set_npoints():
    """ Tests cube generated with specified npoints """
    npoints = 500
    cube = generate_metadata(npoints=npoints)

    assert cube.shape == (ENSEMBLE_MEMBERS_DEFAULT, npoints, npoints)

    # Assert that cube shape is different from default cube shape but metadata unchanged
    _check_cube_shape_different(cube)


def test_set_ensemble_members():
    """ Tests cube generated with specified number of ensemble members """
    ensemble_members = 4
    cube = generate_metadata(ensemble_members=ensemble_members)

    assert cube.ndim == 3
    assert cube.shape == (ensemble_members, NPOINTS_DEFAULT, NPOINTS_DEFAULT)

    # Assert that cube shape is different from default cube shape but metadata unchanged
    _check_cube_shape_different(cube)


def test_disable_ensemble():
    """ Tests cube generated without realizations dimension """
    ensemble_members = 1
    cube = generate_metadata(ensemble_members=ensemble_members)

    assert cube.ndim == 2
    assert cube.shape == (NPOINTS_DEFAULT, NPOINTS_DEFAULT)

    # Assert that cube shape is different from default cube shape but metadata unchanged
    _check_cube_shape_different(cube)


def test_set_height_levels():
    """ Tests cube generated with specified height levels as an additional dimension """
    height_levels = [1.5, 3.0, 4.5]
    cube = generate_metadata(height_levels=height_levels)

    assert cube.ndim == 4
    assert cube.shape == (
        ENSEMBLE_MEMBERS_DEFAULT,
        len(height_levels),
        NPOINTS_DEFAULT,
        NPOINTS_DEFAULT,
    )

    expected_spatial_grid_values = _spatial_grid_defaults(SPATIAL_GRID_DEFAULT)
    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == "height"
    assert cube.coords()[2].name() == expected_spatial_grid_values["y"]
    assert cube.coords()[3].name() == expected_spatial_grid_values["x"]

    assert np.array_equal(cube.coord("height").points, height_levels)

    # Assert that cube shape is different from default cube shape but metadata unchanged
    _check_cube_shape_different(cube)


def test_set_height_levels_single_value():
    """ Tests cube generated with single height level is demoted from dimension to scalar coordinate """
    height_levels = [1.5]
    cube = generate_metadata(height_levels=height_levels)

    assert cube.ndim == 3
    assert cube.shape == (ENSEMBLE_MEMBERS_DEFAULT, NPOINTS_DEFAULT, NPOINTS_DEFAULT,)

    expected_spatial_grid_values = _spatial_grid_defaults(SPATIAL_GRID_DEFAULT)
    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == expected_spatial_grid_values["y"]
    assert cube.coords()[2].name() == expected_spatial_grid_values["x"]

    assert np.array_equal(cube.coord("height").points, height_levels)

    # Assert that no other values have unexpectedly changed by returning changed values to defaults and comparing against default cube
    default_cube = generate_metadata()
    cube.remove_coord("height")
    _check_cubes_are_same(cube, default_cube)


def test_disable_ensemble_set_height_levels():
    """ Tests cube generated without realizations dimension but with height dimension """
    ensemble_members = 1
    height_levels = [1.5, 3.0, 4.5]
    cube = generate_metadata(
        ensemble_members=ensemble_members, height_levels=height_levels
    )

    assert cube.ndim == 3
    assert cube.shape == (len(height_levels), NPOINTS_DEFAULT, NPOINTS_DEFAULT,)

    expected_spatial_grid_values = _spatial_grid_defaults(SPATIAL_GRID_DEFAULT)
    assert cube.coords()[0].name() == "height"
    assert cube.coords()[1].name() == expected_spatial_grid_values["y"]
    assert cube.coords()[2].name() == expected_spatial_grid_values["x"]

    assert np.array_equal(cube.coord("height").points, height_levels)

    # Assert that cube shape is different from default cube shape but metadata unchanged
    _check_cube_shape_different(cube)
