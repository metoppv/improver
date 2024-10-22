# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for metadata cube generation."""

from datetime import datetime

import iris
import numpy as np
import pytest

from improver.grids import GLOBAL_GRID_CCRS, STANDARD_GRID_CCRS
from improver.metadata.constants.attributes import (
    MANDATORY_ATTRIBUTE_DEFAULTS,
    MANDATORY_ATTRIBUTES,
)
from improver.metadata.probabilistic import probability_is_above_or_below
from improver.synthetic_data.generate_metadata import generate_metadata
from improver.utilities.temporal import datetime_to_iris_time, iris_time_to_datetime

NAME_DEFAULT = "air_pressure_at_sea_level"
UNITS_DEFAULT = "Pa"
SPATIAL_GRID_DEFAULT = "latlon"
NPOINTS_DEFAULT = 71
ENSEMBLE_MEMBERS_DEFAULT = 8
TIME_DEFAULT = datetime(2017, 11, 10, 4, 0)
FRT_DEFAULT = datetime(2017, 11, 10, 0, 0)
FORECAST_PERIOD_DEFAULT = 14400
NDIMS_DEFAULT = 3
RELATIVE_TO_THRESHOLD_DEFAULT = "greater_than"
SPATIAL_GRID_ATTRIBUTE_DEFAULTS = {
    "latlon": {
        **{
            "y": "latitude",
            "x": "longitude",
            "grid_spacing": 0.02,
            "units": "degrees",
            "coord_system": GLOBAL_GRID_CCRS,
        }
    },
    "equalarea": {
        **{
            "y": "projection_y_coordinate",
            "x": "projection_x_coordinate",
            "grid_spacing": 2000,
            "units": "metres",
            "coord_system": STANDARD_GRID_CCRS,
        }
    },
}


def _check_cube_shape_different(cube):
    """Asserts that cube shape has been changed from default but name, units and
    attributes are unchanged"""
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)
    assert cube.shape != default_cube.shape
    assert iris.util.describe_diff(cube, default_cube) is None


def test_default():
    """Tests default metadata cube generated"""
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)

    assert cube.name() == NAME_DEFAULT
    assert cube.standard_name == NAME_DEFAULT
    assert cube.units == UNITS_DEFAULT

    assert cube.ndim == NDIMS_DEFAULT
    assert cube.shape == (ENSEMBLE_MEMBERS_DEFAULT, NPOINTS_DEFAULT, NPOINTS_DEFAULT)

    spatial_grid_values = SPATIAL_GRID_ATTRIBUTE_DEFAULTS[SPATIAL_GRID_DEFAULT]
    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == spatial_grid_values["y"]
    assert cube.coords()[2].name() == spatial_grid_values["x"]

    for axis in ("y", "x"):
        assert cube.coord(axis=axis).units == spatial_grid_values["units"]
        assert cube.coord(axis=axis).coord_system == spatial_grid_values["coord_system"]
        assert np.diff(cube.coord(axis=axis).points)[0] == pytest.approx(
            spatial_grid_values["grid_spacing"]
        )

    assert np.count_nonzero(cube.data) == 0

    assert iris_time_to_datetime(cube.coord("time"))[0] == TIME_DEFAULT
    assert cube.coord("time").bounds is None
    assert (
        iris_time_to_datetime(cube.coord("forecast_reference_time"))[0] == FRT_DEFAULT
    )
    assert cube.coord("forecast_period").points == FORECAST_PERIOD_DEFAULT

    assert cube.attributes == MANDATORY_ATTRIBUTE_DEFAULTS


def test_set_name_no_units():
    """Tests cube generated with specified name, automatically setting units, and the
    rest of the values set as default values"""
    name = "air_pressure"
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, name=name)

    assert cube.name() == name
    assert cube.standard_name == name
    assert cube.units == "Pa"

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)
    cube.standard_name = default_cube.standard_name
    cube.units = default_cube.units

    assert cube == default_cube


def test_set_name_units():
    """Tests cube generated with specified name and units, and the rest of the values
    set as default values"""
    name = "air_pressure"
    units = "pascal"
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, name=name, units=units)

    assert cube.name() == name
    assert cube.standard_name == name
    assert cube.units == units

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)
    cube.standard_name = default_cube.standard_name
    cube.units = default_cube.units

    assert cube == default_cube


def test_name_unknown_no_units():
    """Tests error raised if output variable name not in iris.std_names.STD_NAMES and
    no unit provided"""
    name = "temperature"

    with pytest.raises(ValueError, match=name):
        generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, name)


def test_name_unknown_with_units():
    """Tests cube generated with specified name which isn't a CF standard name,
    specified units, and the rest of the values set as default values"""
    name = "lapse_rate"
    units = "K m-1"
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, name=name, units=units)

    # "lapse_rate" not CF standard so standard_name expected None
    assert cube.name() == name
    assert cube.standard_name is None
    assert cube.units == units

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)
    # Iris.cube.Cube.rename() assigns standard_name if valid
    cube.rename(default_cube.standard_name)
    cube.units = default_cube.units

    assert cube == default_cube


@pytest.mark.parametrize("spatial_grid", ["latlon", "equalarea"])
def test_set_spatial_grid(spatial_grid):
    """Tests different spatial grids generates cubes with default values for that
    spatial grid"""
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, spatial_grid=spatial_grid)

    expected_spatial_grid_attributes = SPATIAL_GRID_ATTRIBUTE_DEFAULTS[spatial_grid]

    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == expected_spatial_grid_attributes["y"]
    assert cube.coords()[2].name() == expected_spatial_grid_attributes["x"]

    for axis in ("y", "x"):
        assert cube.coord(axis=axis).units == expected_spatial_grid_attributes["units"]
        assert (
            cube.coord(axis=axis).coord_system
            == expected_spatial_grid_attributes["coord_system"]
        )
        assert np.diff(cube.coord(axis=axis).points)[0] == pytest.approx(
            expected_spatial_grid_attributes["grid_spacing"]
        )

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)

    if spatial_grid != SPATIAL_GRID_DEFAULT:
        default_spatial_grid_values = SPATIAL_GRID_ATTRIBUTE_DEFAULTS[
            SPATIAL_GRID_DEFAULT
        ]
        cube.coords()[1].rename(default_spatial_grid_values["y"])
        cube.coords()[2].rename(default_spatial_grid_values["x"])

        for axis in ("y", "x"):
            cube.coord(axis=axis).points = default_cube.coord(axis=axis).points
            cube.coord(axis=axis).bounds = default_cube.coord(axis=axis).bounds
            cube.coord(axis=axis).units = default_spatial_grid_values["units"]
            cube.coord(axis=axis).coord_system = default_spatial_grid_values[
                "coord_system"
            ]

        assert cube == default_cube
    else:
        assert cube == default_cube


def test_spatial_grid_not_supported():
    """Tests error raised if spatial grid not supported"""
    spatial_grid = "other"

    with pytest.raises(ValueError, match=spatial_grid):
        generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, spatial_grid=spatial_grid)


def test_set_time():
    """Tests cube generated with specified time and the rest of the values set as
    default values"""
    time = datetime(2020, 1, 1, 0, 0)
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, time=time)

    assert iris_time_to_datetime(cube.coord("time"))[0] == time
    assert cube.coord("forecast_period").points > FORECAST_PERIOD_DEFAULT

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)
    cube.coord("time").points = default_cube.coord("time").points
    cube.coord("forecast_period").points = default_cube.coord("forecast_period").points

    assert cube == default_cube


def test_set_time_period():
    """Tests cube generated with time bounds calculated using specified time_period
    and the rest of the values set as default values"""
    time_period = 150
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, time_period=time_period)

    assert iris_time_to_datetime(cube.coord("time"))[0] == TIME_DEFAULT
    assert cube.coord("forecast_period").points == FORECAST_PERIOD_DEFAULT
    assert cube.coord("time").bounds[0][0] == datetime_to_iris_time(
        datetime(2017, 11, 10, 1, 30)
    )
    assert cube.coord("time").bounds[0][1] == datetime_to_iris_time(TIME_DEFAULT)

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)
    cube.coord("time").bounds = None
    cube.coord("forecast_period").bounds = None

    assert cube == default_cube


def test_set_frt():
    """Tests cube generated with specified forecast reference time and the rest of the
    values set as default values"""
    frt = datetime(2017, 1, 1, 0, 0)
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, frt=frt)

    assert iris_time_to_datetime(cube.coord("forecast_reference_time"))[0] == frt
    assert cube.coord("forecast_period").points > 0

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)

    cube.coord("forecast_reference_time").points = default_cube.coord(
        "forecast_reference_time"
    ).points
    cube.coord("forecast_period").points = default_cube.coord("forecast_period").points

    assert cube == default_cube


def test_set_ensemble_members():
    """Tests cube generated with specified number of ensemble members"""
    ensemble_members = 4
    cube = generate_metadata(
        MANDATORY_ATTRIBUTE_DEFAULTS, ensemble_members=ensemble_members
    )

    assert cube.ndim == 3
    assert cube.shape == (ensemble_members, NPOINTS_DEFAULT, NPOINTS_DEFAULT)
    assert cube.coords()[0].name() == "realization"

    # Assert that cube shape is different from default cube shape but metadata unchanged
    _check_cube_shape_different(cube)


@pytest.mark.parametrize("ensemble_members", (0, 1))
def test_disable_ensemble(ensemble_members):
    """Tests cube generated without realizations dimension"""
    cube = generate_metadata(
        MANDATORY_ATTRIBUTE_DEFAULTS, ensemble_members=ensemble_members
    )

    assert cube.ndim == 2
    assert cube.shape == (NPOINTS_DEFAULT, NPOINTS_DEFAULT)

    spatial_grid_values = SPATIAL_GRID_ATTRIBUTE_DEFAULTS[SPATIAL_GRID_DEFAULT]
    assert cube.coords()[0].name() == spatial_grid_values["y"]
    assert cube.coords()[1].name() == spatial_grid_values["x"]

    # Assert that cube shape is different from default cube shape but metadata unchanged
    _check_cube_shape_different(cube)


@pytest.mark.parametrize(
    "cube_type", ("variable", "percentile", "probability", "other")
)
@pytest.mark.parametrize(
    "spp__relative_to_threshold", ("greater_than", "less_than", None)
)
def test_leading_dimension(cube_type, spp__relative_to_threshold):
    """Tests cube generated with leading dimension specified using percentile and
    probability flags, and different values for spp__relative_to_threshold"""
    if cube_type == "other":
        # Tests that error is raised when cube type isn't supported
        msg = (
            "Cube type {} not supported. "
            'Specify one of "variable", "percentile" or "probability".'
        ).format(cube_type)

        with pytest.raises(ValueError, match=msg):
            generate_metadata(
                MANDATORY_ATTRIBUTE_DEFAULTS,
                cube_type=cube_type,
                spp__relative_to_threshold=spp__relative_to_threshold,
            )
    else:
        leading_dimension = [10, 20, 30, 40, 50, 60, 70, 80]

        if spp__relative_to_threshold is not None:
            cube = generate_metadata(
                MANDATORY_ATTRIBUTE_DEFAULTS,
                leading_dimension=leading_dimension,
                cube_type=cube_type,
                spp__relative_to_threshold=spp__relative_to_threshold,
            )
        else:
            cube = generate_metadata(
                MANDATORY_ATTRIBUTE_DEFAULTS,
                leading_dimension=leading_dimension,
                cube_type=cube_type,
            )
            spp__relative_to_threshold = RELATIVE_TO_THRESHOLD_DEFAULT

        if cube_type == "percentile":
            cube_name = NAME_DEFAULT
            coord_name = "percentile"
        elif cube_type == "probability":
            cube_name = "probability_of_{}_{}_threshold".format(
                NAME_DEFAULT, probability_is_above_or_below(cube)
            )
            coord_name = NAME_DEFAULT

            assert cube.coord(coord_name).attributes == {
                "spp__relative_to_threshold": spp__relative_to_threshold
            }
            cube.coord(coord_name).attributes = {}
        else:
            cube_name = NAME_DEFAULT
            coord_name = "realization"

        assert cube.name() == cube_name
        assert cube.ndim == 3
        assert cube.shape == (len(leading_dimension), NPOINTS_DEFAULT, NPOINTS_DEFAULT)
        assert cube.coords()[0].name() == coord_name
        np.testing.assert_array_equal(cube.coord(coord_name).points, leading_dimension)

        # Assert that no other values have unexpectedly changed by returning changed
        # values to defaults and comparing against default cube
        default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)

        cube.coord(coord_name).points = default_cube.coord("realization").points
        cube.coord(coord_name).rename("realization")
        cube.coord("realization").units = "1"
        cube.rename(default_cube.standard_name)
        cube.units = default_cube.units

        assert cube == default_cube


def test_set_attributes():
    """Tests cube generated with specified attributes and the rest of the values set
    as default values"""
    attributes = {"test_attribute": "kittens"}
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, attributes=attributes)
    expected_attributes = MANDATORY_ATTRIBUTE_DEFAULTS.copy()
    expected_attributes["test_attribute"] = "kittens"
    assert cube.attributes == expected_attributes

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)
    cube.attributes = default_cube.attributes

    assert cube == default_cube


@pytest.mark.parametrize(
    "use_attributes",
    (MANDATORY_ATTRIBUTES[-1:], MANDATORY_ATTRIBUTES[:2], MANDATORY_ATTRIBUTES[:1]),
)
def test_missing_mandatory_attributes(use_attributes):
    """Tests that unspecified mandatory attributes raise an error"""
    attributes = {}
    for key in use_attributes:
        attributes[key] = MANDATORY_ATTRIBUTE_DEFAULTS[key]
    msg = "No values for these mandatory attributes: "
    with pytest.raises(KeyError, match=msg):
        generate_metadata(attributes)


def test_set_grid_spacing():
    """Tests cube generated with specified grid_spacing and the rest of the values set
    as default values"""
    grid_spacing = 5
    cube = generate_metadata(
        MANDATORY_ATTRIBUTE_DEFAULTS,
        x_grid_spacing=grid_spacing,
        y_grid_spacing=grid_spacing,
    )

    assert np.diff(cube.coord(axis="y").points)[0] == grid_spacing
    assert np.diff(cube.coord(axis="x").points)[0] == grid_spacing

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)
    for axis in ("y", "x"):
        cube.coord(axis=axis).points = default_cube.coord(axis=axis).points
        cube.coord(axis=axis).bounds = default_cube.coord(axis=axis).bounds

    assert cube == default_cube


def test_set_domain_corner():
    """Tests cube generated with specified domain corner and the rest of the values
    set as default values"""
    domain_corner = (0, 0)
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, domain_corner=domain_corner)

    assert cube.coord(axis="y").points[0] == domain_corner[0]
    assert cube.coord(axis="x").points[0] == domain_corner[1]

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)
    cube.coord(axis="y").points = default_cube.coord(axis="y").points
    cube.coord(axis="x").points = default_cube.coord(axis="x").points
    cube.coord(axis="y").bounds = default_cube.coord(axis="y").bounds
    cube.coord(axis="x").bounds = default_cube.coord(axis="x").bounds
    assert cube == default_cube


@pytest.mark.parametrize("domain_corner", ("0", "0,0,0"))
def test_domain_corner_incorrect_length(domain_corner):
    """Tests error raised if domain corner not length 2"""
    msg = "Domain corner"
    with pytest.raises(ValueError, match=msg):
        generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, domain_corner=domain_corner)


def test_set_npoints():
    """Tests cube generated with specified npoints"""
    npoints = 500
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, npoints=npoints)

    assert cube.shape == (ENSEMBLE_MEMBERS_DEFAULT, npoints, npoints)

    # Assert that cube shape is different from default cube shape but metadata unchanged
    _check_cube_shape_different(cube)


def test_set_height_levels():
    """Tests cube generated with specified height levels as an additional dimension"""
    height_levels = [1.5, 3.0, 4.5]
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, height_levels=height_levels)

    assert cube.ndim == 4
    assert cube.shape == (
        ENSEMBLE_MEMBERS_DEFAULT,
        len(height_levels),
        NPOINTS_DEFAULT,
        NPOINTS_DEFAULT,
    )

    expected_spatial_grid_attributes = SPATIAL_GRID_ATTRIBUTE_DEFAULTS[
        SPATIAL_GRID_DEFAULT
    ]
    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == "height"
    assert cube.coords()[2].name() == expected_spatial_grid_attributes["y"]
    assert cube.coords()[3].name() == expected_spatial_grid_attributes["x"]

    np.testing.assert_array_equal(cube.coord("height").points, height_levels)

    # Assert that cube shape is different from default cube shape but metadata unchanged
    _check_cube_shape_different(cube)


def test_set_height_levels_single_value():
    """Tests cube generated with single height level is demoted from dimension to
    scalar coordinate"""
    height_levels = [1.5]
    cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS, height_levels=height_levels)

    assert cube.ndim == 3
    assert cube.shape == (ENSEMBLE_MEMBERS_DEFAULT, NPOINTS_DEFAULT, NPOINTS_DEFAULT)

    expected_spatial_grid_attributes = SPATIAL_GRID_ATTRIBUTE_DEFAULTS[
        SPATIAL_GRID_DEFAULT
    ]
    assert cube.coords()[0].name() == "realization"
    assert cube.coords()[1].name() == expected_spatial_grid_attributes["y"]
    assert cube.coords()[2].name() == expected_spatial_grid_attributes["x"]

    np.testing.assert_array_equal(cube.coord("height").points, height_levels)

    # Assert that no other values have unexpectedly changed by returning changed values
    # to defaults and comparing against default cube
    default_cube = generate_metadata(MANDATORY_ATTRIBUTE_DEFAULTS)
    cube.remove_coord("height")
    assert cube == default_cube


def test_disable_ensemble_set_height_levels():
    """Tests cube generated without realizations dimension but with height dimension"""
    ensemble_members = 1
    height_levels = [1.5, 3.0, 4.5]
    cube = generate_metadata(
        MANDATORY_ATTRIBUTE_DEFAULTS,
        ensemble_members=ensemble_members,
        height_levels=height_levels,
    )

    assert cube.ndim == 3
    assert cube.shape == (len(height_levels), NPOINTS_DEFAULT, NPOINTS_DEFAULT)

    expected_spatial_grid_attributes = SPATIAL_GRID_ATTRIBUTE_DEFAULTS[
        SPATIAL_GRID_DEFAULT
    ]
    assert cube.coords()[0].name() == "height"
    assert cube.coords()[1].name() == expected_spatial_grid_attributes["y"]
    assert cube.coords()[2].name() == expected_spatial_grid_attributes["x"]

    np.testing.assert_array_equal(cube.coord("height").points, height_levels)

    # Assert that cube shape is different from default cube shape but metadata unchanged
    _check_cube_shape_different(cube)
