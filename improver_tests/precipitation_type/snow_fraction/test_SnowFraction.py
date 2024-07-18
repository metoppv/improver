# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""" Tests of SnowFraction plugin"""

from datetime import datetime

import iris
import numpy as np
import pytest
from cf_units import Unit

from improver.precipitation_type.snow_fraction import SnowFraction
from improver.synthetic_data.set_up_test_cubes import (
    construct_scalar_time_coords,
    set_up_variable_cube,
)

COMMON_ATTRS = {
    "source": "Unit test",
    "institution": "Met Office",
    "title": "Post-Processed IMPROVER unit test",
}
RAIN_DATA = np.array([[0, 0.0], [0.5, 1]], dtype=np.float32)
SNOW_DATA = np.array([[0, 0.5], [1, 0.0]], dtype=np.float32)
EXPECTED_DATA = np.array([[1.0, 1.0], [2.0 / 3.0, 0.0]], dtype=np.float32)


def setup_cubes(rain_data=RAIN_DATA, snow_data=SNOW_DATA, name="{phase}rate"):
    """Make CF-compliant rain and snow cubes from supplied arrays"""
    if "rate" in name:
        units = "m s-1"
        time_bounds = None
    else:
        units = "m"
        time_bounds = (datetime(2017, 11, 10, 3, 0), datetime(2017, 11, 10, 4, 0))
    rain = set_up_variable_cube(
        rain_data,
        name=name.format(phase="rain"),
        units=units,
        time_bounds=time_bounds,
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="uk_ens",
    )
    snow = set_up_variable_cube(
        snow_data,
        name=name.format(phase="lwe_snow"),
        units=units,
        time_bounds=time_bounds,
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="uk_ens",
    )
    return rain, snow


@pytest.mark.parametrize("model_id_attr", (None, "mosg__model_configuration"))
@pytest.mark.parametrize("mask_what", ("none", "rain", "snow", "rain and snow"))
@pytest.mark.parametrize(
    "cube_name", ("{phase}rate", "thickness_of_{phase}fall_amount")
)
def test_basic(cube_name, mask_what, model_id_attr):
    """Run a test with four values, including one with no precip that will trigger
    divide-by-zero.
    Check data and metadata of result. Check with and without masked arrays (because
    divide-by-zero gives a different result, even when input mask is all-False as in
    this test) and with and without a model_id_attr value.
    """
    rain, snow = setup_cubes(name=cube_name)
    if "rain" in mask_what:
        rain.data = np.ma.masked_array(rain.data)
    if "snow" in mask_what:
        snow.data = np.ma.masked_array(snow.data)

    expected_attributes = COMMON_ATTRS.copy()
    if model_id_attr:
        expected_attributes[model_id_attr] = rain.attributes[model_id_attr]
    result = SnowFraction(model_id_attr=model_id_attr)(iris.cube.CubeList([rain, snow]))
    assert isinstance(result, iris.cube.Cube)
    assert not isinstance(result.data, np.ma.masked_array)
    assert str(result.units) == "1"
    assert result.name() == "snow_fraction"
    assert result.attributes == expected_attributes
    np.testing.assert_allclose(result.data, EXPECTED_DATA)


def test_acclen_mismatch_error():
    """Test the process function with mismatched accumulation lengths"""
    rain, snow = setup_cubes(name="thickness_of_{phase}fall_amount")
    time_coords = construct_scalar_time_coords(
        [c.point for c in snow.coord("time").cells()],
        (datetime(2017, 11, 10, 1, 0), datetime(2017, 11, 10, 4, 0)),
        snow.coord("forecast_reference_time").cell(0).point,
    )
    _ = [snow.replace_coord(coord) for coord, _ in time_coords]
    with pytest.raises(
        ValueError, match="Rain and snow cubes do not have the same time coord"
    ):
        SnowFraction()(iris.cube.CubeList([rain, snow]))


def test_dims_mismatch_error():
    """Test the process function with mismatched dimensions"""
    rain, snow = setup_cubes(snow_data=np.array([[0, 0.5]], dtype=np.float32))
    with pytest.raises(
        ValueError, match="Rain and snow cubes are not on the same grid"
    ):
        SnowFraction()(iris.cube.CubeList([rain, snow]))


@pytest.mark.parametrize("mask_what", ("rain", "snow", "rain and snow"))
def test_masked_data_error(mask_what):
    """Test the process function with masked data points"""
    rain, snow = setup_cubes()
    mask = [[False, False], [False, True]]
    if "rain" in mask_what:
        rain.data = np.ma.masked_array(rain.data, mask)
    if "snow" in mask_what:
        snow.data = np.ma.masked_array(snow.data, mask)
    with pytest.raises(ValueError, match=r"Unexpected masked data in input cube\(s\)"):
        SnowFraction()(iris.cube.CubeList([rain, snow]))


def test_missing_cube_error():
    """Test the process function with one cube missing"""
    rain, _ = setup_cubes()
    with pytest.raises(ValueError, match="Expected exactly 2 input cubes, found 1"):
        SnowFraction()(iris.cube.CubeList([rain]))


def test_wrong_input_names_error():
    """Test the process function with incorrect input cubes"""
    rain, snow = setup_cubes()
    rain.rename("kittens")
    snow.rename("puppies")
    with pytest.raises(
        ValueError,
        match=r"Could not find both rain and snow in \['kittens', 'puppies'\]",
    ):
        SnowFraction()(iris.cube.CubeList([rain, snow]))


def test_input_name_matches_both_phases_error():
    """Test the process function with an input cube that has both rain and snow in its name"""
    rain, snow = setup_cubes()
    rain.rename("its_raining_snowy_kittens")
    snow.rename("puppies")
    with pytest.raises(
        ValueError,
        match=(
            "Failed to find unique rain and snow cubes from "
            r"\['its_raining_snowy_kittens', 'puppies'\]"
        ),
    ):
        SnowFraction()(iris.cube.CubeList([rain, snow]))


def test_coercing_units():
    """Test the process function with input cubes of different but compatible units"""
    rain, snow = setup_cubes()
    rain.convert_units("mm h-1")
    result = SnowFraction()(iris.cube.CubeList([rain, snow]))
    assert str(result.units) == "1"
    np.testing.assert_allclose(result.data, EXPECTED_DATA)
    assert rain.units == "mm h-1"
    assert snow.units == "m s-1"


def test_non_coercing_units_error():
    """Test the process function with input cubes of incompatible units"""
    rain, snow = setup_cubes()
    rain.units = Unit("K")
    with pytest.raises(
        ValueError, match=r"Unable to convert from 'Unit\('m s-1'\)' to 'Unit\('K'\)'."
    ):
        SnowFraction()(iris.cube.CubeList([rain, snow]))
