# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the SnowSplitter plugin"""

from datetime import datetime
from unittest.mock import patch, sentinel

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.precipitation_type.snow_splitter import SnowSplitter
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "unit test data",
    "source": "unit test",
    "institution": "somewhere",
}


class HaltExecution(Exception):
    pass


@patch("improver.precipitation_type.snow_splitter.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    mock_as_cubelist.side_effect = HaltExecution
    try:
        SnowSplitter(output_is_rain=sentinel.output_is_rain)(
            sentinel.cube1, sentinel.cube2, sentinel.cube3
        )
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(
        sentinel.cube1, sentinel.cube2, sentinel.cube3
    )


@pytest.fixture()
def snow_cube() -> Cube:
    """Set up a r, y, x cube of probability of snow at surface"""
    data = np.full((2, 2, 2), fill_value=1, dtype=np.float32)
    snow_phase_prob_cube = set_up_variable_cube(
        data,
        name="probability_of_snow_at_surface",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return snow_phase_prob_cube


@pytest.fixture()
def rain_cube() -> Cube:
    """Set up a r, y, x cube of probability of rain at surface"""
    data = np.full((2, 2, 2), fill_value=0, dtype=np.float32)
    rain_phase_prob_cube = set_up_variable_cube(
        data,
        name="probability_of_rain_at_surface",
        units="1",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return rain_phase_prob_cube


@pytest.fixture()
def precip_rate_cube() -> Cube:
    """Set up a r, y, x cube of precipitation rate"""
    data = np.full((2, 2, 2), fill_value=1, dtype=np.float32)
    precip_cube = set_up_variable_cube(
        data,
        name="lwe_precipitation_rate",
        units="m/s",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return precip_cube


@pytest.fixture()
def precip_acc_cube() -> Cube:
    """Set up a r, y, x cube of precipitation accumulation"""
    data = np.full((2, 2, 2), fill_value=1, dtype=np.float32)
    precip_cube = set_up_variable_cube(
        data,
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        time_bounds=(datetime(2017, 11, 10, 3, 0), datetime(2017, 11, 10, 4, 0)),
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return precip_cube


def run_test(
    cube_name,
    expected,
    output_is_rain,
    precip_cube,
    rain_cube,
    rain_value,
    snow_cube,
    snow_value,
):
    """Used in two tests, this method applies the rain_value and snow_value to the relevant data,
    runs the SnowSplitter plugin and checks the results are as expected, including data, name,
    units and attributes."""
    rain_cube.data = np.full_like(rain_cube.data, rain_value)
    snow_cube.data = np.full_like(snow_cube.data, snow_value)
    result = SnowSplitter(output_is_rain=output_is_rain)(
        CubeList([snow_cube, rain_cube, precip_cube])
    )
    if expected == "dependent":
        expected = rain_value if output_is_rain else snow_value
    assert np.isclose(result.data, expected).all()
    assert result.name() == cube_name
    assert result.units == precip_cube.units
    assert result.attributes == LOCAL_MANDATORY_ATTRIBUTES


@pytest.mark.parametrize(
    "output_is_rain,cube_name", ((True, "rainfall_rate"), (False, "lwe_snowfall_rate"))
)
@pytest.mark.parametrize(
    "rain_value,snow_value,expected",
    ((0, 0, 0.5), (0, 1, "dependent"), (1, 0, "dependent")),
)
def test_rates(
    snow_cube,
    rain_cube,
    precip_rate_cube,
    snow_value,
    rain_value,
    output_is_rain,
    cube_name,
    expected,
):
    """Check that for all possible combinations of rain and snow probabilities the correct
    rain/snow rate is returned. The correct output will sometimes depend on whether the
    output_is_rain is True or False. Also check the name of the returned cube has been
    updated correctly"""
    run_test(
        cube_name,
        expected,
        output_is_rain,
        precip_rate_cube,
        rain_cube,
        rain_value,
        snow_cube,
        snow_value,
    )


@pytest.mark.parametrize(
    "output_is_rain,cube_name",
    (
        (True, "thickness_of_rainfall_amount"),
        (False, "lwe_thickness_of_snowfall_amount"),
    ),
)
@pytest.mark.parametrize(
    "rain_value,snow_value,expected",
    ((0, 0, 0.5), (0, 1, "dependent"), (1, 0, "dependent")),
)
def test_accumulations(
    snow_cube,
    rain_cube,
    precip_acc_cube,
    snow_value,
    rain_value,
    output_is_rain,
    cube_name,
    expected,
):
    """Check that for all possible combinations of rain and snow probabilities the correct
    rain/snow accumulation is returned. The correct output will sometimes depend on whether the
    output_is_rain is True or False. Also check the name of the returned cube has been
    updated correctly"""
    run_test(
        cube_name,
        expected,
        output_is_rain,
        precip_acc_cube,
        rain_cube,
        rain_value,
        snow_cube,
        snow_value,
    )


def test_both_phases_1(snow_cube, rain_cube, precip_rate_cube):
    """Test an error is raised if both snow and rain_cube have a probability of
    1"""

    rain_cube.data = np.full_like(rain_cube.data, 1)
    snow_cube.data = np.full_like(snow_cube.data, 1)
    with pytest.raises(ValueError, match="1 grid square where the probability of snow"):
        SnowSplitter(output_is_rain=False)(
            CubeList([snow_cube, rain_cube, precip_rate_cube])
        )
