# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests of FreezingRain plugin"""

import itertools
from unittest.mock import patch, sentinel

import iris
import pytest
from iris.cube import Cube
from numpy.testing import assert_almost_equal

from improver.precipitation.freezing_rain import FreezingRain

RATE_NAME = "lwe_freezing_rainrate"
ACCUM_NAME = "thickness_of_lwe_freezing_rainfall_amount"
PROB_NAME = "probability_of_{}_above_threshold"
TIME_WINDOW_TYPE = ["instantaneous", "period"]


class HaltExecution(Exception):
    pass


@patch("improver.precipitation.freezing_rain.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    mock_as_cubelist.side_effect = HaltExecution
    try:
        FreezingRain()(sentinel.cube1, sentinel.cube2, sentinel.cube3)
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(
        sentinel.cube1, sentinel.cube2, sentinel.cube3
    )


def modify_coordinate(cube, coord):
    """Modify a coordinate to enable testing for mismatches."""
    points = cube.coord(coord).points.copy()
    points += 1
    cube.coord(coord).points = points


def test_expected_result(input_cubes, expected_probabilities, expected_attributes):
    """Test that the returned probabilities are correct. This test also loops
    through all permutations of input cube order to ensure the result is
    consistent. The input_cubes fixture provides a set of either instantaneous
    or period diagnostics, such that both are tested here."""

    input_cubes = list(itertools.permutations(input_cubes))
    for cubes in input_cubes:
        result = FreezingRain(model_id_attr="mosg__model_configuration")(
            iris.cube.CubeList(cubes)
        )

        assert_almost_equal(result.data, expected_probabilities)
        assert isinstance(result, Cube)
        assert result.attributes == expected_attributes
        assert result.coord("time") == cubes[-1].coord("time")
        if result.coord("time").has_bounds():
            assert result.name() == PROB_NAME.format(ACCUM_NAME)
            assert result.coord(var_name="threshold").name() == ACCUM_NAME
            assert result.coord(var_name="threshold").units == "mm"
            assert len(result.cell_methods) == 1
            assert result.cell_methods[0].method == "sum"
            assert "time" in result.cell_methods[0].coord_names
        else:
            assert result.name() == PROB_NAME.format(RATE_NAME)
            assert result.coord(var_name="threshold").name() == RATE_NAME
            assert result.coord(var_name="threshold").units == "mm hr-1"
            assert len(result.cell_methods) == 0


@pytest.mark.parametrize("period", ["instantaneous"])
def test_model_config_attribute(
    precipitation_only, temperature_only, expected_attributes
):
    """Test that the returned model configuration attribute is correct when the
    inputs are derived from a mixture of models. The uk_ens attribute of the
    temperature input should be ignored."""

    temperature_only.attributes["mosg__model_configuration"] = "uk_ens"
    cubes = iris.cube.CubeList([*precipitation_only, temperature_only])
    result = FreezingRain(model_id_attr="mosg__model_configuration")(
        iris.cube.CubeList(cubes)
    )
    assert result.attributes == expected_attributes


@pytest.mark.parametrize("period", ["instantaneous"])
@pytest.mark.parametrize("n_cubes", [1, 2, 4])
def test_wrong_number_of_cubes(temperature_only, n_cubes):
    """Test that an exception is raised if fewer or more than 3 cubes are
    provided as input."""
    cubes = iris.cube.CubeList([*[temperature_only] * n_cubes])
    with pytest.raises(
        ValueError, match=f"Expected exactly 3 input cubes, found {n_cubes}"
    ):
        FreezingRain()(cubes)


@pytest.mark.parametrize("period", ["instantaneous"])
def test_duplicate_cubes(temperature_only):
    """Test that an exception is raised if duplicate cubes are passed in."""
    cubes = iris.cube.CubeList([*[temperature_only] * 3])
    with pytest.raises(ValueError, match="Duplicate input cube provided."):
        FreezingRain()(cubes)


def test_wrong_cubes(input_cubes):
    """Test that an exception is raised if the cubes provided do not relate to
    the expected diagnostics."""
    input_cubes[0].rename("kittens_playing_snooker")
    with pytest.raises(ValueError, match="Could not find unique"):
        FreezingRain()(input_cubes)


def test_inseperable_cubes(input_cubes):
    """Test that an exception is raised if the cubes cannot be differentiated
    using expected name components."""
    input_cubes[0].rename("kittens_frolicking_in_rain_and_sleet")
    input_cubes[1].rename("kittens_watching_the_rain_and_sleet")
    with pytest.raises(ValueError, match="Could not find unique"):
        FreezingRain()(input_cubes)


def test_mismatched_grids(input_cubes):
    """Test that an exception is raised if the input cubes do not all share the
    same grid."""
    modify_coordinate(input_cubes[0], input_cubes[0].coord(axis="x").name())
    with pytest.raises(ValueError, match="Input cubes are not on the same grid"):
        FreezingRain()(input_cubes)


def test_mismatched_time_coord(input_cubes):
    """Test that an exception is raised if the input cubes do not all share the
    same time coordinate."""

    if input_cubes[0].coord("time").has_bounds():
        input_cubes[0].coord("time").bounds = None
    else:
        modify_coordinate(input_cubes[0], "time")

    with pytest.raises(ValueError, match="Input cubes do not have the same time coord"):
        FreezingRain()(input_cubes)


def test_compatible_precipitation_units(input_cubes, expected_probabilities):
    """Test that differing but compatible precipitation units are handled,
    giving the expected probabilities."""
    if input_cubes[0].coord("time").has_bounds():
        input_cubes[0].coord(var_name="threshold").convert_units("cm")
    else:
        input_cubes[0].coord(var_name="threshold").convert_units("cm s-1")

    result = FreezingRain()(input_cubes)
    assert_almost_equal(result.data, expected_probabilities)
    assert (
        input_cubes[0].coord(var_name="threshold").units
        == result.coord(var_name="threshold").units
    )


def test_incompatible_precipitation_units(input_cubes):
    """Test that an exception is raised if the precipitation diagnostics do not
    have compatible units."""
    input_cubes[0].coord(var_name="threshold").units = "K"

    with pytest.raises(
        ValueError, match="Rain and sleet cubes have incompatible units"
    ):
        FreezingRain()(input_cubes)


def test_differing_precipitation_thresholds(input_cubes):
    """Test that an exception is raised if the precipitation diagnostics do not
    have identical thresholds."""
    modify_coordinate(input_cubes[0], input_cubes[0].coord(var_name="threshold").name())

    with pytest.raises(
        ValueError, match="Rain and sleet cubes have different threshold values"
    ):
        FreezingRain()(input_cubes)


@pytest.mark.parametrize("period", TIME_WINDOW_TYPE)
def test_temperatures_below_thresholds(
    precipitation_only, temperature_below, expected_probabilities
):
    """Test that temperature probabilities that are calculated below thresholds
    give the expected result. In all other tests the probabilities are being
    inverted within the plugin."""
    cubes = iris.cube.CubeList([*precipitation_only, temperature_below])

    result = FreezingRain()(cubes)
    assert_almost_equal(result.data, expected_probabilities)


def test_no_freezing_threshold(input_cubes):
    """Test that an exception is raised if the temperature diagnostic does not
    include a threshold at the freezing point of water."""
    modify_coordinate(
        input_cubes[-1], input_cubes[-1].coord(var_name="threshold").name()
    )

    with pytest.raises(ValueError, match="No 0 Celsius or equivalent threshold"):
        FreezingRain()(input_cubes)


@pytest.mark.parametrize("period", TIME_WINDOW_TYPE)
def test_realization_matching(
    precipitation_multi_realization,
    temperature_multi_realization,
    expected_probabilities_multi_realization,
):
    """Test that only common realizations are used when combining precipitation
    and temperature cubes. In this case the temperature cube has an additional
    realization that is not used, with only 2 realizations present in the
    resulting freezing rain cube."""
    cubes = iris.cube.CubeList(
        [*precipitation_multi_realization, temperature_multi_realization]
    )
    plugin = FreezingRain()
    plugin._get_input_cubes(cubes)
    plugin._extract_common_realizations()
    result = FreezingRain()(cubes)

    assert all(plugin.rain.coord("realization").points == [0, 1])
    assert all(plugin.sleet.coord("realization").points == [0, 1])
    assert all(plugin.temperature.coord("realization").points == [0, 1])

    assert_almost_equal(result.data, expected_probabilities_multi_realization)


@pytest.mark.parametrize("period", TIME_WINDOW_TYPE)
def test_no_realization_matching(
    precipitation_multi_realization, temperature_multi_realization
):
    """Test that an error is raised if the inputs have no common realizations."""
    cubes = iris.cube.CubeList(
        [*precipitation_multi_realization, temperature_multi_realization]
    )
    temperature_multi_realization.coord("realization").points = [10, 11, 12]
    plugin = FreezingRain()
    plugin._get_input_cubes(cubes)

    with pytest.raises(ValueError, match="Input cubes share no common realizations."):
        plugin._extract_common_realizations()
