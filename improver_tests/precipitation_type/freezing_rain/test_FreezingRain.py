# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
""" Tests of FreezingRain plugin"""

import itertools

import iris
import pytest
from iris.cube import Cube
from numpy.testing import assert_almost_equal

from improver.precipitation_type.freezing_rain import FreezingRain

RATE_NAME = "lwe_freezing_rainrate"
ACCUM_NAME = "thickness_of_lwe_freezing_rainfall_amount"
PROB_NAME = "probability_of_{}_above_threshold"
TIME_WINDOW_TYPE = ["instantaneous", "period"]


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
        result = FreezingRain()(iris.cube.CubeList(cubes))

        assert_almost_equal(result.data, expected_probabilities)
        assert isinstance(result, Cube)
        assert result.attributes == expected_attributes
        assert result.coord("time") == cubes[-1].coord("time")
        if result.coord("time").has_bounds():
            assert result.name() == PROB_NAME.format(ACCUM_NAME)
            assert result.coord(var_name="threshold").name() == ACCUM_NAME
            assert result.coord(var_name="threshold").units == "mm"
        else:
            assert result.name() == PROB_NAME.format(RATE_NAME)
            assert result.coord(var_name="threshold").name() == RATE_NAME
            assert result.coord(var_name="threshold").units == "mm hr-1"


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
    expected_probabilities,
):
    """Test that only common realizations are used when combining precipitation
    and temperature cubes. In this case the temperature cube has an additional
    realization that is not used, with only 2 realizations present in the
    resulting freezing rain cube."""
    cubes = iris.cube.CubeList(
        [*precipitation_multi_realization, temperature_multi_realization]
    )
    result = FreezingRain()(cubes)

    assert all(result.coord("realization").points == [1])
    assert_almost_equal(result.data, expected_probabilities)


@pytest.mark.parametrize("period", TIME_WINDOW_TYPE)
def test_no_realization_matching(
    precipitation_multi_realization, temperature_multi_realization,
):
    """Test that an error is raised if the inputs have no common realizations."""
    cubes = iris.cube.CubeList(
        [*precipitation_multi_realization, temperature_multi_realization]
    )
    temperature_multi_realization.coord("realization").points = [10, 11, 12]
    with pytest.raises(ValueError, match="Input cubes share no common realizations."):
        FreezingRain()(cubes)
