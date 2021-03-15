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
"""Unit tests for the MOMetadataInterpreter plugin"""

import pytest
from iris.coords import CellMethod

from improver.developer_tools.metadata_interpreter import MOMetadataInterpreter
from improver_tests.developer_tools import (
    ensemble_fixture,
    percentile_fixture,
    probability_above_fixture,
    probability_below_fixture,
    snow_level_fixture,
    spot_fixture,
    wind_direction_fixture,
    wxcode_fixture,
)


@pytest.fixture(name="interpreter")
def interpreter_fixture():
    return MOMetadataInterpreter()


def test_probabilities_above(probability_above_cube, interpreter):
    """Test interpretation of probability of temperature above threshold
    from UKV"""
    interpreter.run(probability_above_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "probability"
    assert interpreter.diagnostic == "air temperature"
    assert interpreter.relative_to_threshold == "greater than"
    assert not interpreter.methods
    assert interpreter.post_processed == "some"
    assert interpreter.model == "UKV"
    assert not interpreter.blended
    assert interpreter.blendable


def test_probabilities_below(probability_below_cube, interpreter):
    """Test interpretation of blended probability of max temperature in hour
    below threshold"""
    interpreter.run(probability_below_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "probability"
    assert interpreter.diagnostic == "air temperature"
    assert interpreter.relative_to_threshold == "less than"
    assert interpreter.methods == " maximum over time"
    assert interpreter.post_processed == "some"
    assert interpreter.model == "UKV, MOGREPS-UK"
    assert interpreter.blended
    assert interpreter.blendable is None


def test_percentiles(percentile_cube, interpreter):
    """Test interpretation of wind gust percentiles from MOGREPS-UK"""
    interpreter.run(percentile_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "percentile"
    assert interpreter.diagnostic == "wind gust"
    assert interpreter.relative_to_threshold is None
    assert not interpreter.methods
    assert interpreter.post_processed == "no"
    assert interpreter.model == "MOGREPS-UK"
    assert not interpreter.blended
    assert interpreter.blendable


def test_realizations(ensemble_cube, interpreter):
    """Test interpretation of temperature realizations from MOGREPS-UK"""
    interpreter.run(ensemble_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "realization"
    assert interpreter.diagnostic == "air temperature"
    assert interpreter.relative_to_threshold is None
    assert not interpreter.methods
    assert interpreter.post_processed == "no"
    assert interpreter.model == "MOGREPS-UK"
    assert not interpreter.blended
    assert interpreter.blendable


def test_snow_level(snow_level_cube, interpreter):
    """Test interpretation of diagnostic cube with "probability" in the name,
    which is not designed for blending with other models"""
    interpreter.run(snow_level_cube)
    assert interpreter.prod_type == "gridded"
    assert interpreter.field_type == "realization"
    assert (
        interpreter.diagnostic == "probability of snow falling level below ground level"
    )
    assert interpreter.relative_to_threshold is None
    assert not interpreter.methods
    assert interpreter.post_processed == "some"
    assert interpreter.model is None
    assert not interpreter.blended
    assert not interpreter.blendable


def test_spot_median(spot_cube, interpreter):
    """Test interpretation of spot median"""
    interpreter.run(spot_cube)
    assert interpreter.prod_type == "spot"
    assert interpreter.field_type == "percentile"
    assert interpreter.diagnostic == "air temperature"
    assert interpreter.relative_to_threshold is None
    assert not interpreter.methods
    assert interpreter.post_processed == "some"
    assert interpreter.model == "UKV, MOGREPS-UK"
    assert interpreter.blended
    assert interpreter.blendable is None


def test_error_invalid_probability_name(probability_above_cube, interpreter):
    """Test error raised if probability cube name is invalid"""
    probability_above_cube.rename("probability_air_temperature_is_above_threshold")
    with pytest.raises(ValueError, match="is not a valid probability cube name"):
        interpreter.run(probability_above_cube)


def test_error_no_threshold_coordinate(probability_above_cube, interpreter):
    """Test error raised if probability cube has no threshold coordinate"""
    cube = next(probability_above_cube.slices_over("air_temperature"))
    cube.remove_coord("air_temperature")
    with pytest.raises(ValueError, match="no coord with var_name='threshold' found"):
        interpreter.run(cube)


def test_error_invalid_threshold_name(probability_above_cube, interpreter):
    """Test error raised if threshold coordinate name does not match cube name"""
    probability_above_cube.coord("air_temperature").rename("screen_temperature")
    probability_above_cube.coord("screen_temperature").var_name = "threshold"
    with pytest.raises(ValueError, match="expected threshold coord.*incorrect name"):
        interpreter.run(probability_above_cube)


def test_error_no_threshold_var_name(probability_above_cube, interpreter):
    """Test error raised if threshold coordinate does not have var_name='threshold'"""
    probability_above_cube.coord("air_temperature").var_name = None
    with pytest.raises(ValueError, match="does not have var_name='threshold'"):
        interpreter.run(probability_above_cube)


def test_error_inconsistent_relative_to_threshold(probability_above_cube, interpreter):
    """Test error raised if the spp__relative_to_threshold attribute is inconsistent
    with the cube name"""
    probability_above_cube.coord("air_temperature").attributes[
        "spp__relative_to_threshold"
    ] = "less_than"
    with pytest.raises(
        ValueError, match="name.*above.*is not consistent with.*less_than"
    ):
        interpreter.run(probability_above_cube)


# TODO test error cases: permute compliant cubes to cover the following:
"""
Probabilities:
- Invalid probability name - DONE
- No threshold coordinate - DONE
- Invalid threshold coordinate name - DONE
- Invalid threshold coordinate var name - DONE
- Incorrect relative_to_threshold information - DONE
- Multiple errors

Percentiles:
- Incorrect percentile coordinate name TODO implement?
- Incorrect percentile coordinate units

Attributes:
- Missing required attributes (mandatory - TODO or diagnostic-specific - DONE)
- Has forbidden attributes
- Has unexpected attributes
- Inconsistent model ID and title attributes

Cell methods
- Has acceptable cell method
- Has forbidden cell method
- Has a mixture / more than one cell method

Missing / incorrect coordinates (blended, non-blended?, spot)

Time coordinate units
"""


def test_weather_code_success(wxcode_cube, interpreter):
    """Test interpretation of weather code field"""
    interpreter.run(wxcode_cube)
    assert interpreter.diagnostic == "weather code"
    assert interpreter.model == "UKV, MOGREPS-UK"
    assert interpreter.blended


def test_error_weather_code_unexpected_cell_methods(wxcode_cube, interpreter):
    """Test error if exception cubes have a cell method that would usually be
    permitted"""
    wxcode_cube.add_cell_method(CellMethod(method="maximum", coords="time"))
    with pytest.raises(ValueError, match="Unexpected cell methods"):
        interpreter.run(wxcode_cube)


def test_error_weather_code_missing_attribute(wxcode_cube, interpreter):
    """Test error when weather code required attributes are missing"""
    wxcode_cube.attributes.pop("weather_code")
    interpreter = MOMetadataInterpreter()
    with pytest.raises(ValueError, match="missing .* required values"):
        interpreter.run(wxcode_cube)


def test_error_wind_gust_missing_attribute(percentile_cube, interpreter):
    """Test error when a wind gust percentile cube is missing a required attribute"""
    percentile_cube.attributes.pop("wind_gust_diagnostic")
    with pytest.raises(ValueError, match="missing .* required values"):
        interpreter.run(percentile_cube)


def test_wind_direction_success(wind_direction_cube, interpreter):
    """Test interpretation of wind direction field with mean over realizations
    cell method"""
    interpreter.run(wind_direction_cube)
    assert interpreter.diagnostic == "wind from direction"
    assert interpreter.model == "MOGREPS-UK"
    assert not interpreter.blended


def test_error_wind_direction_unexpected_cell_methods(wind_direction_cube, interpreter):
    """Test error if exception cubes have a cell method that would usually be
    permitted"""
    wind_direction_cube.add_cell_method(CellMethod(method="maximum", coords="time"))
    with pytest.raises(ValueError, match="Unexpected cell methods"):
        interpreter.run(wind_direction_cube)
