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

from improver.developer_tools.metadata_interpreter import MOMetadataInterpreter
from improver_tests.developer_tools import (
    ensemble_fixture,
    percentile_fixture,
    probability_above_fixture,
    probability_below_fixture,
    spot_fixture,
    wind_direction_fixture,
    wx_fixture,
)

# Test all aspects of common file types


def test_probabilities_above(probability_above_cube):
    """Test interpretation of probability of temperature above threshold
    from UKV"""
    interpreter = MOMetadataInterpreter()
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


def test_probabilities_below(probability_below_cube):
    """Test interpretation of blended probability of max temperature in hour
    below threshold"""
    interpreter = MOMetadataInterpreter()
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


def test_percentiles(percentile_cube):
    """Test interpretation of wind gust percentiles from MOGREPS-UK"""
    interpreter = MOMetadataInterpreter()
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


def test_realizations(ensemble_cube):
    """Test interpretation of temperature realizations from MOGREPS-UK"""
    interpreter = MOMetadataInterpreter()
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


def test_spot_median(spot_cube):
    """Test interpretation of spot median"""
    interpreter = MOMetadataInterpreter()
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


# TODO test error cases: permute compliant cubes to cover the following:
"""
Probabilities:
- Invalid probability name
- No threshold coordinate
- Invalid threshold coordinate name
- Invalid threshold coordinate var name
- Incorrect relative_to_threshold information

Percentiles:
- Incorrect percentile coordinate name TODO implement?
- Incorrect percentile coordinate units

Attributes:
- Missing required attributes (mandatory or diagnostic-specific)
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


# Test specific treatment of weather codes and wind direction


def test_weather_code_success(wx_cube):
    """Test interpretation of weather code file"""
    interpreter = MOMetadataInterpreter()
    interpreter.run(wx_cube)
    assert interpreter.diagnostic == "weather code"
    assert interpreter.blended
    assert interpreter.model == "UKV, MOGREPS-UK"


def test_weather_code_missing_attribute(wx_cube):
    """Test error when weather code required attributes are missing"""
    wx_cube.attributes.pop("weather_code")
    interpreter = MOMetadataInterpreter()
    with pytest.raises(ValueError, match="missing .* required values"):
        interpreter.run(wx_cube)


# TODO wind direction
