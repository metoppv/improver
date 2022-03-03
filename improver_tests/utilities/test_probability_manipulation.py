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
"""Tests of probability_manipulation utilities"""

import operator
from collections import namedtuple

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube
from improver.utilities.probability_manipulation import (
    comparison_operator_dict,
    invert_probabilities,
    to_threshold_inequality,
)

ComparisonResult = namedtuple("ComparisonResult", ["function", "spp_string", "inverse"])


@pytest.mark.parametrize(
    "inequalities, expected",
    (
        (
            ("ge", "GE", ">="),
            ComparisonResult(
                function=operator.ge,
                spp_string="greater_than_or_equal_to",
                inverse="lt",
            ),
        ),
        (
            ("gt", "GT", ">"),
            ComparisonResult(
                function=operator.gt, spp_string="greater_than", inverse="le",
            ),
        ),
        (
            ("le", "LE", "<="),
            ComparisonResult(
                function=operator.le, spp_string="less_than_or_equal_to", inverse="gt",
            ),
        ),
        (
            ("lt", "LT", "<"),
            ComparisonResult(
                function=operator.lt, spp_string="less_than", inverse="ge",
            ),
        ),
    ),
)
def test_comparison_operator_dict(inequalities, expected):
    """Test that calling this function returns a dictionary containing
    the expected values / functions."""
    for inequality in inequalities:
        result = comparison_operator_dict()[inequality]

        assert isinstance(result, dict)
        assert result["function"] == expected.function
        assert result["spp_string"] == expected.spp_string
        assert result["inverse"] == expected.inverse


def test_comparison_operator_keys():
    """Test that only the expected keys are contained within the comparison
    operator dictionary, and that each contains a further dictionary containing
    the expected types."""
    expected_keys = sorted(
        ["ge", "GE", ">=", "gt", "GT", ">", "le", "LE", "<=", "lt", "LT", "<"]
    )
    expected_subkeys = sorted(["function", "spp_string", "inverse"])
    result = comparison_operator_dict()

    assert isinstance(result, dict)
    assert sorted(result.keys()) == expected_keys
    for k, v in result.items():
        assert sorted(v.keys()) == expected_subkeys
        assert v["function"].__module__ == "_operator"
        assert isinstance(v["spp_string"], str)
        assert isinstance(v["inverse"], str)


@pytest.fixture
def probability_cube(inequality):
    """Set up probability cube"""
    data = np.linspace(0.0, 0.7, 8).reshape(2, 2, 2).astype(np.float32)
    cube = set_up_probability_cube(
        data,
        thresholds=[273.15, 278.15],
        spatial_grid="equalarea",
        spp__relative_to_threshold=inequality,
    )
    return cube


@pytest.fixture
def expected_inverted_probabilities():
    return np.linspace(1.0, 0.3, 8).reshape(2, 2, 2).astype(np.float32)


@pytest.mark.parametrize("above", [True, False])
@pytest.mark.parametrize(
    "inequality, etype, inverted_attr",
    (
        ("greater_than_or_equal_to", "above", "less_than"),
        ("greater_than", "above", "less_than_or_equal_to"),
        ("less_than_or_equal_to", "below", "greater_than"),
        ("less_than", "below", "greater_than_or_equal_to"),
    ),
)
def test_to_threshold_inequality(
    probability_cube, etype, inverted_attr, above, expected_inverted_probabilities
):
    """Test function returns probabilities with the target threshold inequality."""

    def threshold_attr(cube):
        return cube.coord(var_name="threshold").attributes["spp__relative_to_threshold"]

    ref_attr = threshold_attr(probability_cube)
    result = to_threshold_inequality(probability_cube, above=above)
    result_attr = threshold_attr(result)

    if (etype == "above" and above) or (etype == "below" and not above):
        assert result_attr == ref_attr
    else:
        assert result_attr == inverted_attr


@pytest.mark.parametrize(
    "inequality, expected_attr, expected_name",
    (
        ("greater_than_or_equal_to", "less_than", "below"),
        ("greater_than", "less_than_or_equal_to", "below"),
        ("less_than_or_equal_to", "greater_than", "above"),
        ("less_than", "greater_than_or_equal_to", "above"),
    ),
)
def test_invert_probabilities(
    probability_cube, expected_attr, expected_name, expected_inverted_probabilities
):
    """Test function inverts probabilities and updates cube metadata."""
    result = invert_probabilities(probability_cube)

    assert (
        result.coord(var_name="threshold").attributes["spp__relative_to_threshold"]
        == expected_attr
    )
    assert expected_name in result.name()
    assert_almost_equal(result.data, expected_inverted_probabilities)


@pytest.mark.parametrize("inequality", ["greater_than"])
def test_no_threshold_coordinate(probability_cube):
    """Test an exception is raised if no threshold coordinate is found."""

    cube = probability_cube[0]
    threshold = cube.coord(var_name="threshold")
    cube.remove_coord(threshold)

    with pytest.raises(ValueError, match="Cube does not have a threshold coordinate"):
        invert_probabilities(cube)
