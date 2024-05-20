# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests of probability_manipulation utilities"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube
from improver.utilities.probability_manipulation import (
    comparison_operator_dict,
    invert_probabilities,
    to_threshold_inequality,
)


def test_comparison_operator_keys():
    """Test that only the expected keys are contained within the comparison
    operator dictionary, and that each contains a namedtuple containing
    the expected elements."""
    expected_keys = sorted(
        ["ge", "GE", ">=", "gt", "GT", ">", "le", "LE", "<=", "lt", "LT", "<"]
    )
    expected_items = ("function", "spp_string", "inverse")
    result = comparison_operator_dict()

    assert isinstance(result, dict)
    assert sorted(result.keys()) == expected_keys
    for k, v in result.items():
        assert v._fields == expected_items
        assert v.function.__module__ == "_operator"
        assert isinstance(v.spp_string, str)
        assert isinstance(v.inverse, str)


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
    "inequality, expected_above, inverted_attr",
    (
        ("greater_than_or_equal_to", True, "less_than"),
        ("greater_than", True, "less_than_or_equal_to"),
        ("less_than_or_equal_to", False, "greater_than"),
        ("less_than", False, "greater_than_or_equal_to"),
    ),
)
def test_to_threshold_inequality(
    probability_cube,
    expected_above,
    inverted_attr,
    above,
    expected_inverted_probabilities,
):
    """Test function returns probabilities with the target threshold inequality."""

    def threshold_attr(cube):
        return cube.coord(var_name="threshold").attributes["spp__relative_to_threshold"]

    ref_attr = threshold_attr(probability_cube)
    result = to_threshold_inequality(probability_cube, above=above)
    result_attr = threshold_attr(result)

    if expected_above == above:
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
