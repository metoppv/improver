# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the
`ensemble_copula_coupling.EnsembleCopulaCouplingUtilities` class.
"""

import numpy as np
import pytest
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
)
from improver.utilities.threshold_interpolation import ThresholdInterpolation


def basic_input_cube() -> Cube:
    """Set up input cube with sparse thresholds."""
    data = np.array(
        [
            [[1.0, 0.9, 1.0], [0.8, 0.9, 0.5], [0.5, 0.2, 0.0]],
            [[1.0, 0.5, 1.0], [0.5, 0.5, 0.3], [0.2, 0.0, 0.0]],
            [[1.0, 0.2, 0.5], [0.2, 0.0, 0.1], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    input_cube = set_up_probability_cube(
        data,
        thresholds=[100, 200, 300],
        variable_name="visibility_in_air",
        threshold_units="m",
        spp__relative_to_threshold="less_than",
    )
    return input_cube


@pytest.fixture
def input_cube() -> Cube:
    """Return an input cube with sparse thresholds."""
    return basic_input_cube()


@pytest.fixture
def masked_cube() -> Cube:
    """Set up a masked cube which is consistent for every threshold."""
    masked_cube = basic_input_cube()

    mask = np.array(
        [
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.int8,
    )
    masked_cube.data = np.ma.masked_array(masked_cube.data, mask=mask)
    return masked_cube


@pytest.mark.parametrize("input", ["input_cube", "masked_cube"])
def test_cube_returned(request, input):
    """
    Test that the plugin returns an Iris.cube.Cube with suitable units.
    """
    cube = request.getfixturevalue(input)
    thresholds = [100, 150, 200, 250, 300]
    result = ThresholdInterpolation(thresholds)(cube)
    assert result, Cube
    assert result.units == cube.units


@pytest.mark.parametrize("input", ["input_cube", "masked_cube"])
def test_interpolated_values(request, input):
    """
    Test that the interpolated values are as expected.
    """
    cube = request.getfixturevalue(input)
    thresholds = [100, 150, 200, 250, 300]
    result = ThresholdInterpolation(thresholds)(cube)
    expected_interpolated_values = np.array(
        [
            [[1.0, 0.9, 1.0], [0.8, 0.9, 0.5], [0.5, 0.2, 0.0]],
            [[1.0, 0.7, 1.0], [0.65, 0.7, 0.4], [0.35, 0.1, 0.0]],
            [[1.0, 0.5, 1.0], [0.5, 0.5, 0.3], [0.2, 0.0, 0.0]],
            [[1.0, 0.35, 0.75], [0.35, 0.25, 0.2], [0.1, 0.0, 0.0]],
            [[1.0, 0.2, 0.5], [0.2, 0.0, 0.1], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result.data, expected_interpolated_values)


def test_empty_threshold_list():
    """
    Test that a ValueError is raised if the threshold list is empty.
    """
    with pytest.raises(ValueError, match="The thresholds list cannot be empty."):
        ThresholdInterpolation([])


def test_metadata_copy(input_cube):
    """
    Test that the metadata dictionaries within the input cube are
    also present on the output cube.
    """
    input_cube.attributes = {"source": "ukv"}
    thresholds = [100, 150, 200, 250, 300]
    result = ThresholdInterpolation(thresholds)(input_cube)
    assert input_cube.metadata._asdict() == result.metadata._asdict()


def test_thresholds_different_mask(masked_cube):
    """
    Testing that a value error message is raised if masks are different across thresholds.
    """
    masked_cube.data.mask[0, 0, 0] = True
    thresholds = [100, 150, 200, 250, 300]
    error_msg = "The mask is expected to be constant across different slices of the"

    with pytest.raises(ValueError, match=error_msg):
        ThresholdInterpolation(thresholds)(masked_cube)


def test_mask_consistency(masked_cube):
    """
    Test that the mask is the same before and after ThresholdInterpolation.
    """
    thresholds = [100, 150, 200, 250, 300]
    original_mask = masked_cube.data.mask.copy()
    result = ThresholdInterpolation(thresholds)(masked_cube).data.mask
    np.testing.assert_array_equal(original_mask[0], result[0])
