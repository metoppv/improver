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
from iris.exceptions import CoordinateNotFoundError

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
)
from improver.utilities.threshold_interpolation import ThresholdInterpolation


@pytest.fixture(name="input_cube")
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


@pytest.fixture(name="masked_cube_same")
def masked_cube_same() -> Cube:
    """Set up a masked cube which is consistent for every threshold."""
    data = np.array(
        [
            [[1.0, 0.9, 1.0], [0.8, 0.9, 0.5], [0.5, 0.2, 0.0]],
            [[1.0, 0.5, 1.0], [0.5, 0.5, 0.3], [0.2, 0.0, 0.0]],
            [[1.0, 0.2, 0.5], [0.2, 0.0, 0.1], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.int8,
    )

    masked_data = np.ma.masked_array(data, mask=mask)

    masked_cube_same = set_up_probability_cube(
        masked_data,
        thresholds=[100, 200, 300],
        variable_name="visibility_in_air",
        threshold_units="m",
        spp__relative_to_threshold="less_than",
    )
    return masked_cube_same


@pytest.fixture(name="masked_cube_diff")
def masked_cube_diff() -> Cube:
    """Set up a masked cube that has a different mask per threshold."""
    data = np.array(
        [
            [[1.0, 0.9, 1.0], [0.8, 0.9, 0.5], [0.5, 0.2, 0.0]],
            [[1.0, 0.5, 1.0], [0.5, 0.5, 0.3], [0.2, 0.0, 0.0]],
            [[1.0, 0.2, 0.5], [0.2, 0.0, 0.1], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.int8,
    )

    masked_data = np.ma.masked_array(data, mask=mask)

    masked_cube_diff = set_up_probability_cube(
        masked_data,
        thresholds=[100, 200, 300],
        variable_name="visibility_in_air",
        threshold_units="m",
        spp__relative_to_threshold="less_than",
    )
    return masked_cube_diff


def test_basic(input_cube):
    """Test that the plugin returns an Iris.cube.Cube with suitable units."""
    thresholds = [100, 150, 200, 250, 300]
    result = ThresholdInterpolation(thresholds)(input_cube)
    assert result, Cube
    assert result.units == input_cube.units

def test_empty_threshold_list():
    """
    Test that a ValueError is raised if the threshold list is empty.
    """
    with pytest.raises(ValueError, match="The thresholds list cannot be empty."):
        ThresholdInterpolation([])

def test_metadata_copy(input_cube):
    """
    Test that the metadata dictionaries within the input cube, are
    also present on the output cube.
    """
    input_cube.attributes = {"source": "ukv"}
    thresholds = [100, 150, 200, 250, 300]
    result = ThresholdInterpolation(thresholds)(input_cube)
    assert input_cube.metadata._asdict() == result.metadata._asdict()

def test_thresholds_different_mask(masked_cube_diff):
    """
    Testing that a value error message is raised if masks are different across thresholds.
    """
    thresholds = [100, 150, 200, 250, 300]
    error_msg = "The mask is expected to be constant across different slices of the"

    with pytest.raises(ValueError, match=error_msg):
        ThresholdInterpolation(thresholds)(masked_cube_diff)

def test_masked_cube(masked_cube_same):
    """
    Testing that a Cube is returned when inputting a masked cube.
    """
    thresholds = [100, 150, 200, 250, 300]
    result = ThresholdInterpolation(thresholds)(masked_cube_same)
    assert isinstance(result, Cube)

def test_mask_consistency(masked_cube_same):
    """
    Test that the mask is the same before and after ThresholdInterpolation.
    """
    thresholds = [100, 150, 200, 250, 300]
    original_mask = masked_cube_same.data.mask
    print(original_mask)
    result = ThresholdInterpolation(thresholds)(masked_cube_same).data.mask
    print(result)
    np.testing.assert_array_equal(original_mask[0], result[0])
