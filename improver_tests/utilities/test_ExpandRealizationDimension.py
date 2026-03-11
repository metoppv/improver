# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ExpandRealizationDimension class."""

import numpy as np

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.expand_realization_dimension import ExpandRealizationDimension


def basic_cube_with_realization_coord():
    """Create a basic cube with a realization coordinate."""
    data = np.zeros((2, 1, 1), dtype=np.float32)
    data[0, 0, 0] = 1.0
    cube = set_up_variable_cube(data)
    return cube


def test_expand_realization_dimension():
    """Test that the ExpandRealizationDimension plugin correctly expands the realization
    dimension of a cube."""
    cube = basic_cube_with_realization_coord()
    plugin = ExpandRealizationDimension(n_realizations_required=5)
    expanded_cube = plugin.process(cube)

    # Check size of realization correct
    assert expanded_cube.coord("realization").points.size == 5
    # Check realization points monotonically increasing
    np.testing.assert_array_equal(
        expanded_cube.coord("realization").points, np.array([0, 1, 2, 3, 4])
    )
    # Check output data matches expected cycling behaviour
    np.testing.assert_array_equal(
        expanded_cube.data,
        np.array([[[1.0]], [[0.0]], [[1.0]], [[0.0]], [[1.0]]], dtype=np.float32),
    )


def test_no_realization_coord():
    """Test that an error is raised if the input cube does not contain a realization
    coordinate."""
    cube = set_up_variable_cube(np.zeros((1, 1), dtype=np.float32))
    plugin = ExpandRealizationDimension(n_realizations_required=3)

    try:
        plugin.process(cube)
    except ValueError as err:
        assert str(err) == "The input cube does not contain a realization coordinate."
