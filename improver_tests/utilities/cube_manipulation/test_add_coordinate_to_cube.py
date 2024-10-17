# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the function "cube_manipulation.add_coordinate_to_cube"."""

import numpy as np
import pytest
from iris import Constraint
from iris.coords import DimCoord
from iris.cube import Cube
from iris.util import new_axis

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import add_coordinate_to_cube

ATTRIBUTES = {
    "title": "Test forecast",
    "source": "IMPROVER",
}


@pytest.fixture
def input_cube():
    """Test cube to which to add coordinate."""
    data = np.arange(80, dtype=np.float32).reshape(10, 8)
    return set_up_variable_cube(data, attributes=ATTRIBUTES)


@pytest.fixture
def expected_cube():
    """Test cube to which to add coordinate."""
    data = np.stack(
        [np.arange(80, dtype=np.float32).reshape(10, 8) for realization in range(5)],
        axis=0,
    )
    return set_up_variable_cube(data, realizations=np.arange(5), attributes=ATTRIBUTES)


@pytest.fixture
def realization_coord():
    return DimCoord(np.arange(5), standard_name="realization", units=1)


def test_basic(input_cube, expected_cube, realization_coord):
    """Test the basic usage of cube and coordinate arguments."""
    output_cube = add_coordinate_to_cube(input_cube, realization_coord)
    assert output_cube == expected_cube


def test_non_float32_data(input_cube, expected_cube, realization_coord):
    """Test the case where input data is not float32."""
    input_cube.data = input_cube.data.astype(int)
    expected_cube.data = expected_cube.data.astype(int)
    output_cube = add_coordinate_to_cube(input_cube, realization_coord)
    assert output_cube == expected_cube


def test_add_unit_dimension(input_cube, expected_cube):
    """Test case where added dimension is of length 1."""
    realization_coord = DimCoord([0], standard_name="realization", units=1)
    expected_cube = new_axis(
        expected_cube.extract(Constraint(realization=0)), "realization"
    )
    output_cube = add_coordinate_to_cube(input_cube, realization_coord)
    assert output_cube == expected_cube


@pytest.mark.parametrize(
    "new_dim_location, expected_order", [(0, (0, 1, 2)), (1, (1, 0, 2)), (2, (1, 2, 0))]
)
def test_dim_location(
    input_cube, expected_cube, realization_coord, new_dim_location, expected_order
):
    """Test the usage when new_dim_location is provided."""
    # leading position
    expected_cube.transpose(expected_order)
    output_cube = add_coordinate_to_cube(
        input_cube, realization_coord, new_dim_location=new_dim_location
    )
    assert output_cube == expected_cube


def test_dim_location_beyond_bounds(input_cube, realization_coord):
    """Test ValueError raised when new_dim_location is outside expected bounds."""
    # Beyond expected dimension bounds
    with pytest.raises(ValueError):
        add_coordinate_to_cube(input_cube, realization_coord, new_dim_location=3)
    with pytest.raises(ValueError):
        add_coordinate_to_cube(input_cube, realization_coord, new_dim_location=-1)


def test_no_copy_metadata(input_cube, expected_cube, realization_coord):
    """Test case where copy_metadata is not carried over."""
    output_cube = add_coordinate_to_cube(
        input_cube, realization_coord, copy_metadata=False
    )
    # Check data matches
    np.testing.assert_equal(output_cube.data, expected_cube.data)
    # Check coords match
    assert output_cube.coords() == expected_cube.coords()
    # Check that cube metadata is empty
    assert output_cube.metadata == Cube(np.array([])).metadata
