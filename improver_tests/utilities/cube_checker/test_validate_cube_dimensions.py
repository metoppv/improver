# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_checker import validate_cube_dimensions


@pytest.fixture
def cube_3d():
    """
    Set up a 3D cube with 2 'realization' points, and 'longitude' and 'latitude'.
    """
    data = np.ones((2, 1, 1), dtype=np.float32)
    return set_up_variable_cube(data)


@pytest.mark.parametrize(
    "required_dimensions, forbidden_dimensions, exact_match",
    [
        (["realization", "x", "y"], None, True),
        (["x", "y"], None, False),
    ],
)
def test_validate_cube_dimensions_passes(
    cube_3d, required_dimensions, forbidden_dimensions, exact_match
):
    """
    Test that validate_cube_dimensions passes for valid dimension configurations:
    - When the cube dimensions include at least the required dimensions when exact_match
    is False
    - When the cube dimensions match the required dimensions exactly when exact_match
    is True
    """
    validate_cube_dimensions(
        cube=cube_3d,
        required_dimensions=required_dimensions,
        forbidden_dimensions=forbidden_dimensions,
        exact_match=exact_match,
    )


@pytest.mark.parametrize(
    "required_dimensions, forbidden_dimensions, exact_match, error_message",
    [
        # Dimension included in both required and forbidden lists
        (
            ["x", "y"],
            ["x"],
            False,
            r"Dimension\(s\) cannot be both required and forbidden",
        ),
        # Forbidden dimensions present
        (["x", "y"], ["realization"], False, "Forbidden dimension"),
        # Missing required dimensions
        (["time", "x", "y"], [], False, "Missing required dimension"),
        # Missing required dimensions with exact match
        (
            ["time"],
            [],
            True,
            "Missing required dimension",
        ),
        # Exact_match True but surplus dimensions on input cube
        (["x", "y"], [], True, "Extra dimension"),
    ],
)
def test_validate_cube_dimensions_raises(
    cube_3d, required_dimensions, forbidden_dimensions, exact_match, error_message
):
    """
    Test that validate_cube_dimensions raises ValueError when the cube dimension
    configuration does not match the required and forbidden dimensions for the specified
    mode:
    - When a dimension is included in both required and forbidden lists
    - When forbidden dimensions are present on the cube
    - When required dimensions are missing from the cube
    - When exact_match is True but the cube has extra dimensions not listed as required
    """
    with pytest.raises(ValueError, match=error_message):
        validate_cube_dimensions(
            cube=cube_3d,
            required_dimensions=required_dimensions,
            forbidden_dimensions=forbidden_dimensions,
            exact_match=exact_match,
        )
