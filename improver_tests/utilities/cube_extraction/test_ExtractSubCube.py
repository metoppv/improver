# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from unittest.mock import patch, sentinel

import pytest
from iris.cube import Cube

from improver.utilities.cube_extraction import ExtractSubCube


@patch("improver.utilities.cube_extraction.as_cube")
@patch("improver.utilities.cube_extraction.extract_subcube")
def test_ui(mock_extract_subcube, mock_as_cube):
    """Ensure 'extract_subcube' is called with the correct arguments."""
    mock_as_cube.side_effect = lambda x: x  # identity function
    plugin = ExtractSubCube(
        sentinel.constraints,
        units=sentinel.units,
        use_original_units=sentinel.use_original_units,
        ignore_failure=sentinel.ignore_failure,
    )
    plugin(sentinel.cube)
    mock_as_cube.assert_called_once_with(sentinel.cube)
    mock_extract_subcube.assert_called_once_with(
        sentinel.cube, sentinel.constraints, sentinel.units, sentinel.use_original_units
    )


def test_no_matching_constraint_exception():
    """
    Test that a ValueError is raised when no constraints match.
    """
    plugin = ExtractSubCube(["dummy_name=dummy_value"], ignore_failure=False)
    with pytest.raises(ValueError) as excinfo:
        plugin(Cube(0))
    assert str(excinfo.value) == "Constraint(s) could not be matched in input cube"


def test_no_matching_constraint_ignore():
    """
    Test that the original cube is returned when no constraints match and ignore specified.
    """
    plugin = ExtractSubCube(["dummy_name=dummy_value"], ignore_failure=True)
    src_cube = Cube(0)
    res = plugin(src_cube)
    assert res is src_cube
