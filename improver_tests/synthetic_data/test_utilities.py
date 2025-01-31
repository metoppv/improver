# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for synthetic data utilities.
"""

import numpy as np
import pytest

from improver.synthetic_data import utilities


@pytest.mark.parametrize(
    "coord_data,expected_cube_type",
    [
        ({"realizations": [0], "thresholds": [0]}, None),
        ({"realizations": [0, 1, 2]}, "variable"),
        ({"percentiles": [10, 90]}, "percentile"),
        ({"thresholds": [2, 3, 4]}, "probability"),
    ],
)
def test_get_leading_dimension(coord_data, expected_cube_type):
    """Tests leading dimension data extracted from dictionary and the correct cube
    type is assigned, or if more than one leading dimension present raises an error"""
    if expected_cube_type is None:
        msg = "Only one"
        with pytest.raises(ValueError, match=msg):
            utilities.get_leading_dimension(coord_data=coord_data)
    else:
        leading_dimension, cube_type = utilities.get_leading_dimension(
            coord_data=coord_data
        )

        dimension_key = list(coord_data)[0]
        np.testing.assert_array_equal(coord_data[dimension_key], leading_dimension)
        assert expected_cube_type == cube_type


@pytest.mark.parametrize(
    "coord_data,expected_pressure",
    [
        ({"realizations": [0]}, False),
        ({"heights": [0, 1, 2]}, False),
        ({"pressures": [10, 20, 30]}, True),
    ],
)
def test_get_vertical_levels(coord_data, expected_pressure):
    """Tests vertical level data extracted successfully and pressure and height flags set correctly"""
    dimension_key = list(coord_data)[0]

    if dimension_key == "realizations":
        expected_vertical_levels = None
    else:
        expected_vertical_levels = coord_data[dimension_key]

    vertical_levels, pressure, height = utilities.get_vertical_levels(coord_data=coord_data)

    np.testing.assert_array_equal(expected_vertical_levels, vertical_levels)
    assert expected_pressure == pressure
