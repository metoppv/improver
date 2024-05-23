# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Unit tests for spot data utilities"""

import pytest

from improver.spotdata.utilities import get_neighbour_finding_method_name


@pytest.mark.parametrize(
    "land_constraint, minimum_dz, expected",
    (
        (False, False, "nearest"),
        (True, False, "nearest_land"),
        (True, True, "nearest_land_minimum_dz"),
        (False, True, "nearest_minimum_dz"),
    ),
)
def test_get_neighbour_finding_method_name(land_constraint, minimum_dz, expected):
    """Test the function for generating the name that describes the neighbour
    finding method."""

    result = get_neighbour_finding_method_name(land_constraint, minimum_dz)
    assert result == expected
