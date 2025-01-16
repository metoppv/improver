# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function apply_mask.
"""

import iris
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.mask import apply_mask


@pytest.fixture
def wind_gust_cube():
    data = np.full((2, 3), 10)
    return set_up_variable_cube(
        data=data, attributes={"wind_gust_type": "10m_ratio"}, name="wind_gust"
    )


@pytest.fixture
def mask():
    data = np.array([[0, 0, 1], [1, 1, 0]])
    return set_up_variable_cube(data=data, name="land_sea_mask")


@pytest.mark.parametrize("invert_mask", [True, False])
@pytest.mark.parametrize("switch_coord_order", [True, False])
def test_basic(wind_gust_cube, mask, switch_coord_order, invert_mask):
    """
    Test the basic functionality of the apply_mask plugin. Checks that the
    mask is correctly applied and inverted if requested. Also checks plugin
    can cope with different orderings of coordinates on the input cubes."""

    expected_data = np.full((2, 3), 10)
    expected_mask = np.array([[False, False, True], [True, True, False]])
    if switch_coord_order:
        enforce_coordinate_ordering(wind_gust_cube, ["longitude", "latitude"])
        expected_data = expected_data.transpose()
        expected_mask = expected_mask.transpose()
    if invert_mask:
        expected_mask = np.invert(expected_mask)

    input_list = [wind_gust_cube, mask]

    result = apply_mask(
        iris.cube.CubeList(input_list),
        mask_name="land_sea_mask",
        invert_mask=invert_mask,
    )

    assert np.allclose(result.data, expected_data)
    assert np.allclose(result.data.mask, expected_mask)


def test_different_dimensions(wind_gust_cube, mask):
    """Test that the function will raise an error if the mask cube has different
    dimensions to other cube."""
    mask = mask[0]
    input_list = [wind_gust_cube, mask]
    with pytest.raises(
        ValueError, match="Input cube and mask cube must have the same dimensions"
    ):
        apply_mask(iris.cube.CubeList(input_list), mask_name="land_sea_mask")


def test_too_many_cubes(wind_gust_cube, mask):
    """
    Test that the function will raise an error if more than two cubes are provided.
    """
    input_list = [wind_gust_cube, wind_gust_cube, wind_gust_cube]
    with pytest.raises(ValueError, match="Two cubes are required for masking"):
        apply_mask(iris.cube.CubeList(input_list), mask_name="land_sea_mask")
