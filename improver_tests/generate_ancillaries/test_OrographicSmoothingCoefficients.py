# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the OrographicSmoothingCoefficients utility.

"""

import numpy as np
import pytest
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy.testing import assert_array_almost_equal, assert_array_equal

from improver.generate_ancillaries.generate_orographic_smoothing_coefficients import (
    OrographicSmoothingCoefficients,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


@pytest.fixture(name="orography")
def orography_fixture() -> Cube:
    """Orography cube with three gradients in each dimension."""

    data = np.array([[0, 0, 0], [1, 3, 5], [2, 6, 10]], dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="surface_altitude",
        units="m",
        spatial_grid="equalarea",
        x_grid_spacing=1000,
        y_grid_spacing=1000,
        domain_corner=(-1000, -1000),
    )
    return cube


@pytest.fixture(name="gradient")
def gradient_fixture() -> Cube:
    """Gradient cube with several gradient values."""

    data = np.array([0, 0.5, -1.0, 5.0], dtype=np.float32)
    data = np.stack([data] * 2)
    cube = set_up_variable_cube(
        data, name="gradient_of_surface_altitude", units="1", spatial_grid="equalarea"
    )
    return cube


@pytest.fixture(name="smoothing_coefficients")
def smoothing_coefficients_fixture() -> CubeList:
    """Returns example smoothing coefficients."""

    x_data = np.linspace(0, 1.0, 5, dtype=np.float32)
    x_data = np.stack([x_data] * 6)
    y_data = np.linspace(0.05, 0.9, 5, dtype=np.float32)
    y_data = np.stack([y_data] * 6).T

    x_cube = set_up_variable_cube(
        x_data, name="smoothing_coefficient_x", units="1", spatial_grid="equalarea"
    )
    y_cube = set_up_variable_cube(
        y_data, name="smoothing_coefficient_y", units="1", spatial_grid="equalarea"
    )
    return CubeList([x_cube, y_cube])


@pytest.fixture(name="mask")
def mask_fixture() -> Cube:
    """Returns an example mask."""

    data = np.zeros((6, 6), dtype=np.int32)
    data[2:-2, 2:-2] = 1
    mask = set_up_variable_cube(
        data,
        name="land_binary_mask",
        units="1",
        spatial_grid="equalarea",
        x_grid_spacing=1000,
        y_grid_spacing=1000,
        domain_corner=(-2000, -2000),
    )
    return mask


def test_init():
    """Test default attribute initialisation"""
    result = OrographicSmoothingCoefficients()
    assert result.min_gradient_smoothing_coefficient == 0.5
    assert result.max_gradient_smoothing_coefficient == 0.0
    assert result.power == 1.0
    assert result.use_mask_boundary is False


@pytest.mark.parametrize("min_value, max_value", ((1.0, 0.0), (-0.1, 0.0)))
def test_init_value_error(min_value, max_value):
    """Test a ValueError is raised if the chosen smoothing coefficient limits
    are outside the range 0 to 0.5 inclusive."""

    msg = "min_gradient_smoothing_coefficient and max_gradient_smoothing_coefficient"
    with pytest.raises(ValueError, match=msg):
        OrographicSmoothingCoefficients(
            min_gradient_smoothing_coefficient=min_value,
            max_gradient_smoothing_coefficient=max_value,
        )


def test_scale_smoothing_coefficients(smoothing_coefficients):
    """Test the scale_smoothing_coefficients function"""

    # Test using the default max and min coefficient values
    plugin = OrographicSmoothingCoefficients()
    result = plugin.scale_smoothing_coefficients(smoothing_coefficients)
    expected_x = np.linspace(0.5, 0, 5, dtype=np.float32)
    expected_y = np.linspace(0.475, 0.05, 5, dtype=np.float32)
    assert_array_almost_equal(result[0].data[0, :], expected_x)
    assert_array_almost_equal(result[1].data[:, 0], expected_y)

    # Test using custom max and min coefficient values
    plugin = OrographicSmoothingCoefficients(
        min_gradient_smoothing_coefficient=0.4, max_gradient_smoothing_coefficient=0.1
    )
    result = plugin.scale_smoothing_coefficients(smoothing_coefficients)
    expected_x = np.linspace(0.4, 0.1, 5, dtype=np.float32)
    expected_y = np.linspace(0.385, 0.13, 5, dtype=np.float32)
    assert_array_almost_equal(result[0].data[0, :], expected_x)
    assert_array_almost_equal(result[1].data[:, 0], expected_y)


@pytest.mark.parametrize("power", (1, 2, 0.5))
def test_unnormalised_smoothing_coefficients(gradient, power):
    """Test the unnormalised_smoothing_coefficients function using various
    powers."""

    plugin = OrographicSmoothingCoefficients(power=power)
    expected = np.abs(gradient.data.copy()) ** power
    result = plugin.unnormalised_smoothing_coefficients(gradient)
    assert_array_almost_equal(result, expected)


def test_zero_masked_use_mask_boundary(smoothing_coefficients, mask):
    """Test the zero_masked function to set smoothing coefficients to zero
    around the boundary of a mask."""

    expected_x = smoothing_coefficients[0].data.copy()
    expected_x[2:4, 1] = 0
    expected_x[2:4, 3] = 0
    expected_y = smoothing_coefficients[1].data.copy()
    expected_y[1, 2:4] = 0
    expected_y[3, 2:4] = 0

    plugin = OrographicSmoothingCoefficients(use_mask_boundary=True)
    plugin.zero_masked(*smoothing_coefficients, mask)
    assert_array_equal(smoothing_coefficients[0].data, expected_x)
    assert_array_equal(smoothing_coefficients[1].data, expected_y)


def test_zero_masked_whole_area(smoothing_coefficients, mask):
    """Test the zero_masked function to set smoothing coefficients to zero
    across the entire masked area (where the mask is 1)."""

    expected_x = smoothing_coefficients[0].data.copy()
    expected_x[2:4, 1:4] = 0
    expected_y = smoothing_coefficients[1].data.copy()
    expected_y[1:4, 2:4] = 0

    plugin = OrographicSmoothingCoefficients()
    plugin.zero_masked(*smoothing_coefficients, mask)
    assert_array_equal(smoothing_coefficients[0].data, expected_x)
    assert_array_equal(smoothing_coefficients[1].data, expected_y)


def test_zero_masked_whole_area_inverted(smoothing_coefficients, mask):
    """Test the zero_masked function to set smoothing coefficients to zero
    across the entire unmasked area (where the mask is 0)."""

    expected_x = np.zeros((6, 5), dtype=np.float32)
    expected_x[2:4, 2] = 0.5
    expected_y = np.zeros((5, 6), dtype=np.float32)
    expected_y[2, 2:4] = 0.475

    plugin = OrographicSmoothingCoefficients(invert_mask=True)
    plugin.zero_masked(*smoothing_coefficients, mask)
    assert_array_equal(smoothing_coefficients[0].data, expected_x)
    assert_array_equal(smoothing_coefficients[1].data, expected_y)


def test_process_exceptions(orography, mask):
    """Test exceptions raised by the process method."""

    plugin = OrographicSmoothingCoefficients()

    # Test the orography is a cube
    msg = "expects cube"
    with pytest.raises(ValueError, match=msg):
        plugin.process(None, None)

    # Test the orography cube has 2 dimensions
    msg = "Expected orography on 2D grid"
    with pytest.raises(ValueError, match=msg):
        plugin.process(orography[0], None)

    # Test the mask cube shares the orography grid
    msg = "If a mask is provided it must have the same grid"
    mask.coord(axis="x").points = mask.coord(axis="x").points + 1.0
    with pytest.raises(ValueError, match=msg):
        plugin.process(orography, mask)


def test_process_metadata(orography):
    """Test that the cube returned by process has the expected metadata and
    coordinates. These tests cover the functionality of create_coefficient_cube."""

    expected_x_coord = 0.5 * (
        orography.coord(axis="x").points[1:] + orography.coord(axis="x").points[:-1]
    )
    expected_y_coord = 0.5 * (
        orography.coord(axis="y").points[1:] + orography.coord(axis="y").points[:-1]
    )

    plugin = OrographicSmoothingCoefficients()
    orography.attributes["history"] = "I have a history"
    result = plugin.process(orography)

    assert isinstance(result, CubeList)
    assert isinstance(result[0], Cube)
    assert result[0].attributes["title"] == "Recursive filter smoothing coefficients"
    assert result[0].attributes.get("history", None) is None
    assert result[0].shape == (orography.shape[0], orography.shape[1] - 1)
    assert result[1].shape == (orography.shape[0] - 1, orography.shape[1])
    # Check x-y coordinates of x coefficients cube
    assert_array_almost_equal(result[0].coord(axis="x").points, expected_x_coord)
    assert result[0].coord(axis="y") == orography.coord(axis="y")
    # Check x-y coordinates of y coefficients cube
    assert result[1].coord(axis="x") == orography.coord(axis="x")
    assert_array_almost_equal(result[1].coord(axis="y").points, expected_y_coord)
    # Check there are no time coordinates on the returned cube
    with pytest.raises(CoordinateNotFoundError):
        result[0].coord(axis="t")


@pytest.mark.parametrize(
    "min_value, max_value, expected_x, expected_y",
    (
        (0.25, 0.0, [0.25, 0.15, 0.05], [0.20, 0.10, 0.00]),
        (0.0, 0.25, [0.00, 0.10, 0.20], [0.05, 0.15, 0.25]),
    ),
)
def test_process_no_mask(orography, min_value, max_value, expected_x, expected_y):
    """Test generation of smoothing coefficients from orography returns the
    expected values.

    The orography is such that the steepest gradient is found in the y-direction,
    whilst the minimum gradient is found in the x-direction. The maximum gradient
    in x is 4/5ths of the maximum gradient in y.

    In the first test the smoothing coefficient is largest where the orography
    gradient is at its minimum, giving the largest smoothing coefficients in the
    x-dimension. In the second test smoothing coefficient is largest where the
    orography gradient is at its maximum, giving the largest smoothing
    coefficients in the y-dimension."""

    plugin = OrographicSmoothingCoefficients(
        max_gradient_smoothing_coefficient=max_value,
        min_gradient_smoothing_coefficient=min_value,
    )
    result_x, result_y = plugin.process(orography)
    assert_array_almost_equal(result_x.data[:, 0], expected_x)
    assert_array_almost_equal(result_y.data[0, :], expected_y)


def test_process_with_mask(orography, mask):
    """Test generation of smoothing coefficients from orography returns the
    expected values when a mask is used to zero some elements. Here we are using
    a mask that covers a 2x2 area of the 3x3 orography cube in the bottom right
    hand corner.

    The masking is performed after scaling, so masking values does not lead to
    different scaling of the smoothing coefficients."""

    def compare_outputs(plugin, expected_x, expected_y):
        result_x, result_y = plugin.process(orography, mask[1:4, 1:4])
        assert_array_almost_equal(result_x.data, expected_x)
        assert_array_almost_equal(result_y.data, expected_y)

    # Using use_mask_boundary=True which zeroes the edges of the masked region.
    # For x this means the coefficients between the left and central columns
    # in the bottom two rows are zeroed.
    # For y this means the coefficients between the top and middle rows
    # in the right hand two columns are zeroed.
    expected_x = [[0.5, 0.5], [0.0, 0.3], [0.0, 0.1]]
    expected_y = [[0.4, 0.0, 0.0], [0.4, 0.2, 0.0]]

    plugin = OrographicSmoothingCoefficients(use_mask_boundary=True)
    compare_outputs(plugin, expected_x, expected_y)

    # Using use_mask_boundary=False gives the same result as above modified to
    # zero the smoothing coefficients that are entirely covered by the mask as
    # well. This is the bottom right corner.
    expected_x = [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]]
    expected_y = [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]

    plugin = OrographicSmoothingCoefficients()
    compare_outputs(plugin, expected_x, expected_y)

    # Using use_mask_boundary=False and invert_mask=True. In this case only
    # the values in the bottom right corner that are entirely covered by the
    # mask will be returned.
    expected_x = [[0.0, 0.0], [0.0, 0.3], [0.0, 0.1]]
    expected_y = [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]]

    plugin = OrographicSmoothingCoefficients(invert_mask=True)
    compare_outputs(plugin, expected_x, expected_y)

    # This test repeats the one above but with the coordinates of the input
    # cubes reversed. The result should also have reversed coordinates.
    expected_x = [[0.0, 0.0, 0.0], [0.0, 0.3, 0.1]]
    expected_y = [[0.0, 0.0], [0.0, 0.2], [0.0, 0.0]]

    order = ["projection_x_coordinate", "projection_y_coordinate"]
    enforce_coordinate_ordering(orography, order)
    enforce_coordinate_ordering(mask, order)

    plugin = OrographicSmoothingCoefficients(invert_mask=True)
    compare_outputs(plugin, expected_x, expected_y)
