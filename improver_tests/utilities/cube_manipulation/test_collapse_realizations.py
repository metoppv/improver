# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function collapse_realizations.
"""

import iris
import numpy as np
import pytest
from iris.exceptions import CoordinateCollapseError

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import collapse_realizations


@pytest.fixture
def temperature_cube():
    data = 281 * np.random.random_sample((3, 3, 3)).astype(np.float32)
    return set_up_variable_cube(data, realizations=[0, 1, 2])


def test_basic(temperature_cube):
    """Test that a collapsed cube is returned with no realization coord."""
    result = collapse_realizations(temperature_cube)
    assert "realization" not in [coord.name() for coord in result.coords()]
    expected = temperature_cube.collapsed(["realization"], iris.analysis.MEAN)
    np.testing.assert_allclose(result.data, expected.data)


def test_invalid_dimension(temperature_cube):
    """Test that an error is raised if realization dimension
    does not exist."""
    sub_cube = temperature_cube.extract(iris.Constraint(realization=0))
    with pytest.raises(CoordinateCollapseError):
        collapse_realizations(sub_cube, "mean")


def test_different_aggregators(temperature_cube):
    """Test aggregators other than mean."""
    aggregator_dict = {
        "sum": iris.analysis.SUM,
        "median": iris.analysis.MEDIAN,
        "std_dev": iris.analysis.STD_DEV,
        "min": iris.analysis.MIN,
        "max": iris.analysis.MAX,
    }
    for key, value in aggregator_dict.items():
        result = collapse_realizations(temperature_cube, key)
        expected = temperature_cube.collapsed(["realization"], value)
        np.testing.assert_allclose(result.data, expected.data)


def test_invalid_aggregators(temperature_cube):
    """Test that an error is raised if aggregator is not
    one of the allowed types."""

    msg = "method must be one of"
    with pytest.raises(ValueError, match=msg):
        collapse_realizations(temperature_cube, method="product")


def test_1d_std_dev():
    """Test that when std_dev is calculated over a dimension of size 1,
    output is all masked and underlying value is np.nan.
    """
    data = 281 * np.random.random_sample((1, 3, 3)).astype(np.float32)
    cube_1d = set_up_variable_cube(data, realizations=[0])
    result = collapse_realizations(cube_1d, "std_dev")
    assert np.all(np.ma.getmask(result.data))
    assert np.all(np.isnan(result.data.data))
