#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function manipulate_realization_dimension.
"""

import iris
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import manipulate_n_realizations


@pytest.fixture
def temperature_cube():
    data = 281 * np.random.random_sample((3, 3, 3)).astype(np.float32)
    return set_up_variable_cube(data, realizations=[0, 1, 2])


@pytest.mark.parametrize("n_realizations", (2, 3, 4))
def test_basic(temperature_cube, n_realizations):
    """Test that a cube is returned with expected data and realization coordinate."""
    input_len = len(temperature_cube.coord("realization").points)
    expected_realizations = np.array([r for r in range(n_realizations)])
    result = manipulate_n_realizations(temperature_cube, n_realizations)

    assert len(result.coord("realization").points) == n_realizations
    assert np.all(result.coord("realization").points == expected_realizations)
    for realization in result.coord("realization").points:
        input_constr = iris.Constraint(realization=realization % input_len)
        result_constr = iris.Constraint(realization=realization)

        input_slice = temperature_cube.extract(input_constr)
        result_slice = result.extract(result_constr)

        np.testing.assert_allclose(result_slice.data, input_slice.data)


def test_realizations_start_from_one(temperature_cube):
    """Test that correct cube is returned when the realizations in the input cube are
    not numbered starting from zero.
    """
    input_cube = temperature_cube.copy()
    input_cube.coord("realization").points = np.array([1, 2, 3])
    n_realizations = 6
    expected_realizations = [1, 2, 3, 4, 5, 6]
    expected_recycling = [1, 2, 3, 1, 2, 3]

    result = manipulate_n_realizations(input_cube, n_realizations)

    assert len(result.coord("realization").points) == n_realizations
    assert np.all(result.coord("realization").points == expected_realizations)
    for index in range(n_realizations):
        input_constr = iris.Constraint(realization=expected_recycling[index])
        result_constr = iris.Constraint(realization=expected_realizations[index])

        input_slice = input_cube.extract(input_constr)
        result_slice = result.extract(result_constr)

        np.testing.assert_allclose(result_slice.data, input_slice.data)


def test_non_realization_cube(temperature_cube):
    """Test that the correct exception is raised when input cube does not contain
    a realization dimension.
    """
    temperature_cube.coord("realization").rename("percentile")
    msg = (
        "Input cube does not contain realizations. The following dimension "
        "coordinates were found: "
    )

    with pytest.raises(ValueError, match=msg):
        manipulate_n_realizations(temperature_cube, n_realizations=3)
