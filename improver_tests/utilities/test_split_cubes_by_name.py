# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the "forecast_reference_enforcement.split_cubes_by_name" function."""

import iris
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.forecast_reference_enforcement import split_cubes_by_name


def create_test_cubes(n_cubes, name) -> iris.cube.CubeList:
    """Create a cubelist containing a number of cubes equal to n_cubes each with the
    input name as their name.
    """
    shape = (2, 3, 3)
    data = np.ones(shape, dtype=np.float32)
    output = iris.cube.CubeList()

    if n_cubes != 0:
        cube = set_up_variable_cube(data, name=name, units="m s-1")
        for index in range(n_cubes):
            output.append(cube.copy())

    return output


@pytest.mark.parametrize(
    "n_match, n_other", ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (1, 2))
)
def test_split_cubes_by_name_basic(n_match, n_other):
    """Test that the plugin correctly splits the input cubelist when this cubelist
    contains different amounts of matching and non_matching cubes.
    """
    match_name = "rainfall_rate"
    other_name = "sleetfall_rate"

    match_cubes = create_test_cubes(n_match, match_name)
    other_cubes = create_test_cubes(n_other, other_name)
    all_cubes = match_cubes.copy()
    all_cubes.extend(other_cubes)

    match_result, other_result = split_cubes_by_name(all_cubes, match_name)

    assert match_result == match_cubes
    assert other_result == other_cubes


@pytest.mark.parametrize(
    "names",
    ("rainfall_rate,sleetfall_rate", "rainfall_rate,sleetfall_rate,snowfall_rate"),
)
def test_multiple_cube_names(names):
    """Test that the plugin correctly splits the input cubelist when this cubelist
    contains more than two distinct names and more than one matching name is provided.
    """
    match_names = names.split(",")
    other_name = "precipitation_rate"

    match_cubes = iris.cube.CubeList()
    for name in match_names:
        match_cubes.extend(create_test_cubes(1, name))
    other_cubes = create_test_cubes(1, other_name)
    all_cubes = match_cubes.copy()
    all_cubes.extend(other_cubes)

    match_result, other_result = split_cubes_by_name(all_cubes, match_names)

    assert match_result == match_cubes
    assert other_result == other_cubes


def test_none_cube_name():
    """Test that the plugin correctly returns the full input cubelist when no
    matching name is provided.
    """
    match_name = "rainfall_rate"
    other_name = "sleetfall_rate"

    match_cubes = create_test_cubes(1, match_name)
    other_cubes = create_test_cubes(1, other_name)
    all_cubes = match_cubes
    all_cubes.extend(other_cubes)

    match_result, other_result = split_cubes_by_name(all_cubes)

    assert match_result == all_cubes
    assert other_result == iris.cube.CubeList()
