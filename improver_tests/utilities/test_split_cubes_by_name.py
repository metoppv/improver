# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
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
