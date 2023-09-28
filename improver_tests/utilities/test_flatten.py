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
"""Unit tests for flattening an arbitrarily nested iterable."""

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.flatten import flatten


@pytest.fixture
def cube() -> Cube:
    """Sets up a cube for testing"""
    return set_up_variable_cube(
        np.zeros((2, 2), dtype=np.float32),
    )


@pytest.mark.parametrize(
    "nested,expected",
    (
        ([0, 1, 2], [0, 1, 2]),
        ([0, 1, [2, 3]], [0, 1, 2, 3]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", "b", ["c", "d"]], ["a", "b", "c", "d"]),
        (
            [np.array([0]), np.array([1]), [np.array([2]), np.array([2])]],
            [np.array([0]), np.array([1]), np.array([2]), np.array([2])],
        ),
        ([cube, cube, [cube, cube]], [cube, cube, cube, cube]),
        (CubeList([cube, cube, CubeList([cube, cube])]), [cube, cube, cube, cube]),
        ([0, [1, [2, [3]]]], [0, 1, 2, 3]),
        ([0, [1, 2], [3]], [0, 1, 2, 3]),
        (["cat"], ["cat"]),
        ((0, 1, (2, 3)), [0, 1, 2, 3]),
    ),
)
def test_basic(nested, expected):
    """Test flattening an arbitrarily nested iterable."""
    result = flatten(nested)
    assert result == expected
    assert isinstance(result, list)


@pytest.mark.parametrize(
    "nested",
    (
        (0),
        ("cat"),
        ({0: {1: "cat"}, 1: {2: "dog"}}),
    ),
)
def test_exception(nested):
    """Test an exception is raised if inappropriate types
    are provided for flattening."""
    msg = "Expected object of type list or tuple"
    with pytest.raises(ValueError, match=msg):
        flatten(nested)
