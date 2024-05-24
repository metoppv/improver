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

from improver.utilities.flatten import flatten


def get_cube(name) -> Cube:
    """Sets up a cube for testing"""
    return Cube(0, long_name=name)


@pytest.mark.parametrize(
    "nested_iterable, expected",
    [
        # Simple nested lists
        ([1, [2, [3, 4], 5], [6, [7, 8]], 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        # Cubes and CubeLists
        (
            CubeList(
                [get_cube("0"), get_cube("1"), CubeList([get_cube("2"), get_cube("3")])]
            ),
            [get_cube("0"), get_cube("1"), get_cube("2"), get_cube("3")],
        ),
        # Numpy arrays
        (
            [np.array([0]), np.array([1]), [np.array([2]), np.array([2])]],
            [np.array([0]), np.array([1]), np.array([2]), np.array([2])],
        ),
        # Nested tuples
        ((1, (2, (3, 4), 5), (6, (7, 8)), 9), [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        # Nested sets (using frozensets for inner sets) - we sort the output to compare
        ({1, 2, frozenset([3, 4]), frozenset([5, 6]), 7}, [1, 2, 3, 4, 5, 6, 7]),
        # Mixed nested iterables
        ([1, (2, {3, 4}, [5, 6]), 7], [1, 2, 3, 4, 5, 6, 7]),
        # Iterators and generators
        (iter([1, iter([2, iter([3, 4]), 5]), 6]), [1, 2, 3, 4, 5, 6]),
        ((x for x in [1, [2, [3, 4], 5], [6, 7], 8]), [1, 2, 3, 4, 5, 6, 7, 8]),
        # Edge cases
        ([], []),
        ([1], [1]),
        (iter([]), []),
        ({frozenset([])}, []),
        # Non-iterables - just yield the item, don't attempt to flatten
        (42, [42]),
        ("string", ["string"]),
        (b"bytes", [b"bytes"]),
    ],
)
def test_all(nested_iterable, expected):
    res = list(flatten(nested_iterable))
    if isinstance(nested_iterable, (set, frozenset)):
        # Sets are unordered, so we need to sort the output
        res = sorted(res)
        expected = sorted(expected)
    assert res == expected
