# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
            CubeList([get_cube("0"), get_cube("1"), get_cube("2"), get_cube("3")]),
            CubeList([get_cube("0"), get_cube("1"), get_cube("2"), get_cube("3")]),
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
    """Check permutations of input types and nested structures to flatten."""
    res = list(flatten(nested_iterable))
    if isinstance(nested_iterable, (set, frozenset)):
        # Sets are unordered, so we need to sort the output
        res = sorted(res)
        expected = sorted(expected)
    assert res == expected
