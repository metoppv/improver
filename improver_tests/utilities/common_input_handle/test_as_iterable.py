# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from improver.utilities.common_input_handle import as_iterable


def test_none():
    """Test that None returns an empty list."""
    result = as_iterable(None)
    assert result == []


def test_non_iterable():
    """Test that a non-iterable object is returned as a list."""
    result = as_iterable(1)
    assert result == [1]


def test_string():
    """Test that a string is returned as a list."""
    result = as_iterable("1")
    assert result == ["1"]


def test_bytes():
    """Test that bytes are returned as a list."""
    result = as_iterable(b"1")
    assert result == [b"1"]


def test_iterable():
    """Test that an iterable object is returned as is."""
    src = tuple(range(1))
    result = as_iterable(src)
    assert result is src
