# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the example_data_path function."""

from pathlib import Path

import pytest

from improver import example_data_path


def test_example_data_path_success(monkeypatch):
    """Test that example_data_path returns the correct path when
    improver_example_data is available."""
    # Setup mock
    mock_path = Path("/mock/path/to/data")
    mock_module = type("module", (), {"path": mock_path})()
    monkeypatch.setattr("improver.improver_example_data", mock_module)

    # Call function
    result = example_data_path("threshold", "file.nc")

    # Assertions
    assert result == mock_path / "threshold" / "file.nc"


def test_example_data_path_single_argument(monkeypatch):
    """Test example_data_path with a single path component."""
    mock_path = Path("/mock/path/to/data")
    mock_module = type("module", (), {"path": mock_path})()
    monkeypatch.setattr("improver.improver_example_data", mock_module)

    result = example_data_path("data")

    assert result == mock_path / "data"


def test_example_data_path_multiple_arguments(monkeypatch):
    """Test example_data_path with multiple path components."""
    mock_path = Path("/mock/path/to/data")
    mock_module = type("module", (), {"path": mock_path})()
    monkeypatch.setattr("improver.improver_example_data", mock_module)

    result = example_data_path("sub", "dir", "file.nc")

    assert result == mock_path / "sub" / "dir" / "file.nc"


def test_example_data_path_not_installed(monkeypatch):
    """Test that example_data_path raises ImportError when
    improver_example_data is not available."""
    monkeypatch.setattr("improver.improver_example_data", None)

    with pytest.raises(
        ImportError,
        match="Please install the 'improver_example_data' package to access example data.",
    ):
        example_data_path("some", "path")


def test_example_data_path_no_arguments(monkeypatch):
    """Test example_data_path with no path components returns base path."""
    mock_path = Path("/mock/path/to/base")
    mock_module = type("module", (), {"path": mock_path})()
    monkeypatch.setattr("improver.improver_example_data", mock_module)

    result = example_data_path()

    assert result == mock_path
