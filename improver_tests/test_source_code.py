# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Checks on source code files."""

from pathlib import Path

TOP_LEVEL_DIR = (Path(__file__).parent / "..").resolve()
DIRECTORIES_COVERED = [TOP_LEVEL_DIR / "improver", TOP_LEVEL_DIR / "improver_tests"]


def self_licence():
    """Collect licence text from this file"""
    self_lines = Path(__file__).read_text().splitlines()
    licence_lines = list()
    for line in self_lines:
        if not line.startswith("#"):
            break
        licence_lines.append(line)
    licence = "\n".join(licence_lines)
    return licence


def test_py_licence():
    """
    Check that non-empty python files contain 3-clause BSD licence text
    """
    failed_files = []
    licence_text = self_licence()
    for directory in DIRECTORIES_COVERED:
        for file in directory.glob("**/*.py"):
            contents = file.read_text()
            # skip zero-byte empty files such as __init__.py
            if len(contents) > 0 and licence_text not in contents:
                failed_files.append(str(file))
    assert len(failed_files) == 0, "\n".join(failed_files)


def test_init_files_exist():
    """Check for missing __init__.py files."""
    failed_directories = []
    for directory in DIRECTORIES_COVERED:
        for path in directory.glob("**"):
            if not path.is_dir():
                continue
            # ignore hidden directories and their sub-directories
            if any([part.startswith(".") for part in path.parts]):
                continue
            # in-place running will produce pycache directories, these should be ignored
            if path.name == "__pycache__":
                continue
            expected_init = path / "__init__.py"
            if not expected_init.exists():
                failed_directories.append(str(path))
    assert len(failed_directories) == 0, "\n".join(failed_directories)
