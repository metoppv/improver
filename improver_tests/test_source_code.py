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
"""Checks on source code files."""

from pathlib import Path

TOP_LEVEL_DIR = (Path(__file__).parent / "..").resolve()
DIRECTORIES_COVERED = [TOP_LEVEL_DIR / "improver", TOP_LEVEL_DIR / "improver_tests"]
EXCLUDED_DIRECTORIES = [TOP_LEVEL_DIR / "improver_tests" / "acceptance" / "resources"]


def _has_common_route_with_any(path1, paths):
    for path2 in paths:
        try:
            # Attempt to get the relative path from path1 to path2
            path1.relative_to(path2)
            return True
        except ValueError:
            # No common route found, continue to the next path
            continue
    return False


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
    Check that non-empty python files contain the utf8 header and
    3-clause BSD licence text
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
            if _has_common_route_with_any(path, EXCLUDED_DIRECTORIES):
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
