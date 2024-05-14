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
"""Bulk checking and updating of checksum file"""

import difflib
import functools
import locale
import os
import pathlib
from contextlib import contextmanager
from typing import Callable, Dict, List

import pytest

from . import acceptance as acc


EXCLUDE_PATTERNS = ["README.md", "LICENSE"]


@contextmanager
def temporary_sort_locale(collate: str) -> Callable[[str, str], bool]:
    """
    Set a temporary locale for sorting.

    Args:
        collate: LC_COLLATE locale name

    Yields:
        String comparison function
    """
    old = locale.getlocale(locale.LC_COLLATE)
    try:
        locale.setlocale(locale.LC_COLLATE, collate)
        yield locale.strcoll
    finally:
        locale.setlocale(locale.LC_COLLATE, old)


def checksums_to_text(
    path_csums: Dict[pathlib.Path, str], sort: bool = False
) -> List[str]:
    """
    Convert checksum dict to sha256sum like format.

    Args:
        path_csums: per-file checksums
        sort: apply sorting to file paths

    Returns:
        List of checksum file lines, no newline at end of each line.
    """
    pathstr_csums = {str(path): csum for path, csum in path_csums.items()}
    if sort:
        # sorting uses C locale to avoid locale-specific variation
        with temporary_sort_locale("C") as strcoll:
            paths = sorted(pathstr_csums.keys(), key=functools.cmp_to_key(strcoll))
    else:
        paths = pathstr_csums.keys()
    lines = [f"{pathstr_csums[path]}  {path}" for path in paths]
    return lines


@pytest.mark.acc
@pytest.mark.checksum
@acc.skip_if_kgo_missing
def test_kgo_checksums():
    """Bulk check of all KGO checksums independent of other tests"""
    kgo_root = acc.kgo_root()
    data_paths = []
    # walk the KGO directory and gather all files and symlinks to files
    for directory, subdirectories, filenames in os.walk(kgo_root, topdown=True):
        # exclude dotfiles such as .git
        subdirectories[:] = [d for d in subdirectories if not d.startswith(".")]
        filenames = [f for f in filenames if not f.startswith(".")]
        for filename in filenames:
            if filename in EXCLUDE_PATTERNS:
                continue
            data_paths.append(pathlib.Path(directory) / filename)
    # generate checksums for all the found files
    path_checksums = {
        dpath.relative_to(kgo_root): acc.calculate_checksum(dpath)
        for dpath in data_paths
    }

    # convert to SHA256SUMS-like text format for comparison and diff output
    expected_text = checksums_to_text(acc.acceptance_checksums(), sort=False)
    actual_text = checksums_to_text(path_checksums, sort=True)
    diff_generator = difflib.unified_diff(
        expected_text,
        actual_text,
        fromfile=str(acc.DEFAULT_CHECKSUM_FILE),
        tofile=str(kgo_root),
        n=1,
        lineterm="",
    )
    print("\n".join(diff_generator))
    assert actual_text == expected_text, (
        f"Files in {kgo_root} don't match checksums in {acc.DEFAULT_CHECKSUM_FILE}"
        " - see diff in stdout for details"
    )


def test_checksums_sorted():
    """
    Test that the checksums file is sorted.

    This test doesn't depend on having the acceptance test data available, so
    can run with the unit tests.
    """
    try:
        csum_paths = [str(path) for path in acc.acceptance_checksums().keys()]
    except FileNotFoundError:
        pytest.skip("no checksum file, likely due to package being installed")
    with temporary_sort_locale("C") as strcoll:
        csum_paths_sorted = sorted(csum_paths, key=functools.cmp_to_key(strcoll))
    assert csum_paths == csum_paths_sorted
