# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
import os
import pathlib

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]


@pytest.mark.checksum
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
            data_paths.append(pathlib.Path(directory) / filename)
    # generate checksums for all the found files
    path_checksums = {
        dpath.relative_to(kgo_root): acc.calculate_checksum(dpath)
        for dpath in data_paths
    }

    def format_checksums(path_csums):
        lines = [f"{path_csums[path]}  {path}" for path in sorted(path_csums.keys())]
        return lines

    # convert to SHA256SUMS-like text format for comparison and diff output
    expected_text = format_checksums(acc.acceptance_checksums())
    actual_text = format_checksums(path_checksums)
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
