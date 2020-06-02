# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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

import os
import pathlib

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]


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
    try:
        # check that all the data files are in the list
        assert len(data_paths) == len(acc.acceptance_checksums())
        # check each file's checksum
        for path in data_paths:
            acc.verify_checksum(path)
    except Exception as e:
        if acc.kgo_recreate():
            recreate_checksum_file(data_paths)
        raise e


def recreate_checksum_file(kgo_paths, checksum_path=None):
    """
    Recreate the KGO checksum file.
    The checksum file is in plain text format as produced by the sha256sum
    tool, with paths relative to the KGO root directory.

    Args:
        kgo_paths (Sequence[pathlib.Path]): Absolute path to each KGO data
            file to include in the checksum file. Paths should be inside
            the KGO root directory.
        checksum_path (Optional[pathlib.Path]): Path to checksum file.
            Default is provided by DEFAULT_CHECKSUM_FILE constant.
    """
    if checksum_path is None:
        checksum_path = acc.DEFAULT_CHECKSUM_FILE
    kgo_root = acc.kgo_root()
    new_checksum_lines = []
    for path in sorted(kgo_paths):
        checksum = acc.calculate_checksum(path)
        rel_path = path.relative_to(kgo_root)
        new_checksum_lines.append(f"{checksum}  ./{rel_path}\n")
    with open(checksum_path, mode="w") as checksum_file:
            checksum_file.writelines(new_checksum_lines)
    print(f"Checksum file {checksum_path} recreated")
    print("This test and any others with checksum failures should now pass when re-run")
