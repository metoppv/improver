# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""
Tests for the update-grid-metadata CLI
"""

import shutil

import pytest

from improver.tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic updating"""
    kgo_dir = acc.kgo_root() / "update-grid-metadata/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_no_change(tmp_path):
    """Test updating with no change"""
    kgo_dir = acc.kgo_root() / "update-grid-metadata/basic"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [kgo_path, output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_file_not_updated(tmp_path):
    """Test no change situation not writing to file"""
    # TODO: this test is based on modification time reported by the filesystem
    # resolution will vary on different filesystems and may cause test to fail
    kgo_dir = acc.kgo_root() / "update-grid-metadata/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = tmp_path / "input.nc"
    shutil.copy(kgo_path, input_path)
    before_mtime = input_path.stat().st_mtime
    args = [input_path, input_path]
    run_cli(args)
    after_mtime = input_path.stat().st_mtime
    assert after_mtime == before_mtime


def test_basic_file_updated(tmp_path):
    """Test basic updating causes new file modification time"""
    # TODO: this test is based on modification time reported by the filesystem
    # resolution will vary on different filesystems and may cause test to fail
    kgo_dir = acc.kgo_root() / "update-grid-metadata/basic"
    orig_input_path = kgo_dir / "input.nc"
    input_path = tmp_path / "input.nc"
    shutil.copy(orig_input_path, input_path)
    before_mtime = input_path.stat().st_mtime
    args = [input_path, input_path]
    run_cli(args)
    after_mtime = input_path.stat().st_mtime
    assert after_mtime > before_mtime
