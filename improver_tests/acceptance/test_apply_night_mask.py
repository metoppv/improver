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
"""Tests for the apply-night-mask CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic_uk(tmp_path):
    """Test UK apply night mask operation (multiple realizations)"""
    kgo_dir = acc.kgo_root() / "apply-night-mask/uk_basic"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [kgo_dir / "input.nc", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_basic_global(tmp_path):
    """Test global apply night mask operation (multiple realizations)"""
    kgo_dir = acc.kgo_root() / "apply-night-mask/global_basic"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [kgo_dir / "input.nc", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_uk_prob_above(tmp_path):
    """Test apply night mask operation to probabilities above threshold"""
    kgo_dir = acc.kgo_root() / "apply-night-mask/uk_prob"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [kgo_dir / "valid_input.nc", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_uk_prob_below(tmp_path):
    """Test error raised for probabilities below threshold"""
    kgo_dir = acc.kgo_root() / "apply-night-mask/uk_prob"
    output_path = tmp_path / "output.nc"
    args = [kgo_dir / "invalid_input.nc", "--output", output_path]
    with pytest.raises(ValueError):
        run_cli(args)
