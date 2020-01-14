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
"""Tests for the percentile CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
run_cli = acc.run_cli("percentile")


def test_basic(tmp_path):
    """Test basic percentile processing"""
    kgo_dir = acc.kgo_root() / "percentile/basic"
    kgo_path = kgo_dir / "kgo.nc"
    perc_input = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [perc_input, "--output", output_path,
            "--coordinates", "realization",
            "--percentiles", "25.0,50,75.0"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
@pytest.mark.parametrize("count", ("single", "multi"))
def test_probconvert(tmp_path, count):
    """Test probability conversion"""
    kgo_dir = acc.kgo_root() / "percentile/probability_convert"
    kgo_path = kgo_dir / f"{count}_realization_kgo.nc"
    prob_input = kgo_dir / f"{count}_realization.nc"
    output_path = tmp_path / "output.nc"
    args = [prob_input, "--output", output_path,
            "--coordinates", "realization",
            "--percentiles", "25,50,75"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_num_percentiles(tmp_path):
    """Test setting number of percentiles"""
    kgo_dir = acc.kgo_root() / "percentile/basic"
    kgo_path = kgo_dir / "kgo.nc"
    perc_input = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [perc_input, "--output", output_path,
            "--coordinates", "realization",
            "--percentiles-count", "3"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_masked(tmp_path):
    """Test use of a mask"""
    kgo_dir = acc.kgo_root() / "percentile/basic"
    kgo_path = kgo_dir / "kgo.nc"
    perc_input = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [perc_input, "--output", output_path,
            "--coordinates", "realization",
            "--percentiles", "25.0,50,75.0"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_eccbounds(tmp_path):
    """Test ECC bounds warning option"""
    kgo_dir = acc.kgo_root() / "percentile/ecc_bounds_warning"
    kgo_path = kgo_dir / "kgo.nc"
    perc_input = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [perc_input, "--output", output_path,
            "--coordinates", "realization",
            "--percentiles", "25,50,75",
            "--ignore-ecc-bounds"]
    run_cli(args)
    acc.compare(output_path, kgo_path)
