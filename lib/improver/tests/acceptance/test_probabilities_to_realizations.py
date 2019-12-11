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
Tests for the probabilities-to-realizations CLI
"""

import pytest

from improver.tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
run_cli = acc.run_cli("probabilities-to-realizations")


@pytest.mark.slow
def test_basic(tmp_path):
    """Test rebadging to generate realizations from probabilities"""
    kgo_dir = acc.kgo_root() / "probabilities-to-realizations/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--rebadging", input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_n_realizations(tmp_path):
    """Test specified number of realizations"""
    kgo_dir = acc.kgo_root() / "probabilities-to-realizations/12_realizations"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--rebadging",
            "--no-of-realizations=12",
            input_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_reordering_requires_raw(tmp_path):
    """Test requiring a raw ensemble for reordering"""
    kgo_dir = acc.kgo_root() / "probabilities-to-realizations/basic"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--reordering", input_path, "--output", output_path]
    with pytest.raises(RuntimeError, match=".*raw forecast.*"):
        run_cli(args)


def test_reordering_requires_realization(tmp_path):
    """Test requiring a realization coordinate for reordering"""
    kgo_dir = acc.kgo_root() / "probabilities-to-realizations/basic"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--reordering", input_path,
            input_path,
            "--output", output_path]
    with pytest.raises(RuntimeError, match=".*realization coordinate.*"):
        run_cli(args)


@pytest.mark.slow
def test_basic_reordering(tmp_path):
    """Test reordering to generate realizations from probabilities"""
    kgo_dir = acc.kgo_root() / "probabilities-to-realizations/basic_reordering"
    kgo_path = kgo_dir / "kgo.nc"
    raw_path = kgo_dir / "raw_ens.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--reordering",
            "--random-seed", "0",
            input_path,
            raw_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_rebadging_wrong_args_random(tmp_path):
    """Test wrong random seed argument"""
    kgo_dir = acc.kgo_root() / "probabilities-to-realizations/basic_reordering"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--rebadging",
            input_path,
            "--random-seed", "0",
            "--output", output_path]
    with pytest.raises(RuntimeError):
        run_cli(args)


def test_rebadging_wrong_args_raw(tmp_path):
    """Test wrong raw forecast argument"""
    kgo_dir = acc.kgo_root() / "probabilities-to-realizations/basic_reordering"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--rebadging",
            input_path,
            input_path,
            "--output", output_path]
    with pytest.raises(RuntimeError):
        run_cli(args)


@pytest.mark.slow
def test_reordering_n_realizations(tmp_path):
    """Test reordering with specified number of realizations"""
    kgo_dir = acc.kgo_root() / \
        "probabilities-to-realizations/reordering_6_reals"
    kgo_path = kgo_dir / "kgo.nc"
    raw_path = kgo_dir / "../basic_reordering/raw_ens.nc"
    input_path = kgo_dir / "../basic_reordering/input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--reordering",
            "--random-seed", "0",
            "--no-of-realizations", "6",
            input_path,
            raw_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_reordering_ecc_bounds(tmp_path):
    """Test reordering to generate realizations from probabilities"""
    kgo_dir = acc.kgo_root() / \
        "probabilities-to-realizations/ecc_bounds_warning"
    kgo_path = kgo_dir / "kgo.nc"
    raw_path = kgo_dir / "raw_ens.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--reordering",
            "--ecc-bounds-warning",
            "--random-seed", "0",
            input_path,
            raw_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
