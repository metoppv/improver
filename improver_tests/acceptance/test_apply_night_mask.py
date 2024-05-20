# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
