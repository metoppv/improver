# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the apply-ensemble-copula-coupling CLI."""

import json
import sys

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_probability_input(tmp_path):
    kgo_dir = acc.kgo_root() / "apply-ensemble-copula-coupling"
    input_path_raw = kgo_dir / "input_raw.nc"
    input_path_calib = kgo_dir / "input_threshold.nc"
    kgo_path = kgo_dir / "kgo_threshold.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path_calib, input_path_raw, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_input(tmp_path):
    kgo_dir = acc.kgo_root() / "apply-ensemble-copula-coupling"
    input_path_raw = kgo_dir / "input_raw.nc"
    input_path_calib = kgo_dir / "input_percentile.nc"
    kgo_path = kgo_dir / "kgo_percentile.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path_calib, input_path_raw, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_invalid_input(tmp_path):
    """Test that a warning is raised when input does not contain a threshold
    or percentile coordinate."""
    kgo_dir = acc.kgo_root() / "apply-ensemble-copula-coupling"
    input_path = kgo_dir / "kgo_threshold.nc"
    args = [input_path, input_path]
    with pytest.raises(ValueError, match="Post-processed forecast must be either"):
        run_cli(args)
