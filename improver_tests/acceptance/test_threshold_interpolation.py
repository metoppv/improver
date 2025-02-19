# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Test for the threshold interpolation CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic invocation with threshold argument"""
    thresholds = "50.0,200.0,400.0,600.0," "1000.0,2000.0," "10000.0,25000.0," "40000.0"
    kgo_dir = acc.kgo_root() / "threshold-interpolation"
    kgo_path = kgo_dir / "extra_thresholds_kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--thresholds", thresholds, "--output", f"{output_path}"]

    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_realization_collapse(tmp_path):
    """Test realization coordinate is collapsed"""
    thresholds = "50.0,200.0,400.0,600.0," "1000.0,2000.0," "10000.0,25000.0," "40000.0"
    kgo_dir = acc.kgo_root() / "threshold-interpolation"
    kgo_path = kgo_dir / "realization_collapse_kgo.nc"
    input_path = kgo_dir / "input_realization.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--thresholds", thresholds, "--output", f"{output_path}"]

    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_masked_cube(tmp_path):
    """Test masked cube"""
    thresholds = "50.0,200.0,400.0,600.0," "1000.0,2000.0," "10000.0,25000.0," "40000.0"
    kgo_dir = acc.kgo_root() / "threshold-interpolation"
    kgo_path = kgo_dir / "masked_cube_kgo.nc"
    input_path = kgo_dir / "masked_input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--thresholds", thresholds, "--output", f"{output_path}"]

    run_cli(args)
    acc.compare(output_path, kgo_path)
