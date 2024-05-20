# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the between-thresholds CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic between-thresholds usage"""
    kgo_dir = acc.kgo_root() / "between-thresholds"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "input.nc",
        "--threshold-ranges",
        kgo_dir / "threshold_ranges_m.json",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_units(tmp_path):
    """Test between-thresholds with different units"""
    kgo_dir = acc.kgo_root() / "between-thresholds"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "input.nc",
        "--threshold-ranges",
        kgo_dir / "threshold_ranges_km.json",
        "--threshold-units",
        "km",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
