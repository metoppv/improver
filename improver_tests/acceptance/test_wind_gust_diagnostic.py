# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the wind-gust-diagnostic CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_average_wind_gust(tmp_path):
    """Test basic wind gust diagnostic processing"""
    kgo_dir = acc.kgo_root() / "wind-gust-diagnostic/basic"
    kgo_path = kgo_dir / "kgo_average_wind_gust.nc"
    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "wind_gust_perc.nc",
        kgo_dir / "wind_speed_perc.nc",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_extreme_wind_gust(tmp_path):
    """Test basic wind gust diagnostic processing"""
    kgo_dir = acc.kgo_root() / "wind-gust-diagnostic/basic"
    kgo_path = kgo_dir / "kgo_extreme_wind_gust.nc"
    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "wind_gust_perc.nc",
        kgo_dir / "wind_speed_perc.nc",
        "--wind-gust-percentile",
        "95.0",
        "--wind-speed-percentile",
        "100.0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
