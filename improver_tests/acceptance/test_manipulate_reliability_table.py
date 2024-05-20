# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the manipulate-reliability-table CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_manipulate(tmp_path):
    """
    Test manipulation of a reliability table
    """
    kgo_dir = acc.kgo_root() / "manipulate-reliability-table/basic"
    kgo_path = kgo_dir / "kgo_precip.nc"
    table_path = kgo_dir / "reliability_table_precip.nc"
    output_path = tmp_path / "output.nc"
    args = [table_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_manipulate_minimum_forecast_count(tmp_path):
    """
    Test manipulation of a reliability table with an increased minimum forecast count
    """
    kgo_dir = acc.kgo_root() / "manipulate-reliability-table/basic"
    kgo_path = kgo_dir / "kgo_300_min_count.nc"
    table_path = kgo_dir / "reliability_table_cloud.nc"
    output_path = tmp_path / "output.nc"
    args = [table_path, "--minimum-forecast-count", "300", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_manipulate_point_by_point(tmp_path):
    """
    Test manipulation of a reliability table using point_by_point functionality
    """
    kgo_dir = acc.kgo_root() / "manipulate-reliability-table/point_by_point"
    kgo_path = kgo_dir / "kgo_point_by_point.nc"
    table_path = kgo_dir / "reliability_table_point_by_point.nc"
    output_path = tmp_path / "output.nc"
    args = [table_path, "--point-by-point", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
