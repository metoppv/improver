# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the apply-reliability-calibration CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_calibration(tmp_path):
    """
    Test calibration of a forecast using a reliability calibration table.
    """
    kgo_dir = acc.kgo_root() / "apply-reliability-calibration/basic"
    kgo_path = kgo_dir / "kgo.nc"
    forecast_path = kgo_dir / "forecast.nc"
    table_path = kgo_dir / "collapsed_table.nc"
    output_path = tmp_path / "output.nc"
    args = [forecast_path, table_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_calibration_cubelist_input(tmp_path):
    """
    Test calibration of a forecast using a reliability calibration table input
    as a cubelist with a separate cube for each threshold.
    """
    kgo_dir = acc.kgo_root() / "apply-reliability-calibration/basic"
    kgo_path = kgo_dir / "kgo.nc"
    forecast_path = kgo_dir / "forecast.nc"
    table_path = kgo_dir / "cubelist_table.nc"
    output_path = tmp_path / "output.nc"
    args = [forecast_path, table_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_no_calibration(tmp_path):
    """
    Test applying reliability calibration without a reliability table.
    """
    kgo_dir = acc.kgo_root() / "apply-reliability-calibration/basic"
    forecast_path = kgo_dir / "forecast.nc"
    output_path = tmp_path / "output.nc"
    args = [forecast_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, forecast_path)


def test_calibration_point_by_point(tmp_path):
    """
    Test point by point calibration of a forecast using a reliability
    calibration table.
    """
    kgo_dir = acc.kgo_root() / "apply-reliability-calibration/point_by_point"
    kgo_path = kgo_dir / "kgo_point_by_point.nc"
    forecast_path = kgo_dir / "forecast_point_by_point.nc"
    table_path = kgo_dir / "cubelist_table_point_by_point.nc"
    output_path = tmp_path / "output.nc"
    args = [forecast_path, table_path, True, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
