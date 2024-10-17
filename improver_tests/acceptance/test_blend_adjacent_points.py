# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the blend-adjacent-points CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic_mean(tmp_path):
    """Test basic blend-adjacent-points usage"""
    kgo_dir = acc.kgo_root() / "blend-adjacent-points/basic_mean"
    kgo_path = kgo_dir / "kgo.nc"
    multi_prob = sorted(kgo_dir.glob("multiple_probabilities_rain_*H.nc"))
    output_path = tmp_path / "output.nc"
    args = [
        "--coordinate",
        "forecast_period",
        "--central-point",
        "2",
        "--units",
        "hours",
        "--width",
        "3",
        *multi_prob,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_time_bounds(tmp_path):
    """Test triangular time blending with matched time bounds"""
    kgo_dir = acc.kgo_root() / "blend-adjacent-points/time_bounds"
    kgo_path = kgo_dir / "kgo.nc"
    multi_prob = sorted(kgo_dir.glob("*wind_gust*.nc"))
    output_path = tmp_path / "output.nc"
    args = [
        "--coordinate",
        "forecast_period",
        "--central-point",
        "4",
        "--units",
        "hours",
        "--width",
        "2",
        *multi_prob,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_mismatched_bounds_ranges(tmp_path):
    """Test triangular time blending with mismatched time bounds"""
    kgo_dir = acc.kgo_root() / "blend-adjacent-points/basic_mean"
    multi_prob = sorted(kgo_dir.glob("multiple_probabilities_rain_*H.nc"))
    output_path = tmp_path / "output.nc"
    args = [
        "--coordinate",
        "forecast_period",
        "--central-point",
        "2",
        "--units",
        "hours",
        "--width",
        "3",
        "--blend-time-using-forecast-period",
        *multi_prob,
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match=".*mismatching time bounds.*"):
        run_cli(args)


def test_mismatched_args(tmp_path):
    """Test triangular time blending with inappropriate arguments"""
    kgo_dir = acc.kgo_root() / "blend-adjacent-points/basic_mean"
    multi_prob = sorted(kgo_dir.glob("multiple_probabilities_rain_*H.nc"))
    output_path = tmp_path / "output.nc"
    args = [
        "--coordinate",
        "model",
        "--central-point",
        "2",
        "--units",
        "None",
        "--width",
        "3",
        "--blend-time-using-forecast-period",
        *multi_prob,
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match=".*blend-time-using-forecast.*"):
        run_cli(args)


def test_time(tmp_path):
    """Test time coordinate blending"""
    kgo_dir = acc.kgo_root() / "blend-adjacent-points/time_bounds"
    multi_prob = sorted(kgo_dir.glob("*wind_gust*.nc"))
    output_path = tmp_path / "output.nc"
    args = [
        "--coordinate",
        "time",
        "--central-point",
        "1536908400",
        "--units",
        "seconds since 1970-01-01 00:00:00",
        "--width",
        "7200",
        *multi_prob,
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match=".*Cannot blend over time.*"):
        run_cli(args)
