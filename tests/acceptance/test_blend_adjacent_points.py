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
    args = ["--coordinate", "forecast_period",
            "--central-point", "2",
            "--units", "hours",
            "--width", "3",
            *multi_prob,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_time_bounds(tmp_path):
    """Test triangular time blending with matched time bounds"""
    kgo_dir = acc.kgo_root() / "blend-adjacent-points/time_bounds"
    kgo_path = kgo_dir / "kgo.nc"
    multi_prob = sorted(kgo_dir.glob("*wind_gust*.nc"))
    output_path = tmp_path / "output.nc"
    args = ["--coordinate", "forecast_period",
            "--central-point", "4",
            "--units", "hours",
            "--width", "2",
            *multi_prob,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_mismatched_bounds_ranges(tmp_path):
    """Test triangular time blending with mismatched time bounds"""
    kgo_dir = acc.kgo_root() / "blend-adjacent-points/basic_mean"
    multi_prob = sorted(kgo_dir.glob("multiple_probabilities_rain_*H.nc"))
    output_path = tmp_path / "output.nc"
    args = ["--coordinate", "forecast_period",
            "--central-point", "2",
            "--units", "hours",
            "--width", "3",
            "--blend-time-using-forecast-period",
            *multi_prob,
            "--output", output_path]
    with pytest.raises(ValueError, match=".*mismatching time bounds.*"):
        run_cli(args)


def test_mismatched_args(tmp_path):
    """Test triangular time blending with inappropriate arguments"""
    kgo_dir = acc.kgo_root() / "blend-adjacent-points/basic_mean"
    multi_prob = sorted(kgo_dir.glob("multiple_probabilities_rain_*H.nc"))
    output_path = tmp_path / "output.nc"
    args = ["--coordinate", "model",
            "--central-point", "2",
            "--units", "None",
            "--width", "3",
            "--blend-time-using-forecast-period",
            *multi_prob,
            "--output", output_path]
    with pytest.raises(ValueError, match=".*blend-time-using-forecast.*"):
        run_cli(args)


def test_time(tmp_path):
    """Test time coordinate blending"""
    kgo_dir = acc.kgo_root() / "blend-adjacent-points/time_bounds"
    multi_prob = sorted(kgo_dir.glob("*wind_gust*.nc"))
    output_path = tmp_path / "output.nc"
    args = ["--coordinate", "time",
            "--central-point", "1536908400",
            "--units", "seconds since 1970-01-01 00:00:00",
            "--width", "7200",
            *multi_prob,
            "--output", output_path]
    with pytest.raises(ValueError, match=".*Cannot blend over time.*"):
        run_cli(args)
