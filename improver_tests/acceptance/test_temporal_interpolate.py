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
Tests for the temporal-interpolate CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
T2M = "temperature_at_screen_level"
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic interpolation"""
    kgo_dir = acc.kgo_root() / "temporal-interpolate/basic"
    kgo_path = kgo_dir / "kgo_t1.nc"
    input_paths = [kgo_dir / f"20190116T{v:04}Z-PT{l:04}H00M-{T2M}.nc"
                   for v, l in ((900, 33), (1200, 36))]
    output_path = tmp_path / "output.nc"
    args = [*input_paths,
            "--times", "20190116T1000Z",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_linear_interval(tmp_path):
    """Test linear time intervals"""
    kgo_dir = acc.kgo_root() / "temporal-interpolate/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [kgo_dir / f"20190116T{v:04}Z-PT{l:04}H00M-{T2M}.nc"
                   for v, l in ((900, 33), (1200, 36))]
    output_path = tmp_path / "output.nc"
    args = [*input_paths,
            "--interval-in-mins", "60",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_solar_uv_index(tmp_path):
    """Test interpolation of UV index"""
    kgo_dir = acc.kgo_root() / "temporal-interpolate/uv"
    kgo_path = kgo_dir / "kgo_t1.nc"
    input_paths = [kgo_dir / f"20181220T{v:04}Z-PT{l:04}H00M-uv_index.nc"
                   for v, l in ((900, 21), (1200, 24))]
    output_path = tmp_path / "output.nc"
    args = [*input_paths,
            "--times", "20181220T1000Z",
            "--interpolation-method", "solar",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_daynight(tmp_path):
    """Test interpolation of UV index with day/night method"""
    kgo_dir = acc.kgo_root() / "temporal-interpolate/uv"
    kgo_path = kgo_dir / "kgo_t1_daynight.nc"
    input_paths = [kgo_dir / f"20181220T{v:04}Z-PT{l:04}H00M-uv_index.nc"
                   for v, l in ((900, 21), (1200, 24))]
    output_path = tmp_path / "output.nc"
    args = [*input_paths,
            "--times", "20181220T1000Z",
            "--interpolation-method", "daynight",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
