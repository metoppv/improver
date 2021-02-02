# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Tests for the nowcast-optical-flow-from-winds CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

RADAR_EXT = "u1096_ng_radar_precip_ratecomposite_2km"


def test_basic(tmp_path):
    """Test optical flow calculation by perturbing model winds"""
    pytest.importorskip("pysteps")
    kgo_dir = acc.kgo_root() / "nowcast-optical-flow-from-winds"
    kgo_path = kgo_dir / "kgo_15min.nc"
    input_paths = [
        kgo_dir / f"20190101T{hhmm}Z-{RADAR_EXT}.nc" for hhmm in ("0645", "0700")
    ]
    flow_path = (
        kgo_dir / "20190101T0700Z-PT0000H00M-wind_components_on_pressure_levels.nc"
    )
    oe_path = kgo_dir / "20190101T0700Z-PT0000H00M-orographic_enhancement.nc"
    output_path = tmp_path / "output.nc"
    args = [flow_path, oe_path, *input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_longer_interval(tmp_path):
    pytest.importorskip("pysteps")
    """Test optical flow calculation by perturbing model winds over a 30 minute
    time interval"""
    kgo_dir = acc.kgo_root() / "nowcast-optical-flow-from-winds"
    kgo_path = kgo_dir / "kgo_30min.nc"
    input_paths = [
        kgo_dir / f"20190101T{hhmm}Z-{RADAR_EXT}.nc" for hhmm in ("0630", "0700")
    ]
    flow_path = (
        kgo_dir / "20190101T0700Z-PT0000H00M-wind_components_on_pressure_levels.nc"
    )
    oe_path = kgo_dir / "20190101T0700Z-PT0000H00M-orographic_enhancement.nc"
    output_path = tmp_path / "output.nc"
    args = [flow_path, oe_path, *input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_too_many_inputs(tmp_path):
    """Test an error is thrown if too many radar cubes are provided"""
    kgo_dir = acc.kgo_root() / "nowcast-optical-flow-from-winds"
    input_paths = [
        kgo_dir / f"20190101T{hhmm}Z-{RADAR_EXT}.nc"
        for hhmm in ("0630", "0645", "0700")
    ]
    flow_path = (
        kgo_dir / "20190101T0700Z-PT0000H00M-wind_components_on_pressure_levels.nc"
    )
    oe_path = kgo_dir / "20190101T0700Z-PT0000H00M-orographic_enhancement.nc"
    output_path = tmp_path / "output.nc"
    args = [flow_path, oe_path, *input_paths, "--output", output_path]
    with pytest.raises(ValueError, match="Expected 2 radar cubes - got 3"):
        run_cli(args)
