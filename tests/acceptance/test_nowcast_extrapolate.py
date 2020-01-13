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
Tests for the nowcast-extrapolate CLI
"""

import pytest

from improver.tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

RAINRATE_NC = "201811031600_radar_rainrate_composite_UK_regridded.nc"
OE = "orographic_enhancement_standard_resolution"


def test_basic(tmp_path):
    """Test basic extrapolation nowcast"""
    kgo_dir = acc.kgo_root() / "nowcast-extrapolate/extrapolate"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = acc.kgo_root() / "nowcast-optical-flow/basic"
    input_path = input_dir / RAINRATE_NC
    oe_path = kgo_dir / "orographic_enhancement.nc"
    uv_path = input_dir / "kgo.nc"

    output_path = tmp_path / "output.nc"

    args = [input_path, uv_path, oe_path,
            "--max-lead-time", "90",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_metadata(tmp_path):
    """Test basic extrapolation nowcast with json metadata"""
    kgo_dir = acc.kgo_root() / "nowcast-extrapolate/extrapolate"
    kgo_path = kgo_dir / "kgo_with_metadata.nc"
    input_dir = acc.kgo_root() / "nowcast-optical-flow/basic"
    input_path = input_dir / RAINRATE_NC
    oe_path = input_dir / f"20181103T1600Z-PT0003H00M-{OE}.nc"
    meta_path = input_dir / "../metadata/precip.json"
    uv_path = input_dir / "kgo.nc"

    output_path = tmp_path / "output.nc"

    args = [input_path, uv_path, oe_path,
            "--attributes-config", meta_path,
            "--max-lead-time", "30",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_basic_no_orographic(tmp_path):
    """Test basic extrapolation nowcast without orographic enhancement"""
    kgo_path = (acc.kgo_root() / "nowcast-extrapolate" /
                "extrapolate_no_orographic_enhancement/kgo.nc")
    input_dir = (acc.kgo_root() / "nowcast-optical-flow" /
                 "basic_no_orographic_enhancement")
    input_path = input_dir / RAINRATE_NC
    uv_path = input_dir / "../basic/kgo.nc"

    output_path = tmp_path / "output.nc"

    args = [input_path, uv_path,
            "--max-lead-time", "30",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
