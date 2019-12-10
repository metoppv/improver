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
WSPD = "wind_speed_on_pressure_levels"
WDIR = "wind_direction_on_pressure_levels"


def test_basic(tmp_path):
    """Test basic extrapolation nowcast"""
    kgo_path = acc.kgo_root() / "nowcast-extrapolate/extrapolate/kgo.nc"
    input_dir = acc.kgo_root() / "nowcast-optical-flow/basic"
    input_path = input_dir / RAINRATE_NC
    oe_paths = [input_dir / f"20181103T{x}00Z-PT000{y}H00M-{OE}.nc"
                for x, y in ((16, 3), (17, 4))]
    uv_path = input_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            output_path,
            "--max_lead_time", "90",
            "--u_and_v_filepath", uv_path,
            "--orographic_enhancement_filepaths", *oe_paths]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_metadata(tmp_path):
    """Test basic extrapolation nowcast with json metadata"""
    kgo_dir = acc.kgo_root() / "nowcast-extrapolate/extrapolate"
    kgo_path = kgo_dir / "kgo_with_metadata.nc"
    input_dir = acc.kgo_root() / "nowcast-optical-flow/basic"
    input_path = input_dir / RAINRATE_NC
    oe_paths = [input_dir / f"20181103T1600Z-PT0003H00M-{OE}.nc"]
    meta_path = input_dir / "../metadata/precip.json"
    uv_path = input_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            output_path,
            "--json_file", meta_path,
            "--max_lead_time", "30",
            "--u_and_v_filepath", uv_path,
            "--orographic_enhancement_filepaths", *oe_paths]
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
    args = [input_path,
            output_path,
            "--max_lead_time", "30",
            "--u_and_v_filepath", uv_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_model_winds(tmp_path):
    """Test extrapolation using model winds on pressure levels"""
    kgo_dir = acc.kgo_root() / "nowcast-extrapolate/model_winds"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = acc.kgo_root() / "nowcast-optical-flow/basic"
    input_path = input_dir / RAINRATE_NC
    output_path = tmp_path / "output.nc"
    wspd_path = kgo_dir / f"20181103T1600Z-PT0001H00M-{WSPD}.nc"
    wdir_path = kgo_dir / f"20181103T1600Z-PT0001H00M-{WDIR}.nc"
    oe_paths = [input_dir / f"20181103T1600Z-PT0003H00M-{OE}.nc"]
    args = [input_path,
            output_path,
            "--max_lead_time", "30",
            "--advection_speed_filepath", wspd_path,
            "--advection_direction_filepath", wdir_path,
            "--orographic_enhancement_filepaths", *oe_paths]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_invalid_advection(tmp_path):
    """Test invalid advection and UV combination"""
    kgo_dir = acc.kgo_root() / "nowcast-extrapolate/model_winds"
    input_dir = acc.kgo_root() / "nowcast-optical-flow/basic"
    input_path = input_dir / RAINRATE_NC
    output_path = tmp_path / "output.nc"
    wspd_path = kgo_dir / f"20181103T1600Z-PT0001H00M-{WSPD}.nc"
    wdir_path = kgo_dir / f"20181103T1600Z-PT0001H00M-{WDIR}.nc"
    uv_path = input_dir / "kgo.nc"
    args = [input_path,
            output_path,
            "--max_lead_time", "30",
            "--advection_speed_filepath", wspd_path,
            "--advection_direction_filepath", wdir_path,
            "--u_and_v_filepath", uv_path]
    with pytest.raises(ValueError, match=".*mix advection.*"):
        run_cli(args)


def test_unavailable_pressure_level(tmp_path):
    """Test use of a pressure level that is not available"""
    kgo_dir = acc.kgo_root() / "nowcast-extrapolate/model_winds"
    input_dir = acc.kgo_root() / "nowcast-optical-flow/basic"
    input_path = input_dir / RAINRATE_NC
    output_path = tmp_path / "output.nc"
    wspd_path = kgo_dir / f"20181103T1600Z-PT0001H00M-{WSPD}.nc"
    wdir_path = kgo_dir / f"20181103T1600Z-PT0001H00M-{WDIR}.nc"
    oe_paths = [input_dir / f"20181103T1600Z-PT0003H00M-{OE}.nc"]
    args = [input_path,
            output_path,
            "--max_lead_time", "30",
            "--advection_speed_filepath", wspd_path,
            "--advection_direction_filepath", wdir_path,
            "--pressure_level", "1234",
            "--orographic_enhancement_filepaths", *oe_paths]
    with pytest.raises(ValueError, match=".*specified pressure level.*"):
        run_cli(args)


def test_invalid_cubes(tmp_path):
    """Test an invalid amount of cubes"""
    kgo_dir = acc.kgo_root() / "nowcast-extrapolate/model_winds"
    input_dir = acc.kgo_root() / "nowcast-optical-flow/basic"
    input_path = input_dir / RAINRATE_NC
    output_path = tmp_path / "output.nc"
    wdir_path = kgo_dir / f"20181103T1600Z-PT0001H00M-{WDIR}.nc"
    args = [input_path,
            output_path,
            "--max_lead_time", "30",
            "--advection_direction_filepath", wdir_path]
    with pytest.raises(ValueError, match=".*speed and direction.*u and v.*"):
        run_cli(args)
