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
Tests for the nowcast-optical-flow CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

RADAR_REGRID = "radar_rainrate_composite_UK_regridded"
RADAR_REMASK = "radar_rainrate_remasked_composite_2km_UK"
P_RATE = "lwe_precipitation_rate"
OE = "20181103T1600Z-PT0003H00M-orographic_enhancement_standard_resolution.nc"


@pytest.mark.slow
def test_basic(tmp_path):
    """Test basic optical flow nowcast"""
    kgo_dir = acc.kgo_root() / "nowcast-optical-flow/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [kgo_dir / f"20181103{hhmm}_{RADAR_REGRID}.nc"
                   for hhmm in ("1530", "1545", "1600")]
    oe_path = kgo_dir / OE
    output_path = tmp_path / "output.nc"
    args = [oe_path, *input_paths,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_remasked(tmp_path):
    """Test remasked optical flow"""
    kgo_dir = acc.kgo_root() / "nowcast-optical-flow/remasked"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [kgo_dir / f"20181127{hhmm}_{RADAR_REMASK}.nc"
                   for hhmm in ("1330", "1345", "1400")]
    oe_path = kgo_dir / "20181127T1400Z-PT0004H00M-orographic_enhancement_" \
                        "standard_resolution.nc"
    output_path = tmp_path / "output.nc"
    args = [oe_path, *input_paths,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
