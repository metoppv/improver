# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
OE = "orographic_enhancement_standard_resolution"


@pytest.mark.slow
def test_basic(tmp_path):
    """Test basic optical flow nowcast"""
    kgo_dir = acc.kgo_root() / "nowcast-optical-flow/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"20181103{hhmm}_{RADAR_REGRID}.nc"
        for hhmm in ("1530", "1545", "1600")
    ]
    oe_path = kgo_dir / f"20181103T1600Z-PT0003H00M-{OE}.nc"
    output_path = tmp_path / "output.nc"
    args = [oe_path, *input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_remasked(tmp_path):
    """Test remasked optical flow"""
    kgo_dir = acc.kgo_root() / "nowcast-optical-flow/remasked"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"20181127{hhmm}_{RADAR_REMASK}.nc"
        for hhmm in ("1330", "1345", "1400")
    ]
    oe_path = kgo_dir / f"20181127T1400Z-PT0004H00M-{OE}.nc"
    output_path = tmp_path / "output.nc"
    args = [oe_path, *input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
