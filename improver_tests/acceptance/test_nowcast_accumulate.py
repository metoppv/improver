# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests the nowcast-accumulate CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]

CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.slow
def test_optical_flow_inputs(tmp_path):
    """Test creating a nowcast accumulation using optical flow inputs"""
    pytest.importorskip("pysteps")
    kgo_dir = acc.kgo_root() / "nowcast-accumulate/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "201811031600_radar_rainrate_composite_UK_regridded.nc"
    uv_path = kgo_dir / "optical_flow_uv.nc"
    oe_path = kgo_dir / "20181103T1600Z-PT0003H00M-orographic_enhancement.nc"
    output_path = tmp_path / "output.nc"

    args = [
        input_path,
        uv_path,
        oe_path,
        "--max-lead-time",
        "30",
        "--output",
        output_path,
    ]

    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_wind_inputs(tmp_path):
    """Test creating a nowcast accumulation using wind component inputs"""
    pytest.importorskip("pysteps")
    kgo_dir = acc.kgo_root() / "nowcast-accumulate/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "201811031600_radar_rainrate_composite_UK_regridded.nc"
    uv_path = kgo_dir / "wind_uv.nc"
    oe_path = kgo_dir / "20181103T1600Z-PT0003H00M-orographic_enhancement.nc"
    output_path = tmp_path / "output.nc"

    args = [
        input_path,
        uv_path,
        oe_path,
        "--max-lead-time",
        "30",
        "--output",
        output_path,
    ]

    run_cli(args)
    acc.compare(output_path, kgo_path)
