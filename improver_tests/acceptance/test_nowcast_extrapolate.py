# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the nowcast-extrapolate CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

RAINRATE_NC = "201811031600_radar_rainrate_composite_UK_regridded.nc"
OE = "orographic_enhancement_standard_resolution"


def test_optical_flow_inputs(tmp_path):
    """Test extrapolation nowcast using optical flow inputs"""
    pytest.importorskip("pysteps")
    kgo_dir = acc.kgo_root() / "nowcast-extrapolate/extrapolate"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = acc.kgo_root() / "nowcast-extrapolate"
    input_path = input_dir / RAINRATE_NC
    oe_path = input_dir / "orographic_enhancement.nc"
    uv_path = input_dir / "optical_flow_uv.nc"

    output_path = tmp_path / "output.nc"

    args = [
        input_path,
        uv_path,
        oe_path,
        "--max-lead-time",
        "90",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_wind_inputs(tmp_path):
    """Test extrapolation nowcast using wind component inputs"""
    pytest.importorskip("pysteps")
    kgo_dir = acc.kgo_root() / "nowcast-extrapolate/extrapolate"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = acc.kgo_root() / "nowcast-extrapolate"
    input_path = input_dir / RAINRATE_NC
    oe_path = input_dir / "orographic_enhancement.nc"
    uv_path = input_dir / "wind_uv.nc"

    output_path = tmp_path / "output.nc"

    args = [
        input_path,
        uv_path,
        oe_path,
        "--max-lead-time",
        "90",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_metadata(tmp_path):
    """Test basic extrapolation nowcast with json metadata"""
    pytest.importorskip("pysteps")
    kgo_dir = acc.kgo_root() / "nowcast-extrapolate/metadata"
    kgo_path = kgo_dir / "kgo_with_metadata.nc"
    input_dir = acc.kgo_root() / "nowcast-extrapolate"
    input_path = input_dir / RAINRATE_NC
    oe_path = input_dir / "orographic_enhancement.nc"
    meta_path = input_dir / "metadata/precip.json"
    uv_path = input_dir / "optical_flow_uv.nc"

    output_path = tmp_path / "output.nc"

    args = [
        input_path,
        uv_path,
        oe_path,
        "--attributes-config",
        meta_path,
        "--max-lead-time",
        "30",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
