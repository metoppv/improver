# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
