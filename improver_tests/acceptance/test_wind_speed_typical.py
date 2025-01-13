# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the wind-speed-typical CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_wind_speed_typical(tmp_path):
    """Test basic typical wind speed processing"""
    kgo_dir = acc.kgo_root() / "wind-speed-typical"
    kgo_path = kgo_dir / "kgo_typical_wind.nc"
    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "limited_cube.nc",
        "--output",
        output_path,
    ]    
    run_cli(args)
    acc.compare(output_path, kgo_path)
