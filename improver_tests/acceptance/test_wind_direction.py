# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the wind-direction CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.slow
def test_basic(tmp_path):
    """Test basic wind direction operation"""
    kgo_dir = acc.kgo_root() / "wind_direction/basic"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [kgo_dir / "input.nc", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_global(tmp_path):
    """Test global wind direction operation"""
    kgo_dir = acc.kgo_root() / "wind_direction/global"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "input.nc",
        "--backup-method=first_realization",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
