# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the extend-radar-mask CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test radar mask extension"""
    kgo_dir = acc.kgo_root() / "extend-radar-mask/basic"
    kgo_path = kgo_dir / "kgo.nc"
    nimrod_prefix = "201811271330_nimrod_ng_radar"
    rainrate_path = kgo_dir / f"{nimrod_prefix}_rainrate_composite_2km_UK.nc"
    arc_path = kgo_dir / f"{nimrod_prefix}_arc_composite_2km_UK.nc"
    output_path = tmp_path / "output.nc"
    args = [rainrate_path, arc_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
