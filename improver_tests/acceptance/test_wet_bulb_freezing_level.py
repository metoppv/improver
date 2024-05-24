# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the wet-bulb-freezing-level CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic wet bulb freezing level calculation"""
    kgo_dir = acc.kgo_root() / "wet-bulb-freezing-level"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"

    args = [
        kgo_dir / "wet_bulb_temperature.nc",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
