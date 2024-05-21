# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the apply-lapse-rate CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test lapse rate adjustment for a deterministic cube"""
    kgo_dir = acc.kgo_root() / "apply-lapse-rate/basic"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "ukvx_temperature.nc",
        kgo_dir / "ukvx_lapse_rate.nc",
        kgo_dir / "ukvx_orography.nc",
        kgo_dir / "../highres_orog.nc",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_realizations(tmp_path):
    """Test lapse rate adjustment for a cube with realizations"""
    kgo_dir = acc.kgo_root() / "apply-lapse-rate/realizations"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "enukx_temperature.nc",
        kgo_dir / "enukx_lapse_rate.nc",
        kgo_dir / "enukx_orography.nc",
        kgo_dir / "../highres_orog.nc",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
