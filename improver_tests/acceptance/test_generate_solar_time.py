# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the generate-solar-time CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test generation of local solar time derived field."""
    kgo_dir = acc.kgo_root() / "generate-solar-time"
    kgo_path = kgo_dir / "basic" / "kgo.nc"
    input_path = kgo_dir / "surface_altitude.nc"  # Use this as target_grid
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--time",
        "20220607T0000Z",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_new_title_attribute(tmp_path):
    """Test new-title attribute set correctly when generating local solar
    time derived field."""
    kgo_dir = acc.kgo_root() / "generate-solar-time"
    kgo_path = kgo_dir / "new_title_attribute" / "kgo.nc"
    input_path = kgo_dir / "surface_altitude.nc"  # Use this as target_grid
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--time",
        "20220607T0000Z",
        "--new-title",
        "IMPROVER ancillary on Australia 9.6 km Albers Grid",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
