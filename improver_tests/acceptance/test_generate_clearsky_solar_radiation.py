# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the generate-clearsky-solar-radiation CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic generation of clearsky solar radiation derived field."""
    kgo_dir = acc.kgo_root() / "generate-clearsky-solar-radiation"
    kgo_path = kgo_dir / "basic" / "kgo.nc"
    input_path = kgo_dir / "surface_altitude.nc"  # Use this as target_grid
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--time",
        "20220506T0000Z",
        "--accumulation-period",
        "24",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_with_altitude_and_lt(tmp_path):
    """Test generation of clearsky solar radiation derived field with input
    surface_altitude and linke_turbidity."""
    kgo_dir = acc.kgo_root() / "generate-clearsky-solar-radiation"
    kgo_path = kgo_dir / "with_altitude_and_lt" / "kgo.nc"
    input_path = kgo_dir / "surface_altitude.nc"  # Use this as target_grid
    surface_altitude_path = kgo_dir / "surface_altitude.nc"
    linke_turbidity_path = kgo_dir / "linke_turbidity.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        surface_altitude_path,
        linke_turbidity_path,
        "--time",
        "20220506T0000Z",
        "--accumulation-period",
        "24",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_temporal_spacing(tmp_path):
    """Test generation of clearsky solar radiation derived field with input
    temporal-spacing."""
    kgo_dir = acc.kgo_root() / "generate-clearsky-solar-radiation"
    kgo_path = kgo_dir / "basic" / "kgo.nc"
    input_path = kgo_dir / "surface_altitude.nc"  # Use this as target_grid
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--time",
        "20220506T0000Z",
        "--accumulation-period",
        "24",
        "--temporal-spacing",
        "60",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=0.005)


def test_new_title_attribute(tmp_path):
    """Test new-title attribute set correctly when generating clearsky solar
    radiation derived field."""
    kgo_dir = acc.kgo_root() / "generate-clearsky-solar-radiation"
    kgo_path = kgo_dir / "new_title_attribute" / "kgo.nc"
    input_path = kgo_dir / "surface_altitude.nc"  # Use this as target_grid
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--time",
        "20220506T0000Z",
        "--accumulation-period",
        "24",
        "--new-title",
        "IMPROVER ancillary on Australia 9.6 km Albers Grid",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
