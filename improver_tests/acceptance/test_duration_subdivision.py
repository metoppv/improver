# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the duration_subdivision CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test no mask application to duration is split evenly into shorter
    periods."""
    kgo_dir = acc.kgo_root() / "duration-subdivision"
    kgo_path = kgo_dir / "kgo_nomask.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--target-period=10800",
        "--fidelity=900",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_nightmask(tmp_path):
    """Test night mask application to duration so period is unequally split
    to ensure no daylight duration is spread into night time hours."""
    kgo_dir = acc.kgo_root() / "duration-subdivision"
    kgo_path = kgo_dir / "kgo_nightmask.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--target-period=10800",
        "--fidelity=900",
        "--night-mask",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_daymask(tmp_path):
    """Test day mask application to duration. As we are working with a
    sunshine duration diagnostic this effectively shows up the
    disagreements between our night mask and the irradiation limit of the
    radiation scheme used in generating the diagnostic."""
    kgo_dir = acc.kgo_root() / "duration-subdivision"
    kgo_path = kgo_dir / "kgo_daymask.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--target-period=10800",
        "--fidelity=900",
        "--day-mask",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
