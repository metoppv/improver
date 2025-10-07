# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the vicinity CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("vicinity", ("10000", "50000"))
def test_basic(tmp_path, vicinity):
    """Test application with single radii, two values"""
    kgo_dir = acc.kgo_root() / "vicinity"
    kgo_path = kgo_dir / f"kgo_{vicinity}.nc"
    input_path = kgo_dir / "lightning.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--vicinity", vicinity, "--output", f"{output_path}"]

    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_vicinities(tmp_path):
    """Test application with two vicinity radii provided simultaneously"""
    kgo_dir = acc.kgo_root() / "vicinity"
    kgo_path = kgo_dir / "kgo_multiple_radii.nc"
    input_path = kgo_dir / "lightning.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--vicinity", "10000,20000", "--output", f"{output_path}"]

    run_cli(args)
    acc.compare(output_path, kgo_path)

def test_masked_vicinity(tmp_path):
    """Test application with landmask ancillary"""
    kgo_dir = acc.kgo_root() / "vicinity"
    kgo_path = kgo_dir / "kgo_masked.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path, "--vicinity", "10000", "--output", f"{output_path}"]

    run_cli(args)
    acc.compare(output_path, kgo_path)