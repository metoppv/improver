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
    kgo_path = kgo_dir / "kgo_50000_masked.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path, "--vicinity", "50000", "--output", f"{output_path}"]

    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("operator", ("max", "min", "mean", "std"))
def test_different_vicinity_operators(tmp_path, operator):
    kgo_dir = acc.kgo_root() / "vicinity"
    kgo_path = kgo_dir / "operator" / f"kgo_{operator}.nc"
    input_path = kgo_dir / "lightning.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--vicinity",
        "20000",
        "--operator",
        operator,
        "--output",
        f"{output_path}",
    ]

    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_invalid_vicinity_operator(tmp_path):
    """Test application with single radii, two values"""
    kgo_dir = acc.kgo_root() / "vicinity"
    input_path = kgo_dir / "lightning.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--vicinity",
        "20000",
        "--operator",
        "mode",
        "--output",
        f"{output_path}",
    ]

    with pytest.raises(ValueError, match="Unsupported operator.*"):
        run_cli(args)


def test_vicinity_cube_rename(tmp_path):
    kgo_dir = acc.kgo_root() / "vicinity"
    kgo_path = kgo_dir / "kgo_new_name.nc"
    input_path = kgo_dir / "lightning.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--vicinity",
        "20000",
        "--operator",
        "mean",
        "--new-name",
        "mean_probability_in_vicinity_of_lightning_flash_density_above_threshold",
        "--output",
        f"{output_path}",
    ]

    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_no_cell_method(tmp_path):
    """Test exclusion of vicinity cell_method when apply_cell_method=False"""
    kgo_dir = acc.kgo_root() / "vicinity"
    kgo_path = kgo_dir / f"kgo_10000_no_cell_method.nc"
    input_path = kgo_dir / "lightning.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--vicinity", "10000", "--apply-cell-method", "False", "--output", f"{output_path}"]

    run_cli(args)
    acc.compare(output_path, kgo_path)