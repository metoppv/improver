# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the precipitation duration CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_precipitation_duration_basic(tmp_path):
    """Test basic precipitation duration calculation"""
    kgo_dir = acc.kgo_root() / "precipitation_duration/standard_names"
    kgo_path = kgo_dir / "kgo_basic.nc"
    input_cubes = kgo_dir.glob("2025*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *input_cubes,
        "--min-accumulation-per-hour",
        "0.1",
        "--critical-rate",
        "4.0",
        "--target-period",
        "24",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("min_accumulation, critical_rate", [(0.1 / 3, 2.0), (0.1, 4.0)])
def test_different_threshold_parameters(tmp_path, min_accumulation, critical_rate):
    """Test precipitation duration with different parameters"""
    kgo_dir = acc.kgo_root() / "precipitation_duration/standard_names"
    kgo_path = kgo_dir / f"kgo_acc_{min_accumulation:.2f}_rate_{critical_rate:.1f}.nc"
    input_cubes = kgo_dir.glob("2025*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *input_cubes,
        "--min-accumulation-per-hour",
        f"{min_accumulation}",
        "--critical-rate",
        f"{critical_rate}",
        "--target-period",
        "24",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_renamed_diagnostics(tmp_path):
    """Test precipitation duration with different diagnostic names."""
    kgo_dir = acc.kgo_root() / "precipitation_duration/renamed"
    kgo_path = kgo_dir / "kgo.nc"
    input_cubes = kgo_dir.glob("2025*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *input_cubes,
        "--min-accumulation-per-hour",
        "0.1",
        "--critical-rate",
        "4.0",
        "--target-period",
        "24",
        "--rate-diagnostic",
        "probability_of_lwe_kittens_above_threshold",
        "--accumulation-diagnostic",
        "probability_of_lwe_thickness_of_kittens_above_threshold",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)

