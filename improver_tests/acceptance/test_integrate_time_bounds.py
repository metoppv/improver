# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the integrate_time_bounds CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test conversion of lightning frequency to count."""
    kgo_dir = acc.kgo_root() / "integrate-time-bounds/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "lightning_frequency.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_rename(tmp_path):
    """Test renaming the output diagnostic using the new-name kwarg."""
    kgo_dir = acc.kgo_root() / "integrate-time-bounds/basic"
    kgo_path = kgo_dir / "kgo_renamed.nc"
    input_path = kgo_dir / "lightning_frequency.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--new-name",
        "number_of_lightning_flashes_per_unit_area",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
