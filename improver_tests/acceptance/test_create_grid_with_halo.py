# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the create-grid-with-halo-points CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic create grid with halo"""
    kgo_dir = acc.kgo_root() / "create-grid-with-halo/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "source_grid.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_halo_size(tmp_path):
    """Test basic create grid with halo"""
    kgo_dir = acc.kgo_root() / "create-grid-with-halo/halo_size"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/source_grid.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--halo-radius", "75000", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
