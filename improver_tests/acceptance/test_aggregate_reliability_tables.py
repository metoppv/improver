# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the aggregate-reliability-tables CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_coordinate_collapse(tmp_path):
    """
    Test aggregation of values by collapsing spatial coordinates.
    """
    kgo_dir = acc.kgo_root() / "aggregate-reliability-tables/basic"
    kgo_path = kgo_dir / "collapse_lat_lon_kgo.nc"
    input_path = kgo_dir / "reliability_table.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--coordinates", "latitude,longitude", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_tables(tmp_path):
    """
    Test aggregation of multiple reliability calibration tables.
    """
    kgo_dir = acc.kgo_root() / "aggregate-reliability-tables/basic"
    kgo_path = kgo_dir / "multiple_tables_kgo.nc"
    input_paths = sorted(kgo_dir.glob("reliability_table*.nc"))
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
