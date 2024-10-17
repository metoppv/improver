# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the extract CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_1d_table(tmp_path):
    """Test extracting values from a table when the table only has one dimension"""
    kgo_dir = acc.kgo_root() / "extract-from-table"
    kgo_path = kgo_dir / "kgo_lapse_class.nc"
    row_path = kgo_dir / "lapse_rate.nc"
    column_path = (
        kgo_dir / "wind_speed_800m.nc"
    )  # This could be any cube as the table only has one dimension
    table = kgo_dir / "table_1d.json"

    output_path = tmp_path / "output.nc"
    args = [
        row_path,
        column_path,
        "--table",
        table,
        "--row-name",
        "lapse_rate",
        "--new-name",
        "lapse_class",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_2d_table(tmp_path):
    """Test extracting values from a table when the table only has two dimensions"""
    kgo_dir = acc.kgo_root() / "extract-from-table"
    kgo_path = kgo_dir / "kgo_gust_ratio.nc"
    row_path = kgo_dir / "lapse_class.nc"
    column_path = kgo_dir / "wind_speed_800m.nc"
    table = kgo_dir / "table_2d.json"

    output_path = tmp_path / "output.nc"
    args = [
        row_path,
        column_path,
        "--table",
        table,
        "--row-name",
        "lapse_class",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
