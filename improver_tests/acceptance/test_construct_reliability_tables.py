# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the construct-reliability-tables CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_no_single_value_bins(tmp_path):
    """
    Test construction of reliability tables without the single value lower and
    upper bins at 0 and 1.
    """
    kgo_dir = acc.kgo_root() / "construct-reliability-tables/basic"
    kgo_path = kgo_dir / "kgo_without_single_value_bins.nc"
    history_path = kgo_dir / "forecast*.nc"
    truth_path = kgo_dir / "truth*.nc"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_aggregate(tmp_path):
    """
    Test construction of reliability tables with aggregation over
    latitude/longitude coordinates.
    """
    kgo_dir = acc.kgo_root() / "construct-reliability-tables/basic"
    kgo_path = kgo_dir / "kgo_aggregated.nc"
    history_path = kgo_dir / "forecast*.nc"
    truth_path = kgo_dir / "truth*.nc"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--aggregate-coordinates",
        "latitude,longitude",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_single_value_bins(tmp_path):
    """
    Test construction of reliability tables with the single value lower and
    upper bins at 0 and 1.
    """
    kgo_dir = acc.kgo_root() / "construct-reliability-tables/basic"
    kgo_path = kgo_dir / "kgo_single_value_bins.nc"
    history_path = kgo_dir / "forecast*.nc"
    truth_path = kgo_dir / "truth*.nc"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--single-value-lower-limit",
        "--single-value-upper-limit",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
