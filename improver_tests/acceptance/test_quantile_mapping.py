# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the quantile-mapping CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_floor_no_threshold(tmp_path):
    """Test quantile mapping with floor method and no preservation threshold."""
    kgo_dir = acc.kgo_root() / "quantile-mapping/basic/"
    kgo_path = kgo_dir / "kgo.nc"
    reference_path = acc.kgo_root() / "quantile-mapping/reference.nc"
    forecast_path = acc.kgo_root() / "quantile-mapping/forecast.nc"
    output_path = tmp_path / "output.nc"

    args = [
        reference_path,
        forecast_path,
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_floor_with_threshold(tmp_path):
    """Test quantile mapping with floor method and preservation threshold."""
    kgo_dir = acc.kgo_root() / "quantile-mapping/with_preservation_threshold/"
    kgo_path = kgo_dir / "kgo.nc"
    reference_path = acc.kgo_root() / "quantile-mapping/reference.nc"
    forecast_path = acc.kgo_root() / "quantile-mapping/forecast.nc"
    output_path = tmp_path / "output.nc"

    args = [
        reference_path,
        forecast_path,
        "--preservation-threshold",
        "2.0",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
