# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the VisibilityCombineCloudBase CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("forecast_type", ("spot", "gridded"))
def test_basic(tmp_path, forecast_type):
    """Test combine visibility and cloud bases plugin for spot and gridded forecasts"""
    test_dir = acc.kgo_root() / "visibility-combine-cloud-base"
    output_path = tmp_path / "output.nc"
    args = [
        test_dir / f"{forecast_type}" / "visibility.nc",
        test_dir / f"{forecast_type}" / "cloud_base_ground.nc",
        "--first-unscaled-threshold",
        "5000",
        "--initial-scaling-value",
        "0.6",
        "--output",
        output_path,
    ]
    kgo_path = test_dir / f"{forecast_type}" / "kgo.nc"
    run_cli(args)
    acc.compare(output_path, kgo_path)
