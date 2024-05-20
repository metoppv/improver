# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the feels-like-temp CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic feels like temperature processing"""
    kgo_dir = acc.kgo_root() / "feels_like_temp/ukvx"
    kgo_path = kgo_dir / "kgo.nc"
    params = [
        "temperature_at_screen_level",
        "wind_speed_at_10m",
        "relative_humidity_at_screen_level",
        "pressure_at_mean_sea_level",
    ]
    input_paths = [
        kgo_dir / f"20181121T1200Z-PT0012H00M-{param}.nc" for param in params
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--output",
        output_path,
        "--model-id-attr",
        "mosg__model_configuration",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
