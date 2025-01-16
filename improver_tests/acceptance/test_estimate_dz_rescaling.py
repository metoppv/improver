# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the estimate-dz-rescaling CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "forecast, truth, kgo",
    (
        (
            "T1200Z-PT0006H00M-wind_speed_at_10m.nc",
            "T1200Z-srfc_wind_sped_spot_truths.nc",
            "T1200Z_kgo.nc",
        ),
        (
            "T1500Z-PT0132H00M-wind_speed_at_10m.nc",
            "T1500Z-srfc_wind_sped_spot_truths.nc",
            "T1500Z_kgo.nc",
        ),
    ),
)
def test_estimate_dz_rescaling(tmp_path, forecast, truth, kgo):
    """Test estimate_dz_rescaling CLI."""
    kgo_dir = acc.kgo_root() / "estimate-dz-rescaling/"
    kgo_path = kgo_dir / kgo
    forecast_path = kgo_dir / forecast
    truth_path = kgo_dir / truth
    neighbour_path = kgo_dir / "neighbour.nc"
    output_path = tmp_path / "output.nc"
    args = [
        forecast_path,
        truth_path,
        neighbour_path,
        "--forecast-period",
        "6",
        "--dz-lower-bound",
        "-550",
        "--dz-upper-bound",
        "550",
        "--land-constraint",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
