# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
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
