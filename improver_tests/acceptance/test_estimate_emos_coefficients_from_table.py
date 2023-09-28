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
Tests for the estimate-emos-coefficients-from-table CLI

"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

for mod in ["fastparquet", "statsmodels"]:
    pytest.importorskip(mod)

# The EMOS estimation tolerance is defined in units of the variable being
# calibrated - not in terms of the EMOS coefficients produced by
# estimate-emos-coefficients and compared against KGOs here.
# See comments and CLI help messages in
# improver/cli/estimate_emos_coefficients.py for more detail.
EST_EMOS_TOLERANCE = 1e-4

# The EMOS coefficients are expected to vary by at most one order of magnitude
# more than the CRPS tolerance specified.
COMPARE_EMOS_TOLERANCE = EST_EMOS_TOLERANCE * 10

# Pre-convert to string for easier use in each test
EST_EMOS_TOL = str(EST_EMOS_TOLERANCE)


@pytest.mark.parametrize(
    "forecast_input,distribution,diagnostic,percentiles,additional_predictor,kgo_name",
    [
        (
            "forecast_table",
            "norm",
            "temperature_at_screen_level",
            "10,20,30,40,50,60,70,80,90",
            None,
            "screen_temperature",
        ),
        (
            "forecast_table",
            "truncnorm",
            "wind_speed_at_10m",
            "20,40,60,80",
            None,
            "wind_speed",
        ),
        (
            "forecast_table_quantiles",
            "norm",
            "temperature_at_screen_level",
            None,
            None,
            "screen_temperature_input_quantiles",
        ),
        (
            "forecast_table",
            "norm",
            "temperature_at_screen_level",
            "10,20,30,40,50,60,70,80,90",
            "altitude.nc",
            "screen_temperature_additional_predictor",
        ),
    ],
)
@pytest.mark.slow
def test_basic(
    tmp_path,
    forecast_input,
    distribution,
    diagnostic,
    percentiles,
    additional_predictor,
    kgo_name,
):
    """
    Test estimate-emos-coefficients-from-table with an example forecast and truth
    table for screen temperature and wind speed.
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    kgo_path = kgo_dir / f"{kgo_name}_kgo.nc"
    history_path = kgo_dir / forecast_input
    truth_path = kgo_dir / "truth_table"
    output_path = tmp_path / "output.nc"
    compulsory_args = [history_path, truth_path]
    named_args = [
        "--diagnostic",
        diagnostic,
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--distribution",
        distribution,
        "--tolerance",
        EST_EMOS_TOL,
        "--output",
        output_path,
    ]
    if additional_predictor:
        compulsory_args += [kgo_dir / additional_predictor]
    if percentiles:
        named_args += ["--percentiles", percentiles]
    run_cli(compulsory_args + named_args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


@pytest.mark.slow
def test_invalid_truth_filter(
    tmp_path,
):
    """
    Test using an invalid diagnostic name to filter the truth table.
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = kgo_dir / "forecast_table"
    truth_path = kgo_dir / "truth_table_misnamed_diagnostic"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--distribution",
        "norm",
        "--tolerance",
        EST_EMOS_TOL,
        "--output",
        output_path,
    ]
    with pytest.raises(
        IOError, match="The requested filepath.*temperature_at_screen_level.*"
    ):
        run_cli(args)


@pytest.mark.slow
def test_return_none(
    tmp_path,
):
    """
    Test that None is returned if a non-existent forecast period is requested.
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = kgo_dir / "forecast_table_quantiles"
    truth_path = kgo_dir / "truth_table"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "3600000",
        "--training-length",
        "5",
        "--distribution",
        "norm",
        "--tolerance",
        EST_EMOS_TOL,
        "--output",
        output_path,
    ]
    assert run_cli(args) is None
