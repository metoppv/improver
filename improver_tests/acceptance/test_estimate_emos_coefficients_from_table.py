# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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

Many of these tests use globs which are expanded by IMPROVER code itself,
rather than by shell glob expansion. There are also a some directory globs
which expand directory names in addition to filenames.
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


@pytest.mark.slow
def test_screen_temperature(tmp_path):
    """
    Test estimate-emos-coefficients-from-table with an example forecast and truth
    table for screen temperature.
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    kgo_path = kgo_dir / "kgo.nc"
    history_path = kgo_dir / "forecast_table"
    truth_path = kgo_dir / "truth_table"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210815T0000Z",
        "--forecast-period",
        "60",
        "--training-length",
        "5",
        "--distribution",
        "norm",
        "--tolerance",
        EST_EMOS_TOL,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


@pytest.mark.slow
def test_daytime_max_temperature(tmp_path):
    """
    Test estimate-emos-coefficients-from-table with an example forecast and truth
    table for a period diagnostic.
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    kgo_path = kgo_dir / "kgo.nc"
    history_path = kgo_dir / "forecast_table"
    truth_path = kgo_dir / "truth_table"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--diagnostic",
        "temperature_at_screen_level_max-daytime",
        "--cycletime",
        "20210815T0000Z",
        "--forecast-period",
        "60",
        "--training-length",
        "5",
        "--distribution",
        "norm",
        "--tolerance",
        EST_EMOS_TOL,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


@pytest.mark.slow
def test_wind_speed(tmp_path):
    """
    Test estimate-emos-coefficients-from-table with an example forecast and truth
    table for wind speed.
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    kgo_path = kgo_dir / "kgo.nc"
    history_path = kgo_dir / "forecast_table"
    truth_path = kgo_dir / "truth_table"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--diagnostic",
        "wind_speed_at_10m",
        "--cycletime",
        "20210815T0000Z",
        "--forecast-period",
        "60",
        "--training-length",
        "5",
        "--distribution",
        "truncnorm",
        "--tolerance",
        EST_EMOS_TOL,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )
