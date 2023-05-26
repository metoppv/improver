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
"""Tests for the enforce-consistent-forecasts CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "forecast_type, ref_name, additive_amount, multiplicative_amount, comparison_operator",
    (
        (
            "probability",
            "probability_of_cloud_area_fraction_above_threshold",
            "",
            "",
            "<=",
        ),
        ("percentile", "wind_speed", "0.0", "1.1", ">="),
        ("realization", "surface_temperature", "0.0", "0.9", "<="),
    ),
)
def test_enforce_consistent_forecasts(
    tmp_path,
    ref_name,
    forecast_type,
    additive_amount,
    multiplicative_amount,
    comparison_operator,
):
    """
    Test enforcement of consistent forecasts between cubes
    """
    kgo_dir = acc.kgo_root() / "enforce-consistent-forecasts"
    kgo_path = kgo_dir / f"{forecast_type}_kgo.nc"

    forecast = kgo_dir / f"{forecast_type}_forecast.nc"
    reference = kgo_dir / f"{forecast_type}_reference.nc"
    output_path = tmp_path / "output.nc"

    if forecast_type == "probability":
        args = [
            forecast,
            reference,
            "--ref-name",
            ref_name,
            "--comparison-operator",
            comparison_operator,
            "--output",
            output_path,
        ]
    else:
        args = [
            forecast,
            reference,
            "--ref-name",
            ref_name,
            "--additive-amount",
            additive_amount,
            "--multiplicative-amount",
            multiplicative_amount,
            "--comparison-operator",
            comparison_operator,
            "--output",
            output_path,
        ]

    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_too_many_cubes(tmp_path):
    """
    Test to ensure an error is raised if too many cubes are provided.
    """
    kgo_dir = acc.kgo_root() / "enforce-consistent-forecasts"

    forecast = kgo_dir / "probability_forecast.nc"
    reference = kgo_dir / "probability_reference.nc"
    output_path = tmp_path / "output.nc"

    args = [
        forecast,
        forecast,
        reference,
        "--ref-name",
        "probability_of_cloud_area_fraction_above_threshold",
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match="Exactly two cubes"):
        run_cli(args)
