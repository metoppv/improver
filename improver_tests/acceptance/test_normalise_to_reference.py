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
"""Tests for the normalise-to-reference CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "forecast_type,ignore_zero_total",
    (("probability", False), ("percentile", False), ("percentile", True)),
)
def test_normalise_to_reference(
    tmp_path,
    forecast_type,
    ignore_zero_total,
):
    """
    Test input cubes are updated correctly so that their total equals the reference cube
    """
    kgo_dir = acc.kgo_root() / f"normalise-to-reference/{forecast_type}"
    kgo_path = kgo_dir / "kgo.nc"

    output_path = tmp_path / "output.nc"

    if forecast_type == "probability":
        inputs = kgo_dir / "*acc.nc"
        reference_name = (
            "probability_of_lwe_thickness_of_precipitation_amount_above_threshold"
        )
        return_name = "probability_of_thickness_of_rainfall_amount_above_threshold"
    else:
        inputs = kgo_dir / "*rate.nc"
        reference_name = "lwe_precipitation_rate"
        return_name = "rainfall_rate"

    args = [
        inputs,
        "--reference-name",
        reference_name,
        "--return-name",
        return_name,
        "--output",
        output_path,
    ]
    if forecast_type == "probability":
        run_cli(args)
        acc.compare(output_path, kgo_path)
    elif ignore_zero_total:
        args.append("--ignore-zero-total")
        run_cli(args)
        acc.compare(output_path, kgo_path)
    else:
        with pytest.raises(
            ValueError, match="There are instances where the total of input"
        ):
            run_cli(args)


def test_incorrect_reference(tmp_path):
    """
    Test correct error is raised when incorrect number of reference cubes are found.
    """
    kgo_dir = acc.kgo_root() / "normalise-to-reference/percentile"

    inputs = kgo_dir / "input*.nc"
    output_path = tmp_path / "output.nc"

    reference_name = "lwe_precipitation_rate"
    return_name = "rainfall_rate"

    args = [
        inputs,
        "--reference-name",
        reference_name,
        "--return-name",
        return_name,
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match="Exactly one cube "):
        run_cli(args)
