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
Tests for the generate-realizations CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_percentiles(tmp_path):
    """Test basic percentile to realization conversion"""
    kgo_dir = acc.kgo_root() / "generate-realizations/percentiles_rebadging"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "multiple_percentiles_wind_cube.nc"

    output_path = tmp_path / "output.nc"

    args = [input_path, "--realizations-count", "12", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentiles_reordering(tmp_path):
    """Test percentile to realization conversion with reordering"""
    kgo_dir = acc.kgo_root() / "generate-realizations/percentiles_reordering"
    kgo_path = kgo_dir / "kgo.nc"
    forecast_path = kgo_dir / "raw_forecast.nc"
    percentiles_path = kgo_dir / "multiple_percentiles_wind_cube.nc"
    output_path = tmp_path / "output.nc"
    args = [
        "--realizations-count",
        "12",
        "--random-seed",
        "0",
        percentiles_path,
        forecast_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize(
    "bounds_option, kgo",
    (
        ("", "with_ecc_bounds_kgo.nc"),
        ("--skip-ecc-bounds", "without_ecc_bounds_kgo.nc"),
    ),
)
def test_extreme_percentiles(tmp_path, bounds_option, kgo):
    """Test percentile to percentile conversion where outputs are more extreme than inputs
    (lowest percentile of inputs is 31, outputs have lowest of 20)"""
    kgo_dir = acc.kgo_root() / "generate-realizations/percentiles_extremes"
    kgo_path = kgo_dir / kgo
    percentiles_path = kgo_dir / "few_percentiles_wind_cube.nc"
    output_path = tmp_path / "output.nc"
    args = [
        "--realizations-count",
        "5",
        percentiles_path,
        "--output",
        output_path,
    ]
    if bounds_option:
        args += [bounds_option]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_probabilities(tmp_path):
    """Test basic probabilities to realization conversion"""
    kgo_dir = acc.kgo_root() / "generate-realizations/probabilities_12_realizations"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--realizations-count", "12", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_probabilities_reordering(tmp_path):
    """Test probabilities to realization conversion with reordering"""
    kgo_dir = acc.kgo_root() / "generate-realizations/probabilities_reordering"
    kgo_path = kgo_dir / "kgo.nc"
    raw_path = kgo_dir / "raw_ens.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--random-seed", "0", input_path, raw_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_realizations(tmp_path):
    """Test basic null realization to realization conversion"""
    kgo_dir = acc.kgo_root() / "generate-realizations/probabilities_12_realizations"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_path
    output_path = tmp_path / "output.nc"
    args = [input_path, "--realizations-count", "12", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, input_path)


def test_ecc_bounds_warning(tmp_path):
    """
    Test use of ECC to convert one set of percentiles to another set of
    percentiles, and then rebadge the percentiles to be ensemble realizations.
    Data in this input exceeds the ECC bounds and so tests ecc_bounds_warning
    functionality.
    """
    kgo_dir = acc.kgo_root() / "generate-realizations/ecc_bounds_warning"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "multiple_percentiles_wind_cube_out_of_bounds.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--realizations-count",
        "5",
        "--ignore-ecc-bounds-exceedance",
        "--output",
        output_path,
    ]
    with pytest.warns(UserWarning, match="Forecast values exist that fall outside"):
        run_cli(args)
    acc.compare(output_path, kgo_path)


def test_error_no_realizations_count(tmp_path):
    """Test a helpful error is raised if wrong args are set"""
    kgo_dir = acc.kgo_root() / "generate-realizations/probabilities_12_realizations"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    with pytest.raises(ValueError, match=".*realizations_count or raw_cube.*"):
        run_cli(args)


def test_invalid_dataset(tmp_path):
    """Test unhandlable conversion failure"""
    input_dir = acc.kgo_root() / "generate-realizations/invalid/"
    input_path = input_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    with pytest.raises(ValueError, match=".*Unable to convert.*"):
        run_cli(args)
