# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
Tests for the percentiles-to-realizations CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
run_cli = acc.run_cli("percentiles-to-realizations")


@pytest.mark.slow
def test_percentiles_reordering(tmp_path):
    """
    Test use of ECC to convert one set of percentiles to another set of
    percentiles, and then reorder the ensemble using the raw ensemble
    realizations
    """
    kgo_dir = acc.kgo_root() / \
        "percentiles-to-realizations/percentiles_reordering"
    kgo_path = kgo_dir / "kgo.nc"
    forecast_path = kgo_dir / "raw_forecast.nc"
    percentiles_path = kgo_dir / "multiple_percentiles_wind_cube.nc"
    output_path = tmp_path / "output.nc"
    args = ["--sampling-method", "quantile",
            "--realizations-count", "12",
            "--random-seed", "0",
            percentiles_path,
            forecast_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_percentiles_rebadging(tmp_path):
    """
    Test use of ECC to convert one set of percentiles to another set of
    percentiles, and then rebadge the percentiles to be ensemble realizations
    """
    kgo_dir = acc.kgo_root() / \
        "percentiles-to-realizations/percentiles_rebadging"
    kgo_path = kgo_dir / "kgo.nc"
    percentiles_path = kgo_dir / "multiple_percentiles_wind_cube.nc"
    output_path = tmp_path / "output.nc"
    args = ["--sampling-method", "quantile",
            "--realizations-count", "12",
            percentiles_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_percentiles_rebadging_numbers(tmp_path):
    """Test rebadging with specified realization numbers"""
    kgo_dir = acc.kgo_root() / \
        "percentiles-to-realizations/percentiles_rebadging_extra_option"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir / "../percentiles_rebadging"
    percentiles_path = input_dir / "multiple_percentiles_wind_cube.nc"
    output_path = tmp_path / "output.nc"
    args = [percentiles_path,
            "--output", output_path,
            "--sampling-method", "quantile",
            "--realizations-count", "12",
            "--realizations", ",".join(str(n) for n in range(100, 112))]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_ecc_bounds_warning(tmp_path):
    """
    Test use of ECC to convert one set of percentiles to another set of
    percentiles, and then rebadge the percentiles to be ensemble realizations.
    Data in this input exceeds the ECC bounds and so tests ecc_bounds_warning
    functionality.
    """
    kgo_dir = acc.kgo_root() / \
        "percentiles-to-realizations/ecc_bounds_warning"
    kgo_path = kgo_dir / "kgo.nc"
    percentiles_path = kgo_dir / \
        "multiple_percentiles_wind_cube_out_of_bounds.nc"
    output_path = tmp_path / "output.nc"
    args = ["--sampling-method", "quantile",
            "--realizations-count", "5",
            "--ignore-ecc-bounds",
            percentiles_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
