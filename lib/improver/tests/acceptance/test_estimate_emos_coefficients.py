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
Tests for the estimate-emos-coefficients CLI

Many of these tests use globs which are expanded by IMPROVER code itself,
rather than by shell glob expansion. There are also a some directory globs
which expand directory names in addition to filenames.
"""

import pytest

from improver.tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.slow
def test_gaussian(tmp_path):
    """
    Test estimate-emos-coefficients for diagnostic with assumed
    gaussian distribution
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/gaussian"
    kgo_path = kgo_dir / "kgo.nc"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = ["gaussian", "20170605T0300Z", output_path,
            "--historic_filepath", history_path,
            "--truth_filepath", truth_path]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


@pytest.mark.slow
def test_trunc_gaussian(tmp_path):
    """
    Test estimate-emos-coefficients for diagnostic with assumed
    truncated gaussian distribution
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/truncated_gaussian"
    kgo_path = kgo_dir / "kgo.nc"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = ["truncated_gaussian", "20170605T0300Z", output_path,
            "--historic_filepath", history_path,
            "--truth_filepath", truth_path]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


@pytest.mark.slow
def test_units(tmp_path):
    """Test prescribed units that may not match inputs"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/gaussian"
    kgo_path = kgo_dir / "kgo.nc"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = ["gaussian", "20170605T0300Z", output_path,
            "--historic_filepath", history_path,
            "--truth_filepath", truth_path,
            "--units", "K", "--max_iterations", "600"]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


@pytest.mark.slow
@acc.skip_if_statsmodels
def test_predictor_of_mean_no_sm(tmp_path):
    """Test using non-default predictor realizations"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    kgo_path = kgo_dir / "realizations/without_statsmodels_kgo.nc"
    history_path = kgo_dir / "gaussian/history/*.nc"
    truth_path = kgo_dir / "gaussian/truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = ["gaussian", "20170605T0300Z", output_path,
            "--predictor_of_mean", "realizations",
            "--historic_filepath", history_path,
            "--truth_filepath", truth_path,
            "--max_iterations", "150"]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


@acc.skip_if_no_statsmodels
def test_predictor_of_mean_sm(tmp_path):
    """Test using non-default predictor realizations"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    kgo_path = kgo_dir / "realizations/with_statsmodels_kgo.nc"
    history_path = kgo_dir / "gaussian/history/*.nc"
    truth_path = kgo_dir / "gaussian/truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = ["gaussian", "20170605T0300Z", output_path,
            "--predictor_of_mean", "realizations",
            "--historic_filepath", history_path,
            "--truth_filepath", truth_path,
            "--max_iterations", "150"]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


@pytest.mark.slow
def test_combined_inputs(tmp_path):
    """Test combined historic forecasts and truths"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    kgo_path = kgo_dir / "gaussian/kgo.nc"
    combined_path = kgo_dir / "gaussian/*/*.nc"
    historic_path = kgo_dir / "combined_input/historic_forecast.json"
    truth_path = kgo_dir / "combined_input/truth.json"
    output_path = tmp_path / "output.nc"
    args = ["gaussian", "20170605T0300Z", output_path,
            "--combined_filepath", combined_path,
            "--historic_forecast_identifier", historic_path,
            "--truth_identifier", truth_path]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


def test_only_truth_dataset(tmp_path):
    """Test error when only the truth dataset is provided"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    truth_path = kgo_dir / "gaussian/truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = ["gaussian", "20170605T0300Z", output_path,
            "--truth_filepath", truth_path]
    with pytest.raises(ValueError,
                       match=".*historic_filepath.*truth_filepath.*"):
        run_cli(args)


def test_too_many_inputs(tmp_path):
    """Test error when too many inputs are provided"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    historic_path = kgo_dir / "gaussian/history/*.nc"
    truth_path = kgo_dir / "gaussian/truth/*.nc"
    combined_path = kgo_dir / "gaussian/*/*.nc"
    output_path = tmp_path / "output.nc"
    args = ["gaussian", "20170605T0300Z", output_path,
            "--historic_filepath", historic_path,
            "--truth_filepath", truth_path,
            "--combined_filepath", combined_path]
    with pytest.raises(ValueError,
                       match=".*historic_filepath.*truth_filepath.*"):
        run_cli(args)


def test_too_few_combined(tmp_path):
    """
    Test error when too few arguments to identify historic
    forecasts and truths
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    gaussian_all_path = kgo_dir / "gaussian/*/*.nc"
    output_path = tmp_path / "output.nc"
    args = ["gaussian", "20170605T0300Z", output_path,
            "--combined_filepath", gaussian_all_path]
    with pytest.raises(ValueError, match=".*combined.*"):
        run_cli(args)


def test_mismatching_validity_times(tmp_path):
    """
    Test warning when unable to identify both historic forecasts and truths
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    combined_path = kgo_dir / "gaussian/truth/*.nc"
    historic_path = kgo_dir / "combined_input/historic_forecast.json"
    truth_path = kgo_dir / "combined_input/truth.json"
    output_path = tmp_path / "output.nc"
    args = ["gaussian", "20170605T0300Z", output_path,
            "--combined_filepath", combined_path,
            "--historic_forecast_identifier", historic_path,
            "--truth_identifier", truth_path]
    with pytest.warns(UserWarning, match=".*metadata to identify.*"):
        run_cli(args)


@pytest.mark.slow
def test_land_points_only(tmp_path):
    """Test use of land-sea mask"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    kgo_path = kgo_dir / "gaussian/land_only_kgo.nc"
    lsmask_path = kgo_dir / "landmask.nc"
    historic_path = kgo_dir / "gaussian/history/*.nc"
    truth_path = kgo_dir / "gaussian/truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = ["gaussian", "20170605T0300Z", output_path,
            "--historic_filepath", historic_path,
            "--truth_filepath", truth_path,
            "--landsea_mask", lsmask_path,
            "--tolerance", "1e-4"]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)
