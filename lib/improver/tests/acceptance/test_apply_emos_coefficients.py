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
"""Tests for the apply-emos-coefficients CLI"""

import pytest

from improver.tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_gaussian(tmp_path):
    """Test diagnostic with assumed gaussian distribution"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/gaussian"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_dir = kgo_dir / "../../estimate-emos-coefficients"
    emos_est_path = emos_est_dir / "gaussian/kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, emos_est_path, output_path,
            "norm", "--random_seed", "0"]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


def test_truncated_gaussian(tmp_path):
    """Test diagnostic with assumed gaussian distribution"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/truncated_gaussian"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_dir = kgo_dir / "../../estimate-emos-coefficients"
    emos_est_path = emos_est_dir / "truncated_gaussian/kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, emos_est_path, output_path,
            "truncnorm", "--random_seed", "0",
            "--shape_parameters", "0", "inf"]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


def predictor_of_mean(tmp_path, status):
    """Implementation of test using non-default predictor realizations"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/realizations"
    kgo_path = kgo_dir / f"{status}_statsmodels_kgo.nc"
    input_path = kgo_dir / "../gaussian/input.nc"
    emos_est_dir = kgo_dir / "../../estimate-emos-coefficients"
    emos_est_path = emos_est_dir / f"realizations/{status}_statsmodels_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, emos_est_path, output_path,
            "norm", "--predictor_of_mean", "realizations",
            "--random_seed", "0"]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


@acc.skip_if_statsmodels
def test_predictor_of_mean_no_sm(tmp_path):
    """Test using non-default predictor realizations"""
    sm_status = "without"
    predictor_of_mean(tmp_path, sm_status)


@acc.skip_if_no_statsmodels
def test_predictor_of_mean_sm(tmp_path):
    """Test using non-default predictor realizations"""
    sm_status = "with"
    predictor_of_mean(tmp_path, sm_status)


def test_probabilities(tmp_path):
    """Test using probabilities as input"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/probabilities"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_dir = kgo_dir / "../../estimate-emos-coefficients"
    emos_est_path = emos_est_dir / "gaussian/kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, emos_est_path, output_path,
            "norm", "--num_realizations=18"]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


def test_probabilities_error(tmp_path):
    """Test using probabilities as input without num_realizations"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/probabilities"
    input_path = kgo_dir / "input.nc"
    emos_est_dir = kgo_dir / "../../estimate-emos-coefficients"
    emos_est_path = emos_est_dir / "gaussian/kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, emos_est_path, output_path, "norm"]
    with pytest.raises(ValueError, match=".*provided as probabilities.*"):
        run_cli(args)


def test_percentiles(tmp_path):
    """Test using percentiles as input"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/percentiles"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_dir = kgo_dir / "../../estimate-emos-coefficients"
    emos_est_path = emos_est_dir / "gaussian/kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, emos_est_path, output_path,
            "norm", "--num_realizations=18"]
    run_cli(args)
    acc.compare(output_path, kgo_path,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


def test_percentiles_error(tmp_path):
    """Test using percentiles as input"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/percentiles"
    input_path = kgo_dir / "input.nc"
    emos_est_dir = kgo_dir / "../../estimate-emos-coefficients"
    emos_est_path = emos_est_dir / "gaussian/kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, emos_est_path, output_path, "norm"]
    with pytest.raises(ValueError):
        run_cli(args)


def test_rebadged_percentiles(tmp_path):
    """Test using realizations rebadged as percentiles as input"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/percentiles"
    kgo_path = kgo_dir / "kgo.nc"
    emos_est_dir = kgo_dir / "../../estimate-emos-coefficients"
    emos_est_path = emos_est_dir / "gaussian/kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [kgo_dir / "../rebadged_percentiles/input.nc",
            emos_est_path,
            output_path,
            "norm", "--num_realizations=18"]
    run_cli(args)
    # The known good output in this case is the same as when passing in
    # percentiles directly, apart from a difference in the coordinates, such
    # that the percentile input will have a percentile coordinate, whilst the
    # rebadged percentile input will result in a realization coordinate.
    acc.compare(output_path, kgo_path,
                exclude_dims=["realization", "percentile"], options="-dNs",
                atol=acc.LOOSE_TOLERANCE, rtol=None)


def test_no_coefficients(tmp_path):
    """Test no coefficients provided"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/gaussian"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, output_path, "norm", "--random_seed", "0"]
    with pytest.warns(UserWarning, match=".*no coefficients provided.*"):
        run_cli(args)
    acc.compare(output_path, input_path, recreate=False,
                atol=acc.LOOSE_TOLERANCE, rtol=None)


def test_wrong_coefficients(tmp_path):
    """Test wrong coefficients provided"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/gaussian"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, input_path, output_path, "norm", "--random_seed", "0"]
    with pytest.raises(ValueError, match=".*coefficients cube.*"):
        run_cli(args)


def test_wrong_forecast(tmp_path):
    """Test forecast cube being a coefficients cube"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/gaussian"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [kgo_path, kgo_path, output_path, "norm", "--random_seed", "0"]
    with pytest.raises(ValueError, match=".*forecast cube.*"):
        run_cli(args)
