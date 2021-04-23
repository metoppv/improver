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
Tests for the estimate-emos-coefficients CLI

Many of these tests use globs which are expanded by IMPROVER code itself,
rather than by shell glob expansion. There are also a some directory globs
which expand directory names in addition to filenames.
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

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
def test_normal(tmp_path):
    """
    Test estimate-emos-coefficients for diagnostic with assumed
    normal distribution
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/normal"
    kgo_path = kgo_dir / "kgo.nc"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--distribution",
        "norm",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
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
def test_truncated_normal(tmp_path):
    """
    Test estimate-emos-coefficients for diagnostic with assumed
    truncated normal distribution
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/truncated_normal"
    kgo_path = kgo_dir / "kgo.nc"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--distribution",
        "truncnorm",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
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
def test_normal_default_initial_guess(tmp_path):
    """
    Test estimate-emos-coefficients for diagnostic with assumed
    normal distribution with the default initial guess.
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/normal"
    kgo_path = kgo_dir / "default_initial_guess_kgo.nc"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--distribution",
        "norm",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--tolerance",
        EST_EMOS_TOL,
        "--use-default-initial-guess",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


@pytest.mark.slow
def test_units(tmp_path):
    """Test prescribed units that may not match inputs"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/normal"
    kgo_path = kgo_dir / "kgo.nc"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--distribution",
        "norm",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--units",
        "K",
        "--max-iterations",
        "600",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


@pytest.mark.slow
@acc.skip_if_statsmodels
def test_using_realizations_as_predictor_no_sm(tmp_path):
    """Test using non-default predictor realizations"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    kgo_path = kgo_dir / "normal/realizations/without_statsmodels_kgo.nc"
    history_path = kgo_dir / "normal/history/*.nc"
    truth_path = kgo_dir / "normal/truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--distribution",
        "norm",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--predictor",
        "realizations",
        "--max-iterations",
        "150",
        "--tolerance",
        EST_EMOS_TOL,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


@acc.skip_if_no_statsmodels
def test_using_realizations_as_predictor_sm(tmp_path):
    """Test using non-default predictor realizations"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    kgo_path = kgo_dir / "normal/realizations/with_statsmodels_kgo.nc"
    history_path = kgo_dir / "normal/history/*.nc"
    truth_path = kgo_dir / "normal/truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--distribution",
        "norm",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--predictor",
        "realizations",
        "--max-iterations",
        "150",
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
def test_land_points_only(tmp_path):
    """Test use of land-sea mask"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    kgo_path = kgo_dir / "normal/land_only_kgo.nc"
    lsmask_path = kgo_dir / "landmask.nc"
    history_path = kgo_dir / "normal/history/*.nc"
    truth_path = kgo_dir / "normal/truth/*.nc"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        lsmask_path,
        "--distribution",
        "norm",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
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
def test_normal_point_by_point_sites(tmp_path):
    """
    Test estimate-emos-coefficients for diagnostic with assumed
    normal distribution where coefficients are computed independently at each
    site location (initial guess and minimisation).
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/normal/sites"
    kgo_path = kgo_dir / "point_by_point" / "kgo.nc"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    output_path = tmp_path / "output.nc"
    est_emos_tol = str(0.01)
    compare_emos_tolerance = 0.1
    args = [
        history_path,
        truth_path,
        "--distribution",
        "norm",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--tolerance",
        est_emos_tol,
        "--point-by-point",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=compare_emos_tolerance, rtol=compare_emos_tolerance
    )


@pytest.mark.slow
@acc.skip_if_statsmodels
def test_normal_realizations_point_by_point_sites(tmp_path):
    """
    Test estimate-emos-coefficients for diagnostic with assumed
    normal distribution where coefficients are computed independently at each
    site location (initial guess and minimisation).
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/normal/sites"
    kgo_path = kgo_dir / "point_by_point" / "realizations_kgo.nc"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    output_path = tmp_path / "output.nc"
    est_emos_tol = str(0.01)
    compare_emos_tolerance = 0.1
    args = [
        history_path,
        truth_path,
        "--distribution",
        "norm",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--predictor",
        "realizations",
        "--tolerance",
        est_emos_tol,
        "--point-by-point",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=compare_emos_tolerance, rtol=compare_emos_tolerance
    )


@pytest.mark.slow
def test_normal_point_by_point_default_initial_guess_sites(tmp_path):
    """
    Test estimate-emos-coefficients for diagnostic with assumed
    normal distribution where coefficients are computed independently at each
    site location (minimisation only).
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients/normal/sites"
    kgo_path = kgo_dir / "point_by_point_default_initial_guess" / "kgo.nc"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    output_path = tmp_path / "output.nc"
    est_emos_tol = str(0.01)
    compare_emos_tolerance = 0.1
    args = [
        history_path,
        truth_path,
        "--distribution",
        "norm",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--tolerance",
        est_emos_tol,
        "--point-by-point",
        "--use-default-initial-guess",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=compare_emos_tolerance, rtol=compare_emos_tolerance
    )
