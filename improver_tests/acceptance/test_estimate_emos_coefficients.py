# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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

pytest.importorskip("statsmodels")

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


def test_using_realizations_as_predictor(tmp_path):
    """Test using non-default predictor realizations"""
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients"
    kgo_path = kgo_dir / "normal/realizations/kgo.nc"
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
