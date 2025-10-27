# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the estimate-samos-coefficients CLI

Many of these tests use globs which are expanded by IMPROVER code itself,
rather than by shell glob expansion. There are also some directory globs
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
def test_coordinates(tmp_path):
    """
    Test estimate-samos-coefficients for diagnostic with assumed
    normal distribution and additional gam features are coordinates on
    the input cubes.
    """
    # The source data is from the estimate-emos-coefficients acceptance tests
    source_emos_dir = acc.kgo_root() / "estimate-emos-coefficients/normal"
    history_path = source_emos_dir / "history/*.nc"
    truth_path = source_emos_dir / "truth/*.nc"

    kgo_dir = acc.kgo_root() / "estimate-samos-coefficients"
    output_path = tmp_path / "output.nc"
    kgo_path = kgo_dir / "kgo_coordinates.nc"
    gam_path = kgo_dir / "gam_configs/samos_gam.pkl"
    gam_features = "projection_y_coordinate,projection_x_coordinate,height"

    args = [
        history_path,
        truth_path,
        gam_path,
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--gam-features",
        gam_features,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


def test_normal_cube_gam_features(tmp_path):
    """
    Test estimate-samos-coefficients for diagnostic with assumed
    normal distribution and additional features provided as a cube.
    """
    source_emos_dir = acc.kgo_root() / "estimate-emos-coefficients/normal"
    history_path = source_emos_dir / "history/*.nc"
    truth_path = source_emos_dir / "truth/*.nc"

    kgo_dir = acc.kgo_root() / "estimate-samos-coefficients"
    additional_features_path = kgo_dir / "additional_features/roughness_length.nc"
    output_path = tmp_path / "output.nc"
    kgo_path = kgo_dir / "kgo_extra_gam_feature.nc"
    gam_path = kgo_dir / "gam_configs/samos_gam_additional_features.pkl"
    gam_features = (
        "projection_y_coordinate,projection_x_coordinate,vegetative_roughness_length"
    )

    args = [
        history_path,
        truth_path,
        additional_features_path,
        gam_path,
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--gam-features",
        gam_features,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


def test_estimate_samos_coefficients_sites(tmp_path):
    """
    Test estimate-samos-coefficients for site data with a diagnostic with and assumed
    normal distribution and additional gam features are coordinates on
    the input cubes.
    """
    # The source data is from the estimate-emos-coefficients acceptance tests
    source_emos_dir = acc.kgo_root() / "estimate-emos-coefficients/normal/sites"
    history_path = source_emos_dir / "history/*.nc"
    truth_path = source_emos_dir / "truth/*.nc"

    kgo_dir = acc.kgo_root() / "estimate-samos-coefficients"
    output_path = tmp_path / "output.nc"
    kgo_path = kgo_dir / "kgo_sites.nc"
    gam_path = kgo_dir / "gam_configs/samos_gam_sites.pkl"
    gam_features = "latitude,longitude,height"

    args = [
        history_path,
        truth_path,
        gam_path,
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--gam-features",
        gam_features,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


def test_estimate_samos_coefficients_no_gam(tmp_path):
    """
    Test estimate-samos-coefficients when no GAM is provided. The CLI should return
    None in this instance.
    """
    # The source data is from the estimate-emos-coefficients acceptance tests
    source_emos_dir = acc.kgo_root() / "estimate-emos-coefficients/normal/sites"
    history_path = source_emos_dir / "history/*.nc"
    truth_path = source_emos_dir / "truth/*.nc"

    output_path = tmp_path / "output.nc"
    gam_features = "latitude,longitude,height"

    args = [
        history_path,
        truth_path,
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--gam-features",
        gam_features,
        "--output",
        output_path,
    ]
    assert run_cli(args) is None
