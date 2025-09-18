# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the apply-samos-coefficients CLI"""

import pytest

from improver.constants import LOOSE_TOLERANCE

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_normal_gam_coordinates(tmp_path):
    """Test diagnostic with assumed normal distribution and GAM feature
    provided as coordinates on the input cube."""
    kgo_dir = acc.kgo_root() / "apply-samos-coefficients"
    kgo_path = kgo_dir / "kgo_coord.nc"

    input_path = kgo_dir / "forecast.nc"
    samos_est_path = kgo_dir / "samos_coefficients/coefficients_coordinates.nc"
    gam_path = kgo_dir / "gam_configs/samos_gam.pkl"
    output_path = tmp_path / "output.nc"

    gam_features = "projection_y_coordinate,projection_x_coordinate,height"

    args = [
        input_path,
        samos_est_path,
        gam_path,
        "--gam-features",
        gam_features,
        "--realizations-count",
        "9",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_normal_gam_cubes(tmp_path):
    """
    Test apply-samos-coefficients for diagnostic with assumed
    normal distribution and additional features provided as cubes.
    """
    kgo_dir = acc.kgo_root() / "apply-samos-coefficients"
    kgo_path = kgo_dir / "kgo_cubes_additional_cubes.nc"

    input_path = kgo_dir / "forecast.nc"
    samos_est_path = kgo_dir / "samos_coefficients/coefficients_extra_features.nc"
    gam_path = kgo_dir / "gam_configs/samos_gam_additional_features.pkl"
    output_path = tmp_path / "output.nc"
    additional_features_path = kgo_dir / "additional_features/roughness_length.nc"
    gam_features = (
        "projection_y_coordinate,projection_x_coordinate,vegetative_roughness_length"
    )

    args = [
        input_path,
        samos_est_path,
        additional_features_path,
        gam_path,
        "--gam-features",
        gam_features,
        "--realizations-count",
        "9",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_normal_sites(tmp_path):
    """
    Test apply-samos-coefficients for diagnostic with assumed
    normal distribution at sites.
    """
    kgo_dir = acc.kgo_root() / "apply-samos-coefficients"
    kgo_path = kgo_dir / "kgo_sites.nc"

    input_path = kgo_dir / "site_forecast.nc"
    samos_est_path = kgo_dir / "samos_coefficients/coefficients_sites.nc"
    gam_path = kgo_dir / "gam_configs/samos_gam_sites.pkl"
    output_path = tmp_path / "output.nc"
    gam_features = "latitude,longitude,height"

    args = [
        input_path,
        samos_est_path,
        gam_path,
        "--gam-features",
        gam_features,
        "--realizations-count",
        "9",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_no_coefficients(tmp_path):
    """Test apply-samos-coefficients when no coefficients provided"""
    kgo_dir = acc.kgo_root() / "apply-samos-coefficients/"
    kgo_path = kgo_dir / "kgo_with_comment.nc"

    input_path = kgo_dir / "site_forecast.nc"
    gam_path = kgo_dir / "gam_configs/samos_gam_sites.pkl"
    output_path = tmp_path / "output.nc"
    additional_features_path = kgo_dir / "additional_features/roughness_length.nc"
    gam_features = (
        "projection_y_coordinate,projection_x_coordinate,vegetative_roughness_length"
    )

    args = [
        input_path,
        additional_features_path,
        gam_path,
        "--gam-features",
        gam_features,
        "--realizations-count",
        "9",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]

    with pytest.warns(UserWarning, match=".*no coefficients provided.*"):
        run_cli(args)

    # Check output matches kgo.
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)
