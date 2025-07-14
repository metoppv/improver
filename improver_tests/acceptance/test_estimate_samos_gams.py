# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
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

TOLERANCE = str(1e-4)


@pytest.mark.slow
def test_gam_features_on_cube(tmp_path):
    """
    Test estimate-samos-gams-coefficients for diagnostic with assumed
    normal distribution
    """
    kgo_dir = acc.kgo_root() / "estimate-samos-gam"
    kgo_path = kgo_dir / "kgo.pkl"
    history_path = kgo_dir / "history/*.nc"
    truth_path = kgo_dir / "truth/*.nc"
    model_specification_path = kgo_dir / "samos_model_spec_simple.json"
    output_path = tmp_path / "output.pkl"

    gam_features = "projection_y_coordinate,projection_x_coordinate,height"
    args = [
        history_path,
        truth_path,
        "--distribution",
        "normal",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--tolerance",
        TOLERANCE,
        "--gam-features",
        gam_features,
        "--model-specification",
        model_specification_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare_pickled_objects(output_path, kgo_path)


def test_gam_cube_gam_features(tmp_path):
    """
    Test estimate-samos-gams-coefficients for diagnostic with assumed
    normal distribution and additional features provided as a cube
    """
    emos_source_dir = acc.kgo_root() / "estimate-emos-coefficients/normal"
    kgo_dir = acc.kgo_root() / "estimate-samos-gam"
    kgo_path = kgo_dir / "kgo.pkl"
    history_path = emos_source_dir / "history/*.nc"
    truth_path = emos_source_dir / "truth/*.nc"
    additional_features_path = kgo_dir / "additional_features/*.nc"
    model_specification_path = kgo_dir / "samos_model_spec_simple.json"
    output_path = tmp_path / "output_extra_cube.pkl"

    gam_features = (
        "projection_x_coordinate,projection_y_coordinate,vegetative_roughness_length"
    )
    args = [
        history_path,
        truth_path,
        additional_features_path,
        "--distribution",
        "normal",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--tolerance",
        TOLERANCE,
        "--gam-features",
        gam_features,
        "--model-specification",
        model_specification_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare_pickled_objects(output_path, kgo_path)


def test_gam_at_sites():
    """
    Test estimate-samos-gams-coefficients for diagnostic with assumed
    normal distribution and additional features provided as a cube
    """
    emos_source_dir = acc.kgo_root() / "estimate-emos-coefficients/normal/sites"
    kgo_dir = acc.kgo_root() / "estimate-samos-gam"
    kgo_path = kgo_dir / "kgo_sites.pkl"
    history_path = emos_source_dir / "history/*.nc"
    truth_path = emos_source_dir / "truth/*.nc"
    model_specification_path = kgo_dir / "samos_model_spec_simple.json"
    output_path = emos_source_dir / "output_at_sites.pkl"

    gam_features = "latitude,longitude,height"
    args = [
        history_path,
        truth_path,
        "--distribution",
        "normal",
        "--truth-attribute",
        "mosg__model_configuration=uk_det",
        "--tolerance",
        TOLERANCE,
        "--gam-features",
        gam_features,
        "--model-specification",
        model_specification_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare_pickled_objects(output_path, kgo_path)
