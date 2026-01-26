# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the estimate-samos-gams CLI
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
    normal distribution.
    """
    source_emos_dir = acc.kgo_root() / "estimate-emos-coefficients/normal"
    history_path = source_emos_dir / "history/*.nc"
    truth_path = source_emos_dir / "truth/*.nc"

    kgo_dir = acc.kgo_root() / "estimate-samos-gam"
    kgo_path = kgo_dir / "kgo.pkl"
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
        "--window-length",
        "2",
        "--required-rolling-window-points",
        "2",
        "--output",
        output_path,
    ]
    run_cli(args)

    # Compare the output with the known good output. This
    # comparison only ensures that the string version of the
    # pickled objects are the same, not the actual objects as
    # there is no function to compare the GAM class objects.
    acc.compare(output_path, kgo_path, file_type="generic_pickle")


def test_gam_cube_gam_features(tmp_path):
    """
    Test estimate-samos-gams-coefficients for diagnostic with assumed
    normal distribution and additional features provided as a cube.
    """
    source_emos_dir = acc.kgo_root() / "estimate-emos-coefficients/normal"
    history_path = source_emos_dir / "history/*.nc"
    truth_path = source_emos_dir / "truth/*.nc"

    kgo_dir = acc.kgo_root() / "estimate-samos-gam"
    kgo_path = kgo_dir / "kgo.pkl"
    additional_features_path = kgo_dir / "roughness_length.nc"
    model_specification_path = kgo_dir / "samos_model_spec_simple.json"
    output_path = tmp_path / "output.pkl"

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
        "--window-length",
        "2",
        "--required-rolling-window-points",
        "2",
        "--output",
        output_path,
    ]
    run_cli(args)
    # Compare the output with the known good output. This
    # comparison only ensures that the string version of the
    # pickled objects are the same, not the actual objects as
    # there is no function to compare the GAM class objects.
    acc.compare(output_path, kgo_path, file_type="generic_pickle")


def test_gam_at_sites(tmp_path):
    """
    Test estimate-samos-gams-coefficients for diagnostic with assumed
    normal distribution and additional features provided as a cube.
    """
    source_emos_dir = acc.kgo_root() / "estimate-emos-coefficients/normal/sites"
    history_path = source_emos_dir / "history/*.nc"
    truth_path = source_emos_dir / "truth/*.nc"

    kgo_dir = acc.kgo_root() / "estimate-samos-gam"
    kgo_path = kgo_dir / "kgo_sites.pkl"
    model_specification_path = kgo_dir / "samos_model_spec_simple.json"
    output_path = tmp_path / "output.pkl"

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
        "--window-length",
        "2",
        "--required-rolling-window-points",
        "2",
        "--output",
        output_path,
    ]
    run_cli(args)
    # Compare the output with the known good output. This
    # comparison only ensures that the string version of the
    # pickled objects are the same, not the actual objects as
    # there is no function to compare the GAM class objects.
    acc.compare(output_path, kgo_path, file_type="generic_pickle")


def test_insufficient_data(tmp_path):
    """
    Test estimate-samos-gams returns None when insufficient data is available at all
    sites.

    This test provides 3 days of input data but uses a window length of 10 days. This
    will cause the training data at all sites to be considered insufficient to fit the
    GAMs (at least 6 data points are required in each window). Hence, None should be
    returned.
    """
    source_emos_dir = acc.kgo_root() / "estimate-emos-coefficients/normal/sites"
    history_path = source_emos_dir / "history/*.nc"
    truth_path = source_emos_dir / "truth/*.nc"

    kgo_dir = acc.kgo_root() / "estimate-samos-gam"
    model_specification_path = kgo_dir / "samos_model_spec_simple.json"
    output_path = tmp_path / "output.pkl"

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
        "--window-length",
        "10",
        "--required-rolling-window-points",
        "6",
        "--output",
        output_path,
    ]
    run_cli(args)
    # Check no file has been written to disk.
    assert not output_path.exists()
