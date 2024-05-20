# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the apply-rainforests-calibration CLI."""

import json

import pytest

from . import acceptance as acc

lightgbm = pytest.importorskip("lightgbm")

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.fixture
def create_model_config():
    """Load model-config json containing relative paths, and create a duplicate version
    of the associated dictionary with relative paths replaced with absolute paths."""
    model_file_dir = acc.kgo_root() / "apply-rainforests-calibration/model_files"
    # Load model config containing paths relative to apply-rainforests-calibration
    # within the directory contaning improver acceptance test data.
    relative_path_model_config_file = model_file_dir / "model_config.json"
    with open(relative_path_model_config_file, "r") as model_config_json:
        relative_path_model_config_dict = json.load(model_config_json)

    absolute_path_model_config_dict = {}
    for lead_time in relative_path_model_config_dict.keys():
        absolute_path_model_config_dict[lead_time] = {}
        for threshold, relative_path in relative_path_model_config_dict[
            lead_time
        ].items():
            absolute_path = relative_path["lightgbm_model"].replace(
                "./", str(acc.kgo_root()) + "/"
            )
            absolute_path_model_config_dict[lead_time][threshold] = {
                "lightgbm_model": absolute_path
            }

    return absolute_path_model_config_dict


def test_basic(tmp_path, create_model_config):
    """
    Test calibration of a forecast using a rainforests approach.
    """
    rainforests_dir = acc.kgo_root() / "apply-rainforests-calibration"
    kgo_path = rainforests_dir / "basic" / "kgo.nc"
    forecast_path = (
        rainforests_dir
        / "features"
        / "20200802T0000Z-PT0024H00M-precipitation_accumulation-PT24H.nc"
    )
    feature_paths = (rainforests_dir / "features").glob("20200802T0000Z-PT00*-PT24H.nc")
    model_config = create_model_config
    output_path = tmp_path / "output.nc"
    args = [
        forecast_path,
        *feature_paths,
        "--model-config",
        model_config,
        "--output-thresholds",
        "0.0,0.0005,0.001",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_bin_data(tmp_path, create_model_config):
    """
    Test that the bin_data option does not affect the output.
    """
    rainforests_dir = acc.kgo_root() / "apply-rainforests-calibration"
    kgo_path = rainforests_dir / "basic" / "kgo.nc"
    forecast_path = (
        rainforests_dir
        / "features"
        / "20200802T0000Z-PT0024H00M-precipitation_accumulation-PT24H.nc"
    )
    feature_paths = (rainforests_dir / "features").glob("20200802T0000Z-PT00*-PT24H.nc")
    model_config = create_model_config
    output_path = tmp_path / "output.nc"
    args = [
        forecast_path,
        *feature_paths,
        "--model-config",
        model_config,
        "--output-thresholds",
        "0.0,0.0005,0.001",
        "--bin-data",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_json_threshold_config(tmp_path, create_model_config):
    """
    Test calibration of a forecast using a rainforests approach where
    thresholds are specified with json file.
    """
    rainforests_dir = acc.kgo_root() / "apply-rainforests-calibration"
    kgo_path = rainforests_dir / "basic" / "kgo.nc"
    forecast_path = (
        rainforests_dir
        / "features"
        / "20200802T0000Z-PT0024H00M-precipitation_accumulation-PT24H.nc"
    )
    feature_paths = (rainforests_dir / "features").glob("20200802T0000Z-PT00*-PT24H.nc")
    model_config = create_model_config
    output_path = tmp_path / "output.nc"
    json_path = rainforests_dir / "threshold_config" / "thresholds.json"
    args = [
        forecast_path,
        *feature_paths,
        "--model-config",
        model_config,
        "--output-threshold-config",
        json_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_no_threshold_config(tmp_path):
    """
    Test cli raises an error when no threshold config is specified.
    """
    rainforests_dir = acc.kgo_root() / "apply-rainforests-calibration"
    forecast_path = (
        rainforests_dir
        / "features"
        / "20200802T0000Z-PT0024H00M-precipitation_accumulation-PT24H.nc"
    )
    feature_paths = (rainforests_dir / "features").glob("20200802T0000Z-PT00*-PT24H.nc")
    model_config = create_model_config
    output_path = tmp_path / "output.nc"
    args = [
        forecast_path,
        *feature_paths,
        "--model-config",
        model_config,
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match="must be specified"):
        run_cli(args)
