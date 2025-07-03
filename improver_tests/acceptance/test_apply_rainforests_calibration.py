# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the apply-rainforests-calibration CLI."""

import json
import sys

import pytest

from . import acceptance as acc

lightgbm = pytest.importorskip("lightgbm")
tl2cgen = pytest.importorskip("tl2cgen")
treelite = pytest.importorskip("treelite")

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

RAINFORESTS_DIR = acc.kgo_root() / "apply-rainforests-calibration"
KGO_PATH = RAINFORESTS_DIR / "basic" / "kgo.nc"
FORECAST_PATH = (
    RAINFORESTS_DIR
    / "features"
    / "20200802T0000Z-PT0024H00M-precipitation_accumulation-PT24H.nc"
)
FEATURE_PATHS = (RAINFORESTS_DIR / "features").glob("20200802T0000Z-PT00*-PT24H.nc")


# Run tests with treelite module and with lightgbm module
@pytest.mark.parametrize(
    "model_key",
    ("lightgbm_model", "treelite_model"),
    ids=["use_lightgbm", "use_treelite"],
)
class TestApplyRainforestsCalibration:
    @pytest.fixture(autouse=True)
    def run_before_tests(self, model_key, monkeypatch):
        # Logic for acceptance test setup
        match model_key:
            case "lightgbm_model":
                monkeypatch.setitem(sys.modules, "tl2cgen", None)
            case "treelite_model":
                monkeypatch.setitem(sys.modules, "lightgbm", None)
            case _:
                raise NotImplementedError("Unknown value for model_key")
        yield

    @pytest.fixture(scope="class")
    def create_model_config(self):
        """Load model-config json containing relative paths, and create a duplicate version
        of the associated dictionary with relative paths replaced with absolute paths."""
        model_file_dir = (
            acc.kgo_root() / "apply-rainforests-calibration" / "model_files"
        )
        # Load model config containing paths relative to apply-rainforests-calibration
        # within the directory contaning improver acceptance test data.
        relative_path_model_config_file = model_file_dir / "model_config.json"
        with open(relative_path_model_config_file, "r") as model_config_json:
            relative_path_model_config_dict = json.load(model_config_json)

        absolute_path_model_config_dict = {}
        for lead_time in relative_path_model_config_dict.keys():
            absolute_path_model_config_dict[lead_time] = {}
            for threshold, relative_path_config in relative_path_model_config_dict[
                lead_time
            ].items():
                absolute_path_model_config_dict[lead_time][threshold] = {}
                for model_name in relative_path_config.keys():
                    absolute_path = relative_path_config[model_name].replace(
                        "./", str(acc.kgo_root()) + "/"
                    )
                    absolute_path_model_config_dict[lead_time][threshold][model_name] = absolute_path
        return absolute_path_model_config_dict

    def test_basic(self, tmp_path, model_key, create_model_config):
        """
        Test calibration of a forecast using a rainforests approach.
        """
        model_config = create_model_config
        output_path = tmp_path / "output.nc"
        args = [
            FORECAST_PATH,
            *FEATURE_PATHS,
            "--model-config",
            model_config,
            "--output-thresholds",
            "0.0,0.0005,0.001",
            "--output",
            output_path,
        ]
        run_cli(args)
        acc.compare(output_path, KGO_PATH)

    def test_bin_data(self, tmp_path, model_key, create_model_config):
        """
        Test that the bin_data option does not affect the output.
        """
        model_config = create_model_config
        output_path = tmp_path / "output.nc"
        args = [
            FORECAST_PATH,
            *FEATURE_PATHS,
            "--model-config",
            model_config,
            "--output-thresholds",
            "0.0,0.0005,0.001",
            "--bin-data",
            "--output",
            output_path,
        ]
        run_cli(args)
        acc.compare(output_path, KGO_PATH)

    def test_json_threshold_config(self, tmp_path, model_key, create_model_config):
        """
        Test calibration of a forecast using a rainforests approach where
        thresholds are specified with json file.
        """
        model_config = create_model_config
        output_path = tmp_path / "output.nc"
        json_path = RAINFORESTS_DIR / "threshold_config" / "thresholds.json"
        args = [
            FORECAST_PATH,
            *FEATURE_PATHS,
            "--model-config",
            model_config,
            "--output-threshold-config",
            json_path,
            "--output",
            output_path,
        ]
        run_cli(args)
        acc.compare(output_path, KGO_PATH)

    def test_no_threshold_config(self, tmp_path, model_key, create_model_config):
        """
        Test cli raises an error when no threshold config is specified.
        """
        model_config = create_model_config
        output_path = tmp_path / "output.nc"
        args = [
            FORECAST_PATH,
            *FEATURE_PATHS,
            "--model-config",
            model_config,
            "--output",
            output_path,
        ]
        with pytest.raises(ValueError, match="must be specified"):
            run_cli(args)
