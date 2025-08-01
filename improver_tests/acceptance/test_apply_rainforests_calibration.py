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


# Run tests with treelite module and with lightgbm module
@pytest.mark.parametrize(
    "model_key",
    ("lightgbm_model", "treelite_model"),
    ids=["use_lightgbm", "use_treelite"],
)
class TestApplyRainforestsCalibration:
    @pytest.fixture(autouse=True)
    def patch_available_packages(self, model_key, monkeypatch):
        match model_key:
            case "lightgbm_model":
                monkeypatch.setitem(sys.modules, "tl2cgen", None)
            case "treelite_model":
                monkeypatch.setitem(sys.modules, "lightgbm", None)
            case _:
                raise NotImplementedError("Unknown value for model_key")
        yield

    @pytest.fixture
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
                    absolute_path_model_config_dict[lead_time][threshold][
                        model_name
                    ] = absolute_path
        return absolute_path_model_config_dict

    @pytest.fixture
    def test_data_paths(self):
        rainforests_dir = acc.kgo_root() / "apply-rainforests-calibration"
        kgo_path = rainforests_dir / "basic" / "kgo.nc"
        forecast_path = (
            rainforests_dir
            / "features"
            / "20200802T0000Z-PT0024H00M-precipitation_accumulation-PT24H.nc"
        )
        feature_paths = (rainforests_dir / "features").glob(
            "20200802T0000Z-PT00*-PT24H.nc"
        )
        return rainforests_dir, kgo_path, forecast_path, feature_paths

    def test_basic(self, tmp_path, model_key, create_model_config, test_data_paths):
        """Test calibration of a forecast using a rainforests approach."""
        _, kgo_path, forecast_path, feature_paths = test_data_paths
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

    def test_bin_data(self, tmp_path, model_key, create_model_config, test_data_paths):
        """Test that the bin_data option does not affect the output."""
        _, kgo_path, forecast_path, feature_paths = test_data_paths
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

    def test_json_threshold_config(
        self, tmp_path, model_key, create_model_config, test_data_paths
    ):
        """Test calibration of a forecast using a rainforests approach where
        thresholds are specified with json file.
        """
        rainforests_dir, kgo_path, forecast_path, feature_paths = test_data_paths
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

    def test_no_threshold_config(
        self, tmp_path, model_key, create_model_config, test_data_paths
    ):
        """Test that an error is raised when no threshold config
        is specified.
        """
        _, _, forecast_path, feature_paths = test_data_paths
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
