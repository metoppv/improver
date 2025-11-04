# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

# from pathlib import Path

from improver import BasePlugin
from improver.calibration import (
    lightgbm_package_available,
    treelite_packages_available,
)


class TrainRainForestsCalibration(BasePlugin):
    lightgbm_params = {
        "objective": "binary",
        "num_leaves": 5,
        "num_boost_round": 10,
        "verbose": -1,
        "seed": 0,
    }

    def __init__(self, training_data):
        self.lightgbm_available = lightgbm_package_available()
        if not self.lightgbm_available:
            raise ModuleNotFoundError("Could not find LightGBM module")

        self.training_data = training_data

    def process(
        self, threshold, observation_column, training_columns, output_path=None
    ):
        """Train a model for one threshold."""
        import lightgbm

        threshold_met = (self.training_data[observation_column] >= threshold).astype(
            int
        )
        training_data = self.training_data[training_columns]
        dataset = lightgbm.Dataset(training_data, label=threshold_met)

        model = lightgbm.train(self.lightgbm_params, dataset)
        if output_path:
            model.save_model(output_path)

        return model.model_to_string()


class CompileRainForestsCalibration(BasePlugin):
    treelight_params = {"parallel_comp": 8, "quantize": 1}

    def __init__(self):
        self.treelite_available = treelite_packages_available()
        if not self.treelite_available:
            raise ModuleNotFoundError("Could not find TreeLite module")
        # Also need lightGBM available to read in models
        self.lightgbm_available = lightgbm_package_available()
        if not self.lightgbm_available:
            raise ModuleNotFoundError("Could not find LightGBM module")

    def process(self, lightgbm_filepath, output_filepath):
        """Compile a lightgbm model."""
        import tl2cgen
        import treelite
        from lightgbm import Booster

        lightgbm_model = Booster(model_file=lightgbm_filepath)

        model = treelite.Model.from_lightgbm(lightgbm_model)
        tl2cgen.export_lib(
            model,
            toolchain="gcc",
            libpath=output_filepath,
            verbose=False,
            params=self.treelight_params,
        )
