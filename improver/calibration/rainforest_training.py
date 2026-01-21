# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""RainForests model training plugin."""

from improver import BasePlugin
from improver.calibration import (
    lightgbm_package_available,
)


class TrainRainForestsModel(BasePlugin):
    lightgbm_params = {
        "objective": "binary",
        "num_leaves": 5,
        "num_boost_round": 10,
        "verbose": -1,
        "seed": 0,
    }

    def __init__(self, training_data, observation_column, training_columns):
        """Initialise the options used when compiling models.

        Args:
            training_data (pandas.DataFrame):
                Combined data set used to train models.
            observation_column (str):
                The column in the data set to be trained for.
            training_columns (List(str)):
                Set of columns from the data set to be trained from.
        """
        self.lightgbm_available = lightgbm_package_available()
        if not self.lightgbm_available:
            raise ModuleNotFoundError("Could not find LightGBM module")

        expected_columns = training_columns + [observation_column]

        # Check all specified columns exist in the data.
        for col in expected_columns:
            if col not in training_data:
                raise KeyError(f"Column '{col}' not found in training data.")

        # Check the observation column is not also a training column.
        if observation_column in training_columns:
            raise KeyError(
                f"Observation column '{observation_column}' appears in training data."
            )

        self.observation_column = observation_column
        self.training_columns = training_columns

        # Keep only the columns relevant for training.
        self.training_data = training_data[expected_columns]

    def process(self, thresholds, output_path):
        """Train a model for a particular threshold.

        Args:
            thresholds (list of float):
                Thresholds for which the observation column is trained.
            output_path (str or Path):
                Template file path for export of model files.
                Actual paths will have the threshold appended to the filename.
        """
        import lightgbm

        for threshold in thresholds:
            threshold_met = (
                self.training_data[self.observation_column] >= threshold
            ).astype(int)
            dataset = lightgbm.Dataset(self.training_data, label=threshold_met)

            model = lightgbm.train(self.lightgbm_params, dataset)

            threshold_path = output_path.with_stem(f"{output_path.stem}_{threshold:08.6f}")
            model.save_model(threshold_path)
