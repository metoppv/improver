# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""RainForests model training plugin."""

from improver import BasePlugin
from improver.calibration import (
    lightgbm_package_available,
)


class TrainRainForestsCalibration(BasePlugin):
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
                Combined data set used to train models
            observation_column (str):
                The column in the data set to be used
            training_columns (List(str)):
                Set of columns from the data set to be used as training data.
        """
        self.lightgbm_available = lightgbm_package_available()
        if not self.lightgbm_available:
            raise ModuleNotFoundError("Could not find LightGBM module")

        self.observation_column = observation_column
        self.training_columns = training_columns

        expected_columns = training_columns + [observation_column]
        for col in expected_columns:
            if col not in training_data:
                raise KeyError(f"Column {col} not found in training data")

        self.training_data = training_data[expected_columns]

    def process(
        self, threshold, output_path=None
    ):
        """Train a model for a particular threshold.

        Args:
            threshold (float):
                Threshold for which the observation column is trained.
            output_path (str or Path):
                If provided, the model will be exported to this file path.
        """
        import lightgbm

        threshold_met = (self.training_data[self.observation_column] >= threshold).astype(
            int
        )
        dataset = lightgbm.Dataset(self.training_data, label=threshold_met)

        model = lightgbm.train(self.lightgbm_params, dataset)
        if output_path:
            model.save_model(output_path)

        return model.model_to_string()
