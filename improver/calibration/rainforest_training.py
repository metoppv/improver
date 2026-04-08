# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""RainForests model training plugin."""

from pathlib import Path

from improver import BasePlugin
from improver.calibration import (
    lightgbm_package_available,
)


class TrainRainForestsModel(BasePlugin):
    lightgbm_params = {
        "objective": "binary",
        "num_leaves": 5,
        "seed": 0,
    }

    def __init__(
        self,
        model_config_dict: dict[int, dict[str, dict[str, str]]],
        training_data,
        observation_column,
        training_columns,
        lightgbm_params=None,
    ):
        """Initialise the options used when training models.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            training_data (pandas.DataFrame):
                Combined data set used to train models.
            observation_column (str):
                The column in the data set to be trained for.
            training_columns (List(str)):
                Set of columns from the data set to be trained from.
            lightgbm_params (Dict):
                Optional. Parameters passed into training library.

        Dictionary is of format::

        {
        "24": {
            "0.000010": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
            "0.000050": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
            "0.000100": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
        }

        The keys specify the lead times and model threshold values, while the
        associated values are the path to the corresponding tree-model objects
        for that lead time and threshold.
        """
        lightgbm_available = lightgbm_package_available()
        if not lightgbm_available:
            raise ModuleNotFoundError("Could not find LightGBM module")

        self.config = model_config_dict

        # Check all specified columns exist in the data.
        for col in training_columns:
            if col not in training_data:
                raise KeyError(f"Training column '{col}' not found in training data.")
        if observation_column not in training_data:
            raise KeyError(
                f"Target column '{observation_column}' not found in training data."
            )

        # Check the observation column is not also a training column.
        if observation_column in training_columns:
            raise KeyError(
                f"Observation column '{observation_column}' appears in training data."
            )

        self.observation_column = observation_column
        self.training_columns = training_columns

        # Keep only the columns relevant for training.
        self.training_data = training_data[training_columns + [observation_column]]

        # Merge default params with optional params.
        lightgbm_params = lightgbm_params or {}
        self.lightgbm_params = self.lightgbm_params | lightgbm_params

    def process(self, lead_time, thresholds):
        """Train models for a set of threshold values.

        Args:
            lead_time (int):
                Lead time in hours of training data. Used to get retreive model paths
                from config data.
            thresholds (list of float):
                Threshold values for which the observation column is trained.
        """
        if lead_time not in self.config:
            raise KeyError(f"Lead time {lead_time} not found in model config.")

        for threshold in thresholds:
            if threshold not in self.config[lead_time]:
                raise KeyError(
                    f"Threshold '{threshold}' not found in model config for lead time {lead_time}."
                )

            model_path = Path(self.config[lead_time][threshold]["lightgbm_model"])
            self._train_model(threshold, model_path)

    def _train_model(self, threshold, model_path):
        """Train a model for a particular threshold.

        Args:
            threshold (float):
                Threshold for which the observation column is trained.

        Returns:
            The model object (lightgbm.Booster)
        """
        import lightgbm

        threshold_met_label = (
            self.training_data[self.observation_column] >= float(threshold)
        ).astype(int)

        dataset = lightgbm.Dataset(
            self.training_data[self.training_columns], label=threshold_met_label
        )

        model = lightgbm.train(self.lightgbm_params, dataset)

        Path.mkdir(model_path.parent, parents=True, exist_ok=True)
        model.save_model(model_path)
