# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""RainForests model training plugin."""

import pathlib
from pathlib import Path

import pandas

from improver import BasePlugin
from improver.calibration import lightgbm_package_available


class TrainRainForestsModel(BasePlugin):
    # Default parameters to be passed to lightGBM library
    params = {
        "objective": "binary",
        "num_leaves": 5,
        "seed": 0,
    }

    def __init__(
        self,
        model_config_dict: dict[int, dict[str, dict[str, str]]],
        training_data: pandas.DataFrame,
        observation_column: str,
        training_columns: list[str],
        lightgbm_params: dict | None = None,
    ):
        """Initialise the options used when training models.

        Args:
            model_config_dict:
                Dictionary describing the high-level RainForests model structure;
                - top level key describes the lead-hour,
                - next level key describes the threshold,
                - corresponding values locate the associated model file.
            training_data:
                Combined data set used to train models.
            observation_column:
                The column in the data set to be trained for.
            training_columns:
                Set of columns from the data set to be trained from.
            lightgbm_params:
                Optional. Parameters passed into training library. Any parameters
                here will override the default parameters.

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
            if col not in training_data.columns:
                raise KeyError(f"Training column '{col}' not found in training data.")
        if observation_column not in training_data.columns:
            raise KeyError(
                f"Target column '{observation_column}' not found in training data."
            )

        # Check the observation column is not also a training column.
        if observation_column in training_columns:
            raise KeyError(
                f"Observation column '{observation_column}' is also in training columns."
            )

        self.observation_column = observation_column
        self.training_columns = training_columns

        # Keep only the columns relevant for training.
        self.training_data = training_data[[*training_columns, observation_column]]

        # Merge default params with overrides from constructor argument.
        if lightgbm_params:
            self.params = self.params | lightgbm_params

    def process(self, lead_time: int, thresholds: list[str]):
        """Train models for a set of threshold values.

        Args:
            lead_time:
                Lead time in hours of training data. Used to get retreive model paths
                from config data.
            thresholds:
                Threshold values for which the observation column is trained.
                Formatted to match the keys used in the model_config object.
        """
        if lead_time not in self.config:
            raise KeyError(f"Lead time {lead_time} not found in model config.")

        for threshold in thresholds:
            if threshold not in self.config[lead_time]:
                raise KeyError(
                    f"Threshold '{threshold}' not found in model config for lead time {lead_time}."
                )

        for threshold in thresholds:
            model_path = Path(self.config[lead_time][threshold]["lightgbm_model"])
            self._train_model(float(threshold), model_path)

    def _train_model(self, threshold: float, model_path: pathlib.Path):
        """Train a model for a particular threshold and saves it to disk.

        Args:
            threshold:
                Threshold for which the observation column is trained.
            model_path:
                Full file path where the model should be saved.
        """
        import lightgbm

        threshold_met_label = (
            self.training_data[self.observation_column] >= threshold
        ).astype(int)

        dataset = lightgbm.Dataset(
            self.training_data[self.training_columns], label=threshold_met_label
        )

        model = lightgbm.train(self.params, dataset)

        Path.mkdir(model_path.parent, parents=True, exist_ok=True)
        model.save_model(model_path)
