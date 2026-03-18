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
        training_data,
        observation_column,
        training_columns,
        output_dir,
        lightgbm_params=None,
        compiler=None,
    ):
        """Initialise the options used when compiling models.

        Args:
            training_data (pandas.DataFrame):
                Combined data set used to train models.
            observation_column (str):
                The column in the data set to be trained for.
            training_columns (List(str)):
                Set of columns from the data set to be trained from.
            output_dir (str or Path):
                Directory path where model files will be saved.
                Filenames will be generated based on threshold.
            lightgbm_params (Dict):
                Optional. Parameters passed into training library.
            compiler (CompileRainForestsModel):
                Optional. Object used to compile trained models.
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

        self.output_dir = Path(output_dir)

        # Merge default params with optional params.
        lightgbm_params = lightgbm_params or {}
        self.lightgbm_params = self.lightgbm_params | lightgbm_params

        self.compiler = compiler

        # Set a default filename formatter
        self._model_file_name = lambda threshold: (
            f"lgb_model-threshold_{threshold:04.2f}.txt"
        )

    @property
    def model_file_name_formatter(self):
        return self._model_file_name

    @model_file_name_formatter.setter
    def model_file_name_formatter(self, file_name_fn):
        """Return elapsed time in seconds."""
        self._model_file_name = file_name_fn

    def process(self, thresholds, compile=False):
        """Train models for a set of threshold values.

        Args:
            thresholds (list of float):
                Thresholds for which the observation column is trained.
            compile (Bool):
                Whether to also compile the model.
                Defaults to False.
        """

        if compile and not self.compiler:
            raise ValueError("Compile option used when compiler not present.")

        for threshold in thresholds:
            threshold_path = self.output_dir / self._model_file_name(threshold)

            model = self._train_model(threshold)
            model.save_model(threshold_path)

            if compile:
                self.compiler.process(threshold_path, self.output_dir)

    def _train_model(self, threshold):
        """Train a model for a particular threshold.

        Args:
            threshold (float):
                Threshold for which the observation column is trained.

        Returns:
            The model object (lightgbm.Booster)
        """
        import lightgbm

        threshold_met = (
            self.training_data[self.observation_column] >= threshold
        ).astype(int)

        dataset = lightgbm.Dataset(
            self.training_data[self.training_columns], label=threshold_met
        )

        return lightgbm.train(self.lightgbm_params, dataset)
