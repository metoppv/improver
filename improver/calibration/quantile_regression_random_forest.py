# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugins to perform quantile regression using random forests."""

from typing import Optional

import iris
import joblib
import numpy as np
from iris.cube import Cube, CubeList
from quantile_forest import RandomForestQuantileRegressor

from improver import BasePlugin, PostProcessingPlugin
from improver.constants import DAYS_IN_YEAR, HOURS_IN_DAY


def prep_feature(
    template_cube: Cube,
    feature_cube: Cube,
    feature: str,
):
    """Prepare the feature values for the quantile regression random forest model.
    Args:
        template_cube (cube):
            The forecast cube that acts only as a template.
        feature_cube (cube):
            Feature cube that is either static or dynamic.
        feature:
            The feature to be extracted from the associated feature_cube.
            If "static" then the cube itself acts as the feature.
    Returns:
        feature_values (numpy.ndarray):
            Flattened array of feature values.
    """

    collapsed_cube = template_cube.collapsed(["realization"], iris.analysis.MEAN)

    if "mean" == feature:
        feature_values = feature_cube.collapsed(
            ["realization"], iris.analysis.MEAN
        ).data.flatten()
    elif "std" == feature:
        feature_values = feature_cube.collapsed(
            ["realization"], iris.analysis.STD_DEV
        ).data.flatten()
    elif feature in ["latitude", "longitude", "altitude"]:
        coord_multidim = feature_cube.coord(feature).points[np.newaxis, :]
        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
    elif feature == "forecast_period":
        coord_multidim = feature_cube.coord("forecast_period").points[:, np.newaxis]
        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
    elif feature in ["day_of_year", "day_of_year_sin", "day_of_year_cos"]:
        time_coord = feature_cube.coord("time").copy()
        day_of_year = np.array([c.point.strftime("%j") for c in time_coord.cells()])
        coord_multidim = day_of_year[:, np.newaxis]
        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
        if feature == "day_of_year_sin":
            feature_values = np.sin(2 * np.pi * feature_values / HOURS_IN_DAY)
        elif feature == "day_of_year_cos":
            feature_values = np.cos(2 * np.pi * feature_values / HOURS_IN_DAY)
    elif feature in ["hour_of_day", "hour_of_day_sin", "hour_of_day_cos"]:
        hour_of_day = np.zeros(feature_cube.coord("time").shape)
        for i in range(feature_cube.coord("time").shape[0]):
            for j in range(feature_cube.coord("time").shape[1]):
                hour_of_day[i, j] = feature_cube.coord("time")[i][j].cell(0).point.hour
        hour_of_day = np.array(hour_of_day)[:, np.newaxis, :]
        feature_values = np.broadcast_to(hour_of_day, collapsed_cube.shape).flatten()
        if feature == "day_of_year_sin":
            feature_values = np.sin(2 * np.pi * feature_values / (DAYS_IN_YEAR + 1))
        elif feature == "day_of_year_cos":
            feature_values = np.cos(2 * np.pi * feature_values / (DAYS_IN_YEAR + 1))
    elif feature == "static":
        coord_multidim = feature_cube.data[np.newaxis, :, np.newaxis]
        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape)

    return feature_values


class TrainQuantileRegressionRandomForests(BasePlugin):
    """Plugin to train a model using quantile regression random forests."""

    def __init__(
        self,
        experiment,
        feature_config,
        n_estimators,
        max_depth=None,
        random_state=None,
        transformation=None,
        pre_transform_addition=0,
        compression=5,
        model_output=None,
        **kwargs,
    ):
        """Initialise the plugin.
        Args:
            experiment (str):
                The name of the experiment (step) that calibration is applied to.
            feature_config (dict):
                Feature configuration defining the features to be used for quantile regression.
                The configuration is a dictionary of strings, where the keys are the names of
                the input cube(s) supplied, and the values are a list. This list can contain both
                computed features, such as the mean or standard deviation (std), or static
                features, such as the altitude. The computed features will be computed using
                the cube defined in the dictionary key. If the key is the feature itself e.g.
                a distance to water cube, then the value should state "static". This will ensure
                the cube's data is used as the feature.
                The config will have the structure:
                "DYNAMIC_VARIABLE_NAME": ["FEATURE1", "FEATURE2"] e.g:
                {
                "air_temperature": ["mean", "std", "altitude"],
                "visibility_at_screen_level": ["mean", "std"]
                "distance_to_water": ["static"],
                }
            n_estimators (int):
                Number of trees in the forest.
            max_depth (int):
                Maximum depth of the tree.
            random_state (int):
                Random seed for reproducibility.
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.
            compression (int):
                Compression level for saving the model.
            model_output (str):
                Full path including model file name that will store the pickled model.
            kwargs:
                Additional keyword arguments for the quantile regression model.
        """

        self.feature_config = feature_config
        self.experiment = experiment
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.transformation = transformation
        if self.transformation not in ["log", "log10", "sqrt", "cbrt", None]:
            msg = (
                "Currently the only supported transformations are log, log10, sqrt "
                f"and cbrt. The transformation supplied was {self.transformation}."
            )
            raise ValueError(msg)
        self.pre_transform_addition = pre_transform_addition
        self.compression = compression
        self.output = model_output
        self.kwargs = kwargs

    def fit_qrf(
        self, forecast_features: np.ndarray, target: np.ndarray
    ) -> RandomForestQuantileRegressor:
        """Fit the quantile regression model.
        Args:
            forecast_features (numpy.ndarray):
                Array of forecast features.
            target (numpy.ndarray):
                Array of target values.
        Returns:
            qrf_model (RandomForestQuantileRegressor):
                Fitted quantile regression model.
        """

        qrf_model = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **self.kwargs,
        )
        qrf_model.fit(forecast_features, target)
        return qrf_model

    def process(
        self,
        forecast_cube: Cube,
        truth_cube: Cube,
        feature_cubes: Optional[CubeList] = None,
    ):
        """Train a quantile regression random forests model.
        Args:
            forecast_cube:
                Cube containing the forecasts.
            truth_cube:
                Cube containing the truths.
            feature_cubes:
                List of additional feature cubes.
        """

        if self.transformation:
            forecast_cube.data = getattr(np, self.transformation)(
                forecast_cube.data + self.pre_transform_addition
            )
            truth_cube.data = getattr(np, self.transformation)(
                truth_cube.data + self.pre_transform_addition
            )

        feature_values = []

        for feature_name in self.feature_config.keys():
            feature_cube = feature_cubes.extract(iris.Constraint(feature_name))
            for feature in self.feature_config[feature_name]:
                print(feature)
                feature_values.append(
                    prep_feature(forecast_cube, feature_cube[0], feature)
                )

        feature_values = np.array(feature_values).T
        target_values = truth_cube.data.flatten()

        # Fit the quantile regression model
        qrf_model = self.fit_qrf(feature_values, target_values)

        joblib.dump(qrf_model, self.output, compress=self.compression)


class ApplyQuantileRegressionRandomForests(PostProcessingPlugin):
    """Plugin to apply a trained model using quantile regression random forests."""

    def __init__(
        self,
        feature_config,
        quantiles,
        n_estimators,
        transformation,
        pre_transform_addition=0,
    ):
        """Initialise the plugin.
        Args:
            feature_config (dict):
                Feature configuration defining the features to be used for quantile regression.
                The configuration is a dictionary of strings, where the keys are the names of
                the input cube(s) supplied, and the values are a list. This list can contain both
                computed features, such as the mean or standard deviation (std), or static
                features, such as the altitude. The computed features will be computed using
                the cube defined in the dictionary key. If the key is the feature itself e.g.
                a distance to water cube, then the value should state "static". This will ensure
                the cube's data is used as the feature.
                The config will have the structure:
                "DYNAMIC_VARIABLE_NAME": ["FEATURE1", "FEATURE2"] e.g:
                {
                "air_temperature": ["mean", "std", "altitude"],
                "visibility_at_screen_level": ["mean", "std"]
                "distance_to_water": ["static"],
                }
            quantiles (float):
                Quantiles used for prediction (values ranging from 0 to 1)
            n_estimators (int):
                Number of trees in the forest.
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.
        """

        self.feature_config = feature_config
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.transformation = transformation
        if self.transformation not in ["log", "log10", "sqrt", "cbrt", None]:
            msg = (
                "Currently the only supported transformations are log, log10, sqrt "
                f"and cbrt. The transformation supplied was {self.transformation}."
            )
            raise ValueError(msg)
        self.pre_transform_addition = pre_transform_addition

    def process(
        self,
        forecast_cube: Cube,
        template_forecast_cube,
        qrf_model,
        feature_cubes: Optional[CubeList] = None,
    ):
        if self.transformation:
            if self.transformation == "log":
                forecast_cube.data = (
                    np.exp(forecast_cube.data) - self.pre_transform_addition
                )
            elif self.transformation == "log10":
                forecast_cube.data = (
                    10 ** (forecast_cube.data) - self.pre_transform_addition
                )
            elif self.transformation == "sqrt":
                forecast_cube.data = forecast_cube.data**2 - self.pre_transform_addition
            elif self.transformation == "cbrt":
                forecast_cube.data = forecast_cube.data**3 - self.pre_transform_addition

        feature_values = []

        for feature_name in self.feature_config.keys():
            feature_cube = feature_cubes.extract(iris.Constraint(feature_name))
            for feature in self.feature_config[feature_name]:
                feature_values.append(
                    prep_feature(template_forecast_cube, feature_cube[0], feature)
                )

        feature_values = np.array(feature_values).T
        calibrated_forecast = qrf_model.predict(feature_values, self.quantiles)
        calibrated_forecast = np.float32(calibrated_forecast)
        calibrated_forecast_cube = forecast_cube.copy(data=calibrated_forecast.T)
        return calibrated_forecast_cube
