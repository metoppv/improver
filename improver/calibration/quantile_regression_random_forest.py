# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugins to perform quantile regression using random forests."""

from typing import Optional

import joblib
import numpy as np
import pandas as pd
from iris.cube import Cube
from quantile_forest import RandomForestQuantileRegressor

from improver import BasePlugin, PostProcessingPlugin
from improver.constants import DAYS_IN_YEAR, HOURS_IN_DAY


def prep_feature(
    df: pd.DataFrame,
    feature_name: str,
    feature: str,
) -> pd.DataFrame:
    if feature in ["mean", "std"]:
        representation_name = [
            n for n in ["percentile", "realization"] if n in df.columns
        ][0]
        subset_cols = [
            "forecast_reference_time",
            "forecast_period",
            "wmo_id",
            representation_name,
            feature_name,
        ]
        groupby_cols = ["forecast_reference_time", "forecast_period", "wmo_id"]
        if feature == "mean":
            subset_df = df[subset_cols].groupby(groupby_cols).mean()
        elif feature == "std":
            subset_df = df[subset_cols].groupby(groupby_cols).std()

        subset_df = subset_df.reset_index()
        subset_df.rename(
            columns={feature_name: f"{feature_name}_{feature}"}, inplace=True
        )
        # df.drop(columns=[representation_name], inplace=True)
        # df.drop_duplicates(inplace=True)
        df = df.merge(
            subset_df[groupby_cols + [f"{feature_name}_{feature}"]],
            on=["forecast_reference_time", "forecast_period", "wmo_id"],
            how="left",
        )

    elif feature in ["day_of_year", "day_of_year_sin", "day_of_year_cos"]:
        day_of_year = df["time"].dt.strftime("%j")
        day_of_year = np.array(day_of_year, dtype=np.int32)
        if feature == "day_of_year":
            feature_values = day_of_year
        elif feature == "day_of_year_sin":
            feature_values = np.sin(
                2 * np.pi * day_of_year / (DAYS_IN_YEAR + 1)
            ).astype(np.float32)
        elif feature == "day_of_year_cos":
            feature_values = np.cos(
                2 * np.pi * day_of_year / (DAYS_IN_YEAR + 1)
            ).astype(np.float32)
        df[feature] = feature_values
    elif feature in ["hour_of_day", "hour_of_day_sin", "hour_of_day_cos"]:
        hour_of_day = df["time"].dt.hour
        hour_of_day = np.array(hour_of_day, dtype=np.int32)
        if feature == "hour_of_day":
            feature_values = hour_of_day
        elif feature == "hour_of_day_sin":
            feature_values = np.sin(2 * np.pi * hour_of_day / HOURS_IN_DAY).astype(
                np.float32
            )
        elif feature == "hour_of_day_cos":
            feature_values = np.cos(2 * np.pi * hour_of_day / HOURS_IN_DAY).astype(
                np.float32
            )
        df[feature] = feature_values
    return df


def sanitise_forecast_dataframe(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    representation_name = [n for n in ["percentile", "realization"] if n in df.columns][
        0
    ]
    collapsed_features = []
    for key, values in feature_config.items():
        collapsed_features.extend([key for v in values if v in ["mean", "std"]])
    collapsed_features = list(set(collapsed_features))

    df = df.drop(columns=[representation_name, *collapsed_features]).drop_duplicates()
    return df


def select_features(df, feature_config):
    """Prepare the features from a DataFrame for quantile regression random forest."""

    # Remove columns that are no longer required e.g. as the mean or std have been computed.
    # Removing these columns allows a lot of duplicates to then be removed.

    feature_column_names = []
    for feature_name in feature_config.keys():
        for feature in feature_config[feature_name]:
            if feature in ["mean", "std"]:
                feature_column_names.append(f"{feature_name}_{feature}")
            elif feature in ["static"]:
                feature_column_names.append(feature_name)
            else:
                feature_column_names.append(feature)

    if len(list(set(feature_column_names) - set(df.columns))) > 0:
        msg = f"Feature '{feature}' is not supported."
        raise ValueError(msg)

    return feature_column_names


def _check_valid_transformation(transformation: str):
    """Check if the transformation is one of the supported types.
    Args:
        transformation: Transformation to be checked.
    Raises:
        ValueError: If the transformation is not one of the supported types.
    """
    if transformation not in ["log", "log10", "sqrt", "cbrt", None]:
        msg = (
            "Currently the only supported transformations are log, log10, sqrt "
            f"and cbrt. The transformation supplied was {transformation}."
        )
        raise ValueError(msg)


class TrainQuantileRegressionRandomForests(BasePlugin):
    """Plugin to train a model using quantile regression random forests."""

    def __init__(
        self,
        target_name: str,
        feature_config: dict[str, list[str]],
        n_estimators: int,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
        transformation: Optional[str] = None,
        pre_transform_addition: np.float32 = 0,
        compression: int = 5,
        model_output: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialise the plugin.
        Args:
            target_name (str):
                Name of the target variable to be calibrated e.g. 'air_temperature'.
            feature_config (dict):
                Feature configuration defining the features to be used for quantile
                regression. The configuration is a dictionary of strings, where the
                keys are the names of the columns within the dataframe. Some
                features may be used as initially provided within the dataframe,
                whilst others may be computed from the data e.g. mean, std.
                If the key is the feature itself e.g. distance to water, then the value
                should state "static". In this case, the name of feature e.g.
                'distance_to_water' is expected to be a column name in the input
                dataframe. The config will have the structure:
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
        self.target_name = target_name
        self.feature_config = feature_config
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.transformation = transformation
        _check_valid_transformation(self.transformation)
        self.pre_transform_addition = pre_transform_addition
        self.compression = compression
        self.output = model_output
        self.kwargs = kwargs
        self.expected_coordinate_order = ["forecast_reference_time", "forecast_period"]

    def fit_qrf(
        self, forecast_features: np.ndarray, target: np.ndarray
    ) -> RandomForestQuantileRegressor:
        """Fit the quantile regression random forest model.
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
        forecast_df: pd.DataFrame,
        truth_df: pd.DataFrame,
    ) -> None:
        """Train a quantile regression random forests model.
        Args:
            forecast_df:
                DataFrame containing the forecast information and features.
            truth_df:
                Cube containing the truths. The truths should have the same validity
                times as the forecasts.

        References:
            Johnson. (2024). quantile-forest: A Python Package for Quantile
            Regression Forests. Journal of Open Source Software, 9(93), 5976.
            https://doi.org/10.21105/joss.05976.
            Meinshausen, N. (2006). Quantile regression forests.
            Journal of Machine Learning Research,
            7(35), 983–999. http://jmlr.org/papers/v7/meinshausen06a.html
            Taillardat, M., O. Mestre, M. Zamo, and P. Naveau, 2016: Calibrated
            Ensemble Forecasts Using Quantile Regression Forests and Ensemble Model
            Output Statistics. Mon. Wea. Rev., 144, 2375–2393,
            https://doi.org/10.1175/MWR-D-15-0260.1.
            Taillardat, M. and Mestre, O.: From research to applications – examples of
            operational ensemble post-processing in France using machine learning,
            Nonlin. Processes Geophys., 27, 329–347,
            https://doi.org/10.5194/npg-27-329-2020, 2020.
        """

        if self.transformation:
            forecast_df[self.target_name] = getattr(np, self.transformation)(
                forecast_df[self.target_name] + self.pre_transform_addition
            )
            truth_df["ob_value"] = getattr(np, self.transformation)(
                truth_df["ob_value"] + self.pre_transform_addition
            )

        for feature_name in self.feature_config.keys():
            if feature_name not in forecast_df.columns:
                msg = f"Feature '{feature_name}' is not present in the forecast DataFrame."
                raise ValueError(msg)
            for feature in self.feature_config[feature_name]:
                forecast_df = prep_feature(forecast_df, feature_name, feature)

        forecast_df = sanitise_forecast_dataframe(forecast_df, self.feature_config)
        feature_column_names = select_features(forecast_df, self.feature_config)

        merge_columns = ["wmo_id", "time"]
        combined_df = forecast_df.merge(
            truth_df[merge_columns + ["ob_value"]], on=merge_columns, how="inner"
        )
        feature_values = np.array(combined_df[feature_column_names])
        target_values = combined_df["ob_value"].values

        print(feature_values, target_values)
        # Fit the quantile regression model
        qrf_model = self.fit_qrf(feature_values, target_values)

        joblib.dump(qrf_model, self.output, compress=self.compression)


class ApplyQuantileRegressionRandomForests(PostProcessingPlugin):
    """Plugin to apply a trained model using quantile regression random forests."""

    def __init__(
        self,
        target_name: str,
        feature_config: dict[str, list[str]],
        quantiles: list[np.float32],
        transformation: str = None,
        pre_transform_addition: np.float32 = 0,
    ) -> None:
        """Initialise the plugin.
        Args:
            target_name (str):
                Name of the target variable to be calibrated.
            feature_config (dict):
                Feature configuration defining the features to be used for quantile
                regression. The configuration is a dictionary of strings, where the
                keys are the names of the columns within the dataframe. Some
                features may be used as initially provided within the dataframe,
                whilst others may be computed from the data e.g. mean, std.
                If the key is the feature itself e.g. distance to water, then the value
                should state "static". In this case, the name of feature e.g.
                'distance_to_water' is expected to be a column name in the input
                dataframe. The config will have the structure:
                    "DYNAMIC_VARIABLE_NAME": ["FEATURE1", "FEATURE2"] e.g:
                    {
                    "air_temperature": ["mean", "std", "altitude"],
                    "visibility_at_screen_level": ["mean", "std"]
                    "distance_to_water": ["static"],
                    }
            quantiles (float):
                Quantiles used for prediction (values ranging from 0 to 1)
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.

        Raises:
            ValueError: If the transformation is not one of the supported types.
        """
        self.target_name = target_name
        self.feature_config = feature_config
        self.quantiles = quantiles
        self.transformation = transformation
        _check_valid_transformation(self.transformation)
        self.pre_transform_addition = pre_transform_addition

    def _reverse_transformation(self, forecast: np.ndarray) -> np.ndarray:
        """Reverse the transformation applied to the data prior to fitting the QRF.
        The forecast cube provided is modified in place.

        Args:
            forecast_cube: Forecast to be calibrated.
        """
        if self.transformation:
            if self.transformation == "log":
                forecast = np.exp(forecast) - self.pre_transform_addition
            elif self.transformation == "log10":
                forecast = 10 ** (forecast) - self.pre_transform_addition
            elif self.transformation == "sqrt":
                forecast = forecast**2 - self.pre_transform_addition
            elif self.transformation == "cbrt":
                forecast = forecast**3 - self.pre_transform_addition
        return forecast

    def process(
        self,
        qrf_model: RandomForestQuantileRegressor,
        forecast_df: pd.DataFrame,
    ) -> Cube:
        """Apply a quantile regression random forests model.

        Args:
            qrf_model: A trained QRF model.
            feature_cubes: CubeList of features. This should include the forecast to be
                calibrated, if that has been used as a feature in the training, and
                any features that can be provided as cubes.
            template_forecast_cube: Template forecast cube that provides the required
                metadata and shape for the output cube. The data from this cube will not
                be used.

        Returns:
            Calibrated forecast cube with the same metadata as the template
            forecast cube.

        """
        feature_values = []

        for feature_name in self.feature_config.keys():
            # Transform the feature cube data if a transformation is specified.
            if (
                self.transformation
                and set(["mean", "std"]).intersection(self.feature_config[feature_name])
                and self.target_name in forecast_df.columns
            ):
                forecast_df[self.target_name] = getattr(np, self.transformation)(
                    forecast_df[self.target_name] + self.pre_transform_addition
                )

            for feature in self.feature_config[feature_name]:
                forecast_df = prep_feature(forecast_df, feature_name, feature)

        forecast_df = sanitise_forecast_dataframe(forecast_df, self.feature_config)
        feature_column_names = select_features(forecast_df, self.feature_config)

        feature_values = np.array(forecast_df[feature_column_names])

        calibrated_forecast = qrf_model.predict(
            feature_values, quantiles=self.quantiles
        )
        calibrated_forecast = np.float32(calibrated_forecast)

        calibrated_forecast = self._reverse_transformation(calibrated_forecast)

        return calibrated_forecast
