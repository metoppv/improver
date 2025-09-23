#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to load and apply the trained Quantile Regression Random Forest (QRF)
model."""

import warnings
from typing import Optional

import iris
import numpy as np
import pandas as pd
from iris.cube import Cube, CubeList
from iris.pandas import as_data_frame

from improver import PostProcessingPlugin
from improver.calibration import add_warning_comment
from improver.calibration.quantile_regression_random_forest import (
    ApplyQuantileRegressionRandomForests,
    quantile_forest_package_available,
)
from improver.ensemble_copula_coupling.utilities import choose_set_of_percentiles
from improver.utilities.cube_checker import assert_spatial_coords_match

try:
    from quantile_forest import RandomForestQuantileRegressor
except ModuleNotFoundError:
    # Define empty class to avoid type hint errors.
    class RandomForestQuantileRegressor:
        pass


iris.FUTURE.pandas_ndim = True


class PrepareAndApplyQRF(PostProcessingPlugin):
    """Prepare the input forecast for application of a trained Quantile Regression
    Random Forest (QRF) model and apply the QRF model."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        target_cf_name: str,
        unique_site_id_keys: list[str] = ["wmo_id"],
    ):
        """Initialise the plugin.

        Args:
            feature_config (dict):
                Feature configuration defining the features to be used for quantile
                regression. The configuration is a dictionary of strings, where the
                keys are the names of the input cube(s) supplied, and the values are
                a list. This list can contain both computed features, such as the mean
                or standard deviation (std), or static features, such as the altitude.
                The computed features will be computed using the cube defined in the
                dictionary key. If the key is the feature itself e.g. a distance to
                water cube, then the value should state "static". This will ensure
                the cube's data is used as the feature.
                The config will have the structure:
                "DYNAMIC_VARIABLE_CF_NAME": ["FEATURE1", "FEATURE2"] e.g:
                {
                "air_temperature": ["mean", "std", "altitude"],
                "visibility_at_screen_level": ["mean", "std"]
                "distance_to_water": ["static"],
                }
            target_cf_name (str):
                A string containing the CF name of diagnostic to be calibrated. This
                will be used to separate it from the rest of the dynamic predictors,
                if present.
            unique_site_id_keys (list):
                The names of the coordinates that uniquely identify each site,
                e.g. "wmo_id" or "latitude,longitude".
        """
        self.feature_config = feature_config
        self.target_cf_name = target_cf_name
        self.unique_site_id_keys = unique_site_id_keys
        self.quantile_forest_installed = quantile_forest_package_available()

    def _get_inputs(
        self,
        cube_inputs: iris.cube.CubeList,
        qrf_model: Optional[RandomForestQuantileRegressor] = None,
    ) -> tuple[CubeList, Cube]:
        """Split the forecast to be calibrated from the other features. Handle
        the case where the qrf_model is not provided. In this case, the uncalibrated
        forecast is returned with a warning comment added.

        Args:
            cube_inputs: List of cubes containing the features and the forecast to be
                calibrated.

        Returns:
            CubeList of the features cubes and the forecast cube.

        Raises:
            ValueError: If not target forecast is provided.
            ValueError: If the number of cubes provided does not match the number of
                features expected.
        """
        # Extract all additional cubes which are associated with a feature in the
        # feature_config.
        forecast_constraint = iris.Constraint(name=self.target_cf_name)
        forecast_cube = cube_inputs.extract(forecast_constraint)

        if forecast_cube:
            (forecast_cube,) = forecast_cube
        else:
            msg = (
                "No target forecast provided. An input cube representing the target "
                "must be provided, even if the target will not be used as a feature. "
                f"The target is '{self.target_cf_name}'."
            )
            raise ValueError(msg)

        if not qrf_model:
            # If no model is provided, return the input forecast with a warning.
            forecast_cube = add_warning_comment(forecast_cube)
            return None, forecast_cube

        if len(cube_inputs) != len(self.feature_config.keys()):
            msg = (
                "The number of cubes loaded does not match the number of features "
                "expected. The number of cubes loaded was: "
                f"{len(cube_inputs)}. The number of features expected was: "
                f"{len(self.feature_config.keys())}."
            )
            raise ValueError(msg)

        # If target diagnostic not a feature in the training then remove.
        if self.target_cf_name not in self.feature_config.keys():
            cube_inputs.remove(forecast_cube)

        return cube_inputs, forecast_cube

    @staticmethod
    def _compute_quantile_list(forecast_cube: Cube, coord: str) -> list[float]:
        """Compute the list of quantiles e.g. 0.25, 0.5, 0.75 that will be produced
        by using the forecast cube.

        Args:
            forecast_cube: Forecast to be calibrated.
            coord: Coordinate name. The length of the coordinate will be used to
                determine the number of quantiles to compute.

        Returns:
            List of quantiles (e.g. 0.25, 0.5, 0.75) computed from the forecast cube.
        """
        n_percentiles = len(forecast_cube.coord(coord).points)
        quantiles = (np.array(choose_set_of_percentiles(n_percentiles)) / 100).tolist()
        return quantiles

    def _cube_to_dataframe(self, cube_inputs: CubeList) -> pd.DataFrame:
        """Convert cube inputs to a pandas DataFrame.

        Args:
            cube_inputs: List of cubes containing the features and the forecast to be
                calibrated.
        Returns:
            DataFrame containing the data from the cubes, with auxiliary coordinates
            included as columns.
        """
        # Convert the first cube to a DataFrame.
        df = as_data_frame(cube_inputs[0], add_aux_coords=True).reset_index()

        possible_columns = [
            *self.unique_site_id_keys,
            "time",
            "forecast_reference_time",
            "forecast_period",
        ]

        # Iteratively convert remaining cubes to DataFrame and merge.
        for cube in cube_inputs[1:]:
            temporary_df = as_data_frame(cube, add_aux_coords=True).reset_index()
            merge_columns = [
                col for col in possible_columns if col in temporary_df.columns
            ]
            df = df.merge(
                temporary_df[merge_columns + [cube.name()]],
                on=merge_columns,
                how="left",
            )

        for column in ["forecast_reference_time", "time"]:
            df[column] = df[column].apply(lambda x: x._to_real_datetime())
        return df

    def process(
        self,
        cube_inputs: CubeList,
        qrf_descriptors: Optional[
            tuple[RandomForestQuantileRegressor, str, float]
        ] = None,
    ) -> Cube:
        """Load and applying the trained Quantile Regression Random Forest (QRF) model.
        The model is applied to the forecast supplied to calibrate the forecast.
        The calibrated forecast is written to a cube. If no model is provided the
        input forecast is returned unchanged.

        Args:
            cube_inputs: List of cubes containing the features and the forecast to be
                calibrated.
            qrf_descriptors: The trained QRF model to be applied to the forecast
                and the transformation and pre-transform addition applied during
                training. The descriptors expected are a tuple of:
                (qrf_model, transformation, pre_transform_addition).

        Returns:
            iris.cube.Cube:
                The calibrated forecast cube.
        """
        if qrf_descriptors is None:
            # If no descriptors are provided, return the input forecast with a warning.
            # Descriptors expected: (qrf_model, transformation, pre_transform_addition)
            qrf_descriptors = (None, None, 0)
        qrf_model, transformation, pre_transform_addition = qrf_descriptors

        cube_inputs, forecast_cube = self._get_inputs(cube_inputs, qrf_model=qrf_model)

        if cube_inputs:
            assert_spatial_coords_match(cube_inputs)

        if not self.quantile_forest_installed or not qrf_model:
            msg = "Unable to apply Quantile Regression Random Forest model."
            if not self.quantile_forest_installed:
                msg += " The 'quantile_forest' package is not installed."
            elif not qrf_model:
                msg += " No trained model has been provided."
            msg += " Returning the input forecast without calibration."
            warnings.warn(msg)
            return forecast_cube

        template_forecast_cube = forecast_cube.copy()
        if forecast_cube.coords("realization"):
            quantile_list = self._compute_quantile_list(
                forecast_cube.copy(), "realization"
            )
        elif forecast_cube.coords("percentile"):
            quantile_list = (forecast_cube.coord("percentile").points / 100.0).tolist()

        df = self._cube_to_dataframe(cube_inputs)

        calibrated_forecast = ApplyQuantileRegressionRandomForests(
            target_name=self.target_cf_name,
            feature_config=self.feature_config,
            quantiles=quantile_list,
            transformation=transformation,
            pre_transform_addition=pre_transform_addition,
            unique_site_id_keys=self.unique_site_id_keys,
        )(qrf_model, df)
        calibrated_forecast_cube = template_forecast_cube.copy(
            data=np.broadcast_to(calibrated_forecast.T, template_forecast_cube.shape)
        )

        return calibrated_forecast_cube
