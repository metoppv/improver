#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to load and apply the trained Quantile Regression Random Forest (QRF)
model."""

import pathlib
from typing import Optional

import iris
import joblib
import numpy as np
import pandas as pd
from iris.cube import Cube, CubeList
from iris.pandas import as_data_frame

from improver import PostProcessingPlugin
from improver.calibration import add_warning_comment
from improver.calibration.quantile_regression_random_forest import (
    ApplyQuantileRegressionRandomForests, quantile_forest_package_available
)
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations,
)
from improver.ensemble_copula_coupling.utilities import choose_set_of_percentiles
from improver.utilities.cube_checker import assert_spatial_coords_match

try:
    from quantile_forest import RandomForestQuantileRegressor
except ModuleNotFoundError:
    pass

iris.FUTURE.pandas_ndim = True


class LoadAndApplyQRF(PostProcessingPlugin):
    """Load and apply the trained Quantile Regression Random Forest (QRF) model."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        target_cf_name: str,
        transformation: Optional[str] = None,
        pre_transform_addition: Optional[float] = None,
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
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.

        """

        self.feature_config = feature_config
        self.target_cf_name = target_cf_name
        self.transformation = transformation
        self.pre_transform_addition = pre_transform_addition

    def _get_inputs(
        self, file_paths: list[pathlib.Path]
    ) -> tuple[CubeList, Cube, RandomForestQuantileRegressor]:
        """Get inputs from disk and separate the model and the features.

        Args:
            file_paths: List of paths to the trained QRF model and the forecast to be
                calibrated and the features, as required.

        Returns:
            CubeList of the features cubes, the forecast cube, and the
            trained QRF model.

        Raises:
            ValueError: If no QRF model is found in the provided file paths.
            ValueError: If no features are found in the provided file paths.
            ValueError: If the number of inputs does not match the number of file paths.
        """
        cube_inputs = iris.cube.CubeList([])
        qrf_model = None

        for file_path in file_paths:
            try:
                cube = iris.load_cube(file_path)
                cube_inputs.append(cube)
            except ValueError:
                qrf_model = joblib.load(file_path)

        if not cube_inputs:
            msg = (
                "No features found in the provided file paths. "
                "At least one feature must be provided."
            )
            raise ValueError(msg)

        # Extract all additional cubes which are associated with a feature in the
        # feature_config.
        forecast_constraint = iris.Constraint(name=self.target_cf_name)
        forecast_cube = cube_inputs.extract(forecast_constraint)

        if forecast_cube:
            (forecast_cube,) = forecast_cube
        else:
            msg = (
                "No target forecast provided. An input file representing the target "
                "must be provided, even if the target will not be used as a feature. "
                f"The target is '{self.target_cf_name}'."
            )
            raise ValueError(msg)

        if not qrf_model:
            forecast_cube = add_warning_comment(forecast_cube)
            return None, forecast_cube, None

        if len(cube_inputs) != len(self.feature_config.keys()):
            msg = (
                "The number of cubes loaded does not match the number of features "
                "expected. The number of cubes loaded was: "
                f"{len(cube_inputs)}. The number of features expected was: "
                f"{len(self.feature_config.keys())}."
            )
            raise ValueError(msg)

        if not qrf_model:
            # The specified model doesn't exist and the forecast will not be calibrated
            return forecast_cube

        # If target diagnostic not a feature in the training then remove.
        if self.target_cf_name not in self.feature_config.keys():
            cube_inputs.remove(forecast_cube)

        return cube_inputs, forecast_cube, qrf_model

    @staticmethod
    def _compute_percentiles(forecast_cube: Cube, coord: str) -> list[float]:
        """Compute the percentiles from the forecast cube.

        Args:
            forecast_cube: Forecast to be calibrated.
            coord: Coordinate name. The length of the coordinate will be used to
                determine the number of percentiles to compute.

        Returns:
            List of percentiles computed from the forecast cube.
        """
        n_percentiles = len(forecast_cube.coord(coord).points)
        percentiles = (
            np.array(choose_set_of_percentiles(n_percentiles)) / 100
        ).tolist()
        return percentiles

    @staticmethod
    def _percentiles_to_realizations(cube_inputs: CubeList) -> CubeList:
        """Convert percentiles to realizations. The input forecasts are expected to
        be percentiles but these percentiles are rebadged as realizations.

        Args:
            cube_inputs:
                List of cubes containing the features and the forecast to be calibrated.
                Some may be percentiles.
        Returns:
            cube_inputs:
                List of cubes with percentiles rebadged as realizations,
                where appropriate
        """

        # Ensure there is a realization dimension on all cubes. This assumes a
        # percentile dimension is present.
        realization_cube_inputs = iris.cube.CubeList([])
        for feature_cube in cube_inputs:
            if feature_cube.coords("percentile"):
                feature_cube = RebadgePercentilesAsRealizations()(feature_cube)
            realization_cube_inputs.append(feature_cube)
        return realization_cube_inputs

    @staticmethod
    def _cube_to_dataframe(cube_inputs: CubeList) -> pd.DataFrame:
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

        # Iteratively convert remaining cubes to DataFrame and merge.
        for cube in cube_inputs[1:]:
            temporary_df = as_data_frame(cube, add_aux_coords=True).reset_index()
            possible_columns = [
                "wmo_id",
                "time",
                "forecast_reference_time",
                "forecast_period",
            ]
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
        file_paths: list[pathlib.Path],
    ) -> Cube:
        """Load and applying the trained Quantile Regression Random Forest (QRF) model.
        The model is applied to the forecast supplied to calibrate the forecast.
        The calibrated forecast is written to a cube. If no model is provided the
        input forecast is returned unchanged.

        Args:
            file_paths (cli.inputpaths):
                A list of input paths containing:
                - The path to a QRF trained model in pickle file format to be used
                for calibration.
                - The path to a NetCDF file containing the forecast to be calibrated.
                - Optionally, paths to NetCDF files containing additional predictors.

        Returns:
            iris.cube.Cube:
                The calibrated forecast cube.
        """
        cube_inputs, forecast_cube, qrf_model = self._get_inputs(file_paths)
        import pdb; pdb.set_trace()
        if not quantile_forest_package_available():
            return forecast_cube
        if not qrf_model:
            return forecast_cube

        template_forecast_cube = forecast_cube.copy()
        if forecast_cube.coords("percentile"):
            percentiles = self._compute_percentiles(forecast_cube.copy(), "percentile")
        elif forecast_cube.coords("realization"):
            percentiles = self._compute_percentiles(forecast_cube.copy(), "realization")

        assert_spatial_coords_match(cube_inputs)
        df = self._cube_to_dataframe(cube_inputs)

        calibrated_forecast = ApplyQuantileRegressionRandomForests(
            target_name=self.target_cf_name,
            feature_config=self.feature_config,
            quantiles=percentiles,
            transformation=self.transformation,
            pre_transform_addition=self.pre_transform_addition,
        )(qrf_model, df)
        calibrated_forecast_cube = template_forecast_cube.copy(
            data=np.broadcast_to(calibrated_forecast.T, template_forecast_cube.shape)
        )

        return calibrated_forecast_cube
