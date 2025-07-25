#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to load and apply the trained Quantile Regression Random Forest (QRF) model."""

import pathlib

import iris
import joblib
import numpy as np
from iris.cube import Cube, CubeList
from quantile_forest import RandomForestQuantileRegressor

from improver import PostProcessingPlugin
from improver.calibration.quantile_regression_random_forest import (
    ApplyQuantileRegressionRandomForests,
)
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations,
)
from improver.ensemble_copula_coupling.utilities import choose_set_of_percentiles
from improver.utilities.cube_checker import assert_spatial_coords_match


class LoadAndApplyQRF(PostProcessingPlugin):
    """Load and apply the trained Quantile Regression Random Forest (QRF) model."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        target_cube_name: str,
        transformation: str = None,
        pre_transform_addition: float = 0,
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
                "DYNAMIC_VARIABLE_NAME": ["FEATURE1", "FEATURE2"] e.g:
                {
                "air_temperature": ["mean", "std", "altitude"],
                "visibility_at_screen_level": ["mean", "std"]
                "distance_to_water": ["static"],
                }
            target_cube_name (str):
                A string containing the cube name of the forecast to be
                calibrated. This will be used to separate it from the rest of the
                dynamic predictors, if present.
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.

        """
        self.feature_config = feature_config
        self.target_cube_name = target_cube_name
        self.transformation = transformation
        self.pre_transform_addition = pre_transform_addition

    def _get_inputs(
        self, file_paths: pathlib.Path
    ) -> tuple[CubeList, Cube, RandomForestQuantileRegressor]:
        """Get inputs from disk and separate the model and the features.

        Args:
            file_paths: Path to the trained QRF model and the forecast to be calibrated,
                and the features, as required.

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
        forecast_constraint = iris.Constraint(name=self.target_cube_name)
        forecast_cube = cube_inputs.extract(forecast_constraint)

        if forecast_cube:
            (forecast_cube,) = forecast_cube
        else:
            msg = (
                "No target forecast provided. An input file representing the target "
                "must be provided, even if the target will not be used as a feature. "
                f"The target is '{self.target_cube_name}'."
            )
            raise ValueError(msg)

        if not qrf_model:
            return None, forecast_cube, None

        if len(cube_inputs) != len(self.feature_config.keys()):
            msg = (
                "The number of cubes loaded does not match the number of features "
                "expected. The number of cubes loaded was: "
                f"{len(cube_inputs)}. The number of features expected was: "
                f"{len(self.feature_config.keys())}."
            )
            raise ValueError(msg)

        # If target diagnostic not a feature in the training then remove.
        if self.target_cube_name not in self.feature_config.keys():
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
    def _percentiles_to_realizations(cube_inputs: Cube) -> CubeList:
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

        # Ensure there is a realization dimension on all cubes. This assumes a percentile
        # dimension is present.
        realization_cube_inputs = iris.cube.CubeList([])
        for feature_cube in cube_inputs:
            if feature_cube.coords("percentile"):
                feature_cube = RebadgePercentilesAsRealizations()(feature_cube)
            realization_cube_inputs.append(feature_cube)
        cube_inputs = realization_cube_inputs
        return cube_inputs

    @staticmethod
    def _organise_cubes(
        cube_inputs: CubeList, forecast_cube: Cube
    ) -> tuple[CubeList, Cube]:
        """Promote the forecast period and forecast reference time coordinates to be
        dimension coordinates, if present, on the feature cubes and the template
        forecast cube.

        Args:
            cube_inputs: CubeList of feature cubes, which may include the forecast to be
            forecast_cube: Forecast cube for use as a template.

        Returns:
            Feature cubes and template cube with forecast period and
            forecast reference time promoted to dimension coordinates.
        """
        # Ensure that forecast_period is a dimension on all cubes.
        fp_dim_cube_inputs = iris.cube.CubeList([])
        for feature_cube in cube_inputs:
            if feature_cube.coords("forecast_period", dim_coords=False):
                feature_cube = iris.util.new_axis(feature_cube, "forecast_period")
            if feature_cube.coords("forecast_reference_time", dim_coords=False):
                feature_cube = iris.util.new_axis(
                    feature_cube, "forecast_reference_time"
                )
            fp_dim_cube_inputs.append(feature_cube)
        cube_inputs = fp_dim_cube_inputs

        # Ensure the forecast cube has the same dimensions as the features
        template_forecast_cube = iris.util.new_axis(forecast_cube, "forecast_period")
        template_forecast_cube = iris.util.new_axis(
            template_forecast_cube, "forecast_reference_time"
        )

        # Check that the grids are the same for all dynamic predictors and the forecast
        assert_spatial_coords_match(cube_inputs)
        return cube_inputs, template_forecast_cube

    def process(
        self,
        file_paths: pathlib.Path,
    ) -> Cube:
        """Loading and applying the trained model for Quantile Regression Random Forest.

        Load in the previously trained model for Quantile Regression Random
        Forest (QRF). The model is applied to the forecast that is supplied,
        so as to calibrate the forecast. The calibrated forecast is written
        to a cube. If no model is provided the input forecast is returned unchanged.

        Args:
            file_paths (cli.inputpaths):
                A list of input paths containing:
                - The path to a QRF trained model in pickle file format to be used
                for calibration.
                - The path to a NetCDF file containing the forecast to be calibrated.
                - Optionally, paths to NetCDF files containing additional preictors.

        Returns:
            iris.cube.Cube:
                The calibrated forecast cube.
        """
        cube_inputs, forecast_cube, qrf_model = self._get_inputs(file_paths)
        if not qrf_model:
            return forecast_cube
        if forecast_cube.coords("percentile"):
            percentiles = self._compute_percentiles(forecast_cube, "percentile")
            cube_inputs = self._percentiles_to_realizations(cube_inputs)
        elif forecast_cube.coords("realization"):
            percentiles = self._compute_percentiles(forecast_cube, "realization")

        cube_inputs, template_forecast_cube = self._organise_cubes(
            cube_inputs, forecast_cube
        )

        result = ApplyQuantileRegressionRandomForests(
            feature_config=self.feature_config,
            quantiles=percentiles,
            transformation=self.transformation,
            pre_transform_addition=self.pre_transform_addition,
        )(qrf_model, cube_inputs, template_forecast_cube)
        return result
