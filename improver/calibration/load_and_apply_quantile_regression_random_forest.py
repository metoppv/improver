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
    def process(
        self,
        file_paths: pathlib.Path,
        feature_config: dict,
        target_cube_name: str,
        n_estimators: int = 100,
        transformation: str = None,
        pre_transform_addition: float = 0,
    ):
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
            target_cube_name (str):
                A string containing the cube name of the forecast to be
                calibrated. This will be used to separate it from the rest of the
                dynamic predictors, if present.
            n_estimators (int):
                Number of trees in the forest.
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.
        Returns:
            iris.cube.Cube:
                The calibrated forecast cube.
        """

        cube_inputs = iris.cube.CubeList([])

        for file_path in file_paths:
            try:
                cube = iris.load_cube(file_path)
                cube_inputs.append(cube)
            except ValueError:
                qrf_model = joblib.load(file_path)

        # Extract all additional cubes which are associated with a feature in the
        # feature_config.

        forecast_constraint = iris.Constraint(name=target_cube_name)
        forecast_cube = cube_inputs.extract_cube(forecast_constraint)

        # If target diagnostic not a feature in the training then remove.
        if target_cube_name not in feature_config.keys():
            cube_inputs.remove(forecast_cube)

        # Calculate quantiles for the model fit
        n_percentiles = 19
        percentiles = (
            np.array(choose_set_of_percentiles(n_percentiles)) / 100
        ).tolist()

        # Ensure there is a realization dimension on all cubes. This assumes a percentile
        # dimension is present.
        realization_cube_inputs = iris.cube.CubeList([])
        for feature_cube in cube_inputs:
            try:
                feature_cube.coord("realization")
                realization_cube_inputs.append(feature_cube)
            except iris.exceptions.CoordinateNotFoundError:
                feature_cube = RebadgePercentilesAsRealizations()(feature_cube)
                realization_cube_inputs.append(feature_cube)
        cube_inputs = realization_cube_inputs

        # Ensure the feature cubes have dimensions that can be used in the prep_feature function

        fp_dim_cube_inputs = iris.cube.CubeList([])
        for feature_cube in cube_inputs:
            try:
                feature_cube = iris.util.new_axis(feature_cube, "forecast_period")
                fp_dim_cube_inputs.append(feature_cube)
            except ValueError:
                fp_dim_cube_inputs.append(feature_cube)
        cube_inputs = fp_dim_cube_inputs

        frt_dim_cube_inputs = iris.cube.CubeList([])
        for feature_cube in cube_inputs:
            try:
                feature_cube = iris.util.new_axis(
                    feature_cube, "forecast_reference_time"
                )
                frt_dim_cube_inputs.append(feature_cube)
            except ValueError:
                frt_dim_cube_inputs.append(feature_cube)
        cube_inputs = frt_dim_cube_inputs

        # Ensure the forecast cube has the same dimensions as the features
        template_forecast_cube = iris.util.new_axis(forecast_cube, "forecast_period")
        template_forecast_cube = iris.util.new_axis(
            template_forecast_cube, "forecast_reference_time"
        )

        # Check that the grids are the same for all dynamic predictors and the forecast
        assert_spatial_coords_match(cube_inputs)

        if len(cube_inputs) + 1 != len(file_paths):
            raise ValueError("Unable to identify the correct number of inputs")

        result = ApplyQuantileRegressionRandomForests(
            feature_config=feature_config,
            quantiles=percentiles,
            n_estimators=n_estimators,
            transformation=transformation,
            pre_transform_addition=pre_transform_addition,
        )(forecast_cube, template_forecast_cube, qrf_model, cube_inputs)
        return result
