#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to apply a Quantile Regression Random Forest (QRF) model."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *file_paths: cli.inputpath,
    feature_config: cli.inputjson,
    target_cf_name: str,
    transformation: str = None,
    pre_transform_addition: float = 0,
):
    """Applying the Quantile Regression Random Forest model.

    Loads in arguments for applying a Quantile Regression Random Forest (QRF)
    model which has been previously trained.
    Two sources of input data must be provided: The QRF model and the forecast cube
    to be calibrated. The output is a NetCDF file containing the calibrated forecast.

    Args:
        file_paths (cli.inputpaths):
            A list of input paths containing:
            - The path to a QRF trained model in pickle file format to be used
            for calibration.
            - The path to a NetCDF file containing the forecast to be calibrated.
            - Optionally, paths to NetCDF files containing additional predictors.
        feature_config (dict):
            Feature configuration defining the features to be used for quantile
            regression. The configuration is a dictionary of strings, where the keys
            are the names of the input cube(s) supplied, and the values are a list.
            This list can contain both computed features, such as the mean or
            standard deviation (std), or static features, such as the altitude. The
            computed features will be computed using the cube defined in the
            dictionary key. If the key is the feature itself e.g. a distance to water
            cube, then the value should state "static". This will ensure the cube's
            data is used as the feature. The config will have the structure:
            "DYNAMIC_VARIABLE_CF_NAME": ["FEATURE1", "FEATURE2"] e.g.
            {
            "air_temperature": ["mean", "std", "altitude"],
            "visibility_at_screen_level": ["mean", "std"]
            "distance_to_water": ["static"],
            }
        target_cf_name (str):
            A string containing the CF name of the forecast to be
            calibrated. This will be used to separate it from the rest of the
            feature cubes, if present.
        transformation (str):
            Transformation to be applied to the data before fitting.
        pre_transform_addition (float):
            Value to be added before transformation.
    Returns:
        iris.cube.Cube:
            The calibrated forecast cube.
    """

    from improver.calibration.load_and_apply_quantile_regression_random_forest import (
        LoadAndApplyQRF,
    )

    result = LoadAndApplyQRF(
        feature_config=feature_config,
        target_cf_name=target_cf_name,
        transformation=transformation,
        pre_transform_addition=pre_transform_addition,
    )(
        file_paths,
    )
    return result
