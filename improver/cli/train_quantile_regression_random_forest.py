#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to train a model using Quantile Regression Random Forest (QRF)."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *file_paths: cli.inputpath,
    feature_config: cli.inputjson,
    target_diagnostic_name: str,
    target_cf_name: str,
    forecast_periods: str,
    cycletime: str,
    training_length: int,
    experiment: str = None,
    n_estimators: int = 100,
    max_depth: int = None,
    max_samples: float = None,
    random_state: int = None,
    transformation: str = None,
    pre_transform_addition: float = 0,
):
    """Training a model using Quantile Regression Random Forest.

    Loads in arguments for training a Quantile Regression Random Forest (QRF)
    model which can later be applied to calibrate the forecast.
    Two sources of input data must be provided: historical forecasts and
    historical truth data (to use in calibration). The model is output as a pickle file.

    Args:
        file_paths (cli.inputpaths):
            A list of input paths containing:
            - The path to a Parquet file containing the truths to be used
            for calibration. The expected columns within the
            Parquet file are: ob_value, time, wmo_id, diagnostic, latitude,
            longitude and altitude.
            - The path to a Parquet file containing the forecasts to be used
            for calibration.
            - Optionally, paths to NetCDF files containing additional preictors.
        feature_config (dict):
            Feature configuration defining the features to be used for quantile
            regression. The configuration is a dictionary of strings, where the
            keys are the names of the input cube(s) supplied, and the values are a list.
            This list can contain both computed features, such as the mean or standard
            deviation (std), or static features, such as the altitude. The computed
            features will be computed using the cube defined in the dictionary key.
            If the key is the feature itself e.g. a distance to water cube, then the
            value should state "static". This will ensure the cube's data is used as
            the feature. The config will have the structure:
            "DYNAMIC_VARIABLE_CF_NAME": ["FEATURE1", "FEATURE2"] e.g:
            {
            "air_temperature": ["mean", "std", "altitude"],
            "visibility_at_screen_level": ["mean", "std"]
            "distance_to_water": ["static"],
            }
        target_diagnostic_name (str):
            A string containing the diagnostic name of the forecast to be
            calibrated. This will be used to filter the target forecast and truth
            dataframes.
        target_cf_name (str):
            A string containing the CF name of the forecast to be calibrated.
        forecast_periods (str):
            Range of forecast periods to be calibrated in hours in the form:
            "start:end:interval" e.g. "6:18:6" or a single forecast period e.g. "6".
            The end value is exclusive, so "6:18:6" will calibrate the 6 and 12 hours.
        cycletime (str):
            Cycletime of a format similar to 20170109T0000Z used to filter the
            correct blendtimes from the dataframe on load.
        training_length (int):
            The length of the training period in days.
        experiment (str):
            The name of the experiment (step) that calibration is applied to.
        n_estimators (int):
            Number of trees in the forest.
        max_depth (int):
            Maximum depth of the tree.
        max_samples (float):
            If an int, then it is the number of samples to draw to train
            each tree. If a float, then it is the fraction of samples to draw
            to train each tree. If None, then each tree contains the same
            total number of samples as originally provided.
        random_state (int):
            Random seed for reproducibility.
        transformation (str):
            Transformation to be applied to the data before fitting.
        pre_transform_addition (float):
            Value to be added before transformation.
    Returns:
        None:
            The function creates a pickle file.
    """

    from improver.calibration.load_and_train_quantile_regression_random_forest import (
        LoadForTrainQRF,
        PrepareAndTrainQRF,
    )

    forecast_df, truth_df, cube_inputs = LoadForTrainQRF(
        experiment=experiment,
        feature_config=feature_config,
        target_diagnostic_name=target_diagnostic_name,
        target_cf_name=target_cf_name,
        forecast_periods=forecast_periods,
        cycletime=cycletime,
    )(file_paths)
    result = PrepareAndTrainQRF(
        feature_config=feature_config,
        target_cf_name=target_cf_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_samples=max_samples,
        random_state=random_state,
        transformation=transformation,
        pre_transform_addition=pre_transform_addition,
    )(forecast_df, truth_df, cube_inputs)

    return result
