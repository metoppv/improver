#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to estimate the Generalized Additive Model (GAM) for Standardized Anomaly Model
Output Statistics (SAMOS)."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *file_paths: cli.inputpath,
    gam_features: cli.comma_separated_list,
    model_specification: cli.inputjson,
    tolerance: float = 0.02,
    max_iterations: int = 1000,
    percentiles: cli.comma_separated_list = None,
    experiment: str = None,
    forecast_period: int,
    training_length: int,
    diagnostic: str,
    cycletime: str,
    distribution: str = "normal",
    link: str = "identity",
    fit_intercept: bool = True,
    unique_site_id_key: str = "wmo_id",
):
    """Estimate General Additive Model (GAM) for SAMOS.

    Args:
        file_paths (cli.inputpath):
            A list of input paths containing:
            - The path to a Parquet file containing the historical forecasts
            to be used for calibration.The expected columns within the
            Parquet file are: forecast, blend_time, forecast_period,
            forecast_reference_time, time, wmo_id, percentile, diagnostic,
            latitude, longitude, period, height, cf_name, units.
            - The path to a Parquet file containing the truths to be used
            for calibration. The expected columns within the
            Parquet file are: ob_value, time, wmo_id, diagnostic, latitude,
            longitude and altitude.
            - Optionally paths to additional NetCDF files that contain additional
            features (static predictors) that will be provided when estimating the
            GAM.

        gam_features (list of str):
            A list of the names of the cubes that will be used as additional
            features in the GAM. Additionally, the name of any coordinates
            that are to be used as features in the GAM.
        model_specification (dict):
            A list containing three items (in order):
                1. a string containing a single pyGAM term; one of 'l' (linear),
                's' (spline), 'te' (tensor), or 'f' (factor)
                2. a list of integers which correspond to the features to be
                included in that term
                3. a dictionary of kwargs to be included when defining the term
        tolerance (float):
            The tolerance for the Continuous Ranked Probability Score (CRPS)
            calculated by the minimisation. Once multiple iterations result in
            a CRPS equal to the same value within the specified tolerance, the
            minimisation will terminate.
        max_iterations (int):
            The maximum number of iterations allowed until the minimisation has
            converged to a stable solution. If the maximum number of iterations
            is reached but the minimisation has not yet converged to a stable
            solution, then the available solution is used anyway, and a warning
            is raised. If the predictor is "realizations", then the number of
            iterations may require increasing, as there will be more
            coefficients to solve.
        percentiles (List[float]):
            The set of percentiles to be used for estimating EMOS coefficients.
            These should be a set of equally spaced quantiles.
        experiment (str):
            A value within the experiment column to select from the forecast
            table.
        forecast_period (int):
            Forecast period to be calibrated in seconds.
        training_length (int):
            Number of days within the training period.
        diagnostic (str):
            The name of the diagnostic to be calibrated within the forecast
            and truth tables. This name is used to filter the Parquet file
            when reading from disk.
        cycletime (str):
            Cycletime of a format similar to 20170109T0000Z.
        distribution (str):
            The distribution to be used in the model. Valid options are normal, binomial,
            poisson, gamma, inv-gauss.
        link (str):
            The link function to be used in the model. Valid options are identity, logit, inverse, log
            or inverse-squared.
        fit_intercept (bool):
            Whether to include an intercept term in the model. Default is True.
        unique_site_id_key (str):
            If working with spot data and available, the name of the coordinate
            in the input cubes that contains unique site IDs, e.g. "wmo_id" if
            all sites have a valid wmo_id. For GAM estimation the default is
            "wmo_id" as we expect to have a training data set comprising matched
            obs and forecast sites.

    Returns:
        (list of GAM models):
            A list containing two lists of two fitted GAMs. The first list
            contains two fitted GAMs, one for predicting the climatological mean
            of the historical forecasts and the second predicting the
            climatological standard deviation. The second list contains two
            fitted GAMs, one for predicting the climatological mean of the truths
            and the second predicting the climatological standard deviation.
    """

    from improver.calibration import (
        identify_parquet_type,
        split_netcdf_parquet_pickle,
    )
    from improver.calibration.samos_calibration import TrainGAMsForSAMOS
    from improver.calibration.utilities import convert_parquet_to_cube

    # Split the input paths into cubes and pickles.
    additional_predictors, parquets, _ = split_netcdf_parquet_pickle(file_paths)
    # Determine which parquet path provides truths and which historic forecasts.
    forecast, truth = identify_parquet_type(parquets)

    forecast_cube, truth_cube = convert_parquet_to_cube(
        forecast,
        truth,
        diagnostic=diagnostic,
        cycletime=cycletime,
        forecast_period=forecast_period,
        training_length=training_length,
        percentiles=percentiles,
        experiment=experiment,
    )

    if not forecast_cube or not truth_cube:
        return

    plugin = TrainGAMsForSAMOS(
        model_specification=model_specification,
        max_iter=max_iterations,
        tol=tolerance,
        distribution=distribution,
        link=link,
        fit_intercept=fit_intercept,
        unique_site_id_key=unique_site_id_key,
    )

    truth_gams = plugin.process(
        input_cube=truth_cube,
        features=gam_features,
        additional_fields=additional_predictors,
    )

    forecast_gams = plugin.process(
        input_cube=forecast_cube,
        features=gam_features,
        additional_fields=additional_predictors,
    )

    return [forecast_gams, truth_gams]
