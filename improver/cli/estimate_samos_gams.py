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
    *cubes: cli.inputcube,
    truth_attribute: str,
    gam_features: cli.comma_separated_list,
    model_specification: cli.inputjson,
    max_iterations: int = 100,
    tolerance: float = 0.0001,
    distribution: str = "normal",
    link: str = "identity",
    fit_intercept: bool = True,
    window_length: int = 10,
    required_rolling_window_points: int = 5,
    trailing_window: bool = False,
    unique_site_id_key: str = "wmo_id",
):
    """Estimate Generalized Additive Model (GAM) for SAMOS.

    Args:
        cubes (list of iris.cube.Cube):
            A list of cubes containing the historical forecasts and
            corresponding truth used for calibration. They must have the same
            cube name and will be separated based on the truth attribute. The
            list may also contain additional features (static predictors) that
            will be provided when estimating the GAM.
        truth_attribute (str):
            An attribute and its value in the format of "attribute=value",
            which must be present on historical truth cubes.
        gam_features (list of str):
            A list of the names of the cubes that will be used as additional
            features in the GAM.
        model_specification (dict):
            A list containing three items (in order):
                1. a string containing a single pyGAM term; one of 'l' (linear),
                's' (spline), 'te' (tensor), or 'f' (factor)
                2. a list of integers which correspond to the features to be
                included in that term
                3. a dictionary of kwargs to be included when defining the term
        max_iterations (int):
            The maximum number of iterations to use when estimating the GAM
            coefficients.
        tolerance (float):
            The tolerance for the stopping criteria.
        distribution (str):
            The distribution to be used in the model. Valid options are normal,
            binomial, poisson, gamma, inv-gauss.
        link (str):
            The link function to be used in the model. Valid options are identity,
            logit, inverse, log or inverse-squared.
        fit_intercept (bool):
            Whether to include an intercept term in the model. Default is True.
        window_length (int):
            The length of the rolling window, in days, used to calculate the mean
            and standard deviation of the input cube when the input cube does not
            have a realization dimension coordinate. If using a centred rolling
            window, this must be an even integer greater than 1 in order to allow
            equal numbers of days on either side of the central time point. If
            using a trailing rolling window, this must be an integer greater than 1.
        required_rolling_window_points (int):
            The minimum number of valid data points required within a rolling
            window. If fewer valid points are present, the mean and standard
            deviation will be set to NaN for this window.
        trailing_window (bool):
            If False a centred window is used, which assigns the calculated
            statistic to the central time point in the window. If True a trailing
            window is used, which assigns the calculated statistic to the final time
            point.
        unique_site_id_key (str):
            If working with spot data and available, the name of the coordinate
            in the input cubes that contains unique site IDs, e.g. "wmo_id" if
            all sites have a valid wmo_id. For GAM estimation the default is
            "wmo_id" as we expect to have a training data set comprising matched
            obs and forecast sites.

    Returns:
        List:
            A list containing the fitted GAMs for the forecast and truth cubes in that
            order.
    """

    from improver.calibration import split_cubes_for_samos
    from improver.calibration.samos_calibration import TrainGAMsForSAMOS

    # Split the cubes into forecast and truth cubes, along with any additional fields
    # provided for the GAMs.
    (
        forecast,
        truth,
        gam_additional_fields,
        _,
        _,
        _,
    ) = split_cubes_for_samos(
        cubes=cubes,
        gam_features=gam_features,
        truth_attribute=truth_attribute,
        expect_emos_coeffs=False,
        expect_emos_fields=False,
    )

    if forecast is None or truth is None:
        return

    plugin = TrainGAMsForSAMOS(
        model_specification=model_specification,
        max_iter=max_iterations,
        tol=tolerance,
        distribution=distribution,
        link=link,
        fit_intercept=fit_intercept,
        window_length=window_length,
        required_rolling_window_points=required_rolling_window_points,
        trailing_window=trailing_window,
        unique_site_id_key=unique_site_id_key,
    )

    truth_gams = plugin.process(
        input_cube=truth,
        features=gam_features,
        additional_fields=gam_additional_fields,
    )

    forecast_gams = plugin.process(
        input_cube=forecast,
        features=gam_features,
        additional_fields=gam_additional_fields,
    )

    if forecast_gams is None or truth_gams is None:
        return

    return [forecast_gams, truth_gams]
