#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to estimate coefficients for Ensemble Model Output
Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
Regression (NGR)."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast: cli.inputpath,
    truth: cli.inputpath,
    additional_predictors: cli.inputcubelist = None,
    *,
    diagnostic,
    cycletime,
    forecast_period,
    training_length,
    distribution,
    point_by_point=False,
    use_default_initial_guess=False,
    units=None,
    predictor="mean",
    tolerance: float = 0.02,
    max_iterations: int = 1000,
    percentiles: cli.comma_separated_list = None,
    experiment: str = None,
):
    """Estimate coefficients for Ensemble Model Output Statistics.

    Loads in arguments for estimating coefficients for Ensemble Model
    Output Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). Two sources of input data must be provided: historical
    forecasts and historical truth data (to use in calibration).
    The estimated coefficients are output as a cube.

    Args:
        forecast (pathlib.Path):
            The path to a Parquet file containing the historical forecasts
            to be used for calibration. The expected columns within the
            Parquet file are: forecast, blend_time, forecast_period,
            forecast_reference_time, time, wmo_id, percentile, diagnostic,
            latitude, longitude, period, height, cf_name, units.
        truth (pathlib.Path):
            The path to a Parquet file containing the truths to be used
            for calibration. The expected columns within the
            Parquet file are: ob_value, time, wmo_id, diagnostic, latitude,
            longitude and altitude.
        additional_predictors (iris.cube.Cube):
            A cube for a static additional predictor to be used, in addition
            to the forecast, when estimating the EMOS coefficients.
        diagnostic (str):
            The name of the diagnostic to be calibrated within the forecast
            and truth tables. This name is used to filter the Parquet file
            when reading from disk.
        cycletime (str):
            Cycletime of a format similar to 20170109T0000Z.
        forecast_period (int):
            Forecast period to be calibrated in seconds.
        training_length (int):
            Number of days within the training period.
        distribution (str):
            The distribution that will be used for minimising the
            Continuous Ranked Probability Score when estimating the EMOS
            coefficients. This will be dependent upon the input phenomenon.
        point_by_point (bool):
            If True, coefficients are calculated independently for each point
            within the input cube by creating an initial guess and minimising
            each grid point independently. If False, a single set of
            coefficients is calculated using all points.
            Warning: This option is memory intensive and is unsuitable for
            gridded input. Using a default initial guess may reduce the memory
            overhead option.
        use_default_initial_guess (bool):
            If True, use the default initial guess. The default initial guess
            assumes no adjustments are required to the initial choice of
            predictor to generate the calibrated distribution. This means
            coefficients of 1 for the multiplicative coefficients and 0 for
            the additive coefficients. If False, the initial guess is computed.
        units (str):
            The units that calibration should be undertaken in. The historical
            forecast and truth will be converted as required.
        predictor (str):
            String to specify the form of the predictor used to calculate the
            location parameter when estimating the EMOS coefficients.
            Currently, the ensemble mean ("mean") and the ensemble realizations
            ("realizations") are supported as options.
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

    Returns:
        iris.cube.CubeList:
            CubeList containing the coefficients estimated using EMOS. Each
            coefficient is stored in a separate cube.
    """
    from improver.calibration import get_common_wmo_ids
    from improver.calibration.emos_calibration import (
        EstimateCoefficientsForEnsembleCalibration,
    )
    from improver.calibration.utilities import convert_parquet_to_cube

    forecast_cube, truth_cube = convert_parquet_to_cube(
        forecast,
        truth,
        forecast_period=forecast_period,
        cycletime=cycletime,
        training_length=training_length,
        diagnostic=diagnostic,
        percentiles=percentiles,
        experiment=experiment,
    )

    if not forecast_cube or not truth_cube:
        return

    # Extract WMO IDs from the additional predictors.
    forecast_cube, truth_cube, additional_predictors = get_common_wmo_ids(
        forecast_cube, truth_cube, additional_predictors
    )

    plugin = EstimateCoefficientsForEnsembleCalibration(
        distribution,
        point_by_point=point_by_point,
        use_default_initial_guess=use_default_initial_guess,
        desired_units=units,
        predictor=predictor,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return plugin(forecast_cube, truth_cube, additional_fields=additional_predictors)
