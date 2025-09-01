#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to estimate the Ensemble Model Output Statistics (EMOS) coefficients for
Standardized Anomaly Model Output Statistics (SAMOS)."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast: cli.inputpath,
    truth: cli.inputpath,
    gams: cli.inputpickle,
    *samos_additional_predictors: cli.inputcube,
    gam_features: cli.comma_separated_list,
    use_default_initial_guess=False,
    units=None,
    predictor="mean",
    tolerance: float = 0.02,
    max_iterations: int = 1000,
    percentiles: cli.comma_separated_list = None,
    experiment: str = None,
    forecast_period: int,
    training_length: int,
    diagnostic: str,
    cycletime: str,
):
    """Estimate EMOS coefficients for use with SAMOS.

    Loads in arguments for estimating coefficients for Ensemble Model
    Output Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). Two sources of input data must be provided: historical
    forecasts and historical truth data (to use in calibration).
    The estimated coefficients are output as a cube.

    Args:
        forecast (pathlib.Path):
            The path to a Parquet file containing the historical forecasts
            to be used for calibration.The expected columns within the
            Parquet file are: forecast, blend_time, forecast_period,
            forecast_reference_time, time, wmo_id, percentile, diagnostic,
            latitude, longitude, period, height, cf_name, units.
        truth (pathlib.Path):
            The path to a Parquet file containing the truths to be used
            for calibration. The expected columns within the
            Parquet file are: ob_value, time, wmo_id, diagnostic, latitude,
            longitude and altitude.
        samos_additional_predictors (iris.cube.CubeList):
            A list of cubes for additional predictors to be used when
            estimating the SAMOS coefficients. The name of all cubes in this
            list must be in the gam_features list.
        gams (list of GAM models):
            A list containing two lists of two fitted GAMs. The first list
            contains two fitted GAMs, one for predicting the climatological mean
            of the historical forecasts and the second predicting the
            climatological standard deviation. The second list contains two
            fitted GAMs, one for predicting the climatological mean of the truths
            and the second predicting the climatological standard deviation.
        gam_features (list of str):
            A list of the names of the cubes that will be used as additional
            features in the GAM. Additionaly the name of any coordinates
            that are to be used as features in the GAM.
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
            Currently the ensemble mean ("mean") and the ensemble realizations
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

    Returns:
        iris.cube.CubeList:
            CubeList containing the coefficients estimated using EMOS. Each
            coefficient is stored in a separate cube.
        OR
        None:
            If a forecast or truth cube cannot be created from the parquet table then None is returned.
    """
    # monkey-patch to 'tweak' scipy to prevent errors occuring
    import scipy.sparse

    from improver.calibration.samos_calibration import TrainEMOSForSAMOS
    from improver.ensemble_copula_coupling.utilities import convert_parquet_to_cube

    def to_array(self):
        return self.toarray()

    scipy.sparse.spmatrix.A = property(to_array)

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

    # Train emos coefficients for the SAMOS model.
    emos_kwargs = {
        "use_default_initial_guess": use_default_initial_guess,
        "desired_units": units,
        "predictor": predictor,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
    }

    plugin = TrainEMOSForSAMOS(distribution="norm", emos_kwargs=emos_kwargs)
    return plugin(
        historic_forecasts=forecast_cube,
        truths=truth_cube,
        forecast_gams=gams[0],
        truth_gams=gams[1],
        gam_features=gam_features,
        gam_additional_fields=samos_additional_predictors,
    )
