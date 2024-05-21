#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to estimate coefficients for Ensemble Model Output
Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
Regression (NGR)."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    distribution,
    truth_attribute,
    point_by_point=False,
    use_default_initial_guess=False,
    units=None,
    predictor="mean",
    tolerance: float = 0.02,
    max_iterations: int = 1000,
):
    """Estimate coefficients for Ensemble Model Output Statistics.

    Loads in arguments for estimating coefficients for Ensemble Model
    Output Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). Two sources of input data must be provided: historical
    forecasts and historical truth data (to use in calibration).
    The estimated coefficients are output as a cube.

    Args:
        cubes (list of iris.cube.Cube):
            A list of cubes containing the historical forecasts and
            corresponding truth used for calibration. They must have the same
            cube name and will be separated based on the truth attribute.
            Optionally this may also contain a single land-sea mask cube on the
            same domain as the historic forecasts and truth (where land points
            are set to one and sea points are set to zero).
        distribution (str):
            The distribution that will be used for minimising the
            Continuous Ranked Probability Score when estimating the EMOS
            coefficients. This will be dependent upon the input phenomenon.
        truth_attribute (str):
            An attribute and its value in the format of "attribute=value",
            which must be present on historical truth cubes.
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

    Returns:
        iris.cube.CubeList:
            CubeList containing the coefficients estimated using EMOS. Each
            coefficient is stored in a separate cube.
    """

    from improver.calibration import split_forecasts_and_truth
    from improver.calibration.ensemble_calibration import (
        EstimateCoefficientsForEnsembleCalibration,
    )

    forecast, truth, land_sea_mask = split_forecasts_and_truth(cubes, truth_attribute)

    plugin = EstimateCoefficientsForEnsembleCalibration(
        distribution,
        point_by_point=point_by_point,
        use_default_initial_guess=use_default_initial_guess,
        desired_units=units,
        predictor=predictor,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return plugin(forecast, truth, landsea_mask=land_sea_mask)
