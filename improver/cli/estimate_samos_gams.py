#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to estimate coefficients for Standardized Anomaly Model Output Statistics
(SAMOS)."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    truth_attribute: str,
    gam_features: cli.comma_separated_list,
    model_specification: cli.inputjson,
    max_iterations: int = 100,
    tol: float = 0.0001,
    distribution: str = "normal",
    link: str = "identity",
    fit_intercept: bool = True,
):
    """Estimate EMOS coefficients for SAMOS.

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

    from improver.calibration import split_cubes_for_samos
    from improver.calibration.samos_calibration import TrainGAMsForSAMOS

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

    plugin = TrainGAMsForSAMOS(
        model_specification=model_specification,
        max_iter=max_iterations,
        tol=tol,
        distribution=distribution,
        link=link,
        fit_intercept=fit_intercept,
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

    return [forecast_gams, truth_gams]
