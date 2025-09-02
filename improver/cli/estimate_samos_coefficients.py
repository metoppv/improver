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
    *cubes: cli.inputcube,
    truth_attribute: str,
    gams: cli.inputpickle,
    gam_features: cli.comma_separated_list,
    use_default_initial_guess=False,
    units=None,
    predictor="mean",
    tolerance: float = 0.02,
    max_iterations: int = 1000,
    unique_site_id_key: str = None,
):
    """Estimate EMOS coefficients for use with SAMOS.

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
            Optionally this may also include any other cubes to be used as
            additional features in the GAM or in EMOS. These cubes will be
            identified by comparing the cube name to the names in `gam_features`.
            Any additional cubes that are not in `gam_features` will be
            considered as additional fields for EMOS.
        truth_attribute (str):
            An attribute and its value in the format of "attribute=value",
            which must be present on historical truth cubes.
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
        unique_site_id_key (str):
            If working with spot data and available, the name of the coordinate
            in the input cubes that contains unique site IDs, e.g. "wmo_id" if
            all sites have a valid wmo_id.

    Returns:
        iris.cube.CubeList:
            CubeList containing the coefficients estimated using EMOS. Each
            coefficient is stored in a separate cube.
    """

    # monkey-patch to 'tweak' scipy to prevent errors occuring
    import scipy.sparse

    from improver.calibration import split_cubes_for_samos
    from improver.calibration.samos_calibration import TrainEMOSForSAMOS

    def to_array(self):
        return self.toarray()

    scipy.sparse.spmatrix.A = property(to_array)

    # Split the cubes into forecast and truth cubes, along with any additional fields
    # provided for the GAMs and EMOS.
    (
        forecast,
        truth,
        gam_additional_fields,
        _,
        emos_additional_fields,
        _,
    ) = split_cubes_for_samos(
        cubes=cubes,
        gam_features=gam_features,
        truth_attribute=truth_attribute,
        expect_emos_coeffs=False,
        expect_emos_fields=True,
    )
    # Train emos coefficients for the SAMOS model.
    emos_kwargs = {
        "use_default_initial_guess": use_default_initial_guess,
        "desired_units": units,
        "predictor": predictor,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
    }

    plugin = TrainEMOSForSAMOS(distribution="norm", emos_kwargs=emos_kwargs, unique_site_id_key=unique_site_id_key)
    return plugin(
        historic_forecasts=forecast,
        truths=truth,
        forecast_gams=gams[0],
        truth_gams=gams[1],
        gam_features=gam_features,
        gam_additional_fields=gam_additional_fields,
        emos_additional_fields=emos_additional_fields,
    )
