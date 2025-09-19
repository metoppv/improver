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
    *file_paths: cli.inputpath,
    truth_attribute: str,
    gam_features: cli.comma_separated_list,
    use_default_initial_guess=False,
    units=None,
    predictor="mean",
    tolerance: float = 0.02,
    max_iterations: int = 1000,
    unique_site_id_key: str = "wmo_id",
):
    """Estimate EMOS coefficients for use with SAMOS.

    Loads in arguments for estimating coefficients for Ensemble Model
    Output Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). Two sources of input data must be provided: historical
    forecasts and historical truth data (to use in calibration).
    The estimated coefficients are output as a cube.

    Args:
        file_paths (cli.inputpath):
            A list of input paths containing:
            - Path to a pickle file containing the GAMs to be used. This pickle
            file contains two lists, each containing two fitted GAMs. The first list
            contains GAMS for predicting each of the climatological mean and
            standard deviation of the historical forecasts. The second list contains
            GAMS for predicting each of the climatological mean and standard
            deviation of the truths.
            - Paths to NetCDF files containing the historical forecasts and
            corresponding truths used for calibration. They must have the same
            diagnostic name and will be separated based on the provided truth
            attribute.
            - Optionally, paths to additional NetCDF files that will be provided to
            the emos plugin representing static additional predictors. These static
            additional predictors are expected not to have a time coordinate. These
            will be identified by their omission from the gam_features list.
            - Optionally paths to additional NetCDF files that contain additional
            features (static predictors) that will be provided to the GAM to help
            calculate the climatological statistics. The name of the cubes should
            match one of the names in the gam_features list.

        truth_attribute (str):
            An attribute and its value in the format of "attribute=value",
            which must be present on historical truth cubes.
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
            all sites have a valid wmo_id. For estimation the default is "wmo_id"
            as we expect to be including observation data.

    Returns:
        iris.cube.CubeList:
            CubeList containing the coefficients estimated using EMOS. Each
            coefficient is stored in a separate cube.
    """

    # monkey-patch to 'tweak' scipy to prevent errors occuring
    import scipy.sparse

    from improver.calibration import (
        split_cubes_for_samos,
        split_pickle_parquet_and_netcdf,
    )
    from improver.calibration.samos_calibration import TrainEMOSForSAMOS

    def to_array(self):
        return self.toarray()

    scipy.sparse.spmatrix.A = property(to_array)

    print(file_paths)

    # Split the input paths into cubes and pickles
    cubes, _, gams = split_pickle_parquet_and_netcdf(file_paths)
    print(cubes)
    print(gams)

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

    if forecast is None or truth is None:
        return

    # Train emos coefficients for the SAMOS model.
    emos_kwargs = {
        "use_default_initial_guess": use_default_initial_guess,
        "desired_units": units,
        "predictor": predictor,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
    }

    plugin = TrainEMOSForSAMOS(
        distribution="norm",
        emos_kwargs=emos_kwargs,
        unique_site_id_key=unique_site_id_key,
    )
    return plugin(
        historic_forecasts=forecast,
        truths=truth,
        forecast_gams=gams[0],
        truth_gams=gams[1],
        gam_features=gam_features,
        gam_additional_fields=gam_additional_fields,
        emos_additional_fields=emos_additional_fields,
    )
