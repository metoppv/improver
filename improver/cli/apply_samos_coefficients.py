#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to apply Standardised Anomaly Model Output Statistics (SAMOS) calibration."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *file_paths: cli.inputpath,
    gam_features: cli.comma_separated_list,
    validity_times: cli.comma_separated_list = None,
    realizations_count: int = None,
    randomise=False,
    random_seed: int = None,
    ignore_ecc_bounds_exceedance=False,
    tolerate_time_mismatch=False,
    predictor="mean",
    percentiles: cli.comma_separated_list = None,
    unique_site_id_key: str = None,
    constant_extrapolation: bool = False,
):
    """Apply coefficients for Standardized Anomaly Model Output Statistics (SAMOS).

    The forecast is converted to anomaly data, the forecast mean and standard deviation
    are predicted from the provided GAM models. The anomaly data is calibrated using the
    EMOS plugin and the provided forecast coefficients. The calibrated forecast is then
    regenerated from the distributional information and the data is written to a cube.
    If no coefficients are provided the input forecast is returned unchanged.

    Args:
        file_paths (cli.inputpath):
            A list of input paths containing:
            - Path to a pickle file containing the GAMs to be used. This pickle
            file contains two lists, each containing two fitted GAMs. The first list
            contains GAMS for predicting each of the climatological mean and standard
            deviation of the historical forecasts. The second list contains GAMS for
            predicting each of the climatological mean and standard deviation of the
            truths.
            - Path to a NetCDF file containing the forecast to be calibrated. The
            input forecast could be given as realizations, probabilities or
            percentiles.
            - Path to a NetCDF file containing a cube list that includes the
            coefficients to be used for calibration or None. If none then the input,
            or probability template if provided, is returned unchanged.
            - Optionally, paths to additional NetCDF files that will be provided to
            the emos plugin representing static additional predictors. These static
            additional predictors are expected not to have a time coordinate. These
            will be identified by their omission from the gam_features list.
            - Optionally paths to additional NetCDF files that contain additional
            features (static predictors) that will be provided to the GAM to help
            calculate the climatological statistics. The name of the cubes should
            match one of the names in the gam_features list.
            - Optionally, path to a NetCDF file containing the land-sea mask. This
            is used to ensure that only land points are calibrated. If no land-sea
            mask is provided, all points will be calibrated.
            - Optionally, path to a NetCDF file containing a probability forecast
            that will be used as a template when generating probability output when
            the input format of the forecast cube is not probabilities i.e.
            realizations or percentiles. If no coefficients are provided and a
            probability template is provided, the probability template forecast will
            be returned as the uncalibrated probability forecast.

        gam_features (list of str):
            A list of the names of the cubes that will be used as additional
            features in the GAM. Additionally, the name of any coordinates
            that are to be used as features in the GAM.
        validity_times (List[str]):
            Times at which the forecast must be valid. This must be provided
            as a four digit string (HHMM) where the first two digits represent the hour
            and the last two digits represent the minutes e.g. 0300 or 0315. If the
            forecast provided is at a different validity time then no coefficients
            will be applied.
        realizations_count (int):
            Option to specify the number of ensemble realizations that will be
            created from probabilities or percentiles when applying the SAMOS
            coefficients.
        randomise (bool):
            Option to reorder the post-processed forecasts randomly. If not
            set, the ordering of the raw ensemble is used. This option is
            only valid when the input format is realizations.
        random_seed (int):
            Option to specify a value for the random seed for testing
            purposes, otherwise the default random seen behaviour is utilised.
            The random seed is used in the generation of the random numbers
            used for either the randomise option to order the input
            percentiles randomly, rather than use the ordering from the raw
            ensemble, or for splitting tied values within the raw ensemble,
            so that the values from the input percentiles can be ordered to
            match the raw ensemble.
        ignore_ecc_bounds_exceedance (bool):
            If True, where the percentiles exceed the ECC bounds range,
            raises a warning rather than an exception. This occurs when the
            current forecasts is in the form of probabilities and is
            converted to percentiles, as part of converting the input
            probabilities into realizations.
        tolerate_time_mismatch (bool):
            If True, tolerate a mismatch in validity time and forecast period
            for coefficients vs forecasts. Use with caution!
        predictor (str):
            String to specify the form of the predictor used to calculate
            the location parameter when estimating the EMOS coefficients.
            Currently, the ensemble mean ("mean") and the ensemble
            realizations ("realizations") are supported as the predictors.
        percentiles (List[float]):
            The set of percentiles used to create the calibrated forecast.
        unique_site_id_key (str):
            If working with spot data and available, the name of the coordinate
            in the input cubes that contains unique site IDs, e.g. "wmo_id" if
            all sites have a valid wmo_id.
        constant_extrapolation:
            If True, when predicting mean and standard deviation from the GAMs,
            when the predictor values are outside the range of those used to fit
            the GAM, constant extrapolation (i.e. the nearest boundary value) will
            be used. If False, extrapolation extends the trend of each
            GAM term beyond the range of the training data. Default is False.

    Returns:
        iris.cube.Cube:
            The calibrated forecast cube.
    """
    import scipy.sparse

    from improver.calibration import (
        split_cubes_for_samos,
        split_netcdf_parquet_pickle,
    )
    from improver.calibration.samos_calibration import ApplySAMOS
    from improver.calibration.utilities import prepare_cube_no_calibration

    # monkey-patch to 'tweak' scipy to prevent errors occurring.
    def to_array(self):
        return self.toarray()

    scipy.sparse.spmatrix.A = property(to_array)

    # Split the input paths into cubes and pickles
    cubes, _, gams = split_netcdf_parquet_pickle(file_paths)

    # Split the cubes into forecast cubes, along with any additional fields
    # provided for the GAMs and EMOS, and the coefficients to be used for calibration
    (
        forecast,
        _,
        gam_additional_fields,
        emos_coefficients,
        emos_additional_fields,
        prob_template,
    ) = split_cubes_for_samos(
        cubes=cubes,
        gam_features=gam_features,
        truth_attribute=None,
        expect_emos_coeffs=True,
        expect_emos_fields=True,
    )

    uncalibrated_forecast = prepare_cube_no_calibration(
        forecast,
        emos_coefficients,
        ignore_ecc_bounds_exceedance=ignore_ecc_bounds_exceedance,
        validity_times=validity_times,
        percentiles=percentiles,
        prob_template=prob_template,
    )

    if uncalibrated_forecast is not None:
        return uncalibrated_forecast

    plugin = ApplySAMOS(
        percentiles=percentiles,
        unique_site_id_key=unique_site_id_key,
        constant_extrapolation=constant_extrapolation,
    )
    result = plugin.process(
        forecast=forecast,
        forecast_gams=gams[0],
        truth_gams=gams[1],
        gam_features=gam_features,
        emos_coefficients=emos_coefficients,
        gam_additional_fields=gam_additional_fields,
        emos_additional_fields=emos_additional_fields,
        prob_template=prob_template,
        realizations_count=realizations_count,
        ignore_ecc_bounds=ignore_ecc_bounds_exceedance,
        tolerate_time_mismatch=tolerate_time_mismatch,
        predictor=predictor,
        randomise=randomise,
        random_seed=random_seed,
    )

    return result
