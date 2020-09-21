# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
This module defines all the "plugins" specific for Ensemble Model Output
Statistics (EMOS).

.. Further information is available in:
.. include:: extended_documentation/calibration/ensemble_calibration/
   ensemble_calibration.rst

"""
import functools
import os
import warnings
from multiprocessing import Pool

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm

from improver import BasePlugin, PostProcessingPlugin
from improver.calibration.utilities import (
    check_forecast_consistency,
    check_predictor,
    convert_cube_data_to_2d,
    create_unified_frt_coord,
    filter_non_matching_cubes,
    flatten_ignoring_masked_data,
    forecast_coords_match,
    merge_land_and_sea,
)
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertLocationAndScaleParametersToPercentiles,
    ConvertLocationAndScaleParametersToProbabilities,
    ConvertProbabilitiesToPercentiles,
    EnsembleReordering,
    RebadgePercentilesAsRealizations,
    ResamplePercentiles,
)
from improver.metadata.probabilistic import find_percentile_coordinate
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_manipulation import collapsed, enforce_coordinate_ordering


class ContinuousRankedProbabilityScoreMinimisers(BasePlugin):
    """
    Minimise the Continuous Ranked Probability Score (CRPS)

    Calculate the optimised coefficients for minimising the CRPS based on
    assuming a particular probability distribution for the phenomenon being
    minimised.

    The number of coefficients that will be optimised depend upon the initial
    guess.

    Minimisation is performed using the Nelder-Mead algorithm for 200
    iterations to limit the computational expense.
    Note that the BFGS algorithm was initially trialled but had a bug
    in comparison to comparative results generated in R.

    """

    # The tolerated percentage change for the final iteration when
    # performing the minimisation.
    TOLERATED_PERCENTAGE_CHANGE = 5

    # An arbitrary value set if an infinite value is detected
    # as part of the minimisation.
    BAD_VALUE = np.float64(999999)

    def __init__(self, tolerance=0.02, max_iterations=1000, minimise_each_point=False):
        """
        Initialise class for performing minimisation of the Continuous
        Ranked Probability Score (CRPS).

        Args:
            tolerance (float):
                The tolerance for the Continuous Ranked Probability
                Score (CRPS) calculated by the minimisation. The CRPS is in
                the units of the variable being calibrated. The tolerance is
                therefore representative of how close to the actual value are
                we aiming to forecast for a particular variable. Once multiple
                iterations result in a CRPS equal to the same value within the
                specified tolerance, the minimisation will terminate.
            max_iterations (int):
                The maximum number of iterations allowed until the
                minimisation has converged to a stable solution. If the
                maximum number of iterations is reached, but the minimisation
                has not yet converged to a stable solution, then the available
                solution is used anyway, and a warning is raised. If the
                predictor_of_mean is "realizations", then the number of
                iterations may require increasing, as there will be
                more coefficients to solve for.

        """
        # Dictionary containing the functions that will be minimised,
        # depending upon the distribution requested. The names of these
        # distributions match the names of distributions in scipy.stats.
        self.minimisation_dict = {
            "norm": self.calculate_normal_crps,
            "truncnorm": self.calculate_truncated_normal_crps,
        }
        self.tolerance = tolerance
        # Maximum iterations for minimisation using Nelder-Mead.
        self.max_iterations = max_iterations
        self.minimise_each_point = minimise_each_point

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = (
            "<ContinuousRankedProbabilityScoreMinimisers: "
            "minimisation_dict: {}; tolerance: {}; max_iterations: {}>"
        )
        print_dict = {}
        for key in self.minimisation_dict:
            print_dict.update({key: self.minimisation_dict[key].__name__})
        return result.format(print_dict, self.tolerance, self.max_iterations)

    def process(
        self,
        initial_guess,
        forecast_predictor,
        truth,
        forecast_var,
        predictor,
        distribution,
    ):
        """
        Function to pass a given function to the scipy minimize
        function to estimate optimised values for the coefficients.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            initial_guess (list):
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            forecast_predictor (iris.cube.Cube):
                Cube containing the fields to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            truth (iris.cube.Cube):
                Cube containing the field, which will be used as truth.
            forecast_var (iris.cube.Cube):
                Cube containing the field containing the ensemble variance.
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.
            distribution (str):
                String used to access the appropriate function for use in the
                minimisation within self.minimisation_dict.

        Returns:
            list of float:
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].

        Raises:
            KeyError: If the distribution is not supported.

        Warns:
            Warning: If the minimisation did not converge.

        """

        def calculate_percentage_change_in_last_iteration(allvecs):
            """
            Calculate the percentage change that has occurred within
            the last iteration of the minimisation. If the percentage change
            between the last iteration and the last-but-one iteration exceeds
            the threshold, a warning message is printed.

            Args:
                allvecs (list):
                    List of numpy arrays containing the optimised coefficients,
                    after each iteration.

            Warns:
                Warning: If a satisfactory minimisation has not been achieved.
            """
            last_iteration_percentage_change = (
                np.absolute((allvecs[-1] - allvecs[-2]) / allvecs[-2]) * 100
            )
            if np.any(
                last_iteration_percentage_change > self.TOLERATED_PERCENTAGE_CHANGE
            ):
                np.set_printoptions(suppress=True)
                msg = (
                    "The final iteration resulted in a percentage change "
                    "that is greater than the accepted threshold of 5% "
                    "i.e. {}. "
                    "\nA satisfactory minimisation has not been achieved. "
                    "\nLast iteration: {}, "
                    "\nLast-but-one iteration: {}"
                    "\nAbsolute difference: {}\n"
                ).format(
                    last_iteration_percentage_change,
                    allvecs[-1],
                    allvecs[-2],
                    np.absolute(allvecs[-2] - allvecs[-1]),
                )
                #warnings.warn(msg)

        try:
            minimisation_function = self.minimisation_dict[distribution]
        except KeyError as err:
            msg = (
                "Distribution requested {} is not supported in {}"
                "Error message is {}".format(distribution, self.minimisation_dict, err)
            )
            raise KeyError(msg)

        # Ensure predictor is valid.
        check_predictor(predictor)

        preserve_leading_dimension = False
        if self.minimise_each_point:
            preserve_leading_dimension = True

        # Flatten the data arrays and remove any missing data.
        truth_data = flatten_ignoring_masked_data(
            truth.data, preserve_leading_dimension=preserve_leading_dimension)
        forecast_var_data = flatten_ignoring_masked_data(
            forecast_var.data, preserve_leading_dimension=preserve_leading_dimension)
        if predictor.lower() == "mean":
            forecast_predictor_data = flatten_ignoring_masked_data(
                forecast_predictor.data, preserve_leading_dimension=preserve_leading_dimension
            )
        elif predictor.lower() == "realizations":
            enforce_coordinate_ordering(forecast_predictor, "realization")
            # Need to transpose this array so there are columns for each
            # ensemble member rather than rows.
            forecast_predictor_data = flatten_ignoring_masked_data(
                forecast_predictor.data, preserve_leading_dimension=True
            ).T

        # Increased precision is needed for stable coefficient calculation.
        # The resulting coefficients are cast to float32 prior to output.
        initial_guess = np.array(initial_guess, dtype=np.float64)
        forecast_predictor_data = forecast_predictor_data.astype(np.float64)
        forecast_var_data = forecast_var_data.astype(np.float64)
        truth_data = truth_data.astype(np.float64)
        sqrt_pi = np.sqrt(np.pi).astype(np.float64)

        if self.minimise_each_point:
            argument_list = []
            for index in range(forecast_predictor_data.shape[1]):
                argument_list.append((minimisation_function, initial_guess,
                                      forecast_predictor_data[:, index],
                                      truth_data[:, index],
                                      forecast_var_data[:, index], sqrt_pi,
                                      predictor))

            with Pool(os.cpu_count()) as pool:
                optimised_coeffs = pool.starmap(
                    self.minimise_caller, argument_list)

            optimised_coeffs = np.transpose(optimised_coeffs)
            optimised_coeffs = [x.x.astype(np.float32) for x in optimised_coeffs]
            return np.array(optimised_coeffs).reshape(
                (len(initial_guess),) + forecast_predictor.data.shape[1:])

        else:
            optimised_coeffs = self.minimise_caller(
                minimisation_function, initial_guess, forecast_predictor_data,
                truth_data, forecast_var_data, sqrt_pi, predictor)

            if not optimised_coeffs.success:
                msg = (
                    "Minimisation did not result in convergence after "
                    "{} iterations. \n{}".format(
                        self.max_iterations, optimised_coeffs.message
                    )
                )
                warnings.warn(msg)
            calculate_percentage_change_in_last_iteration(optimised_coeffs.allvecs)
            return optimised_coeffs.x.astype(np.float32)

    def minimise_caller(self, minimisation_function, initial_guess, forecast_predictor_data, truth_data,
                             forecast_var_data, sqrt_pi, predictor):

        optimised_coeffs = minimize(
            minimisation_function, initial_guess,
            args=(forecast_predictor_data, truth_data,
                  forecast_var_data, sqrt_pi, predictor),
            method="Nelder-Mead", tol=self.tolerance,
            options={"maxiter": self.max_iterations, "return_all": True})

        return optimised_coeffs

    def calculate_normal_crps(
        self, initial_guess, forecast_predictor, truth, forecast_var, sqrt_pi, predictor
    ):
        """
        Calculate the CRPS for a normal distribution.

        Scientific Reference:
        Gneiting, T. et al., 2005.
        Calibrated Probabilistic Forecasting Using Ensemble Model Output
        Statistics and Minimum CRPS Estimation.
        Monthly Weather Review, 133(5), pp.1098-1118.

        Args:
            initial_guess (list):
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            forecast_predictor (numpy.ndarray):
                Data to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            truth (numpy.ndarray):
                Data to be used as truth.
            forecast_var (numpy.ndarray):
                Ensemble variance data.
            sqrt_pi (numpy.ndarray):
                Square root of Pi
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.

        Returns:
            float:
                CRPS for the current set of coefficients. This CRPS is a mean
                value across all points.

        """
        if predictor.lower() == "mean":
            a, b, gamma, delta = initial_guess
            a_b = np.array([a, b], dtype=np.float64)
        elif predictor.lower() == "realizations":
            a, b, gamma, delta = (
                initial_guess[0],
                initial_guess[1:-2] ** 2,
                initial_guess[-2],
                initial_guess[-1],
            )
            a_b = np.array([a] + b.tolist(), dtype=np.float64)

        new_col = np.ones(truth.shape, dtype=np.float32)
        all_data = np.column_stack((new_col, forecast_predictor))
        mu = np.dot(all_data, a_b)
        sigma = np.sqrt(gamma ** 2 + delta ** 2 * forecast_var)
        xz = (truth - mu) / sigma
        normal_cdf = norm.cdf(xz)
        normal_pdf = norm.pdf(xz)
        if np.isfinite(np.min(mu / sigma)):
            result = np.nanmean(
                sigma * (xz * (2 * normal_cdf - 1) + 2 * normal_pdf - 1 / sqrt_pi)
            )
        else:
            result = self.BAD_VALUE
        return result

    def calculate_truncated_normal_crps(
        self, initial_guess, forecast_predictor, truth, forecast_var, sqrt_pi, predictor
    ):
        """
        Calculate the CRPS for a truncated normal distribution with zero
        as the lower bound.

        Scientific Reference:
        Thorarinsdottir, T.L. & Gneiting, T., 2010.
        Probabilistic forecasts of wind speed: Ensemble model
        output statistics by using heteroscedastic censored regression.
        Journal of the Royal Statistical Society.
        Series A: Statistics in Society, 173(2), pp.371-388.

        Args:
            initial_guess (list):
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            forecast_predictor (numpy.ndarray):
                Data to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            truth (numpy.ndarray):
                Data to be used as truth.
            forecast_var (numpy.ndarray):
                Ensemble variance data.
            sqrt_pi (numpy.ndarray):
                Square root of Pi
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.

        Returns:
            float:
                CRPS for the current set of coefficients. This CRPS is a mean
                value across all points.

        """
        if predictor.lower() == "mean":
            a, b, gamma, delta = initial_guess
            mu = (forecast_predictor * b) + a

        elif predictor.lower() == "realizations":
            a, b, gamma, delta = (
                initial_guess[0],
                initial_guess[1:-2] ** 2,
                initial_guess[-2],
                initial_guess[-1],
            )
            a_b = np.array([a] + b.tolist(), dtype=np.float64)

            new_col = np.ones(truth.shape, dtype=np.float32)
            all_data = np.column_stack((new_col, forecast_predictor))
            mu = np.dot(all_data, a_b)

        sigma = np.sqrt(gamma ** 2 + delta ** 2 * forecast_var)
        xz = (truth - mu) / sigma
        normal_cdf = norm.cdf(xz)
        normal_pdf = norm.pdf(xz)
        x0 = mu / sigma
        normal_cdf_0 = norm.cdf(x0)
        normal_cdf_root_two = norm.cdf(np.sqrt(2) * x0)

        if np.isfinite(np.min(mu / sigma)) or (np.min(mu / sigma) >= -3):
            result = np.nanmean(
                (sigma / normal_cdf_0 ** 2)
                * (
                    xz * normal_cdf_0 * (2 * normal_cdf + normal_cdf_0 - 2)
                    + 2 * normal_pdf * normal_cdf_0
                    - normal_cdf_root_two / sqrt_pi
                )
            )
        else:
            result = self.BAD_VALUE
        return result


class EstimateCoefficientsForEnsembleCalibration(BasePlugin):
    """
    Class focussing on estimating the optimised coefficients for ensemble
    calibration.
    """

    # Logical flag for whether initial guess estimates for the coefficients
    # will be estimated using linear regression i.e.
    # ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG = True, or whether default
    # values will be used instead i.e.
    # ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG = False.
    ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG = True

    def __init__(
        self,
        distribution,
        desired_units=None,
        predictor="mean",
        tolerance=0.02,
        max_iterations=1000,
        each_point=False,
        pool_size=os.cpu_count(),
    ):
        """
        Create an ensemble calibration plugin that, for Nonhomogeneous Gaussian
        Regression, calculates coefficients based on historical forecasts and
        applies the coefficients to the current forecast.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            distribution (str):
                Name of distribution. Assume that a calibrated version of the
                current forecast could be represented using this distribution.
            desired_units (str or cf_units.Unit):
                The unit that you would like the calibration to be undertaken
                in. The current forecast, historical forecast and truth will be
                converted as required.
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.
            tolerance (float):
                The tolerance for the Continuous Ranked Probability
                Score (CRPS) calculated by the minimisation. The CRPS is in
                the units of the variable being calibrated. The tolerance is
                therefore representative of how close to the actual value are
                we aiming to forecast for a particular variable. Once multiple
                iterations result in a CRPS equal to the same value within the
                specified tolerance, the minimisation will terminate.
            max_iterations (int):
                The maximum number of iterations allowed until the
                minimisation has converged to a stable solution. If the
                maximum number of iterations is reached, but the minimisation
                has not yet converged to a stable solution, then the available
                solution is used anyway, and a warning is raised. If the
                predictor_of_mean is "realizations", then the number of
                iterations may require increasing, as there will be
                more coefficients to solve for.

        """
        self.distribution = distribution
        self.each_point = each_point
        self._validate_distribution()
        self.desired_units = desired_units
        # Ensure predictor is valid.
        check_predictor(predictor)
        self.predictor = predictor
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.pool_size = pool_size
        self.minimiser = ContinuousRankedProbabilityScoreMinimisers(
            tolerance=self.tolerance, max_iterations=self.max_iterations
        )

        # Setting default values for coeff_names.
        self.coeff_names = ["alpha", "beta", "gamma", "delta"]

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = (
            "<EstimateCoefficientsForEnsembleCalibration: "
            "distribution: {}; "
            "desired_units: {}; "
            "predictor: {}; "
            "minimiser: {}; "
            "coeff_names: {}; "
            "tolerance: {}; "
            "max_iterations: {}>"
        )
        return result.format(
            self.distribution,
            self.desired_units,
            self.predictor,
            self.minimiser.__class__,
            self.coeff_names,
            self.tolerance,
            self.max_iterations,
        )

    def _validate_distribution(self):
        """Validate that the distribution supplied has a corresponding method
        for minimising the Continuous Ranked Probability Score.

        Raises:
            ValueError: If the distribution requested is not supported.

        """
        valid_distributions = (
            ContinuousRankedProbabilityScoreMinimisers().minimisation_dict.keys()
        )
        if self.distribution not in valid_distributions:
            msg = (
                "Given distribution {} not available. Available "
                "distributions are {}".format(self.distribution, valid_distributions)
            )
            raise ValueError(msg)

    def _get_statsmodels_availability(self):
        """Import the statsmodels module, if available.

        Returns:
            bool:
                True if the statsmodels module is available. Otherwise, False.

        Warns:
            ImportWarning: If the statsmodels module cannot be imported.
        """
        import importlib

        try:
            importlib.import_module("statsmodels")
        except (ModuleNotFoundError, ImportError):
            sm = None
            if self.predictor.lower() == "realizations":
                msg = (
                    "The statsmodels module cannot be imported. "
                    "Will not be able to calculate an initial guess from "
                    "the individual ensemble realizations. "
                    "A default initial guess will be used without "
                    "estimating coefficients from a linear model."
                )
                warnings.warn(msg, ImportWarning)
        else:
            import statsmodels.api as sm

        return sm

    def _set_attributes(self, historic_forecasts):
        """Set attributes for use on the EMOS coefficients cube.

        Args:
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.

        Returns:
            dict:
                Attributes for an EMOS coefficients cube including
                "diagnostic standard name", "distribution", "shape_parameters"
                and an updated title.
        """
        attributes = generate_mandatory_attributes([historic_forecasts])
        attributes["diagnostic_standard_name"] = historic_forecasts.name()
        attributes["distribution"] = self.distribution
        if self.distribution == "truncnorm":
            # For the CRPS minimisation, the truncnorm distribution is
            # truncated at zero.
            attributes["shape_parameters"] = np.array([0, np.inf], dtype=np.float32)
        attributes["title"] = "Ensemble Model Output Statistics coefficients"
        return attributes

    @staticmethod
    def _create_temporal_coordinates(historic_forecasts):
        """Create forecast reference time and forecast period coordinates
        for the EMOS coefficients cube.

        Args:
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.

        Returns:
            list of tuples:
                List of tuples of the temporal coordinates and the associated
                dimension. This format is suitable for use by iris.cube.Cube.
        """
        # Create forecast reference time coordinate.
        frt_coord = create_unified_frt_coord(
            historic_forecasts.coord("forecast_reference_time")
        )

        fp_coord = historic_forecasts.coord("forecast_period")

        if fp_coord.shape[0] != 1:
            msg = (
                "The forecast period must be the same for all historic forecasts. "
                "Forecast periods found: {}".format(fp_coord.points)
            )
            raise ValueError(msg)

        return [(frt_coord, None), (fp_coord, None)]

    @staticmethod
    def _create_spatial_coordinates(historic_forecasts):
        """Create spatial coordinates for the EMOS coefficients cube.

        Args:
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.

        Returns:
            list of tuples:
                List of tuples of the spatial coordinates and the associated
                dimension. This format is suitable for use by iris.cube.Cube.
        """
        spatial_coords_and_dims = []
        for axis in ["x", "y"]:
            spatial_coords_and_dims.append(
                (historic_forecasts.coord(axis=axis).collapsed(), None)
            )
        return spatial_coords_and_dims

    def _create_cubelist(
        self, optimised_coeffs, historic_forecasts, aux_coords_and_dims, attributes
    ):
        """Create a cubelist by combining the optimised coefficients and the
        appropriate metadata. The units of the alpha and gamma coefficients
        match the units of the historic forecast. If the predictor is the
        realizations, then the beta coefficient cube contains a realization
        coordinate.

        Args:
            optimised_coeffs (numpy.ndarray)
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.
            aux_coords_and_dims (list of tuples):
                List of tuples of the format [(coord, dim), (coord, dim)]
            attributes (dict):
                Attributes for an EMOS coefficients cube including
                "diagnostic standard name" and an updated title.

        Returns:
            cubelist (iris.cube.CubeList):
                CubeList constructed using the coefficients provided and using
                metadata from the historic_forecasts cube. Each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.
        """
        cubelist = iris.cube.CubeList([])
        for optimised_coeff, coeff_name in zip(optimised_coeffs, self.coeff_names):
            coeff_units = "1"
            if coeff_name in ["alpha", "gamma"]:
                coeff_units = historic_forecasts.units
            dim_coords_and_dims = []
            if self.predictor.lower() == "realizations" and coeff_name == "beta":
                dim_coords_and_dims = [
                    (historic_forecasts.coord("realization").copy(), 0)
                ]
            cube = iris.cube.Cube(
                optimised_coeff,
                long_name=f"emos_coefficient_{coeff_name}",
                units=coeff_units,
                dim_coords_and_dims=dim_coords_and_dims,
                aux_coords_and_dims=aux_coords_and_dims,
                attributes=attributes,
            )
            cubelist.append(cube)
        return cubelist

    def create_coefficients_cubelist(self, optimised_coeffs, historic_forecasts):
        """Create a cubelist for storing the coefficients computed using EMOS.

        .. See the documentation for examples of these cubes.
        .. include:: extended_documentation/calibration/
           ensemble_calibration/create_coefficients_cube.rst

        Args:
            optimised_coeffs (list or numpy.ndarray):
                Array or list of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.

        Returns:
            iris.cube.CubeList:
                CubeList constructed using the coefficients provided and using
                metadata from the historic_forecasts cube. Each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.

        Raises:
            ValueError: If the number of coefficients in the optimised_coeffs
                does not match the expected number.
        """
        if self.predictor.lower() == "realizations":
            optimised_coeffs = [
                optimised_coeffs[0],
                optimised_coeffs[1:-2],
                optimised_coeffs[-2],
                optimised_coeffs[-1],
            ]

        if len(optimised_coeffs) != len(self.coeff_names):
            msg = (
                "The number of coefficients in {} must equal the "
                "number of coefficient names {}.".format(
                    optimised_coeffs, self.coeff_names
                )
            )
            raise ValueError(msg)

        aux_coords_and_dims = self._create_temporal_coordinates(historic_forecasts)
        aux_coords_and_dims.extend(self._create_spatial_coordinates(historic_forecasts))
        attributes = self._set_attributes(historic_forecasts)

        return self._create_cubelist(
            optimised_coeffs, historic_forecasts, aux_coords_and_dims, attributes
        )

    def compute_initial_guess(
        self,
        truths,
        forecast_predictor,
        predictor,
        estimate_coefficients_from_linear_model_flag,
    ):
        """
        Function to compute initial guess of the alpha, beta, gamma
        and delta components of the EMOS coefficients by linear regression
        of the forecast predictor and the truths, if requested. Otherwise,
        default values for the coefficients will be used.

        If the predictor is "mean", then the order of the initial_guess is
        [alpha, beta, gamma, delta]. Otherwise, if the predictor is
        "realizations" then the order of the initial_guess is
        [alpha, beta0, beta1, beta2, gamma, delta], where the number of beta
        variables will correspond to the number of realizations. In this
        example initial guess with three beta variables, there will
        correspondingly be three realizations.

        The default values for the initial guesses are in
        [alpha, beta, gamma, delta] ordering:

        * For the ensemble mean, the default initial guess: [0, 1, 0, 1]
          assumes that the raw forecast is skilful and the expected adjustments
          are small.

        * For the ensemble realizations, the default initial guess is
          effectively: [0, 1/3., 1/3., 1/3., 0, 1], such that
          each realization is assumed to have equal weight.

        If linear regression is enabled, the alpha and beta coefficients
        associated with the ensemble mean or ensemble realizations are
        modified based on the results from the linear regression fit.

        Args:
            truths (iris.cube.Cube):
                Cube containing the truth fields.
            forecast_predictor (iris.cube.Cube):
                Cube containing the fields to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.
            estimate_coefficients_from_linear_model_flag (bool):
                Flag whether coefficients should be estimated from
                the linear regression, or static estimates should be used.

        Returns:
            list of float:
                List of coefficients to be used as initial guess.
                Order of coefficients is [alpha, beta, gamma, delta].

        """
        sm = None
        if (
            predictor.lower() == "mean"
            and not estimate_coefficients_from_linear_model_flag
        ):
            initial_guess = [0, 1, 0, 1]
        elif predictor.lower() == "realizations" and (
            not estimate_coefficients_from_linear_model_flag or not sm
        ):
            no_of_realizations = len(forecast_predictor.coord("realization").points)
            initial_beta = np.repeat(
                np.sqrt(1.0 / no_of_realizations), no_of_realizations
            ).tolist()
            initial_guess = [0] + initial_beta + [0, 1]
        elif estimate_coefficients_from_linear_model_flag:
            truths_flattened = flatten_ignoring_masked_data(truths.data)
            if predictor.lower() == "mean":
                forecast_predictor_flattened = flatten_ignoring_masked_data(
                    forecast_predictor.data
                )
                if (truths_flattened.size == 0) or (
                    forecast_predictor_flattened.size == 0
                ):
                    gradient, intercept = [np.nan, np.nan]
                else:
                    gradient, intercept, _, _, _ = stats.linregress(
                        forecast_predictor_flattened, truths_flattened
                    )
                initial_guess = [intercept, gradient, 0, 1]
            elif predictor.lower() == "realizations":
                enforce_coordinate_ordering(forecast_predictor, "realization")
                forecast_predictor_flattened = flatten_ignoring_masked_data(
                    forecast_predictor.data, preserve_leading_dimension=True
                )
                val = sm.add_constant(forecast_predictor_flattened.T)
                est = sm.OLS(truths_flattened, val).fit()
                intercept = est.params[0]
                gradient = est.params[1:]
                initial_guess = [intercept] + gradient.tolist() + [0, 1]

        return np.array(initial_guess, dtype=np.float32)

    @staticmethod
    def mask_cube(cube, landsea_mask):
        """
        Mask the input cube using the given landsea_mask. Sea points are
        filled with nans and masked.

        Args:
            cube (iris.cube.Cube):
                A cube to be masked, on the same grid as the landsea_mask.
                The last two dimensions on this cube must match the dimensions
                in the landsea_mask cube.
            landsea_mask(iris.cube.Cube):
                A cube containing a land-sea mask. Within the
                land-sea mask cube land points should be specified as ones,
                and sea points as zeros.

        Raises:
            IndexError: if the cube and landsea_mask shapes are not compatible.
        """
        try:
            cube.data[..., ~landsea_mask.data.astype(np.bool)] = np.nan
        except IndexError as err:
            msg = "Cube and landsea_mask shapes are not compatible. {}".format(err)
            raise IndexError(msg)
        else:
            cube.data = np.ma.masked_invalid(cube.data)

    def reorganise_pointwise_list(self, coefficients_list):
        """Reorganise the coefficient list to consolidate values for different
        points into a single cube for each coefficient.

        Args:
            coefficients_list (list):
                List of cubes with a separate cube for each coefficient and
                point.

        Returns:
            iris.cube.CubeList:
                CubeList with a single cube for each coefficient.

        """
        cubelist = iris.cube.CubeList(coefficients_list)
        coefficients_cubelist = iris.cube.CubeList()
        for name in self.coeff_names:
            constr = iris.Constraint(f"emos_coefficient_{name}")
            cube = iris.cube.CubeList(
                [c.extract_strict(constr) for c in cubelist]
            ).merge_cube()
            coefficients_cubelist.append(cube)
        return coefficients_cubelist

    def guess_and_minimise(
        self, truths, historic_forecasts, forecast_predictor, forecast_var,
    ):
        """Function to consolidate calls to compute the initial guess, compute
        the optimised coefficients using minimisation and store the resulting
        coefficients within a CubeList.

        Args:
            truths (iris.cube.Cube):
                Truths from the training dataset.
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.
            forecast_predictor (iris.cube.Cube):
                Predictor of the forecast within the minimisation. This
                is either ensemble mean or the ensemble realizations.
            forecast_var (iris.cube.Cube):
                Variance of the forecast for use in the minimisation.

        """
        # Computing initial guess for EMOS coefficients
        initial_guess = self.compute_initial_guess(
            truths,
            forecast_predictor,
            self.predictor,
            self.ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG,
        )

        # Calculate coefficients if there are no nans in the initial guess.
        if np.any(np.isnan(initial_guess)):
            optimised_coeffs = initial_guess
        else:
            optimised_coeffs = self.minimiser(
                initial_guess,
                forecast_predictor,
                truths,
                forecast_var,
                self.predictor,
                self.distribution.lower(),
            )

        coefficients_cubelist = self.create_coefficients_cubelist(
            optimised_coeffs, historic_forecasts
        )

        return coefficients_cubelist

    def process(self, historic_forecasts, truths, landsea_mask=None):
        """
        Using Nonhomogeneous Gaussian Regression/Ensemble Model Output
        Statistics, estimate the required coefficients from historical
        forecasts.

        The main contents of this method is:

        1. Check that the predictor is valid.
        2. Filter the historic forecasts and truths to ensure that these
           inputs match in validity time.
        3. Apply unit conversion to ensure that the historic forecasts and
           truths have the desired units for calibration.
        4. Calculate the variance of the historic forecasts. If the chosen
           predictor is the mean, also calculate the mean of the historic
           forecasts.
        5. If a land-sea mask is provided then mask out sea points in the truths
           and predictor from the historic forecasts.
        6. Calculate initial guess at coefficient values by performing a
           linear regression, if requested, otherwise default values are
           used.
        7. Perform minimisation.

        Args:
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.
            truths (iris.cube.Cube):
                Truths from the training dataset.
            landsea_mask (iris.cube.Cube):
                The optional cube containing a land-sea mask. If provided, only
                land points are used to calculate the coefficients. Within the
                land-sea mask cube land points should be specified as ones,
                and sea points as zeros.

        Returns:
            iris.cube.CubeList:
                CubeList constructed using the coefficients provided and using
                metadata from the historic_forecasts cube. Each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.

        Raises:
            ValueError: If either the historic_forecasts or truths cubes were not
                passed in.
            ValueError: If the units of the historic and truth cubes do not
                match.

        """
        import time
        t0 = time.time()
        print("time1 = ", time.time() - t0)
        if not (historic_forecasts and truths):
            raise ValueError("historic_forecasts and truths cubes must be provided.")

        # Ensure predictor is valid.
        check_predictor(self.predictor)
        #sm = self._get_statsmodels_availability()

        historic_forecasts, truths = filter_non_matching_cubes(
            historic_forecasts, truths
        )
        check_forecast_consistency(historic_forecasts)
        # Make sure inputs have the same units.
        if self.desired_units:
            historic_forecasts.convert_units(self.desired_units)
            truths.convert_units(self.desired_units)

        if historic_forecasts.units != truths.units:
            msg = (
                "The historic forecast units of {} do not match "
                "the truths units {}. These units must match, so that "
                "the coefficients can be estimated."
            )
            raise ValueError(msg)

        if self.predictor.lower() == "mean":
            forecast_predictor = collapsed(
                historic_forecasts, "realization", iris.analysis.MEAN
            )
        elif self.predictor.lower() == "realizations":
            forecast_predictor = historic_forecasts

        forecast_var = collapsed(
            historic_forecasts, "realization", iris.analysis.VARIANCE
        )

        # If a landsea_mask is provided mask out the sea points
        if landsea_mask:
            self.mask_cube(forecast_predictor, landsea_mask)
            self.mask_cube(forecast_var, landsea_mask)
            self.mask_cube(truths, landsea_mask)
        print("time2 = ", time.time() - t0)

        if self.each_point:

            index = [
                forecast_predictor.coord(axis="y"),
                forecast_predictor.coord(axis="x"),
            ]

            # coefficients_list = []
            # for (fp_slice, fv_slice, truths_slice, historic_forecasts_slice) in zip(
            #     forecast_predictor.slices_over(index),
            #     forecast_var.slices_over(index),
            #     truths.slices_over(index),
            #     historic_forecasts.slices_over(index),
            # ):
            #     f1 = functools.partial(self.guess_and_minimise,
            #                            no_of_realizations=no_of_realizations)
            #     coefficients_list.append(f1(
            #         truths_slice,
            #         historic_forecasts_slice,
            #         fp_slice,
            #         fv_slice,
            #     ))
            # coefficients_cube = coefficients_cubelist.merge_cube()

            # print("forecast_predictor = ", forecast_predictor)
            # print("forecast_var = ", forecast_var)
            # print("truths = ", truths)
            # print("historic_forecasts = ", historic_forecasts)

            # Explicitly touch data to avoid slowness when slicing.
            forecast_predictor.data
            forecast_var.data
            truths.data
            #historic_forecasts.data

            # truths = truths[..., :50, :50]
            # #historic_forecasts = historic_forecasts[..., :200, :200]
            # forecast_predictor = forecast_predictor[..., :50, :50]
            # forecast_var = forecast_var[..., :50, :50]


            print("truths = ", truths)
            print("historic_forecasts = ", historic_forecasts)
            print("forecast_predictor = ", forecast_predictor)
            print("forecast_var = ", forecast_var)

            print("time2b = ", time.time() - t0)
            #hf_slice = next(historic_forecasts.slices_over(index)).copy()

            argument_list = (
                (truths_slice, hf_slice, fp_slice, fv_slice,)
                for (fp_slice, fv_slice, truths_slice, hf_slice) in zip(
                    forecast_predictor.slices_over(index),
                    forecast_var.slices_over(index),
                    truths.slices_over(index),
                    historic_forecasts.slices_over(index),
                )
            )
            # argument_list = []
            # for (fp_slice, fv_slice, truths_slice, hf_slice) in zip(
            #     forecast_predictor.slices_over(index),
            #     forecast_var.slices_over(index),
            #     truths.slices_over(index),
            #     historic_forecasts.slices_over(index),
            # ):
            #     argument_list.append((truths_slice, hf_slice, fp_slice, fv_slice,))
            print("argument_list = ", argument_list)

            chunksize = (len(truths.coord(axis="x").points) * len(truths.coord(axis="x").points)) // self.pool_size
            #f1 = functools.partial(self.guess_and_minimise, historic_forecasts=hf_slice)
            print("time3 = ", time.time() - t0)
            with Pool(self.pool_size) as pool:
                coefficients_list = pool.starmap(
                    self.guess_and_minimise, argument_list,
                    chunksize=chunksize)

            print("time4 = ", time.time() - t0)

            coefficients_cubelist = self.reorganise_pointwise_list(coefficients_list)
        else:
            coefficients_cubelist = self.guess_and_minimise(
                truths, historic_forecasts, forecast_predictor, forecast_var,
            )

        print("coefficients_cubelist = ", coefficients_cubelist)
        print("time5 = ", time.time() - t0)
        return coefficients_cubelist


class CalibratedForecastDistributionParameters(BasePlugin):
    """
    Class to calculate calibrated forecast distribution parameters given an
    uncalibrated input forecast and EMOS coefficients.
    """

    def __init__(self, predictor="mean"):
        """
        Create a plugin that uses the coefficients created using EMOS from
        historical forecasts and corresponding truths and applies these
        coefficients to the current forecast to generate location and scale
        parameters that represent the calibrated distribution at each point.

        Args:
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.

        """
        check_predictor(predictor)
        self.predictor = predictor

        self.coefficients_cubelist = None
        self.current_forecast = None

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = "<CalibratedForecastDistributionParameters: predictor: {}>"
        return result.format(self.predictor)

    def _diagnostic_match(self):
        """Check that the forecast diagnostic matches the coefficients used to
        construct the coefficients.

        Raises:
            ValueError: If the forecast diagnostic and coefficients cube
                diagnostic does not match.
        """
        for cube in self.coefficients_cubelist:
            diag = cube.attributes["diagnostic_standard_name"]
            if self.current_forecast.name() != diag:
                msg = (
                    f"The forecast diagnostic ({self.current_forecast.name()}) "
                    "does not match the diagnostic used to construct the "
                    f"coefficients ({diag})"
                )
                raise ValueError(msg)

    def _spatial_domain_match(self):
        """
        Check that the domain of the current forecast and coefficients cube
        match.

        Raises:
            ValueError: If the points or bounds of the specified axis of the
                current_forecast and coefficients_cube do not match.
        """
        msg = (
            "The points or bounds of the {} axis given by the current forecast {} "
            "do not match those given by the coefficients cube {}."
        )

        for axis in ["x", "y"]:
            for coeff_cube in self.coefficients_cubelist:
                if (
                    (
                        self.current_forecast.coord(axis=axis).collapsed().points
                        != coeff_cube.coord(axis=axis).collapsed().points
                    ).all()
                    or (
                        self.current_forecast.coord(axis=axis).collapsed().bounds
                        != coeff_cube.coord(axis=axis).collapsed().bounds
                    ).all()
                ):
                    raise ValueError(
                        msg.format(
                            axis,
                            self.current_forecast.coord(axis=axis).collapsed(),
                            coeff_cube.coord(axis=axis).collapsed(),
                        )
                    )

    def _calculate_location_parameter_from_mean(self):
        """
        Function to calculate the location parameter when the ensemble mean at
        each grid point is the predictor.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Returns:
            numpy.ndarray:
                Location parameter calculated using the ensemble mean as the
                predictor.

        """
        forecast_predictor = collapsed(
            self.current_forecast, "realization", iris.analysis.MEAN
        )

        # Calculate location parameter = a + b*X, where X is the
        # raw ensemble mean. In this case, b = beta.
        location_parameter = (
            self.coefficients_cubelist.extract_strict("emos_coefficient_alpha").data
            + self.coefficients_cubelist.extract_strict("emos_coefficient_beta").data
            * forecast_predictor.data
        ).astype(np.float32)

        return location_parameter

    def _calculate_location_parameter_from_realizations(self):
        """
        Function to calculate the location parameter when the ensemble
        realizations are the predictor.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Returns:
            numpy.ndarray:
                Location parameter calculated using the ensemble realizations
                as the predictor.
        """
        forecast_predictor = self.current_forecast

        # Calculate location parameter = a + b1*X1 .... + bn*Xn, where X is the
        # ensemble realizations. The number of b and X terms depends upon the
        # number of ensemble realizations. In this case, b = beta^2.
        beta_values = np.array([], dtype=np.float32)
        beta_values = self.coefficients_cubelist.extract_strict(
            "emos_coefficient_beta"
        ).data.copy()
        a_and_b = np.append(
            self.coefficients_cubelist.extract_strict("emos_coefficient_alpha").data,
            beta_values ** 2,
        )
        forecast_predictor_flat = convert_cube_data_to_2d(forecast_predictor)
        xy_shape = next(forecast_predictor.slices_over("realization")).shape
        col_of_ones = np.ones(np.prod(xy_shape), dtype=np.float32)
        ones_and_predictor = np.column_stack((col_of_ones, forecast_predictor_flat))
        location_parameter = (
            np.dot(ones_and_predictor, a_and_b).reshape(xy_shape).astype(np.float32)
        )
        return location_parameter

    def _calculate_scale_parameter(self):
        """
        Calculation of the scale parameter using the ensemble variance
        adjusted using the gamma and delta coefficients calculated by EMOS.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Returns:
            numpy.ndarray:
                Scale parameter for defining the distribution of the calibrated
                forecast.

        """
        forecast_var = self.current_forecast.collapsed(
            "realization", iris.analysis.VARIANCE
        )
        # Calculating the scale parameter, based on the raw variance S^2,
        # where predicted variance = c + dS^2, where c = (gamma)^2 and
        # d = (delta)^2
        scale_parameter = (
            self.coefficients_cubelist.extract_strict("emos_coefficient_gamma").data
            ** 2
            + self.coefficients_cubelist.extract_strict("emos_coefficient_delta").data
            ** 2
            * forecast_var.data
        ).astype(np.float32)
        return scale_parameter

    def _create_output_cubes(self, location_parameter, scale_parameter):
        """
        Creation of output cubes containing the location and scale parameters.

        Args:
            location_parameter (numpy.ndarray):
                Location parameter of the calibrated distribution.
            scale_parameter (numpy.ndarray):
                Scale parameter of the calibrated distribution.

        Returns:
            (tuple): tuple containing:
                **location_parameter_cube** (iris.cube.Cube):
                    Location parameter of the calibrated distribution with
                    associated metadata.
                **scale_parameter_cube** (iris.cube.Cube):
                    Scale parameter of the calibrated distribution with
                    associated metadata.
        """
        template_cube = next(self.current_forecast.slices_over("realization"))
        template_cube.remove_coord("realization")

        location_parameter_cube = create_new_diagnostic_cube(
            "location_parameter",
            template_cube.units,
            template_cube,
            template_cube.attributes,
            data=location_parameter,
        )
        scale_parameter_cube = create_new_diagnostic_cube(
            "scale_parameter",
            f"({template_cube.units})^2",
            template_cube,
            template_cube.attributes,
            data=scale_parameter,
        )
        return location_parameter_cube, scale_parameter_cube

    def process(self, current_forecast, coefficients_cubelist, landsea_mask=None):
        """
        Apply the EMOS coefficients to the current forecast, in order to
        generate location and scale parameters for creating the calibrated
        distribution.

        Args:
            current_forecast (iris.cube.Cube):
                The cube containing the current forecast.
            coefficients_cubelist (iris.cube.CubeList):
                CubeList of EMOS coefficients where each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.
            landsea_mask (iris.cube.Cube or None):
                The optional cube containing a land-sea mask. If provided sea
                points will be masked in the output cube.
                This cube needs to have land points set to 1 and
                sea points to 0.

        Returns:
            (tuple): tuple containing:
                **location_parameter_cube** (iris.cube.Cube):
                    Cube containing the location parameter of the calibrated
                    distribution calculated using either the ensemble mean or
                    the ensemble realizations. The location parameter
                    represents the point at which a resulting PDF would be
                    centred.
                **scale_parameter_cube** (iris.cube.Cube):
                    Cube containing the scale parameter of the calibrated
                    distribution calculated using either the ensemble mean or
                    the ensemble realizations. The scale parameter represents
                    the statistical dispersion of the resulting PDF, so a
                    larger scale parameter will result in a broader PDF.

        """
        self.current_forecast = current_forecast
        self.coefficients_cubelist = coefficients_cubelist
        # Check coefficients_cube and forecast cube are compatible.
        self._diagnostic_match()
        for cube in coefficients_cubelist:
            forecast_coords_match(cube, current_forecast)
        self._spatial_domain_match()

        if self.predictor.lower() == "mean":
            location_parameter = self._calculate_location_parameter_from_mean()
        else:
            location_parameter = self._calculate_location_parameter_from_realizations()

        scale_parameter = self._calculate_scale_parameter()

        location_parameter_cube, scale_parameter_cube = self._create_output_cubes(
            location_parameter, scale_parameter
        )

        # Use a mask to confine calibration to land regions by masking the
        # sea.
        if landsea_mask:
            # Calibration is applied to all grid points, but the areas
            # where a mask is valid is then masked out at the end. The cube
            # containing a land-sea mask has sea points defined as zeroes and
            # the land points as ones, so the mask needs to be flipped here.
            flip_mask = np.logical_not(landsea_mask.data)
            scale_parameter_cube.data = np.ma.masked_where(
                flip_mask, scale_parameter_cube.data
            )
            location_parameter_cube.data = np.ma.masked_where(
                flip_mask, location_parameter_cube.data
            )

        return location_parameter_cube, scale_parameter_cube


class ApplyEMOS(PostProcessingPlugin):
    """
    Class to calibrate an input forecast given EMOS coefficients
    """

    @staticmethod
    def _get_attribute(coefficients, attribute_name, optional=False):
        """Get the value for the requested attribute, ensuring that the
        attribute is present consistently across the cubes within the
        coefficients cubelist.

        Args:
            coefficients (iris.cube.CubeList):
                EMOS coefficients
            attribute_name (str):
                Name of expected attribute
            optional (bool):
                Indicate whether the attribute is allowed to be optional.

        Returns:
            None or Any:
                Returns None if the attribute is not present. Otherwise,
                the value of the attribute is returned.

        Raises:
            ValueError: If coefficients do not share the expected attributes.
        """
        attributes = [
            str(c.attributes[attribute_name])
            for c in coefficients
            if c.attributes.get(attribute_name) is not None
        ]

        if not attributes and optional:
            return None
        if not attributes and not optional:
            msg = (
                f"The {attribute_name} attribute must be specified on all "
                "coefficients cubes."
            )
            raise AttributeError(msg)

        if len(set(attributes)) == 1 and len(attributes) == len(coefficients):
            return coefficients[0].attributes[attribute_name]

        msg = (
            "Coefficients must share the same {0} attribute. "
            "{0} attributes provided: {1}".format(attribute_name, attributes)
        )
        raise AttributeError(msg)

    @staticmethod
    def _get_forecast_type(forecast):
        """Identifies whether the forecast is in probability, realization
        or percentile space

        Args:
            forecast (iris.cube.Cube)
        """
        try:
            find_percentile_coordinate(forecast)
        except CoordinateNotFoundError:
            if forecast.name().startswith("probability_of"):
                return "probabilities"
            return "realizations"
        return "percentiles"

    def _convert_to_realizations(self, forecast, realizations_count, ignore_ecc_bounds):
        """Convert an input forecast of probabilities or percentiles into
        pseudo-realizations

        Args:
            forecast (iris.cube.Cube)
            realizations_count (int):
                Number of pseudo-realizations to generate from the input
                forecast
            ignore_ecc_bounds (bool)
        """
        if not realizations_count:
            raise ValueError(
                "The 'realizations_count' argument must be defined "
                "for forecasts provided as {}".format(self.forecast_type)
            )

        if self.forecast_type == "probabilities":
            conversion_plugin = ConvertProbabilitiesToPercentiles(
                ecc_bounds_warning=ignore_ecc_bounds
            )
        if self.forecast_type == "percentiles":
            conversion_plugin = ResamplePercentiles(
                ecc_bounds_warning=ignore_ecc_bounds
            )

        forecast_as_percentiles = conversion_plugin(
            forecast, no_of_percentiles=realizations_count
        )
        forecast_as_realizations = RebadgePercentilesAsRealizations()(
            forecast_as_percentiles
        )

        return forecast_as_realizations

    def _calibrate_forecast(self, forecast, randomise, random_seed):
        """
        Generate calibrated probability, percentile or realization output

        Args:
            forecast (iris.cube.Cube):
                Uncalibrated input forecast
            randomise (bool):
                If True, order realization output randomly rather than using
                the input forecast.  If forecast type is not realizations, this
                is ignored.
            random_seed (int):
                For realizations input if randomise is True, random seed for
                generating re-ordered percentiles.  If randomise is False, the
                random seed may still be used for splitting ties.

        Returns:
            iris.cube.Cube:
                Calibrated forecast
        """
        if self.forecast_type == "probabilities":
            conversion_plugin = ConvertLocationAndScaleParametersToProbabilities(
                distribution=self.distribution["name"],
                shape_parameters=self.distribution["shape"],
            )
            result = conversion_plugin(
                self.distribution["location"], self.distribution["scale"], forecast
            )

        else:
            conversion_plugin = ConvertLocationAndScaleParametersToPercentiles(
                distribution=self.distribution["name"],
                shape_parameters=self.distribution["shape"],
            )

            if self.forecast_type == "percentiles":
                perc_coord = find_percentile_coordinate(forecast)
                result = conversion_plugin(
                    self.distribution["location"],
                    self.distribution["scale"],
                    forecast,
                    percentiles=perc_coord.points,
                )
            else:
                no_of_percentiles = len(forecast.coord("realization").points)
                percentiles = conversion_plugin(
                    self.distribution["location"],
                    self.distribution["scale"],
                    forecast,
                    no_of_percentiles=no_of_percentiles,
                )
                result = EnsembleReordering().process(
                    percentiles,
                    forecast,
                    random_ordering=randomise,
                    random_seed=random_seed,
                )

        return result

    def process(
        self,
        forecast,
        coefficients,
        land_sea_mask=None,
        realizations_count=None,
        ignore_ecc_bounds=True,
        predictor="mean",
        randomise=False,
        random_seed=None,
    ):
        """Calibrate input forecast using pre-calculated coefficients

        Args:
            forecast (iris.cube.Cube):
                Uncalibrated forecast as probabilities, percentiles or
                realizations
            coefficients (iris.cube.CubeList):
                EMOS coefficients
            land_sea_mask (iris.cube.Cube or None):
                Land sea mask where a value of "1" represents land points and
                "0" represents sea.  If set, allows calibration of land points
                only.
            realizations_count (int or None):
                Number of realizations to use when generating the intermediate
                calibrated forecast from probability or percentile inputs
            ignore_ecc_bounds (bool):
                If True, allow percentiles from probabilities to exceed the ECC
                bounds range.  If input is not probabilities, this is ignored.
            predictor (str):
                Predictor to be used to calculate the location parameter of the
                calibrated distribution.  Value is "mean" or "realizations".
            randomise (bool):
                Used in generating calibrated realizations.  If input forecast
                is probabilities or percentiles, this is ignored.
            random_seed (int or None):
                Used in generating calibrated realizations.  If input forecast
                is probabilities or percentiles, this is ignored.

        Returns:
            iris.cube.Cube:
                Calibrated forecast in the form of the input (ie probabilities
                percentiles or realizations)
        """
        self.forecast_type = self._get_forecast_type(forecast)

        forecast_as_realizations = forecast.copy()
        if self.forecast_type != "realizations":
            forecast_as_realizations = self._convert_to_realizations(
                forecast.copy(), realizations_count, ignore_ecc_bounds
            )

        calibration_plugin = CalibratedForecastDistributionParameters(
            predictor=predictor
        )
        location_parameter, scale_parameter = calibration_plugin(
            forecast_as_realizations, coefficients, landsea_mask=land_sea_mask
        )

        self.distribution = {
            "name": self._get_attribute(coefficients, "distribution"),
            "location": location_parameter,
            "scale": scale_parameter,
            "shape": self._get_attribute(
                coefficients, "shape_parameters", optional=True
            ),
        }

        result = self._calibrate_forecast(forecast, randomise, random_seed)

        if land_sea_mask:
            # fill in masked sea points with uncalibrated data
            merge_land_and_sea(result, forecast)

        return result
