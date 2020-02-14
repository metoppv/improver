# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
import datetime
import warnings

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm

from improver import BasePlugin
from improver.calibration.utilities import (
    check_predictor, convert_cube_data_to_2d,
    flatten_ignoring_masked_data, filter_non_matching_cubes)
from improver.metadata.utilities import create_new_diagnostic_cube
from improver.utilities.cube_checker import time_coords_match
from improver.utilities.cube_manipulation import (enforce_coordinate_ordering,
                                                  collapsed)
from improver.utilities.temporal import (
    cycletime_to_datetime, datetime_to_iris_time)


class ContinuousRankedProbabilityScoreMinimisers:
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

    def __init__(self, tolerance=0.01, max_iterations=1000):
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
        # depending upon the distribution requested.
        self.minimisation_dict = {
            "gaussian": self.calculate_normal_crps,
            "truncated_gaussian": self.calculate_truncated_normal_crps}
        self.tolerance = tolerance
        # Maximum iterations for minimisation using Nelder-Mead.
        self.max_iterations = max_iterations

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ContinuousRankedProbabilityScoreMinimisers: '
                  'minimisation_dict: {}; tolerance: {}; max_iterations: {}>')
        print_dict = {}
        for key in self.minimisation_dict:
            print_dict.update({key: self.minimisation_dict[key].__name__})
        return result.format(print_dict, self.tolerance, self.max_iterations)

    def process(
            self, initial_guess, forecast_predictor, truth, forecast_var,
            predictor, distribution):
        """
        Function to pass a given function to the scipy minimize
        function to estimate optimised values for the coefficients.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            initial_guess (list):
                List of optimised coefficients.
                Order of coefficients is [gamma, delta, alpha, beta].
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
                Order of coefficients is [gamma, delta, alpha, beta].

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
            last_iteration_percentage_change = np.absolute(
                (allvecs[-1] - allvecs[-2]) / allvecs[-2])*100
            if (np.any(last_iteration_percentage_change >
                       self.TOLERATED_PERCENTAGE_CHANGE)):
                np.set_printoptions(suppress=True)
                msg = ("The final iteration resulted in a percentage change "
                       "that is greater than the accepted threshold of 5% "
                       "i.e. {}. "
                       "\nA satisfactory minimisation has not been achieved. "
                       "\nLast iteration: {}, "
                       "\nLast-but-one iteration: {}"
                       "\nAbsolute difference: {}\n").format(
                           last_iteration_percentage_change, allvecs[-1],
                           allvecs[-2], np.absolute(allvecs[-2]-allvecs[-1]))
                warnings.warn(msg)

        try:
            minimisation_function = self.minimisation_dict[distribution]
        except KeyError as err:
            msg = ("Distribution requested {} is not supported in {}"
                   "Error message is {}".format(
                       distribution, self.minimisation_dict, err))
            raise KeyError(msg)

        # Ensure predictor is valid.
        check_predictor(predictor)

        # Flatten the data arrays and remove any missing data.
        truth_data = flatten_ignoring_masked_data(truth.data)
        forecast_var_data = flatten_ignoring_masked_data(forecast_var.data)
        if predictor.lower() == "mean":
            forecast_predictor_data = flatten_ignoring_masked_data(
                forecast_predictor.data)
        elif predictor.lower() == "realizations":
            enforce_coordinate_ordering(forecast_predictor, "realization")
            # Need to transpose this array so there are columns for each
            # ensemble member rather than rows.
            forecast_predictor_data = flatten_ignoring_masked_data(
                forecast_predictor.data, preserve_leading_dimension=True).T

        # Increased precision is needed for stable coefficient calculation.
        # The resulting coefficients are cast to float32 prior to output.
        initial_guess = np.array(initial_guess, dtype=np.float64)
        forecast_predictor_data = forecast_predictor_data.astype(np.float64)
        forecast_var_data = forecast_var_data.astype(np.float64)
        truth_data = truth_data.astype(np.float64)
        sqrt_pi = np.sqrt(np.pi).astype(np.float64)
        optimised_coeffs = minimize(
            minimisation_function, initial_guess,
            args=(forecast_predictor_data, truth_data,
                  forecast_var_data, sqrt_pi, predictor),
            method="Nelder-Mead", tol=self.tolerance,
            options={"maxiter": self.max_iterations, "return_all": True})

        if not optimised_coeffs.success:
            msg = ("Minimisation did not result in convergence after "
                   "{} iterations. \n{}".format(
                       self.max_iterations, optimised_coeffs.message))
            warnings.warn(msg)
        calculate_percentage_change_in_last_iteration(optimised_coeffs.allvecs)
        return optimised_coeffs.x.astype(np.float32)

    def calculate_normal_crps(
            self, initial_guess, forecast_predictor, truth, forecast_var,
            sqrt_pi, predictor):
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
                Order of coefficients is [gamma, delta, alpha, beta].
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
            beta = initial_guess[2:]
        elif predictor.lower() == "realizations":
            beta = np.array(
                [initial_guess[2]]+(initial_guess[3:]**2).tolist(),
                dtype=np.float32
            )

        new_col = np.ones(truth.shape, dtype=np.float32)
        all_data = np.column_stack((new_col, forecast_predictor))
        mu = np.dot(all_data, beta)
        sigma = np.sqrt(
            initial_guess[0]**2 + initial_guess[1]**2 * forecast_var)
        xz = (truth - mu) / sigma
        normal_cdf = norm.cdf(xz)
        normal_pdf = norm.pdf(xz)
        if np.isfinite(np.min(mu/sigma)):
            result = np.nanmean(
                sigma * (
                    xz * (2 * normal_cdf - 1) + 2 * normal_pdf - 1 / sqrt_pi))
        else:
            result = self.BAD_VALUE
        return result

    def calculate_truncated_normal_crps(
            self, initial_guess, forecast_predictor, truth, forecast_var,
            sqrt_pi, predictor):
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
                Order of coefficients is [gamma, delta, alpha, beta].
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
            beta = initial_guess[2:]
        elif predictor.lower() == "realizations":
            beta = np.array(
                [initial_guess[2]]+(initial_guess[3:]**2).tolist(),
                dtype=np.float32
            )

        new_col = np.ones(truth.shape, dtype=np.float32)
        all_data = np.column_stack((new_col, forecast_predictor))
        mu = np.dot(all_data, beta)
        sigma = np.sqrt(
            initial_guess[0]**2 + initial_guess[1]**2 * forecast_var)
        xz = (truth - mu) / sigma
        normal_cdf = norm.cdf(xz)
        normal_pdf = norm.pdf(xz)
        x0 = mu / sigma
        normal_cdf_0 = norm.cdf(x0)
        normal_cdf_root_two = norm.cdf(np.sqrt(2) * x0)
        if np.isfinite(np.min(mu / sigma)) or (np.min(mu / sigma) >= -3):
            result = np.nanmean(
                (sigma / normal_cdf_0**2) *
                (xz * normal_cdf_0 * (2 * normal_cdf + normal_cdf_0 - 2) +
                 2 * normal_pdf * normal_cdf_0 -
                 normal_cdf_root_two / sqrt_pi))
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

    def __init__(self, distribution, current_cycle, desired_units=None,
                 predictor="mean", tolerance=0.01, max_iterations=1000):
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
            current_cycle (str):
                The current cycle in YYYYMMDDTHHMMZ format e.g. 20171122T0100Z.
                This is used to create a forecast_reference_time coordinate
                on the resulting EMOS coefficients cube.
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

        Raises:
            ValueError: If the given distribution is not valid.

        Warns:
            ImportWarning: If the statsmodels module can't be imported.
        """
        valid_distributions = (ContinuousRankedProbabilityScoreMinimisers().
                               minimisation_dict.keys())
        if distribution not in valid_distributions:
            msg = ("Given distribution {} not available. Available "
                   "distributions are {}".format(
                       distribution, valid_distributions))
            raise ValueError(msg)
        self.distribution = distribution
        self.current_cycle = current_cycle
        self.desired_units = desired_units
        # Ensure predictor is valid.
        check_predictor(predictor)
        self.predictor = predictor
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.minimiser = ContinuousRankedProbabilityScoreMinimisers(
            tolerance=self.tolerance, max_iterations=self.max_iterations)

        # Setting default values for coeff_names. Beta is the final
        # coefficient name in the list, as there can potentially be
        # multiple beta coefficients if the ensemble realizations, rather
        # than the ensemble mean, are provided as the predictor.
        self.coeff_names = ["gamma", "delta", "alpha", "beta"]

        import imp
        try:
            imp.find_module('statsmodels')
        except ImportError:
            statsmodels_found = False
            if predictor.lower() == "realizations":
                msg = (
                    "The statsmodels can not be imported. "
                    "Will not be able to calculate an initial guess from "
                    "the individual ensemble realizations. "
                    "A default initial guess will be used without "
                    "estimating coefficients from a linear model.")
                warnings.warn(msg, ImportWarning)
        else:
            statsmodels_found = True
            import statsmodels.api as sm
            self.sm = sm
        self.statsmodels_found = statsmodels_found

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<EstimateCoefficientsForEnsembleCalibration: '
                  'distribution: {}; '
                  'current_cycle: {}; '
                  'desired_units: {}; '
                  'predictor: {}; '
                  'minimiser: {}; '
                  'coeff_names: {}; '
                  'tolerance: {}; '
                  'max_iterations: {}>')
        return result.format(
            self.distribution, self.current_cycle, self.desired_units,
            self.predictor, self.minimiser.__class__, self.coeff_names,
            self.tolerance, self.max_iterations)

    def create_coefficients_cube(
            self, optimised_coeffs, historic_forecast):
        """Create a cube for storing the coefficients computed using EMOS.

        .. See the documentation for examples of these cubes.
        .. include:: extended_documentation/calibration/
           ensemble_calibration/create_coefficients_cube.rst

        Args:
            optimised_coeffs (list):
                List of optimised coefficients.
                Order of coefficients is [gamma, delta, alpha, beta].
            historic_forecast (iris.cube.Cube):
                The cube containing the historic forecast.

        Returns:
            iris.cube.Cube:
                Cube constructed using the coefficients provided and using
                metadata from the historic_forecast cube.  The cube contains
                a coefficient_index dimension coordinate where the points
                of the coordinate are integer values and a
                coefficient_name auxiliary coordinate where the points of
                the coordinate are e.g. gamma, delta, alpha, beta.

        Raises:
            ValueError: If the number of coefficients in the optimised_coeffs
                does not match the expected number.
        """
        if self.predictor.lower() == "realizations":
            realization_coeffs = []
            for realization in historic_forecast.coord("realization").points:
                realization_coeffs.append(
                    "{}{}".format(self.coeff_names[-1], np.int32(realization)))
            coeff_names = self.coeff_names[:-1] + realization_coeffs
        else:
            coeff_names = self.coeff_names

        if len(optimised_coeffs) != len(coeff_names):
            msg = ("The number of coefficients in {} must equal the "
                   "number of coefficient names {}.".format(
                       optimised_coeffs, coeff_names))
            raise ValueError(msg)

        coefficient_index = iris.coords.DimCoord(
            np.arange(len(optimised_coeffs), dtype=np.int32),
            long_name="coefficient_index", units="1")
        coefficient_name = iris.coords.AuxCoord(
            coeff_names, long_name="coefficient_name", units="no_unit")
        dim_coords_and_dims = [(coefficient_index, 0)]
        aux_coords_and_dims = [(coefficient_name, 0)]

        # Create a forecast_reference_time coordinate.
        frt_point = cycletime_to_datetime(self.current_cycle)
        try:
            frt_coord = (
                historic_forecast.coord("forecast_reference_time").copy(
                    datetime_to_iris_time(frt_point)))
        except CoordinateNotFoundError:
            pass
        else:
            aux_coords_and_dims.append((frt_coord, None))

        # Create forecast period and time coordinates.
        try:
            fp_point = (
                np.unique(historic_forecast.coord("forecast_period").points))
            fp_coord = (
                historic_forecast.coord("forecast_period").copy(fp_point))
        except CoordinateNotFoundError:
            pass
        else:
            aux_coords_and_dims.append((fp_coord, None))
            if historic_forecast.coords("time"):
                # Ensure that the fp_point is determined with units of seconds.
                copy_of_fp_coord = (
                    historic_forecast.coord("forecast_period").copy())
                copy_of_fp_coord.convert_units("seconds")
                fp_point, = np.unique(copy_of_fp_coord.points)
                time_point = (
                    frt_point + datetime.timedelta(seconds=float(fp_point)))
                time_point = datetime_to_iris_time(time_point)
                time_coord = historic_forecast.coord("time").copy(time_point)
                aux_coords_and_dims.append((time_coord, None))

        # Create x and y coordinates
        for axis in ["x", "y"]:
            historic_coord_points = historic_forecast.coord(axis=axis).points
            coord_point = np.median(historic_coord_points)
            coord_bounds = [historic_coord_points[0],
                            historic_coord_points[-1]]
            new_coord = historic_forecast.coord(axis=axis).copy(
                points=coord_point, bounds=coord_bounds)
            aux_coords_and_dims.append((new_coord, None))

        attributes = {"diagnostic_standard_name": historic_forecast.name()}
        for attribute in historic_forecast.attributes.keys():
            if attribute.endswith("model_configuration"):
                attributes[attribute] = (
                    historic_forecast.attributes[attribute])

        cube = iris.cube.Cube(
            optimised_coeffs, long_name="emos_coefficients", units="1",
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims, attributes=attributes)
        return cube

    def compute_initial_guess(
            self, truth, forecast_predictor, predictor,
            estimate_coefficients_from_linear_model_flag,
            no_of_realizations=None):
        """
        Function to compute initial guess of the alpha, beta, gamma
        and delta components of the EMOS coefficients by linear regression
        of the forecast predictor and the truth, if requested. Otherwise,
        default values for the coefficients will be used.

        If the predictor is "mean", then the order of the initial_guess is
        [gamma, delta, alpha, beta]. Otherwise, if the predictor is
        "realizations" then the order of the initial_guess is
        [gamma, delta, alpha, beta0, beta1, beta2], where the number of beta
        variables will correspond to the number of realizations. In this
        example initial guess with three beta variables, there will
        correspondingly be three realizations.

        The default values for the initial guesses are in
        [gamma, delta, alpha, beta] ordering:

        * For the ensemble mean, the default initial guess: [0, 1, 0, 1]
          assumes that the raw forecast is skilful and the expected adjustments
          are small.

        * For the ensemble realizations, the default initial guess is
          effectively: [0, 1, 0, 1/3., 1/3., 1/3.], such that
          each realization is assumed to have equal weight.

        If linear regression is enabled, the alpha and beta coefficients
        associated with the ensemble mean or ensemble realizations are
        modified based on the results from the linear regression fit.

        Args:
            truth (iris.cube.Cube):
                Cube containing the field, which will be used as truth.
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
            no_of_realizations (int):
                Number of realizations, if ensemble realizations are to be
                used as predictors. Default is None.

        Returns:
            list of float:
                List of coefficients to be used as initial guess.
                Order of coefficients is [gamma, delta, alpha, beta].

        """
        if (predictor.lower() == "mean" and
                not estimate_coefficients_from_linear_model_flag):
            initial_guess = [0, 1, 0, 1]
        elif (predictor.lower() == "realizations" and
              not estimate_coefficients_from_linear_model_flag):
            initial_guess = [0, 1, 0] + np.repeat(
                np.sqrt(1. / no_of_realizations), no_of_realizations).tolist()
        elif estimate_coefficients_from_linear_model_flag:
            truth_flattened = flatten_ignoring_masked_data(truth.data)
            if predictor.lower() == "mean":
                forecast_predictor_flattened = flatten_ignoring_masked_data(
                    forecast_predictor.data)
                if (truth_flattened.size == 0) or (
                        forecast_predictor_flattened.size == 0):
                    gradient, intercept = ([np.nan, np.nan])
                else:
                    gradient, intercept, _, _, _ = (
                        stats.linregress(
                            forecast_predictor_flattened, truth_flattened))
                initial_guess = [0, 1, intercept, gradient]
            elif predictor.lower() == "realizations":
                if self.statsmodels_found:
                    enforce_coordinate_ordering(
                        forecast_predictor, "realization")
                    forecast_predictor_flattened = (
                        flatten_ignoring_masked_data(
                            forecast_predictor.data,
                            preserve_leading_dimension=True))
                    val = self.sm.add_constant(forecast_predictor_flattened.T)
                    est = self.sm.OLS(truth_flattened, val).fit()
                    intercept = est.params[0]
                    gradient = est.params[1:]
                    initial_guess = [0, 1, intercept]+gradient.tolist()
                else:
                    initial_guess = (
                        [0, 1, 0] +
                        np.repeat(np.sqrt(1./no_of_realizations),
                                  no_of_realizations).tolist())
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
            msg = (
                "Cube and landsea_mask shapes are not compatible. {}".format(
                    err))
            raise IndexError(msg)
        else:
            cube.data = np.ma.masked_invalid(cube.data)

    def process(self, historic_forecast, truth, landsea_mask=None):
        """
        Using Nonhomogeneous Gaussian Regression/Ensemble Model Output
        Statistics, estimate the required coefficients from historical
        forecasts.

        The main contents of this method is:

        1. Check that the predictor is valid.
        2. Filter the historic forecasts and truth to ensure that these
           inputs match in validity time.
        3. Apply unit conversion to ensure that the historic forecasts and
           truth have the desired units for calibration.
        4. Calculate the variance of the historic forecasts. If the chosen
           predictor is the mean, also calculate the mean of the historic
           forecasts.
        5. If a land-sea mask is provided then mask out sea points in the truth
           and predictor from the historic forecasts.
        6. Calculate initial guess at coefficient values by performing a
           linear regression, if requested, otherwise default values are
           used.
        7. Perform minimisation.

        Args:
            historic_forecast (iris.cube.Cube):
                The cube containing the historical forecasts used
                for calibration.
            truth (iris.cube.Cube):
                The cube containing the truth used for calibration.
            landsea_mask (iris.cube.Cube):
                The optional cube containing a land-sea mask. If provided, only
                land points are used to calculate the coefficients. Within the
                land-sea mask cube land points should be specified as ones,
                and sea points as zeros.

        Returns:
            iris.cube.Cube:
                Cube containing the coefficients estimated using EMOS.
                The cube contains a coefficient_index dimension coordinate
                and a coefficient_name auxiliary coordinate.

        Raises:
            ValueError: If either the historic_forecast or truth cubes were not
                passed in.
            ValueError: If the units of the historic and truth cubes do not
                match.

        """
        if not (historic_forecast and truth):
            raise ValueError("historic_forecast and truth cubes must be "
                             "provided.")

        # Ensure predictor is valid.
        check_predictor(self.predictor)

        historic_forecast, truth = (
            filter_non_matching_cubes(historic_forecast, truth))

        # Make sure inputs have the same units.
        if self.desired_units:
            historic_forecast.convert_units(self.desired_units)
            truth.convert_units(self.desired_units)

        if historic_forecast.units != truth.units:
            msg = ("The historic forecast units of {} do not match "
                   "the truth units {}. These units must match, so that "
                   "the coefficients can be estimated.")
            raise ValueError(msg)

        if self.predictor.lower() == "mean":
            no_of_realizations = None
            forecast_predictor = collapsed(
                historic_forecast, "realization", iris.analysis.MEAN)
        elif self.predictor.lower() == "realizations":
            no_of_realizations = len(
                historic_forecast.coord("realization").points)
            forecast_predictor = historic_forecast

        forecast_var = collapsed(
            historic_forecast, "realization", iris.analysis.VARIANCE)

        # If a landsea_mask is provided mask out the sea points
        if landsea_mask:
            self.mask_cube(forecast_predictor, landsea_mask)
            self.mask_cube(forecast_var, landsea_mask)
            self.mask_cube(truth, landsea_mask)

        # Computing initial guess for EMOS coefficients
        initial_guess = self.compute_initial_guess(
            truth, forecast_predictor, self.predictor,
            self.ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG,
            no_of_realizations=no_of_realizations)

        # Calculate coefficients if there are no nans in the initial guess.
        if np.any(np.isnan(initial_guess)):
            optimised_coeffs = initial_guess
        else:
            optimised_coeffs = (
                self.minimiser.process(
                    initial_guess, forecast_predictor,
                    truth, forecast_var,
                    self.predictor,
                    self.distribution.lower()))
        coefficients_cube = (
            self.create_coefficients_cube(optimised_coeffs, historic_forecast))
        return coefficients_cube


class ApplyCoefficientsFromEnsembleCalibration(BasePlugin):
    """
    Class to apply the optimised EMOS coefficients to the current forecast.

    """
    def __init__(self, predictor="mean"):
        """
        Create a plugin that uses the coefficients created using EMOS from
        historical forecasts and corresponding truths and applies these
        coefficients to the current forecast to generate a location and scale
        parameter that represents the calibrated distribution.

        Args:
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.

        """
        check_predictor(predictor)
        self.predictor = predictor

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ApplyCoefficientsFromEnsembleCalibration: predictor: {}>')
        return result.format(self.predictor)

    def _spatial_domain_match(self):
        """
        Check that the domain of the current forecast and coefficients cube
        match.

        Raises:
            ValueError: If the domain information of the current_forecast and
                coefficients_cube do not match.
        """
        msg = ("The domain along the {} axis given by the current forecast {} "
               "does not match the domain given by the coefficients cube {}.")

        for axis in ["x", "y"]:
            current_forecast_points = [
                self.current_forecast.coord(axis=axis).points[0],
                self.current_forecast.coord(axis=axis).points[-1]]
            if not np.allclose(current_forecast_points,
                               self.coefficients_cube.coord(axis=axis).bounds):
                raise ValueError(
                    msg.format(axis, current_forecast_points,
                               self.coefficients_cube.coord(axis=axis).bounds))

    def _calculate_location_parameter_from_mean(self, optimised_coeffs):
        """
        Function to calculate the location parameter when the ensemble mean at
        each grid point is the predictor.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            optimised_coeffs (dict):
                A dictionary containing the calibration coefficient names as
                keys with their corresponding values.

        Returns:
            numpy.ndarray:
                Location parameter calculated using the ensemble mean as the
                predictor.

        """
        forecast_predictor = collapsed(self.current_forecast,
                                       "realization",
                                       iris.analysis.MEAN)

        # Calculate location parameter = a + b*X, where X is the
        # raw ensemble mean. In this case, b = beta.
        location_parameter = (
            optimised_coeffs["alpha"] +
            optimised_coeffs["beta"] * forecast_predictor.data).astype(
                np.float32)

        return location_parameter

    def _calculate_location_parameter_from_realizations(
            self, optimised_coeffs):
        """
        Function to calculate the location parameter when the ensemble
        realizations are the predictor.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            optimised_coeffs (dict):
                A dictionary containing the calibration coefficient names as
                keys with their corresponding values.

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
        for key in optimised_coeffs.keys():
            if key.startswith("beta"):
                beta_values = np.append(beta_values, optimised_coeffs[key])
        a_and_b = np.append(optimised_coeffs["alpha"], beta_values**2)
        forecast_predictor_flat = (
            convert_cube_data_to_2d(forecast_predictor))
        xy_shape = next(forecast_predictor.slices_over("realization")).shape
        col_of_ones = np.ones(np.prod(xy_shape), dtype=np.float32)
        ones_and_predictor = (
            np.column_stack((col_of_ones, forecast_predictor_flat)))
        location_parameter = (
            np.dot(ones_and_predictor, a_and_b).reshape(xy_shape).astype(
                np.float32))
        return location_parameter

    def _calculate_scale_parameter(self, optimised_coeffs):
        """
        Calculation of the scale parameter using the ensemble variance
        adjusted using the gamma and delta coefficients calculated by EMOS.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            optimised_coeffs (dict):
                A dictionary containing the calibration coefficient names as
                keys with their corresponding values.

        Returns:
            numpy.ndarray:
                Scale parameter for defining the distribution of the calibrated
                forecast.

        """
        forecast_var = self.current_forecast.collapsed(
            "realization", iris.analysis.VARIANCE)
        # Calculating the scale parameter, based on the raw variance S^2,
        # where predicted variance = c + dS^2, where c = (gamma)^2 and
        # d = (delta)^2
        scale_parameter = (
            optimised_coeffs["gamma"]**2 +
            optimised_coeffs["delta"]**2 * forecast_var.data).astype(
                np.float32)
        return scale_parameter

    def _create_output_cubes(
            self, location_parameter, scale_parameter):
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
            "location_parameter", template_cube.units, template_cube,
            template_cube.attributes, data=location_parameter)
        scale_parameter_cube = create_new_diagnostic_cube(
            "scale_parameter", f"({template_cube.units})^2",
            template_cube, template_cube.attributes, data=scale_parameter)
        return location_parameter_cube, scale_parameter_cube

    def process(self, current_forecast, coefficients_cube, landsea_mask=None):
        """
        Apply the EMOS coefficients to the current forecast, in order to
        generate location and scale parameters for creating the calibrated
        distribution.

        Args:
            current_forecast (iris.cube.Cube):
                The cube containing the current forecast.
            coefficients_cube (iris.cube.Cube):
                Cube containing the coefficients estimated using EMOS.
                The cube contains a coefficient_index dimension coordinate
                where the points of the coordinate are integer values and a
                coefficient_name auxiliary coordinate where the points of
                the coordinate are e.g. gamma, delta, alpha, beta.
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
        self.coefficients_cube = coefficients_cube

        # Check coefficients_cube and forecast cube are compatible.
        time_coords_match(self.current_forecast, self.coefficients_cube)
        self._spatial_domain_match()

        optimised_coeffs = (
            dict(zip(self.coefficients_cube.coord("coefficient_name").points,
                     self.coefficients_cube.data)))

        if self.predictor.lower() == "mean":
            location_parameter = (
                self._calculate_location_parameter_from_mean(optimised_coeffs))
        else:
            location_parameter = (
                self._calculate_location_parameter_from_realizations(
                    optimised_coeffs))

        scale_parameter = self._calculate_scale_parameter(optimised_coeffs)

        location_parameter_cube, scale_parameter_cube = (
            self._create_output_cubes(location_parameter, scale_parameter))

        # Use a mask to confine calibration to land regions by masking the
        # sea.
        if landsea_mask:
            # Calibration is applied to all grid points, but the areas
            # where a mask is valid is then masked out at the end. The cube
            # containing a land-sea mask has sea points defined as zeroes and
            # the land points as ones, so the mask needs to be flipped here.
            flip_mask = np.logical_not(landsea_mask.data)
            scale_parameter_cube.data = np.ma.masked_where(
                flip_mask, scale_parameter_cube.data)
            location_parameter_cube.data = np.ma.masked_where(
                flip_mask, location_parameter_cube.data)

        return location_parameter_cube, scale_parameter_cube
