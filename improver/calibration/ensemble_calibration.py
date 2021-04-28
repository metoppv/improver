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
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import iris
import numpy as np
from cf_units import Unit
from iris.coords import Coord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray
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
    statsmodels_available,
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
    guess. The coefficients will be calculated either using all points provided
    or coefficients will be calculated separately for each point.

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

    def __init__(
        self,
        tolerance: float = 0.02,
        max_iterations: int = 1000,
        point_by_point: bool = False,
    ) -> None:
        """
        Initialise class for performing minimisation of the Continuous
        Ranked Probability Score (CRPS).

        Args:
            tolerance:
                The tolerance for the Continuous Ranked Probability
                Score (CRPS) calculated by the minimisation. The CRPS is in
                the units of the variable being calibrated. The tolerance is
                therefore representative of how close to the actual value are
                we aiming to forecast for a particular variable. Once multiple
                iterations result in a CRPS equal to the same value within the
                specified tolerance, the minimisation will terminate.
            max_iterations:
                The maximum number of iterations allowed until the
                minimisation has converged to a stable solution. If the
                maximum number of iterations is reached, but the minimisation
                has not yet converged to a stable solution, then the available
                solution is used anyway, and a warning is raised. If the
                predictor_of_mean is "realizations", then the number of
                iterations may require increasing, as there will be
                more coefficients to solve for.
            point_by_point:
                If True, coefficients are calculated independently for each
                point within the input cube by minimising each point
                independently.
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
        self.point_by_point = point_by_point

    def _calculate_percentage_change_in_last_iteration(
        self, allvecs: List[ndarray]
    ) -> None:
        """
        Calculate the percentage change that has occurred within
        the last iteration of the minimisation. If the percentage change
        between the last iteration and the last-but-one iteration exceeds
        the threshold, a warning message is printed.

        Args:
            allvecs:
                List of numpy arrays containing the optimised coefficients,
                after each iteration.

        Warns:
            Warning: If a satisfactory minimisation has not been achieved.
        """
        last_iteration_percentage_change = (
            np.absolute((allvecs[-1] - allvecs[-2]) / allvecs[-2]) * 100
        )
        if np.any(last_iteration_percentage_change > self.TOLERATED_PERCENTAGE_CHANGE):
            np.set_printoptions(suppress=True)
            msg = (
                "The final iteration resulted in a percentage change "
                f"that is greater than the accepted threshold of "
                f"{self.TOLERATED_PERCENTAGE_CHANGE}% "
                f"i.e. {last_iteration_percentage_change}. "
                "\nA satisfactory minimisation has not been achieved. "
                f"\nLast iteration: {allvecs[-1]}, "
                f"\nLast-but-one iteration: {allvecs[-2]}"
                f"\nAbsolute difference: {np.absolute(allvecs[-2] - allvecs[-1])}\n"
            )
            warnings.warn(msg)

    def _minimise_caller(
        self,
        minimisation_function: Callable,
        initial_guess: Union[List[float], ndarray],
        forecast_predictor_data: ndarray,
        truth_data: ndarray,
        forecast_var_data: ndarray,
        sqrt_pi: float,
        predictor: str,
    ) -> ndarray:
        """Call scipy minimize with the options provided.

        Args:
            minimisation_function
            initial_guess
            forecast_predictor
            truth
            forecast_var
            sqrt_pi:
                Square root of pi for minimisation.
            predictor

        Return:
            A single set of coefficients with the order [alpha, beta, gamma, delta].
        """
        optimised_coeffs = minimize(
            minimisation_function,
            initial_guess,
            args=(
                forecast_predictor_data,
                truth_data,
                forecast_var_data,
                sqrt_pi,
                predictor,
            ),
            method="Nelder-Mead",
            tol=self.tolerance,
            options={"maxiter": self.max_iterations, "return_all": True},
        )

        return optimised_coeffs

    def _set_float64_precision(
        self,
        initial_guess: Union[List[float], ndarray],
        forecast_predictor_data: ndarray,
        truth_data: ndarray,
        forecast_var_data: ndarray,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray, float]:
        """Set values to float64 precision and define a square root of pi
        variable for later usage. The increased precision is need for stable
        coefficient calculation.

        Args:
            initial_guess
            forecast_predictor_data
            truth_data
            forecast_var_data

        Returns:
            Tuple containing the initial guess, forecast predictor, truth
            and forecast variance that have been set to float64 precision
            as well as defining a square root of pi variable with float64
            precision.
        """
        return (
            np.array(initial_guess, dtype=np.float64),
            forecast_predictor_data.astype(np.float64),
            truth_data.astype(np.float64),
            forecast_var_data.astype(np.float64),
            np.sqrt(np.pi).astype(np.float64),
        )

    def _process_points_independently(
        self,
        minimisation_function: Callable,
        initial_guess: Union[List[float], ndarray],
        forecast_predictor: Cube,
        truth: Cube,
        forecast_var: Cube,
        predictor: str,
    ) -> ndarray:
        """Minimise each point along the spatial dimensions independently to
        create a set of coefficients for each point. The coefficients returned
        can be either gridded (i.e. separate dimensions for x and y) or for a
        list of sites where x and y share a common dimension.

        Args:
            minimisation_function:
                Function to use when minimising.
            initial_guess
            forecast_predictor
            truth
            forecast_var
            predictor

        Returns:
            Separate optimised coefficients for each point. The shape of the
            coefficients array is (number of coefficients, length of spatial dimensions).
            Order of coefficients is [alpha, beta, gamma, delta].
        """
        (
            initial_guess,
            forecast_predictor.data,
            forecast_var.data,
            truth.data,
            sqrt_pi,
        ) = self._set_float64_precision(
            initial_guess, forecast_predictor.data, forecast_var.data, truth.data
        )

        sindex = [
            forecast_predictor.coord(axis="y"),
            forecast_predictor.coord(axis="x"),
        ]

        optimised_coeffs = []
        for index, (fp_slice, truth_slice, fv_slice) in enumerate(
            zip(
                forecast_predictor.slices_over(sindex),
                truth.slices_over(sindex),
                forecast_var.slices_over(sindex),
            )
        ):
            optimised_coeffs.append(
                self._minimise_caller(
                    minimisation_function,
                    initial_guess[index],
                    fp_slice.data.T,
                    truth_slice.data,
                    fv_slice.data,
                    sqrt_pi,
                    predictor,
                ).x.astype(np.float32)
            )

        y_coord = forecast_predictor.coord(axis="y")
        x_coord = forecast_predictor.coord(axis="x")

        if forecast_predictor.coord_dims(y_coord) == forecast_predictor.coord_dims(
            x_coord
        ):
            return np.array(np.transpose(optimised_coeffs)).reshape(
                (len(initial_guess[0]), len(forecast_predictor.coord(axis="y").points),)
            )
        else:
            return np.array(np.transpose(optimised_coeffs)).reshape(
                (
                    len(initial_guess[0]),
                    len(forecast_predictor.coord(axis="y").points),
                    len(forecast_predictor.coord(axis="x").points),
                )
            )

    def _process_points_together(
        self,
        minimisation_function: Callable,
        initial_guess: Union[List[float], ndarray],
        forecast_predictor: Cube,
        truth: Cube,
        forecast_var: Cube,
        predictor: str,
    ) -> ndarray:
        """Minimise all points together in one minimisation to create a single
        set of coefficients.

        Args:
            minimisation_function:
                Function to use when minimising.
            initial_guess
            forecast_predictor
            truth
            forecast_var
            predictor

        Returns:
            The optimised coefficients. Order of coefficients is [alpha, beta, gamma, delta].
        """
        # Flatten the data arrays and remove any missing data.
        truth_data = flatten_ignoring_masked_data(truth.data)
        forecast_var_data = flatten_ignoring_masked_data(forecast_var.data)
        if predictor.lower() == "mean":
            forecast_predictor_data = flatten_ignoring_masked_data(
                forecast_predictor.data
            )
        elif predictor.lower() == "realizations":
            # Need to transpose this array so there are columns for each
            # ensemble member rather than rows.
            forecast_predictor_data = flatten_ignoring_masked_data(
                forecast_predictor.data, preserve_leading_dimension=True
            ).T
        (
            initial_guess,
            forecast_predictor_data,
            forecast_var_data,
            truth_data,
            sqrt_pi,
        ) = self._set_float64_precision(
            initial_guess, forecast_predictor_data, forecast_var_data, truth_data
        )

        optimised_coeffs = self._minimise_caller(
            minimisation_function,
            initial_guess,
            forecast_predictor_data,
            truth_data,
            forecast_var_data,
            sqrt_pi,
            predictor,
        )

        if not optimised_coeffs.success:
            msg = (
                "Minimisation did not result in convergence after "
                "{} iterations. \n{}".format(
                    self.max_iterations, optimised_coeffs.message
                )
            )
            warnings.warn(msg)
        self._calculate_percentage_change_in_last_iteration(optimised_coeffs.allvecs)
        return optimised_coeffs.x.astype(np.float32)

    def process(
        self,
        initial_guess: List[float],
        forecast_predictor: Cube,
        truth: Cube,
        forecast_var: Cube,
        predictor: str,
        distribution: str,
    ) -> ndarray:
        """
        Function to pass a given function to the scipy minimize
        function to estimate optimised values for the coefficients.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            initial_guess:
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            forecast_predictor:
                Cube containing the fields to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            truth:
                Cube containing the field, which will be used as truth.
            forecast_var:
                Cube containing the field containing the ensemble variance.
            predictor:
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.
            distribution:
                String used to access the appropriate function for use in the
                minimisation within self.minimisation_dict.

        Returns:
            The optimised coefficients following the order
            [alpha, beta, gamma, delta]. If point_by_point is False, then
            one set of coefficients are generated. If point_by_point
            is True, then the leading dimension of the numpy array is
            the length of the spatial dimensions within the forecast and
            truth cubes. Each set of coefficients are appropriate for a
            particular point.

        Raises:
            KeyError: If the distribution is not supported.

        Warns:
            Warning: If the minimisation did not converge.

        """
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

        if predictor.lower() == "realizations":
            enforce_coordinate_ordering(forecast_predictor, "realization")

        if self.point_by_point:
            optimised_coeffs = self._process_points_independently(
                minimisation_function,
                initial_guess,
                forecast_predictor,
                truth,
                forecast_var,
                predictor,
            )
        else:
            optimised_coeffs = self._process_points_together(
                minimisation_function,
                initial_guess,
                forecast_predictor,
                truth,
                forecast_var,
                predictor,
            )
        return optimised_coeffs

    def calculate_normal_crps(
        self,
        initial_guess: List[float],
        forecast_predictor: ndarray,
        truth: ndarray,
        forecast_var: ndarray,
        sqrt_pi: float,
        predictor: str,
    ) -> float:
        """
        Calculate the CRPS for a normal distribution.

        Scientific Reference:
        Gneiting, T. et al., 2005.
        Calibrated Probabilistic Forecasting Using Ensemble Model Output
        Statistics and Minimum CRPS Estimation.
        Monthly Weather Review, 133(5), pp.1098-1118.

        Args:
            initial_guess:
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            forecast_predictor:
                Data to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            truth:
                Data to be used as truth.
            forecast_var:
                Ensemble variance data.
            sqrt_pi:
                Square root of Pi
            predictor:
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.

        Returns:
            CRPS for the current set of coefficients. This CRPS is a mean
            value across all points.
        """
        if predictor.lower() == "mean":
            a, b, gamma, delta = initial_guess
            a_b = np.array([a, b], dtype=np.float64)
        elif predictor.lower() == "realizations":
            a, b, gamma, delta = (
                initial_guess[0],
                initial_guess[1:-2] * initial_guess[1:-2],
                initial_guess[-2],
                initial_guess[-1],
            )
            a_b = np.array([a] + b.tolist(), dtype=np.float64)

        new_col = np.ones(truth.shape, dtype=np.float32)
        all_data = np.column_stack((new_col, forecast_predictor))
        mu = np.dot(all_data, a_b)
        sigma = np.sqrt(gamma * gamma + delta * delta * forecast_var)
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
        self,
        initial_guess: List[float],
        forecast_predictor: ndarray,
        truth: ndarray,
        forecast_var: ndarray,
        sqrt_pi: float,
        predictor: str,
    ) -> float:
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
            initial_guess:
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            forecast_predictor:
                Data to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            truth:
                Data to be used as truth.
            forecast_var:
                Ensemble variance data.
            sqrt_pi:
                Square root of Pi
            predictor:
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.

        Returns:
            CRPS for the current set of coefficients. This CRPS is a mean
            value across all points.
        """
        if predictor.lower() == "mean":
            a, b, gamma, delta = initial_guess
            mu = (forecast_predictor * b) + a

        elif predictor.lower() == "realizations":
            a, b, gamma, delta = (
                initial_guess[0],
                initial_guess[1:-2] * initial_guess[1:-2],
                initial_guess[-2],
                initial_guess[-1],
            )
            a_b = np.array([a] + b.tolist(), dtype=np.float64)

            new_col = np.ones(truth.shape, dtype=np.float32)
            all_data = np.column_stack((new_col, forecast_predictor))
            mu = np.dot(all_data, a_b)

        sigma = np.sqrt(gamma * gamma + delta * delta * forecast_var)
        xz = (truth - mu) / sigma
        normal_cdf = norm.cdf(xz)
        normal_pdf = norm.pdf(xz)
        x0 = mu / sigma
        normal_cdf_0 = norm.cdf(x0)
        normal_cdf_root_two = norm.cdf(np.sqrt(2) * x0)

        if np.isfinite(np.min(mu / sigma)) or (np.min(mu / sigma) >= -3):
            result = np.nanmean(
                (sigma / (normal_cdf_0 * normal_cdf_0))
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

    def __init__(
        self,
        distribution: str,
        point_by_point: bool = False,
        use_default_initial_guess: bool = False,
        desired_units: Optional[Union[str, Unit]] = None,
        predictor: str = "mean",
        tolerance: float = 0.02,
        max_iterations: int = 1000,
    ) -> None:
        """
        Create an ensemble calibration plugin that, for Nonhomogeneous Gaussian
        Regression, calculates coefficients based on historical forecasts and
        applies the coefficients to the current forecast.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            distribution:
                Name of distribution. Assume that a calibrated version of the
                current forecast could be represented using this distribution.
            point_by_point:
                If True, coefficients are calculated independently for each
                point within the input cube by creating an initial guess and
                minimising each grid point independently. Please note this
                option is memory intensive and is unsuitable for gridded input.
                Using a default initial guess may reduce the memory overhead
                option.
            use_default_initial_guess:
                If True, use the default initial guess. The default initial
                guess assumes no adjustments are required to the initial
                choice of predictor to generate the calibrated distribution.
                This means coefficients of 1 for the multiplicative
                coefficients and 0 for the additive coefficients. If False,
                the initial guess is computed.
            desired_units:
                The unit that you would like the calibration to be undertaken
                in. The current forecast, historical forecast and truth will be
                converted as required.
            predictor:
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.
            tolerance:
                The tolerance for the Continuous Ranked Probability
                Score (CRPS) calculated by the minimisation. The CRPS is in
                the units of the variable being calibrated. The tolerance is
                therefore representative of how close to the actual value are
                we aiming to forecast for a particular variable. Once multiple
                iterations result in a CRPS equal to the same value within the
                specified tolerance, the minimisation will terminate.
            max_iterations:
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
        self.point_by_point = point_by_point
        self.use_default_initial_guess = use_default_initial_guess
        self._validate_distribution()
        self.desired_units = desired_units
        # Ensure predictor is valid.
        check_predictor(predictor)
        self.predictor = predictor
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.minimiser = ContinuousRankedProbabilityScoreMinimisers(
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            point_by_point=self.point_by_point,
        )

        # Setting default values for coeff_names.
        self.coeff_names = ["alpha", "beta", "gamma", "delta"]

    def _validate_distribution(self) -> None:
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

    def _set_attributes(self, historic_forecasts: Cube) -> Dict[str, Any]:
        """Set attributes for use on the EMOS coefficients cube.

        Args:
            historic_forecasts:
                Historic forecasts from the training dataset.

        Returns:
            Attributes for an EMOS coefficients cube including
            "diagnostic standard name", "distribution", "shape_parameters"
            and an updated title.
        """
        attributes = {}
        attributes["diagnostic_standard_name"] = historic_forecasts.name()
        attributes["distribution"] = self.distribution
        if self.distribution == "truncnorm":
            # For the CRPS minimisation, the truncnorm distribution is
            # truncated at zero.
            attributes["shape_parameters"] = np.array([0, np.inf], dtype=np.float32)
        attributes["title"] = "Ensemble Model Output Statistics coefficients"
        return attributes

    @staticmethod
    def _create_temporal_coordinates(historic_forecasts: Cube) -> List[Coord]:
        """Create forecast reference time and forecast period coordinates
        for the EMOS coefficients cube.

        Args:
            historic_forecasts:
                Historic forecasts from the training dataset.

        Returns:
            List of the temporal coordinates.
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

        return [frt_coord, fp_coord]

    def _get_spatial_associated_coordinates(
        self, historic_forecasts: Cube
    ) -> Tuple[List[int], List[Coord]]:
        """Set-up the spatial dimensions and coordinates for the EMOS
        coefficients cube.

        Args:
            historic_forecasts:
                Historic forecasts from the training dataset.

        Returns:
            List of the spatial dimensions to retain within the
            coefficients cube and a list of the auxiliary coordinates that
            share the same dimension as the spatial coordinates.
        """
        template_dims = []
        if self.point_by_point:
            spatial_coords = [historic_forecasts.coord(axis=axis) for axis in "yx"]
            spatial_dims = {
                n for n, in [historic_forecasts.coord_dims(c) for c in spatial_coords]
            }
            template_dims = [x for x in spatial_dims]
            spatial_associated_coords = [
                c
                for d in template_dims
                for c in historic_forecasts.coords(dimensions=d)
            ]
        else:
            spatial_associated_coords = [
                historic_forecasts.coord(axis=axis).collapsed() for axis in "yx"
            ]
        return template_dims, spatial_associated_coords

    def _create_cubelist(
        self, optimised_coeffs: ndarray, historic_forecasts: Cube
    ) -> CubeList:
        """Create a cubelist by combining the optimised coefficients and the
        appropriate metadata. The units of the alpha and gamma coefficients
        match the units of the historic forecast. If the predictor is the
        realizations, then the beta coefficient cube contains a realization
        coordinate.

        Args:
            optimised_coeffs
            historic_forecasts:
                Historic forecasts from the training dataset.

        Returns:
            CubeList constructed using the coefficients provided and using
            metadata from the historic_forecasts cube. Each cube within the
            cubelist is for a separate EMOS coefficient e.g. alpha, beta,
            gamma, delta.
        """
        (
            template_dims,
            spatial_associated_coords,
        ) = self._get_spatial_associated_coordinates(historic_forecasts)
        coords_to_replace = (
            self._create_temporal_coordinates(historic_forecasts)
            + spatial_associated_coords
        )
        coords_to_replace_names = [c.name() for c in coords_to_replace]

        cubelist = iris.cube.CubeList([])
        for optimised_coeff, coeff_name in zip(optimised_coeffs, self.coeff_names):
            used_dims = template_dims.copy()
            replacements = coords_to_replace_names.copy()
            if self.predictor.lower() == "realizations" and "beta" == coeff_name:
                used_dims = ["realization"] + used_dims
                replacements += ["realization"]
            template_cube = next(historic_forecasts.slices(used_dims))

            for coord in coords_to_replace:
                template_cube.replace_coord(coord)

            for coord in set([c.name() for c in template_cube.coords()]) - set(
                replacements
            ):
                template_cube.remove_coord(coord)

            coeff_units = "1"
            if coeff_name in ["alpha", "gamma"]:
                coeff_units = historic_forecasts.units

            cube = create_new_diagnostic_cube(
                f"emos_coefficient_{coeff_name}",
                coeff_units,
                template_cube,
                generate_mandatory_attributes([historic_forecasts]),
                optional_attributes=self._set_attributes(historic_forecasts),
                data=np.array(optimised_coeff),
            )
            cubelist.append(cube)
        return cubelist

    def create_coefficients_cubelist(
        self, optimised_coeffs: Union[List[float], ndarray], historic_forecasts: Cube
    ) -> CubeList:
        """Create a cubelist for storing the coefficients computed using EMOS.

        .. See the documentation for examples of these cubes.
        .. include:: extended_documentation/calibration/
           ensemble_calibration/create_coefficients_cube.rst

        Args:
            optimised_coeffs:
                Array or list of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            historic_forecasts:
                Historic forecasts from the training dataset.

        Returns:
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

        return self._create_cubelist(optimised_coeffs, historic_forecasts)

    def compute_initial_guess(
        self,
        truths: ndarray,
        forecast_predictor: ndarray,
        predictor: str,
        number_of_realizations: Optional[int],
    ) -> List[float]:
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
            truths:
                Array containing the truth fields.
            forecast_predictor:
                Array containing the fields to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            predictor:
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.
            number_of_realizations:
                Number of realizations within the forecast predictor. If no
                realizations are present, this option is None.

        Returns:
            List of coefficients to be used as initial guess.
            Order of coefficients is [alpha, beta, gamma, delta].
        """
        default_initial_guess = (
            self.use_default_initial_guess
            or np.any(np.isnan(truths))
            or np.any(np.isnan(forecast_predictor))
        )

        if predictor.lower() == "mean" and default_initial_guess:
            initial_guess = [0, 1, 0, 1]
        elif predictor.lower() == "realizations" and (
            default_initial_guess or not statsmodels_available()
        ):
            initial_beta = np.repeat(
                np.sqrt(1.0 / number_of_realizations), number_of_realizations
            ).tolist()
            initial_guess = [0] + initial_beta + [0, 1]
        elif not self.use_default_initial_guess:
            truths_flattened = flatten_ignoring_masked_data(truths)
            if predictor.lower() == "mean":
                forecast_predictor_flattened = flatten_ignoring_masked_data(
                    forecast_predictor
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
                import statsmodels.api as sm

                forecast_predictor_flattened = flatten_ignoring_masked_data(
                    forecast_predictor, preserve_leading_dimension=True
                )
                val = sm.add_constant(forecast_predictor_flattened.T)
                est = sm.OLS(truths_flattened, val).fit()
                intercept = est.params[0]
                gradient = est.params[1:]
                initial_guess = [intercept] + gradient.tolist() + [0, 1]

        return np.array(initial_guess, dtype=np.float32)

    @staticmethod
    def mask_cube(cube: Cube, landsea_mask: Cube) -> None:
        """
        Mask the input cube using the given landsea_mask. Sea points are
        filled with nans and masked.

        Args:
            cube:
                A cube to be masked, on the same grid as the landsea_mask.
                The last two dimensions on this cube must match the dimensions
                in the landsea_mask cube.
            landsea_mask:
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

    def guess_and_minimise(
        self,
        truths: Cube,
        historic_forecasts: Cube,
        forecast_predictor: Cube,
        forecast_var: Cube,
        number_of_realizations: Optional[int],
    ) -> CubeList:
        """Function to consolidate calls to compute the initial guess, compute
        the optimised coefficients using minimisation and store the resulting
        coefficients within a CubeList.

        Args:
            truths:
                Truths from the training dataset.
            historic_forecasts:
                Historic forecasts from the training dataset.
            forecast_predictor:
                Predictor of the forecast within the minimisation. This
                is either the ensemble mean or the ensemble realizations.
            forecast_var:
                Variance of the forecast for use in the minimisation.
            number_of_realizations:
                Number of realizations within the forecast predictor. If no
                realizations are present, this option is None.

        Returns:
            CubeList constructed using the coefficients provided and using
            metadata from the historic_forecasts cube. Each cube within the
            cubelist is for a separate EMOS coefficient e.g. alpha, beta,
            gamma, delta.
        """
        if self.point_by_point and not self.use_default_initial_guess:
            index = [
                forecast_predictor.coord(axis="y"),
                forecast_predictor.coord(axis="x"),
            ]
            initial_guess = []
            for (truths_slice, fp_slice) in zip(
                truths.slices_over(index), forecast_predictor.slices_over(index)
            ):
                initial_guess.append(
                    self.compute_initial_guess(
                        truths_slice.data,
                        fp_slice.data,
                        self.predictor,
                        number_of_realizations,
                    )
                )
        else:
            # Computing initial guess for EMOS coefficients
            initial_guess = self.compute_initial_guess(
                truths.data,
                forecast_predictor.data,
                self.predictor,
                number_of_realizations,
            )
            if self.point_by_point:
                initial_guess = np.broadcast_to(
                    initial_guess,
                    (
                        len(truths.coord(axis="y").points)
                        * len(truths.coord(axis="x").points),
                        len(initial_guess),
                    ),
                )

        # Calculate coefficients if there are no nans in the initial guess.
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

    def process(
        self,
        historic_forecasts: Cube,
        truths: Cube,
        landsea_mask: Optional[Cube] = None,
    ) -> CubeList:
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
            historic_forecasts:
                Historic forecasts from the training dataset.
            truths:
                Truths from the training dataset.
            landsea_mask:
                The optional cube containing a land-sea mask. If provided, only
                land points are used to calculate the coefficients. Within the
                land-sea mask cube land points should be specified as ones,
                and sea points as zeros.

        Returns:
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
        if landsea_mask and self.point_by_point:
            msg = (
                "The use of a landsea mask with the option to compute "
                "coefficients independently at each point is not implemented."
            )
            raise NotImplementedError(msg)

        if not (historic_forecasts and truths):
            raise ValueError("historic_forecasts and truths cubes must be provided.")

        # Ensure predictor is valid.
        check_predictor(self.predictor)

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

        number_of_realizations = None
        if self.predictor.lower() == "mean":
            forecast_predictor = collapsed(
                historic_forecasts, "realization", iris.analysis.MEAN
            )
        elif self.predictor.lower() == "realizations":
            forecast_predictor = historic_forecasts
            number_of_realizations = len(forecast_predictor.coord("realization").points)
            enforce_coordinate_ordering(forecast_predictor, "realization")

        forecast_var = collapsed(
            historic_forecasts, "realization", iris.analysis.VARIANCE
        )

        # If a landsea_mask is provided mask out the sea points
        if landsea_mask:
            self.mask_cube(forecast_predictor, landsea_mask)
            self.mask_cube(forecast_var, landsea_mask)
            self.mask_cube(truths, landsea_mask)

        coefficients_cubelist = self.guess_and_minimise(
            truths,
            historic_forecasts,
            forecast_predictor,
            forecast_var,
            number_of_realizations,
        )

        return coefficients_cubelist


class CalibratedForecastDistributionParameters(BasePlugin):
    """
    Class to calculate calibrated forecast distribution parameters given an
    uncalibrated input forecast and EMOS coefficients.
    """

    def __init__(self, predictor: str = "mean") -> None:
        """
        Create a plugin that uses the coefficients created using EMOS from
        historical forecasts and corresponding truths and applies these
        coefficients to the current forecast to generate location and scale
        parameters that represent the calibrated distribution at each point.

        Args:
            predictor:
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.
        """
        check_predictor(predictor)
        self.predictor = predictor

        self.coefficients_cubelist = None
        self.current_forecast = None

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = "<CalibratedForecastDistributionParameters: predictor: {}>"
        return result.format(self.predictor)

    def _diagnostic_match(self) -> None:
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

    def _spatial_domain_match(self) -> None:
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

    def _calculate_location_parameter_from_mean(self) -> ndarray:
        """
        Function to calculate the location parameter when the ensemble mean at
        each grid point is the predictor.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Returns:
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

    def _calculate_location_parameter_from_realizations(self) -> ndarray:
        """
        Function to calculate the location parameter when the ensemble
        realizations are the predictor.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Returns:
            Location parameter calculated using the ensemble realizations
            as the predictor.
        """
        forecast_predictor = self.current_forecast
        # Calculate location parameter = a + b1*X1 .... + bn*Xn, where X is the
        # ensemble realizations. The number of b and X terms depends upon the
        # number of ensemble realizations. In this case, b = beta^2.
        beta_cube = self.coefficients_cubelist.extract_strict("emos_coefficient_beta")
        beta_values = np.atleast_2d(beta_cube.data * beta_cube.data)
        beta_values = beta_values.T if beta_cube.data.ndim != 1 else beta_values

        a_and_b = np.hstack(
            (
                np.atleast_2d(
                    self.coefficients_cubelist.extract_strict(
                        "emos_coefficient_alpha"
                    ).data
                ).T,
                beta_values,
            )
        )

        forecast_predictor_flat = convert_cube_data_to_2d(forecast_predictor)
        xy_shape = next(forecast_predictor.slices_over("realization")).shape
        col_of_ones = np.ones(np.prod(xy_shape), dtype=np.float32)
        ones_and_predictor = np.column_stack((col_of_ones, forecast_predictor_flat))

        location_parameter = (
            np.sum(ones_and_predictor * a_and_b, axis=-1)
            .reshape(xy_shape)
            .astype(np.float32)
        )

        return location_parameter

    def _calculate_scale_parameter(self) -> ndarray:
        """
        Calculation of the scale parameter using the ensemble variance
        adjusted using the gamma and delta coefficients calculated by EMOS.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Returns:
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
            * self.coefficients_cubelist.extract_strict("emos_coefficient_gamma").data
            + self.coefficients_cubelist.extract_strict("emos_coefficient_delta").data
            * self.coefficients_cubelist.extract_strict("emos_coefficient_delta").data
            * forecast_var.data
        ).astype(np.float32)
        return scale_parameter

    def _create_output_cubes(
        self, location_parameter: ndarray, scale_parameter: ndarray
    ) -> Tuple[Cube, Cube]:
        """
        Creation of output cubes containing the location and scale parameters.

        Args:
            location_parameter:
                Location parameter of the calibrated distribution.
            scale_parameter:
                Scale parameter of the calibrated distribution.

        Returns:
            - Location parameter of the calibrated distribution with
              associated metadata.
            - Scale parameter of the calibrated distribution with
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

    def process(
        self,
        current_forecast: Cube,
        coefficients_cubelist: CubeList,
        landsea_mask: Optional[Cube] = None,
    ) -> Tuple[Cube, Cube]:
        """
        Apply the EMOS coefficients to the current forecast, in order to
        generate location and scale parameters for creating the calibrated
        distribution.

        Args:
            current_forecast:
                The cube containing the current forecast.
            coefficients_cubelist:
                CubeList of EMOS coefficients where each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.
            landsea_mask:
                The optional cube containing a land-sea mask. If provided sea
                points will be masked in the output cube.
                This cube needs to have land points set to 1 and
                sea points to 0.

        Returns:
            - Cube containing the location parameter of the calibrated
              distribution calculated using either the ensemble mean or
              the ensemble realizations. The location parameter
              represents the point at which a resulting PDF would be
              centred.
            - Cube containing the scale parameter of the calibrated
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
    def _get_attribute(
        coefficients: CubeList, attribute_name: str, optional: bool = False
    ) -> Optional[Any]:
        """Get the value for the requested attribute, ensuring that the
        attribute is present consistently across the cubes within the
        coefficients cubelist.

        Args:
            coefficients:
                EMOS coefficients
            attribute_name:
                Name of expected attribute
            optional:
                Indicate whether the attribute is allowed to be optional.

        Returns:
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
    def _get_forecast_type(forecast: Cube) -> str:
        """Identifies whether the forecast is in probability, realization
        or percentile space

        Args:
            forecast

        Returns:
            The forecast type
        """
        try:
            find_percentile_coordinate(forecast)
        except CoordinateNotFoundError:
            if forecast.name().startswith("probability_of"):
                return "probabilities"
            return "realizations"
        return "percentiles"

    def _convert_to_realizations(
        self, forecast: Cube, realizations_count: int, ignore_ecc_bounds: bool
    ) -> Cube:
        """Convert an input forecast of probabilities or percentiles into
        pseudo-realizations

        Args:
            forecast
            realizations_count:
                Number of pseudo-realizations to generate from the input
                forecast
            ignore_ecc_bounds

        Returns:
            Cube with pseudo-realizations
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

    def _calibrate_forecast(
        self, forecast: Cube, randomise: bool, random_seed: int
    ) -> Cube:
        """
        Generate calibrated probability, percentile or realization output

        Args:
            forecast:
                Uncalibrated input forecast
            randomise:
                If True, order realization output randomly rather than using
                the input forecast.  If forecast type is not realizations, this
                is ignored.
            random_seed:
                For realizations input if randomise is True, random seed for
                generating re-ordered percentiles.  If randomise is False, the
                random seed may still be used for splitting ties.

        Returns:
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
        forecast: Cube,
        coefficients: CubeList,
        land_sea_mask: Optional[Cube] = None,
        realizations_count: Optional[int] = None,
        ignore_ecc_bounds: bool = True,
        predictor: str = "mean",
        randomise: bool = False,
        random_seed: Optional[int] = None,
    ) -> Cube:
        """Calibrate input forecast using pre-calculated coefficients

        Args:
            forecast:
                Uncalibrated forecast as probabilities, percentiles or
                realizations
            coefficients:
                EMOS coefficients
            land_sea_mask:
                Land sea mask where a value of "1" represents land points and
                "0" represents sea.  If set, allows calibration of land points
                only.
            realizations_count:
                Number of realizations to use when generating the intermediate
                calibrated forecast from probability or percentile inputs
            ignore_ecc_bounds:
                If True, allow percentiles from probabilities to exceed the ECC
                bounds range.  If input is not probabilities, this is ignored.
            predictor:
                Predictor to be used to calculate the location parameter of the
                calibrated distribution.  Value is "mean" or "realizations".
            randomise:
                Used in generating calibrated realizations.  If input forecast
                is probabilities or percentiles, this is ignored.
            random_seed:
                Used in generating calibrated realizations.  If input forecast
                is probabilities or percentiles, this is ignored.

        Returns:
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
