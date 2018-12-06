# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
This module defines the plugins required for Ensemble Copula Coupling.

"""
import warnings
import numpy as np
from scipy.stats import norm


import iris
from iris.exceptions import CoordinateNotFoundError, InvalidCubeError

from improver.ensemble_calibration.ensemble_calibration_utilities import (
    convert_cube_data_to_2d)
from improver.ensemble_copula_coupling.ensemble_copula_coupling_utilities \
    import (concatenate_2d_array_with_2d_array_endpoints,
            create_cube_with_percentiles, choose_set_of_percentiles,
            get_bounds_of_distribution,
            insert_lower_and_upper_endpoint_to_1d_array,
            restore_non_probabilistic_dimensions)
from improver.utilities.cube_manipulation import (
    concatenate_cubes, enforce_coordinate_ordering)
from improver.utilities.cube_checker import (
    find_percentile_coordinate, check_for_x_and_y_axes,
    check_cube_coordinates)


class RebadgePercentilesAsRealizations(object):
    """
    Class to rebadge percentiles as ensemble realizations.
    This will allow the quantisation to percentiles to be completed, without
    a subsequent EnsembleReordering step to restore spatial correlations,
    if required.
    """
    def __init__(self):
        """
        Initialise the class.
        """
        pass

    @staticmethod
    def process(cube, ensemble_realization_numbers=None):
        """
        Rebadge percentiles as ensemble realizations. The ensemble
        realization numbering will depend upon the number of percentiles in
        the input cube i.e. 0, 1, 2, 3, ..., n-1, if there are n percentiles.

        Args:
            cube (Iris.cube.Cube):
            Cube containing a percentile coordinate, which will be rebadged as
            ensemble realization.

        Raises:
            InvalidCubeError:
                If the realization coordinate already exists on the cube.
        """
        percentile_coord = (
            find_percentile_coordinate(cube).name())

        if ensemble_realization_numbers is None:
            ensemble_realization_numbers = (
                np.arange(
                    len(cube.coord(percentile_coord).points)))

        cube.coord(percentile_coord).points = (
            ensemble_realization_numbers)

        # we can't rebadge if the realization coordinate already exists:
        try:
            realization_coord = cube.coord('realization')
        except CoordinateNotFoundError:
            realization_coord = None

        if realization_coord:
            raise InvalidCubeError(
                "Cannot rebadge percentile coordinate to realization "
                "coordinate because a realization coordinate already exists.")

        cube.coord(percentile_coord).rename("realization")
        cube.coord("realization").units = "1"

        return cube


class ResamplePercentiles(object):
    """
    Class for resampling percentiles from an existing set of percentiles.
    In combination with the Ensemble Reordering plugin, this is a variant of
    Ensemble Copula Coupling.

    This class includes the ability to linearly interpolate from an
    input set of percentiles to a different output set of percentiles.

    """

    def __init__(self):
        """
        Initialise the class.
        """
        pass

    @staticmethod
    def _add_bounds_to_percentiles_and_forecast_at_percentiles(
            percentiles, forecast_at_percentiles, bounds_pairing):
        """
        Padding of the lower and upper bounds of the percentiles for a
        given phenomenon, and padding of forecast values using the
        constant lower and upper bounds.

        Args:
            percentiles (Numpy array):
                Array of percentiles from a Cumulative Distribution Function.
            forecast_at_percentiles (Numpy array):
                Array containing the underlying forecast values at each
                percentile.
            bounds_pairing (Tuple):
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.
        Returns:
            (tuple) : tuple containing:
                **percentiles** (Numpy array):
                    Array of percentiles from a Cumulative Distribution
                    Function.
                **forecast_at_percentiles** (Numpy array):
                    Array containing the underlying forecast values at each
                    percentile.
        """
        lower_bound, upper_bound = bounds_pairing
        percentiles = insert_lower_and_upper_endpoint_to_1d_array(
            percentiles, 0, 100)
        forecast_at_percentiles = concatenate_2d_array_with_2d_array_endpoints(
            forecast_at_percentiles, lower_bound, upper_bound)
        if np.any(np.diff(forecast_at_percentiles) < 0):
            msg = ("The end points added to the forecast at percentiles "
                   "values representing each percentile must result in "
                   "an ascending order. "
                   "In this case, the forecast at percentile values {} "
                   "is outside the allowable range given by the "
                   "bounds {}".format(
                       forecast_at_percentiles, bounds_pairing))
            raise ValueError(msg)
        if np.any(np.diff(percentiles) < 0):
            msg = ("The percentiles must be in ascending order."
                   "The input percentiles were {}".format(percentiles))
            raise ValueError(msg)
        return percentiles, forecast_at_percentiles

    def _interpolate_percentiles(
            self, forecast_at_percentiles, desired_percentiles,
            bounds_pairing, percentile_coord):
        """
        Interpolation of forecast for a set of percentiles from an initial
        set of percentiles to a new set of percentiles. This is constructed
        by linearly interpolating between the original set of percentiles
        to a new set of percentiles.

        Args:
            forecast_at_percentiles (Iris CubeList or Iris Cube):
                Cube or CubeList expected to contain a percentile coordinate.
            desired_percentiles (Numpy array):
                Array of the desired percentiles.
            bounds_pairing (Tuple):
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.
            percentile_coord (String):
                Name of required percentile coordinate.

        Returns:
            percentile_cube (iris cube.Cube):
                Cube containing values for the required diagnostic e.g.
                air_temperature at the required percentiles.

        """
        original_percentiles = (
            forecast_at_percentiles.coord(
                percentile_coord).points)

        # Ensure that the percentile dimension is first, so that the
        # conversion to a 2d array produces data in the desired order.
        forecast_at_percentiles = (
            enforce_coordinate_ordering(
                forecast_at_percentiles, percentile_coord))
        forecast_at_reshaped_percentiles = convert_cube_data_to_2d(
            forecast_at_percentiles, coord=percentile_coord)

        original_percentiles, forecast_at_reshaped_percentiles = (
            self._add_bounds_to_percentiles_and_forecast_at_percentiles(
                original_percentiles, forecast_at_reshaped_percentiles,
                bounds_pairing))

        forecast_at_interpolated_percentiles = (
            np.empty(
                (len(desired_percentiles),
                 forecast_at_reshaped_percentiles.shape[0]),
                dtype=np.float32
            )
        )
        for index in range(forecast_at_reshaped_percentiles.shape[0]):
            forecast_at_interpolated_percentiles[:, index] = np.interp(
                desired_percentiles, original_percentiles,
                forecast_at_reshaped_percentiles[index, :])

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        forecast_at_percentiles_data = (
            restore_non_probabilistic_dimensions(
                forecast_at_interpolated_percentiles, forecast_at_percentiles,
                percentile_coord, len(desired_percentiles)))

        for template_cube in forecast_at_percentiles.slices_over(
                percentile_coord):
            template_cube.remove_coord(percentile_coord)
            break
        percentile_cube = create_cube_with_percentiles(
            desired_percentiles, template_cube, forecast_at_percentiles_data,
            custom_name=percentile_coord)
        return percentile_cube

    def process(self, forecast_at_percentiles, no_of_percentiles=None,
                sampling="quantile"):
        """
        1. Creates a list of percentiles.
        2. Accesses the lower and upper bound pair of the forecast values,
           in order to specify lower and upper bounds for the percentiles.
        3. Interpolate the percentile coordinate into an alternative
           set of percentiles using linear interpolation.

        Args:
            forecast_at_percentiles (Iris Cube):
                Cube expected to contain a percentile coordinate.
            no_of_percentiles (Integer or None):
                Number of percentiles
                If None, the number of percentiles within the input
                forecast_at_percentiles cube is used as the
                number of percentiles.
            sampling (String):
                Type of sampling of the distribution to produce a set of
                percentiles e.g. quantile or random.

                Accepted options for sampling are:

                * Quantile: A regular set of equally-spaced percentiles aimed
                     at dividing a Cumulative Distribution Function into
                     blocks of equal probability.
                * Random: A random set of ordered percentiles.

        Returns:
            forecast_at_percentiles (iris.cube.Cube):
                Cube with forecast values at the desired set of percentiles.
                The percentile coordinate is always the zeroth dimension.

        """
        percentile_coord = (
            find_percentile_coordinate(forecast_at_percentiles).name())

        if no_of_percentiles is None:
            no_of_percentiles = (
                len(forecast_at_percentiles.coord(
                    percentile_coord).points))

        percentiles = choose_set_of_percentiles(
            no_of_percentiles, sampling=sampling)

        cube_units = forecast_at_percentiles.units
        bounds_pairing = (
            get_bounds_of_distribution(
                forecast_at_percentiles.name(), cube_units))

        forecast_at_percentiles = self._interpolate_percentiles(
            forecast_at_percentiles, percentiles, bounds_pairing,
            percentile_coord)
        return forecast_at_percentiles


class GeneratePercentilesFromProbabilities(object):
    """
    Class for generating percentiles from probabilities.
    In combination with the Ensemble Reordering plugin, this is a variant
    Ensemble Copula Coupling.

    This class includes the ability to interpolate between probabilities
    specified using multiple thresholds in order to generate the percentiles,
    see Figure 1 from Flowerdew, 2014.

    Scientific Reference:
    Flowerdew, J., 2014.
    Calibrated ensemble reliability whilst preserving spatial structure.
    Tellus Series A, Dynamic Meteorology and Oceanography, 66, 22662.

    """

    def __init__(self):
        """
        Initialise the class.
        """
        pass

    @staticmethod
    def _add_bounds_to_thresholds_and_probabilities(
            threshold_points, probabilities_for_cdf, bounds_pairing,
            ecc_bounds_warning=False):
        """
        Padding of the lower and upper bounds of the distribution for a
        given phenomenon for the threshold_points, and padding of
        probabilities of 0 and 1 to the forecast probabilities.

        Args:
            threshold_points (Numpy array):
                Array of threshold values used to calculate the probabilities.
            probabilities_for_cdf (Numpy array):
                Array containing the probabilities used for constructing an
                cumulative distribution function i.e. probabilities
                below threshold.
            bounds_pairing (Tuple):
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.

        Keyword Args:
            ecc_bounds_warning (bool):
                If true and ECC bounds are exceeded by the threshold values,
                a warning will be generated rather than an exception.
                Default value is FALSE.
        Returns:
            (tuple) : tuple containing:
                **threshold_points** (Numpy array):
                    Array of threshold values padded with the lower and upper
                    bound of the distribution.
                **probabilities_for_cdf** (Numpy array):
                    Array containing the probabilities padded with 0 and 1 at
                    each end.
        """
        lower_bound, upper_bound = bounds_pairing
        threshold_points_with_endpoints = \
            insert_lower_and_upper_endpoint_to_1d_array(
                threshold_points, lower_bound, upper_bound)
        probabilities_for_cdf = concatenate_2d_array_with_2d_array_endpoints(
            probabilities_for_cdf, 0, 1)

        if np.any(np.diff(threshold_points_with_endpoints) < 0):
            msg = ("The calculated threshold values {} are not in ascending "
                   "order as required for the cumulative distribution "
                   "function (CDF). This is due to the threshold values "
                   "exceeding the range given by the ECC bounds {}."
                   .format(threshold_points_with_endpoints, bounds_pairing))
            # If ecc_bounds_warning has been set, generate a warning message
            # rather than raising an exception so that subsequent processing
            # can continue. Then apply the new bounds as necessary to
            # ensure the threshold values and endpoints are in ascending
            # order and avoid problems further along the processing chain.
            if ecc_bounds_warning:
                warn_msg = msg + (" The threshold points that have "
                                  "exceeded the existing bounds will be used "
                                  "as new bounds.")
                warnings.warn(warn_msg)
                if upper_bound < max(threshold_points_with_endpoints):
                    upper_bound = max(threshold_points_with_endpoints)
                if lower_bound > min(threshold_points_with_endpoints):
                    lower_bound = min(threshold_points_with_endpoints)
                threshold_points_with_endpoints = \
                    insert_lower_and_upper_endpoint_to_1d_array(
                        threshold_points, lower_bound, upper_bound)
            else:
                raise ValueError(msg)
        return threshold_points_with_endpoints, probabilities_for_cdf

    def _probabilities_to_percentiles(
            self, forecast_probabilities, percentiles, bounds_pairing,
            ecc_bounds_warning=False):
        """
        Conversion of probabilities to percentiles through the construction
        of an cumulative distribution function. This is effectively
        constructed by linear interpolation from the probabilities associated
        with each threshold to a set of percentiles.

        Args:
            forecast_probabilities (Iris cube):
                Cube with a threshold coordinate.
            percentiles (Numpy array):
                Array of percentiles, at which the corresponding values will be
                calculated.
            bounds_pairing (Tuple):
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.

        Keyword Args:
            ecc_bounds_warning (bool):
                If true and ECC bounds are exceeded by the threshold values
                from the forecast_probabilities, then a warning will be
                generated rather than an exception. Default value is FALSE.

        Returns:
            percentile_cube (Iris cube):
                Cube containing values for the required diagnostic e.g.
                air_temperature at the required percentiles.

        """
        threshold_coord = forecast_probabilities.coord("threshold")
        threshold_unit = forecast_probabilities.coord("threshold").units
        threshold_points = threshold_coord.points

        # Ensure that the percentile dimension is first, so that the
        # conversion to a 2d array produces data in the desired order.
        forecast_probabilities = (
            enforce_coordinate_ordering(
                forecast_probabilities, threshold_coord.name()))
        prob_slices = convert_cube_data_to_2d(
            forecast_probabilities, coord=threshold_coord.name())

        # The requirement below for a monotonically changing probability
        # across thresholds can be thwarted by precision errors of order 1E-10,
        # as such, here we round to a precision of 9 decimal places.
        prob_slices = np.around(prob_slices, 9)

        # Invert probabilities for data thresholded above thresholds.
        relation = forecast_probabilities.attributes['relative_to_threshold']
        if relation == 'above':
            probabilities_for_cdf = 1 - prob_slices
        elif relation == 'below':
            probabilities_for_cdf = prob_slices
        else:
            msg = ("Probabilities to percentiles only implemented for "
                   "thresholds above or below a given value."
                   "The relation to threshold is given as {}".format(relation))
            raise NotImplementedError(msg)

        threshold_points, probabilities_for_cdf = (
            self._add_bounds_to_thresholds_and_probabilities(
                threshold_points, probabilities_for_cdf, bounds_pairing,
                ecc_bounds_warning))

        if np.any(np.diff(probabilities_for_cdf) < 0):
            msg = ("The probability values used to construct the "
                   "Cumulative Distribution Function (CDF) "
                   "must be ascending i.e. in order to yield "
                   "a monotonically increasing CDF."
                   "The probabilities are {}".format(probabilities_for_cdf))
            warnings.warn(msg)

        # Convert percentiles into fractions.
        percentiles = np.array([x/100.0 for x in percentiles],
                               dtype=np.float32)

        forecast_at_percentiles = (
            np.empty((len(percentiles), probabilities_for_cdf.shape[0]),
                     dtype=np.float32)
        )
        for index in range(probabilities_for_cdf.shape[0]):
            forecast_at_percentiles[:, index] = np.interp(
                percentiles, probabilities_for_cdf[index, :],
                threshold_points)

        # Convert percentiles back into percentages.
        percentiles = np.array([x*100.0 for x in percentiles],
                               dtype=np.float32)

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        forecast_at_percentiles = (
            restore_non_probabilistic_dimensions(
                forecast_at_percentiles, forecast_probabilities,
                threshold_coord.name(), len(percentiles)))

        for template_cube in forecast_probabilities.slices_over(
                threshold_coord.name()):
            template_cube.rename(
                template_cube.name().replace("probability_of_", ""))
            template_cube.remove_coord(threshold_coord.name())
            template_cube.attributes.pop('relative_to_threshold')
            break
        percentile_cube = create_cube_with_percentiles(
            percentiles, template_cube, forecast_at_percentiles,
            custom_name='percentile', cube_unit=threshold_unit)
        return percentile_cube

    def process(self, forecast_probabilities, no_of_percentiles=None,
                percentiles=None, sampling="quantile",
                ecc_bounds_warning=False):
        """
        1. Concatenates cubes with a threshold coordinate.
        2. Creates a list of percentiles.
        3. Accesses the lower and upper bound pair to find the ends of the
           cumulative distribution function.
        4. Convert the threshold coordinate into
           values at a set of percentiles using linear interpolation,
           see Figure 1 from Flowerdew, 2014.

        Args:
            forecast_probabilities (Iris CubeList or Iris Cube):
                Cube or CubeList expected to contain a threshold coordinate.
            no_of_percentiles (Integer or None):
                Number of percentiles. If None and percentiles is not set,
                the number of thresholds within the input
                forecast_probabilities cube is used as the number of
                percentiles. This argument is mutually exclusive with
                percentiles.
            percentiles (list of floats):
                The desired percentile values in the interval [0, 100].
                This argument is mutually exclusive with no_of_percentiles.
            sampling (String):
                Type of sampling of the distribution to produce a set of
                percentiles e.g. quantile or random.

                Accepted options for sampling are:

                * Quantile: A regular set of equally-spaced percentiles aimed
                          at dividing a Cumulative Distribution Function into
                          blocks of equal probability.
                * Random: A random set of ordered percentiles.

        Keyword Args:
            ecc_bounds_warning (bool):
                If True then exceeding ECC bounds will only generate a
                warning rather than an exception. Default value is FALSE.

        Returns:
            forecast_at_percentiles (Iris cube):
                Cube with forecast values at the desired set of percentiles.
                The threshold coordinate is always the zeroth dimension.

        """
        if no_of_percentiles is not None and percentiles is not None:
            raise ValueError(
                "Cannot specify both no_of_percentiles and percentiles to "
                "GeneratePercentilesFromProbabilities")

        forecast_probabilities = concatenate_cubes(
            forecast_probabilities,
            coords_to_slice_over="threshold",
            coordinates_for_association=[])

        threshold_coord = forecast_probabilities.coord("threshold")
        phenom_name = (
            forecast_probabilities.name().replace("probability_of_", ""))

        if no_of_percentiles is None:
            no_of_percentiles = (
                len(forecast_probabilities.coord(
                    threshold_coord.name()).points))

        if percentiles is None:
            percentiles = choose_set_of_percentiles(
                no_of_percentiles, sampling=sampling)
        elif not isinstance(percentiles, (tuple, list)):
            percentiles = [percentiles]
        percentiles = np.array(percentiles, dtype=np.float32)

        cube_units = (
            forecast_probabilities.coord(threshold_coord.name()).units)
        bounds_pairing = (
            get_bounds_of_distribution(
                phenom_name, cube_units))

        # If a cube still has multiple realizations, slice over these to reduce
        # the memory requirements into manageable chunks.
        try:
            slices_over_realization = forecast_probabilities.slices_over(
                "realization")
        except CoordinateNotFoundError:
            slices_over_realization = [forecast_probabilities]

        cubelist = iris.cube.CubeList([])
        for cube_realization in slices_over_realization:
            cubelist.append(self._probabilities_to_percentiles(
                cube_realization, percentiles, bounds_pairing,
                ecc_bounds_warning))

        forecast_at_percentiles = cubelist.merge_cube()
        return forecast_at_percentiles


class GeneratePercentilesFromMeanAndVariance(object):
    """
    Plugin focussing on generating percentiles from mean and variance.
    In combination with the EnsembleReordering plugin, this is Ensemble
    Copula Coupling.
    """

    def __init__(self):
        """
        Initialise the class.
        """
        pass

    @staticmethod
    def _mean_and_variance_to_percentiles(
            calibrated_forecast_predictor, calibrated_forecast_variance,
            percentiles):
        """
        Function returning percentiles based on the supplied
        mean and variance. The percentiles are created by assuming a
        Gaussian distribution and calculating the value of the phenomenon at
        specific points within the distribution.

        Args:
            calibrated_forecast_predictor (cube):
                Predictor for the calibrated forecast i.e. the mean.
            calibrated_forecast_variance (cube):
                Variance for the calibrated forecast.
            percentiles (List):
                Percentiles at which to calculate the value of the phenomenon
                at.

        Returns:
            percentile_cube (Iris cube):
                Cube containing the values for the phenomenon at each of the
                percentiles requested.

        """
        calibrated_forecast_predictor = (
            enforce_coordinate_ordering(
                calibrated_forecast_predictor, "realization"))
        calibrated_forecast_variance = (
            enforce_coordinate_ordering(
                calibrated_forecast_variance, "realization"))

        calibrated_forecast_predictor_data = (
            calibrated_forecast_predictor.data.flatten())
        calibrated_forecast_variance_data = (
            calibrated_forecast_variance.data.flatten())

        # Convert percentiles into fractions.
        percentiles = np.array(
            [x/100.0 for x in percentiles], dtype=np.float32)

        result = np.zeros((len(percentiles),
                           calibrated_forecast_predictor_data.shape[0]),
                          dtype=np.float32)

        # Loop over percentiles, and use a normal distribution with the mean
        # and variance to calculate the values at each percentile.
        for index, percentile in enumerate(percentiles):
            percentile_list = np.repeat(
                percentile, len(calibrated_forecast_predictor_data))
            result[index, :] = norm.ppf(
                percentile_list, loc=calibrated_forecast_predictor_data,
                scale=np.sqrt(calibrated_forecast_variance_data))
            # If percent point function (PPF) returns NaNs, fill in
            # mean instead of NaN values. NaN will only be generated if the
            # variance is zero. Therefore, if the variance is zero, the mean
            # value is used for all gridpoints with a NaN.
            if np.any(calibrated_forecast_variance_data == 0):
                nan_index = np.argwhere(np.isnan(result[index, :]))
                result[index, nan_index] = (
                    calibrated_forecast_predictor_data[nan_index])
            if np.any(np.isnan(result)):
                msg = ("NaNs are present within the result for the {} "
                       "percentile. Unable to calculate the percent point "
                       "function.")
                raise ValueError(msg)

        # Convert percentiles back into percentages.
        percentiles = [x*100.0 for x in percentiles]

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        result = (
            restore_non_probabilistic_dimensions(
                result, calibrated_forecast_predictor,
                "realization", len(percentiles)))

        for template_cube in calibrated_forecast_predictor.slices_over(
                "realization"):
            template_cube.remove_coord("realization")
            break
        percentile_cube = create_cube_with_percentiles(
            percentiles, template_cube, result)
        # Remove cell methods aimed at removing cell methods associated with
        # finding the ensemble mean, which are no longer relevant.
        percentile_cube.cell_methods = {}
        return percentile_cube

    def process(self, calibrated_forecast_predictor_and_variance,
                no_of_percentiles):
        """
        Generate ensemble percentiles from the mean and variance.

        Args:
            calibrated_forecast_predictor_and_variance (Iris CubeList):
                CubeList containing the calibrated forecast predictor and
                calibrated forecast variance.
            raw_forecast (Iris Cube or CubeList):
                Cube or CubeList that is expected to be the raw
                (uncalibrated) forecast.

        Returns:
            calibrated_forecast_percentiles (iris.cube.Cube):
                Cube for calibrated percentiles.
                The percentile coordinate is always the zeroth dimension.

        """
        (calibrated_forecast_predictor, calibrated_forecast_variance) = (
            calibrated_forecast_predictor_and_variance)
        if isinstance(calibrated_forecast_predictor, iris.cube.CubeList):
            calibrated_forecast_predictor = (
                calibrated_forecast_predictor.merge_cube())
        if isinstance(calibrated_forecast_variance, iris.cube.CubeList):
            calibrated_forecast_variance = (
                calibrated_forecast_variance.merge_cube())

        percentiles = choose_set_of_percentiles(no_of_percentiles)
        calibrated_forecast_percentiles = (
            self._mean_and_variance_to_percentiles(
                calibrated_forecast_predictor,
                calibrated_forecast_variance,
                percentiles))

        return calibrated_forecast_percentiles


class GenerateProbabilitiesFromMeanAndVariance(object):
    """
    Plugin to generate probabilities relative to given thresholds from the mean
    and variance of a distribution.
    """

    def __init__(self):
        """
        Initialise the class.
        """
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = '<GenerateProbabilitiesFromMeanAndVariance>'
        return desc

    @staticmethod
    def _check_template_cube(cube):
        """
        The template cube is expected to contain a leading threshold dimension
        followed by spatial (y/x) dimensions. This check raises an error if
        this is not the case. If the cube contains the expected dimensions,
        a threshold leading order is enforced.

        Args:
            cube (iris.cube.Cube):
                A cube whose dimensions are checked to ensure they match what
                is expected.
        Raises:
            ValueError: If cube is not of the expected dimensions.
        """
        check_for_x_and_y_axes(cube, require_dim_coords=True)
        dim_coords = [coord.name() for coord in cube.coords(dim_coords=True)]

        if 'threshold' in dim_coords and len(dim_coords) < 4:
            enforce_coordinate_ordering(cube, 'threshold')
            return

        msg = ('GenerateProbabilitiesFromMeanAndVariance expects a cube with '
               'only a leading threshold dimension, followed by spatial (y/x) '
               'dimensions. Got dimensions: {}'.format(dim_coords))
        raise ValueError(msg)

    @staticmethod
    def _check_unit_compatibility(mean_values, variance_values,
                                  probability_cube_template):
        """
        The mean, variance, and threshold values come from three different
        cubes. They should all be in the same units, but this is a sanity check
        to ensure this is the case, converting units of the means and variances
        if possible. This has been written specifically for this plugin as we
        are comparing squared units in the case of the variance.

        Args:
            mean_values (iris.cube.Cube):
                Cube of mean values.
            variance_values (iris.cube.Cube):
                Cube of variance values.
            probability_cube_template (iris.cube.Cube):
                Cube containing threshold values.
        Raises:
            ValueError: If units of input cubes are not compatible.
        """
        threshold_units = probability_cube_template.coord('threshold').units

        try:
            mean_values.convert_units(threshold_units)
            variance_values.convert_units(threshold_units**2)
        except ValueError as err:
            msg = ('Error: {} This is likely because the mean '
                   'variance and template cube threshold units are '
                   'not equivalent/compatible.'.format(err))
            raise ValueError(msg)

    @staticmethod
    def _mean_and_variance_to_probabilities(mean_values, variance_values,
                                            probability_cube_template):
        """
        Function returning probabilities relative to provided thresholds based
        on the supplied mean and variance. A Gaussian distribution is assumed.

        Args:
            mean_values (iris.cube.Cube):
                Predictor for the calibrated forecast i.e. the mean.
            variance_values (iris.cube.Cube):
                Variance for the calibrated forecast.
            probability_cube_template (iris.cube.Cube):
                A probability cube that has the threshold coordinate, and
                attribute relative_to_threshold, that match the desired output
                cube format.

        Returns:
            probability_cube (iris.cube.Cube):
                Cube containing the data expressed as probabilities relative to
                the provided thresholds in the way described by
                relative_to_threshold.
        """
        thresholds = probability_cube_template.coord('threshold').points
        relative_to_threshold = (
            probability_cube_template.attributes['relative_to_threshold'])

        # Loop over thresholds, and use a normal distribution with the mean
        # and variance to calculate the probabilties relative to each
        # threshold.
        probabilities = np.empty_like(probability_cube_template.data)
        distribution = norm(loc=mean_values.data,
                            scale=np.sqrt(variance_values.data))
        probability_method = distribution.cdf
        if relative_to_threshold == 'above':
            probability_method = distribution.sf

        for index, threshold in enumerate(thresholds):
            probabilities[index, ...] = probability_method(threshold)

        probability_cube = probability_cube_template.copy(data=probabilities)
        return probability_cube

    def process(self, mean_values, variance_values, probability_cube_template):
        """
        Generate probabilties from the mean and variance of distribution.

        Args:
            mean_values (iris.cube.Cube):
                Cube containing the distribution mean values of a diagnostic,
                e.g. the mean over realizations.
            variance_values (iris.cube.Cube):
                Cube containing the distribution variance values of a
                diagnostic, e.g. the variance across realizations.
            probability_cube_template (iris.cube.Cube):
                A probability cube that has the threshold coordinate, and
                attribute relative_to_threshold, that match the desired output
                cube format.

        Returns:
            probability_cube (iris.cube.Cube):
                A cube of diagnostic data expressed as probabilities relative
                to the thresholds found in the probability_cube_template.
        """
        self._check_template_cube(probability_cube_template)
        self._check_unit_compatibility(mean_values, variance_values,
                                       probability_cube_template)

        probability_cube = self._mean_and_variance_to_probabilities(
            mean_values, variance_values, probability_cube_template)

        return probability_cube


class EnsembleReordering(object):
    """
    Plugin for applying the reordering step of Ensemble Copula Coupling,
    in order to generate ensemble realizations with multivariate structure
    from percentiles. The percentiles are assumed to be in ascending order.

    Reference:
    Schefzik, R., Thorarinsdottir, T.L. & Gneiting, T., 2013.
    Uncertainty Quantification in Complex Simulation Models Using Ensemble
    Copula Coupling.
    Statistical Science, 28(4), pp.616-640.

    """
    def __init__(self):
        """Initialise the class"""
        pass

    @staticmethod
    def _recycle_raw_ensemble_realizations(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            percentile_coord):
        """
        Function to determine whether there is a mismatch between the number
        of percentiles and the number of raw forecast realizations. If more
        percentiles are requested than ensemble realizations, then the ensemble
        realizations are recycled. This assumes that the identity of the
        ensemble realizations within the raw ensemble forecast is random, such
        that the raw ensemble realizations are exchangeable. If fewer
        percentiles are requested than ensemble realizations, then only the
        first n ensemble realizations are used.

        Args:
            post_processed_forecast_percentiles  (iris.cube.Cube):
                Cube for post-processed percentiles.
                The percentiles are assumed
                to be in ascending order.
            raw_forecast_realizations (iris.cube.Cube):
                Cube containing the raw (not post-processed) forecasts.
            percentile_coord (String):
                Name of required percentile coordinate.

        Returns:
            raw_forecast_realizations (iris cube.Cube):
                Cube for the raw ensemble forecast, where the raw ensemble
                realizations have either been recycled or constrained,
                depending upon the number of percentiles present
                in the post-processed forecast cube.
        """
        plen = len(
            post_processed_forecast_percentiles.coord(
                percentile_coord).points)
        mlen = len(raw_forecast_realizations.coord("realization").points)
        if plen == mlen:
            pass
        else:
            raw_forecast_realizations_extended = iris.cube.CubeList()
            realization_list = []
            mpoints = raw_forecast_realizations.coord("realization").points
            # Loop over the number of percentiles and finding the
            # corresponding ensemble realization number. The ensemble
            # realization numbers are recycled e.g. 1, 2, 3, 1, 2, 3, etc.
            for index in range(plen):
                realization_list.append(mpoints[index % len(mpoints)])

            # Assume that the ensemble realizations are ascending linearly.
            new_realization_numbers = realization_list[0] + list(range(plen))

            # Extract the realizations required in the realization_list from
            # the raw_forecast_realizations. Edit the realization number as
            # appropriate and append to a cubelist containing rebadged
            # raw ensemble realizations.
            for realization, index in zip(
                    realization_list, new_realization_numbers):
                constr = iris.Constraint(realization=realization)
                raw_forecast_realization = raw_forecast_realizations.extract(
                    constr)
                raw_forecast_realization.coord("realization").points = index
                raw_forecast_realizations_extended.append(
                    raw_forecast_realization)
            raw_forecast_realizations = (
                concatenate_cubes(raw_forecast_realizations_extended))
        return raw_forecast_realizations

    @staticmethod
    def rank_ecc(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            random_ordering=False, random_seed=None):
        """
        Function to apply Ensemble Copula Coupling. This ranks the
        post-processed forecast realizations based on a ranking determined from
        the raw forecast realizations.

        Args:
            post_processed_forecast_percentiles (cube):
                Cube for post-processed percentiles. The percentiles are
                assumed to be in ascending order.
            raw_forecast_realizations (cube):
                Cube containing the raw (not post-processed) forecasts.
                The probabilistic dimension is assumed to be the zeroth
                dimension.
            random_ordering (Logical):
                If random_ordering is True, the post-processed forecasts are
                reordered randomly, rather than using the ordering of the
                raw ensemble.
            random_seed (Integer or None):
                If random_seed is an integer, the integer value is used for
                the random seed.
                If random_seed is None, no random seed is set, so the random
                values generated are not reproducible.

        Returns:
            iris.cube.Cube:
                Cube for post-processed realizations where at a particular grid
                point, the ranking of the values within the ensemble matches
                the ranking from the raw ensemble.

        """
        results = iris.cube.CubeList([])
        for rawfc, calfc in zip(
                raw_forecast_realizations.slices_over("time"),
                post_processed_forecast_percentiles.slices_over("time")):
            if random_seed is not None:
                random_seed = int(random_seed)
            random_seed = np.random.RandomState(random_seed)
            random_data = random_seed.rand(*rawfc.data.shape)
            if random_ordering:
                # Returns the indices that would sort the array.
                # As these indices are from a random dataset, only an argsort
                # is used.
                ranking = np.argsort(random_data, axis=0)
            else:
                # Lexsort returns the indices sorted firstly by the
                # primary key, the raw forecast data (unless random_ordering
                # is enabled), and secondly by the secondary key, an array of
                # random data, in order to split tied values randomly.
                sorting_index = np.lexsort((random_data, rawfc.data), axis=0)
                # Returns the indices that would sort the array.
                ranking = np.argsort(sorting_index, axis=0)
            # Index the post-processed forecast data using the ranking array.
            # np.choose allows indexing of a 3d array using a 3d array,
            calfc.data = np.choose(ranking, calfc.data)
            results.append(calfc)
        # Ensure we haven't lost any dimensional coordinates with only one
        # value in.
        results = results.merge_cube()
        results = check_cube_coordinates(
            post_processed_forecast_percentiles, results)
        return results

    def process(
            self, post_processed_forecast, raw_forecast,
            random_ordering=False, random_seed=None):
        """
        Reorder post-processed forecast using the ordering of the
        raw ensemble.

        Args:
            post_processed_forecast (Iris Cube or CubeList):
                The cube or cubelist containing the post-processed
                forecast realizations.
            raw_forecast (Iris Cube or CubeList):
                The cube or cubelist containing the raw (not post-processed)
                forecast.
            random_ordering (Logical):
                If random_ordering is True, the post-processed forecasts are
                reordered randomly, rather than using the ordering of the
                raw ensemble.
            random_seed (Integer or None):
                If random_seed is an integer, the integer value is used for
                the random seed.
                If random_seed is None, no random seed is set, so the random
                values generated are not reproducible.

        Returns:
            post-processed_forecast_realizations (cube):
                Cube containing the new ensemble realizations where all points
                within the dataset have been reordered in comparison to the
                input percentiles.
        """
        if isinstance(post_processed_forecast, iris.cube.CubeList):
            percentile_coord = (
                find_percentile_coordinate(post_processed_forecast[0]).name())
        else:
            percentile_coord = (
                find_percentile_coordinate(post_processed_forecast).name())

        post_processed_forecast_percentiles = concatenate_cubes(
            post_processed_forecast,
            coords_to_slice_over=[percentile_coord])
        post_processed_forecast_percentiles = (
            enforce_coordinate_ordering(
                post_processed_forecast_percentiles, percentile_coord))
        raw_forecast_realizations = concatenate_cubes(raw_forecast)
        raw_forecast_realizations = enforce_coordinate_ordering(
            raw_forecast_realizations, "realization")
        raw_forecast_realizations = (
            self._recycle_raw_ensemble_realizations(
                post_processed_forecast_percentiles, raw_forecast_realizations,
                percentile_coord))
        post_processed_forecast_realizations = self.rank_ecc(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            random_ordering=random_ordering,
            random_seed=random_seed)
        post_processed_forecast_realizations = (
            RebadgePercentilesAsRealizations.process(
                post_processed_forecast_realizations))

        post_processed_forecast_realizations = (
            enforce_coordinate_ordering(
                post_processed_forecast_realizations, "realization"))
        return post_processed_forecast_realizations
