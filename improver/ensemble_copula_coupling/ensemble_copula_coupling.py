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
This module defines the plugins required for Ensemble Copula Coupling.

"""
import warnings

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError, InvalidCubeError
from scipy import stats

from improver import BasePlugin
from improver.calibration.utilities import convert_cube_data_to_2d
from improver.ensemble_copula_coupling.utilities import (
    choose_set_of_percentiles, concatenate_2d_array_with_2d_array_endpoints,
    create_cube_with_percentiles, get_bounds_of_distribution,
    insert_lower_and_upper_endpoint_to_1d_array,
    restore_non_probabilistic_dimensions)
from improver.metadata.probabilistic import (
    find_percentile_coordinate, find_threshold_coordinate)
from improver.utilities.cube_checker import (
    check_cube_coordinates, check_for_x_and_y_axes)
from improver.utilities.cube_manipulation import (
    concatenate_cubes, enforce_coordinate_ordering, get_dim_coord_names)
from improver.utilities.indexing_operations import choose


class RebadgePercentilesAsRealizations(BasePlugin):
    """
    Class to rebadge percentiles as ensemble realizations.
    This will allow the quantisation to percentiles to be completed, without
    a subsequent EnsembleReordering step to restore spatial correlations,
    if required.
    """

    @staticmethod
    def process(cube, ensemble_realization_numbers=None):
        """
        Rebadge percentiles as ensemble realizations. The ensemble
        realization numbering will depend upon the number of percentiles in
        the input cube i.e. 0, 1, 2, 3, ..., n-1, if there are n percentiles.

        Args:
            cube (iris.cube.Cube):
                Cube containing a percentile coordinate, which will be
                rebadged as ensemble realization.
            ensemble_realization_numbers (numpy.ndarray):
                An array containing the ensemble numbers required in the output
                realization coordinate. Default is None, meaning the
                realization coordinate will be numbered 0, 1, 2 ... n-1 for n
                percentiles on the input cube.
        Raises:
            InvalidCubeError:
                If the realization coordinate already exists on the cube.
        """
        percentile_coord_name = (
            find_percentile_coordinate(cube).name())

        if ensemble_realization_numbers is None:
            ensemble_realization_numbers = (
                np.arange(
                    len(cube.coord(percentile_coord_name).points),
                    dtype=np.int32))

        cube.coord(percentile_coord_name).points = (
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

        cube.coord(percentile_coord_name).rename("realization")
        cube.coord("realization").units = "1"
        cube.coord("realization").points = (
            cube.coord("realization").points.astype(np.int32))
        cube.coord("realization").var_name = "realization"

        return cube


class ResamplePercentiles(BasePlugin):
    """
    Class for resampling percentiles from an existing set of percentiles.
    In combination with the Ensemble Reordering plugin, this is a variant of
    Ensemble Copula Coupling.

    This class includes the ability to linearly interpolate from an
    input set of percentiles to a different output set of percentiles.

    """

    def __init__(self, ecc_bounds_warning=False):
        """
        Initialise the class.

        Args:
            ecc_bounds_warning (bool):
                If true and ECC bounds are exceeded by the percentile values,
                a warning will be generated rather than an exception.
                Default value is FALSE.
        """
        self.ecc_bounds_warning = ecc_bounds_warning

    def _add_bounds_to_percentiles_and_forecast_at_percentiles(
            self, percentiles, forecast_at_percentiles, bounds_pairing):
        """
        Padding of the lower and upper bounds of the percentiles for a
        given phenomenon, and padding of forecast values using the
        constant lower and upper bounds.

        Args:
            percentiles (numpy.ndarray):
                Array of percentiles from a Cumulative Distribution Function.
            forecast_at_percentiles (numpy.ndarray):
                Array containing the underlying forecast values at each
                percentile.
            bounds_pairing (tuple):
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.

        Raises:
            ValueError: If the percentile points are outside the ECC bounds
                and self.ecc_bounds_warning is False.
            ValueError: If the percentiles are not in ascending order.

        Warns:
            Warning:  If the percentile points are outside the ECC bounds
                and self.ecc_bounds_warning is True.
        """
        lower_bound, upper_bound = bounds_pairing
        percentiles = insert_lower_and_upper_endpoint_to_1d_array(
            percentiles, 0, 100)
        forecast_at_percentiles_with_endpoints = \
            concatenate_2d_array_with_2d_array_endpoints(
                forecast_at_percentiles, lower_bound, upper_bound)
        if np.any(np.diff(forecast_at_percentiles_with_endpoints) < 0):
            msg = ("The end points added to the forecast at percentile "
                   "values representing each percentile must result in "
                   "an ascending order. "
                   "In this case, the forecast at percentile values {} "
                   "is outside the allowable range given by the "
                   "bounds {}".format(forecast_at_percentiles, bounds_pairing))

            if self.ecc_bounds_warning:
                warn_msg = msg + (" The percentile values that have "
                                  "exceeded the existing bounds will be used "
                                  "as new bounds.")
                warnings.warn(warn_msg)
                if upper_bound < forecast_at_percentiles_with_endpoints.max():
                    upper_bound = forecast_at_percentiles_with_endpoints.max()
                if lower_bound > forecast_at_percentiles_with_endpoints.min():
                    lower_bound = forecast_at_percentiles_with_endpoints.min()
                forecast_at_percentiles_with_endpoints = \
                    concatenate_2d_array_with_2d_array_endpoints(
                        forecast_at_percentiles, lower_bound, upper_bound)
            else:
                raise ValueError(msg)
        if np.any(np.diff(percentiles) < 0):
            msg = ("The percentiles must be in ascending order."
                   "The input percentiles were {}".format(percentiles))
            raise ValueError(msg)
        return percentiles, forecast_at_percentiles_with_endpoints

    def _interpolate_percentiles(
            self, forecast_at_percentiles, desired_percentiles,
            bounds_pairing, percentile_coord_name):
        """
        Interpolation of forecast for a set of percentiles from an initial
        set of percentiles to a new set of percentiles. This is constructed
        by linearly interpolating between the original set of percentiles
        to a new set of percentiles.

        Args:
            forecast_at_percentiles (iris.cube.Cube):
                Cube containing a percentile coordinate.
            desired_percentiles (numpy.ndarray):
                Array of the desired percentiles.
            bounds_pairing (tuple):
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.
            percentile_coord_name (str):
                Name of required percentile coordinate.
        Returns:
            iris.cube.Cube:
                Cube containing values for the required diagnostic e.g.
                air_temperature at the required percentiles.

        """
        original_percentiles = (
            forecast_at_percentiles.coord(percentile_coord_name).points)

        # Ensure that the percentile dimension is first, so that the
        # conversion to a 2d array produces data in the desired order.
        enforce_coordinate_ordering(
            forecast_at_percentiles, percentile_coord_name)
        forecast_at_reshaped_percentiles = convert_cube_data_to_2d(
            forecast_at_percentiles, coord=percentile_coord_name)

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
                percentile_coord_name, len(desired_percentiles)))

        template_cube = next(forecast_at_percentiles.slices_over(
            percentile_coord_name))
        template_cube.remove_coord(percentile_coord_name)
        percentile_cube = create_cube_with_percentiles(
            desired_percentiles, template_cube, forecast_at_percentiles_data,)
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
            forecast_at_percentiles (iris.cube.Cube):
                Cube expected to contain a percentile coordinate.
            no_of_percentiles (int or None):
                Number of percentiles
                If None, the number of percentiles within the input
                forecast_at_percentiles cube is used as the
                number of percentiles.
            sampling (str):
                Type of sampling of the distribution to produce a set of
                percentiles e.g. quantile or random.

                Accepted options for sampling are:

                * Quantile: A regular set of equally-spaced percentiles aimed
                     at dividing a Cumulative Distribution Function into
                     blocks of equal probability.
                * Random: A random set of ordered percentiles.
        Returns:
            iris.cube.Cube:
                Cube with forecast values at the desired set of percentiles.
                The percentile coordinate is always the zeroth dimension.

        """
        percentile_coord = find_percentile_coordinate(forecast_at_percentiles)

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
            percentile_coord.name())
        return forecast_at_percentiles


class ConvertProbabilitiesToPercentiles(BasePlugin):
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

    def __init__(self, ecc_bounds_warning=False):
        """
        Initialise the class.

        Args:
            ecc_bounds_warning (bool):
                If true and ECC bounds are exceeded by the percentile values,
                a warning will be generated rather than an exception.
                Default value is FALSE.
        """
        self.ecc_bounds_warning = ecc_bounds_warning

    def _add_bounds_to_thresholds_and_probabilities(
            self, threshold_points, probabilities_for_cdf, bounds_pairing):
        """
        Padding of the lower and upper bounds of the distribution for a
        given phenomenon for the threshold_points, and padding of
        probabilities of 0 and 1 to the forecast probabilities.

        Args:
            threshold_points (numpy.ndarray):
                Array of threshold values used to calculate the probabilities.
            probabilities_for_cdf (numpy.ndarray):
                Array containing the probabilities used for constructing an
                cumulative distribution function i.e. probabilities
                below threshold.
            bounds_pairing (tuple):
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.
        Returns:
            (tuple): tuple containing:
                **threshold_points** (numpy.ndarray):
                    Array of threshold values padded with the lower and upper
                    bound of the distribution.
                **probabilities_for_cdf** (numpy.ndarray):
                    Array containing the probabilities padded with 0 and 1 at
                    each end.

        Raises:
            ValueError: If the thresholds exceed the ECC bounds for
                the diagnostic and self.ecc_bounds_warning is False.

        Warns:
            Warning: If the thresholds exceed the ECC bounds for
                the diagnostic and self.ecc_bounds_warning is True.
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
            if self.ecc_bounds_warning:
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
            self, forecast_probabilities, percentiles, bounds_pairing):
        """
        Conversion of probabilities to percentiles through the construction
        of an cumulative distribution function. This is effectively
        constructed by linear interpolation from the probabilities associated
        with each threshold to a set of percentiles.

        Args:
            forecast_probabilities (iris.cube.Cube):
                Cube with a threshold coordinate.
            percentiles (numpy.ndarray):
                Array of percentiles, at which the corresponding values will be
                calculated.
            bounds_pairing (tuple):
                Lower and upper bound to be used as the ends of the
                cumulative distribution function.
        Returns:
            iris.cube.Cube:
                Cube containing values for the required diagnostic e.g.
                air_temperature at the required percentiles.
        Raises:
            NotImplementedError: If the threshold coordinate has an
                spp__relative_to_threshold attribute that is not either
                "above" or "below".
        Warns:
            Warning: If the probability values are not ascending, so the
                resulting cdf is not monotonically increasing.
        """
        threshold_coord = find_threshold_coordinate(forecast_probabilities)
        threshold_unit = threshold_coord.units
        threshold_points = threshold_coord.points

        # Ensure that the percentile dimension is first, so that the
        # conversion to a 2d array produces data in the desired order.
        enforce_coordinate_ordering(
            forecast_probabilities, threshold_coord.name())
        prob_slices = convert_cube_data_to_2d(
            forecast_probabilities, coord=threshold_coord.name())

        # The requirement below for a monotonically changing probability
        # across thresholds can be thwarted by precision errors of order 1E-10,
        # as such, here we round to a precision of 9 decimal places.
        prob_slices = np.around(prob_slices, 9)

        # Invert probabilities for data thresholded above thresholds.
        relation = find_threshold_coordinate(
            forecast_probabilities).attributes['spp__relative_to_threshold']
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
                threshold_points, probabilities_for_cdf, bounds_pairing))

        if np.any(np.diff(probabilities_for_cdf) < 0):
            msg = ("The probability values used to construct the "
                   "Cumulative Distribution Function (CDF) "
                   "must be ascending i.e. in order to yield "
                   "a monotonically increasing CDF."
                   "The probabilities are {}".format(probabilities_for_cdf))
            warnings.warn(msg)

        # Convert percentiles into fractions.
        percentiles_as_fractions = np.array(
            [x/100.0 for x in percentiles], dtype=np.float32)

        forecast_at_percentiles = (
            np.empty((len(percentiles),
                      probabilities_for_cdf.shape[0]), dtype=np.float32)) \
            # pylint: disable=unsubscriptable-object
        for index in range(probabilities_for_cdf.shape[0]): \
                # pylint: disable=unsubscriptable-object
            forecast_at_percentiles[:, index] = np.interp(
                percentiles_as_fractions, probabilities_for_cdf[index, :],
                threshold_points)

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        forecast_at_percentiles = (
            restore_non_probabilistic_dimensions(
                forecast_at_percentiles, forecast_probabilities,
                threshold_coord.name(), len(percentiles)))

        template_cube = next(forecast_probabilities.slices_over(
            threshold_coord.name()))
        template_cube.rename(
            template_cube.name().replace("probability_of_", ""))
        template_cube.rename(
            template_cube.name().replace(
                "_above_threshold", "").replace("_below_threshold", ""))
        template_cube.remove_coord(threshold_coord.name())

        percentile_cube = create_cube_with_percentiles(
            percentiles, template_cube, forecast_at_percentiles,
            cube_unit=threshold_unit)
        return percentile_cube

    def process(self, forecast_probabilities, no_of_percentiles=None,
                percentiles=None, sampling="quantile"):
        """
        1. Concatenates cubes with a threshold coordinate.
        2. Creates a list of percentiles.
        3. Accesses the lower and upper bound pair to find the ends of the
           cumulative distribution function.
        4. Convert the threshold coordinate into
           values at a set of percentiles using linear interpolation,
           see Figure 1 from Flowerdew, 2014.

        Args:
            forecast_probabilities (iris.cube.Cube):
                Cube containing a threshold coordinate.
            no_of_percentiles (int):
                Number of percentiles. If None and percentiles is not set,
                the number of thresholds within the input
                forecast_probabilities cube is used as the number of
                percentiles. This argument is mutually exclusive with
                percentiles.
            percentiles (list of float):
                The desired percentile values in the interval [0, 100].
                This argument is mutually exclusive with no_of_percentiles.
            sampling (str):
                Type of sampling of the distribution to produce a set of
                percentiles e.g. quantile or random.

                Accepted options for sampling are:

                * Quantile: A regular set of equally-spaced percentiles aimed
                          at dividing a Cumulative Distribution Function into
                          blocks of equal probability.
                * Random: A random set of ordered percentiles.

        Returns:
            iris.cube.Cube:
                Cube with forecast values at the desired set of percentiles.
                The threshold coordinate is always the zeroth dimension.

        Raises:
            ValueError: If both no_of_percentiles and percentiles are provided
        """
        if no_of_percentiles is not None and percentiles is not None:
            raise ValueError(
                "Cannot specify both no_of_percentiles and percentiles to "
                "{}".format(self.__class__.__name__))

        threshold_coord = find_threshold_coordinate(forecast_probabilities)

        phenom_name = (
            forecast_probabilities.name().replace(
                "probability_of_", "").replace("_above_threshold", "").replace(
                    "_below_threshold", ""))

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
                cube_realization, percentiles, bounds_pairing))
        forecast_at_percentiles = cubelist.merge_cube()
        return forecast_at_percentiles


class ConvertLocationAndScaleParameters():
    """
    Base Class to support the plugins that compute percentiles and
    probabilities from the location and scale parameters.
    """

    def __init__(self, distribution="norm", shape_parameters=None):
        """
        Initialise the class.

        In order to construct percentiles or probabilities from the location
        or scale parameter, the distribution for the resulting output needs
        to be selected. For use with the outputs from EMOS, where it has been
        assumed that the outputs from minimising the CRPS follow a particular
        distribution, then the same distribution should be selected, as used
        for the CRPS minimisation. The conversion to percentiles and
        probabilities from the location and scale parameter relies upon
        functionality within scipy.stats.

        Args:
            distribution (str):
                Name of a distribution supported by scipy.stats.
            shape_parameters (list or None):
                For use with distributions in scipy.stats (e.g. truncnorm) that
                require the specification of shape parameters to be able to
                define the shape of the distribution. For the truncated normal
                distribution, the shape parameters should be appropriate for
                the distribution constructed from the location and scale
                parameters provided.
                Please note that for use with
                :meth:`~improver.calibration.\
ensemble_calibration.ContinuousRankedProbabilityScoreMinimisers.\
calculate_truncated_normal_crps`,
                the shape parameters for a truncated normal distribution with
                a lower bound of zero should be [0, np.inf].

        """
        try:
            self.distribution = getattr(stats, distribution)
        except AttributeError as err:
            msg = ("The distribution requested {} is not a valid distribution "
                   "in scipy.stats. {}".format(distribution, err))
            raise AttributeError(msg)

        if shape_parameters is None:
            shape_parameters = []
        self.shape_parameters = shape_parameters

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ConvertLocationAndScaleParameters: distribution: {}; '
                  'shape_parameters: {}>')
        return result.format(
            self.distribution.name, self.shape_parameters)

    def _rescale_shape_parameters(self, location_parameter, scale_parameter):
        """
        Rescale the shape parameters for the desired location and scale
        parameters for the truncated normal distribution. The shape parameters
        for any other distribution will remain unchanged.

        For the truncated normal distribution, if the shape parameters are not
        rescaled, then :data:`scipy.stats.truncnorm` will assume that the shape
        parameters are appropriate for a standard normal distribution. As the
        aim is to construct a distribution using specific values for the
        location and scale parameters, the assumption of a standard normal
        distribution is not appropriate. Therefore the shape parameters are
        rescaled using the equations:

        .. math::
          a\\_rescaled = (a - location\\_parameter)/scale\\_parameter

          b\\_rescaled = (b - location\\_parameter)/scale\\_parameter

        Please see :data:`scipy.stats.truncnorm` for some further information.

        Args:
            location_parameter (numpy.ndarray):
                Location parameter to be used to scale the shape parameters.
            scale_parameter (numpy.ndarray):
                Scale parameter to be used to scale the shape parameters.

        """
        if self.distribution.name == "truncnorm":
            if self.shape_parameters:
                rescaled_values = []
                for value in self.shape_parameters:
                    rescaled_values.append((value - location_parameter) /
                                           scale_parameter)
                self.shape_parameters = rescaled_values
            else:
                msg = ("For the truncated normal distribution, "
                       "shape parameters must be specified.")
                raise ValueError(msg)


class ConvertLocationAndScaleParametersToPercentiles(
        BasePlugin, ConvertLocationAndScaleParameters):
    """
    Plugin focusing on generating percentiles from location and scale
    parameters. In combination with the EnsembleReordering plugin, this is
    Ensemble Copula Coupling.
    """

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ConvertLocationAndScaleParametersToPercentiles: '
                  'distribution: {}; shape_parameters: {}>')
        return result.format(self.distribution.name, self.shape_parameters)

    def _location_and_scale_parameters_to_percentiles(
            self, location_parameter, scale_parameter, template_cube,
            percentiles):
        """
        Function returning percentiles based on the supplied location and
        scale parameters.

        Args:
            location_parameter (iris.cube.Cube):
                Location parameter of calibrated distribution.
            scale_parameter (iris.cube.Cube):
                Scale parameter of the calibrated distribution.
            template_cube (iris.cube.Cube):
                Template cube containing either a percentile or realization
                coordinate. All coordinates apart from the percentile or
                realization coordinate will be copied from the template cube.
                Metadata will also be copied from this cube.
            percentiles (list):
                Percentiles at which to calculate the value of the phenomenon
                at.

        Returns:
            iris.cube.Cube:
                Cube containing the values for the phenomenon at each of the
                percentiles requested.

        Raises:
            ValueError: If any of the resulting percentile values are
                nans and these nans are not caused by a scale parameter of
                zero.
        """
        # Remove any mask that may be applied to location and scale parameters
        # and replace with ones
        location_data = np.ma.filled(location_parameter.data, 1).flatten()
        scale_data = np.ma.filled(scale_parameter.data, 1).flatten()

        # Convert percentiles into fractions.
        percentiles = np.array(
            [x/100.0 for x in percentiles], dtype=np.float32)

        result = np.zeros((len(percentiles),
                           location_data.shape[0]), dtype=np.float32)

        self._rescale_shape_parameters(location_data, np.sqrt(scale_data))

        percentile_method = self.distribution(
            *self.shape_parameters, loc=location_data,
            scale=np.sqrt(scale_data))

        # Loop over percentiles, and use the distribution as the
        # "percentile_method" with the location and scale parameter to
        # calculate the values at each percentile.
        for index, percentile in enumerate(percentiles):
            percentile_list = np.repeat(percentile, len(location_data))
            result[index, :] = percentile_method.ppf(percentile_list)
            # If percent point function (PPF) returns NaNs, fill in
            # mean instead of NaN values. NaN will only be generated if the
            # variance is zero. Therefore, if the variance is zero, the mean
            # value is used for all gridpoints with a NaN.
            if np.any(scale_data == 0):
                nan_index = np.argwhere(np.isnan(result[index, :]))
                result[index, nan_index] = (location_data[nan_index])
            if np.any(np.isnan(result)):
                msg = ("NaNs are present within the result for the {} "
                       "percentile. Unable to calculate the percent point "
                       "function.")
                raise ValueError(msg)

        # Convert percentiles back into percentages.
        percentiles = [x*100.0 for x in percentiles]

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        result = result.reshape(
            (len(percentiles),) + location_parameter.data.shape)

        for prob_coord_name in ["realization", "percentile"]:
            if template_cube.coords(prob_coord_name, dim_coords=True):
                prob_coord = template_cube.coord(prob_coord_name)
                template_slice = next(template_cube.slices_over(prob_coord))
                template_slice.remove_coord(prob_coord)

        percentile_cube = create_cube_with_percentiles(
            percentiles, template_slice, result)
        # Define a mask to be reapplied later
        mask = np.logical_or(np.ma.getmaskarray(location_parameter.data),
                             np.ma.getmaskarray(scale_parameter.data))
        # Make the mask defined above fit the data size and then apply to the
        # percentile cube.
        mask_array = np.stack([mask]*len(percentiles))
        percentile_cube.data = np.ma.masked_where(
            mask_array, percentile_cube.data)
        # Remove cell methods associated with finding the ensemble mean
        percentile_cube.cell_methods = {}
        return percentile_cube

    def process(self, location_parameter, scale_parameter, template_cube,
                no_of_percentiles=None, percentiles=None):
        """
        Generate ensemble percentiles from the location and scale parameters.

        Args:
            location_parameter (iris.cube.Cube):
                Cube containing the location parameters.
            scale_parameter (iris.cube.Cube):
                Cube containing the scale parameters.
            template_cube (iris.cube.Cube):
                Template cube containing either a percentile or realization
                coordinate. All coordinates apart from the percentile or
                realization coordinate will be copied from the template cube.
                Metadata will also be copied from this cube.
            no_of_percentiles (int):
                Integer defining the number of percentiles that will be
                calculated from the location and scale parameters.
            percentiles (list):
                List of percentiles that will be generated from the location
                and scale parameters provided.

        Returns:
            iris.cube.Cube:
                Cube for calibrated percentiles.
                The percentile coordinate is always the zeroth dimension.

        Raises:
            ValueError: Ensure that it is not possible to supply
                "no_of_percentiles" and "percentiles" simultaneously
                as keyword arguments.

        """
        if no_of_percentiles and percentiles:
            msg = ("Please specify either the number of percentiles or "
                   "provide a list of percentiles. The number of percentiles "
                   "provided was {} and the list of percentiles "
                   "provided was {}".format(no_of_percentiles, percentiles))
            raise ValueError(msg)

        if no_of_percentiles:
            percentiles = choose_set_of_percentiles(no_of_percentiles)
        calibrated_forecast_percentiles = (
            self._location_and_scale_parameters_to_percentiles(
                location_parameter, scale_parameter, template_cube,
                percentiles))

        return calibrated_forecast_percentiles


class ConvertLocationAndScaleParametersToProbabilities(
        BasePlugin, ConvertLocationAndScaleParameters):
    """
    Plugin to generate probabilities relative to given thresholds from the
    location and scale parameters of a distribution.
    """

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ConvertLocationAndScaleParametersToProbabilities: '
                  'distribution: {}; shape_parameters: {}>')
        return result.format(self.distribution.name, self.shape_parameters)

    def _check_template_cube(self, cube):
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
        dim_coords = get_dim_coord_names(cube)
        msg = ('{} expects a cube with only a leading threshold dimension, '
               'followed by spatial (y/x) dimensions. '
               'Got dimensions: {}'.format(
                   self.__class__.__name__, dim_coords))

        try:
            threshold_coord = find_threshold_coordinate(cube)
        except CoordinateNotFoundError:
            raise ValueError(msg)

        if len(dim_coords) < 4:
            enforce_coordinate_ordering(cube, threshold_coord.name())
            return

        raise ValueError(msg)

    @staticmethod
    def _check_unit_compatibility(
            location_parameter, scale_parameter,
            probability_cube_template):
        """
        The location parameter, scale parameters, and threshold values come
        from three different cubes. They should all be in the same base unit,
        with the units of the scale parameter being the squared units of the
        location parameter and threshold values. This is a sanity check to
        ensure the units are as expected, converting units of the location
        parameter and scale parameter if possible.

        Args:
            location_parameter (iris.cube.Cube):
                Cube of location parameter values.
            scale_parameter (iris.cube.Cube):
                Cube of scale parameter values.
            probability_cube_template (iris.cube.Cube):
                Cube containing threshold values.
        Raises:
            ValueError: If units of input cubes are not compatible.
        """
        threshold_units = (
            find_threshold_coordinate(probability_cube_template).units)

        try:
            location_parameter.convert_units(threshold_units)
            scale_parameter.convert_units(threshold_units**2)
        except ValueError as err:
            msg = ('Error: {} This is likely because the mean '
                   'variance and template cube threshold units are '
                   'not equivalent/compatible.'.format(err))
            raise ValueError(msg)

    def _location_and_scale_parameters_to_probabilities(
            self, location_parameter, scale_parameter,
            probability_cube_template):
        """
        Function returning probabilities relative to provided thresholds based
        on the supplied location and scale parameters.

        Args:
            location_parameter (iris.cube.Cube):
                Predictor for the calibrated forecast location parameter.
            scale_parameter (iris.cube.Cube):
                Scale parameter for the calibrated forecast.
            probability_cube_template (iris.cube.Cube):
                A probability cube that has a threshold coordinate, where the
                probabilities are defined as above or below the threshold by
                the spp__relative_to_threshold attribute. This cube matches
                the desired output cube format.

        Returns:
            iris.cube.Cube:
                Cube containing the data expressed as probabilities relative to
                the provided thresholds in the way described by
                spp__relative_to_threshold.
        """
        # Define a mask to be reapplied later
        loc_mask = np.ma.getmaskarray(location_parameter.data)
        scale_mask = np.ma.getmaskarray(scale_parameter.data)
        mask = np.logical_or(loc_mask, scale_mask)
        # Remove any mask that may be applied to location and scale parameters
        # and replace with ones
        location_parameter.data = np.ma.filled(location_parameter.data, 1)
        scale_parameter.data = np.ma.filled(scale_parameter.data, 1)
        thresholds = (
            find_threshold_coordinate(probability_cube_template).points)
        relative_to_threshold = find_threshold_coordinate(
            probability_cube_template).attributes['spp__relative_to_threshold']

        self._rescale_shape_parameters(
            location_parameter.data.flatten(),
            np.sqrt(scale_parameter.data).flatten())

        # Loop over thresholds, and use the specified distribution with the
        # location and scale parameter to calculate the probabilities relative
        # to each threshold.
        probabilities = np.empty_like(probability_cube_template.data)

        distribution = self.distribution(
            *self.shape_parameters,
            loc=location_parameter.data.flatten(),
            scale=np.sqrt(scale_parameter.data.flatten()))

        probability_method = distribution.cdf
        if relative_to_threshold == 'above':
            probability_method = distribution.sf

        for index, threshold in enumerate(thresholds):
            probabilities[index, ...] = np.reshape(
                probability_method(threshold),
                probabilities.shape[1:]) \
                # pylint: disable=unsubscriptable-object

        probability_cube = probability_cube_template.copy(data=probabilities)
        # Make the mask defined above fit the data size and then apply to the
        # probability cube.
        mask_array = np.array([mask]*len(probabilities))
        probability_cube.data = np.ma.masked_where(
            mask_array, probability_cube.data)
        return probability_cube

    def process(self, location_parameter, scale_parameter,
                probability_cube_template):
        """
        Generate probabilities from the location and scale parameters of the
        distribution.

        Args:
            location_parameter (iris.cube.Cube):
                Cube containing the location parameters.
            scale_parameter (iris.cube.Cube):
                Cube containing the scale parameters.
            probability_cube_template (iris.cube.Cube):
                A probability cube that has a threshold coordinate, where the
                probabilities are defined as above or below the threshold by
                the spp__relative_to_threshold attribute. This cube matches
                the desired output cube format.

        Returns:
            iris.cube.Cube:
                A cube of diagnostic data expressed as probabilities relative
                to the thresholds found in the probability_cube_template.
        """
        self._check_template_cube(probability_cube_template)
        self._check_unit_compatibility(
            location_parameter, scale_parameter,
            probability_cube_template)

        probability_cube = (
            self._location_and_scale_parameters_to_probabilities(
                location_parameter, scale_parameter,
                probability_cube_template))

        return probability_cube


class EnsembleReordering(BasePlugin):
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

    @staticmethod
    def _recycle_raw_ensemble_realizations(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            percentile_coord_name):
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
            percentile_coord_name (str):
                Name of required percentile coordinate.

        Returns:
            iris cube.Cube:
                Cube for the raw ensemble forecast, where the raw ensemble
                realizations have either been recycled or constrained,
                depending upon the number of percentiles present
                in the post-processed forecast cube.
        """
        plen = len(
            post_processed_forecast_percentiles.coord(
                percentile_coord_name).points)
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
            raw_forecast_realizations = concatenate_cubes(
                raw_forecast_realizations_extended,
                coords_to_slice_over=["realization", "time"])
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
            post_processed_forecast_percentiles (iris.cube.Cube):
                Cube for post-processed percentiles. The percentiles are
                assumed to be in ascending order.
            raw_forecast_realizations (iris.cube.Cube):
                Cube containing the raw (not post-processed) forecasts.
                The probabilistic dimension is assumed to be the zeroth
                dimension.
            random_ordering (bool):
                If random_ordering is True, the post-processed forecasts are
                reordered randomly, rather than using the ordering of the
                raw ensemble.
            random_seed (int or None):
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
            # The following uses a custom choose function that reproduces the
            # required elements of the np.choose method without the limitation
            # of having < 32 arrays or a leading dimension < 32 in the
            # input data array. This function allows indexing of a 3d array
            # using a 3d array.
            mask = np.ma.getmask(calfc.data)
            calfc.data = choose(ranking, calfc.data)
            if mask is not np.ma.nomask:
                calfc.data = np.ma.MaskedArray(
                    calfc.data, mask, dtype=np.float32)
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
            post_processed_forecast (iris.cube.Cube):
                The cube containing the post-processed
                forecast realizations.
            raw_forecast (iris.cube.Cube):
                The cube containing the raw (not post-processed)
                forecast.
            random_ordering (bool):
                If random_ordering is True, the post-processed forecasts are
                reordered randomly, rather than using the ordering of the
                raw ensemble.
            random_seed (int):
                If random_seed is an integer, the integer value is used for
                the random seed.
                If random_seed is None, no random seed is set, so the random
                values generated are not reproducible.

        Returns:
            iris.cube.Cube:
                Cube containing the new ensemble realizations where all points
                within the dataset have been reordered in comparison to the
                input percentiles.
        """

        percentile_coord_name = (
            find_percentile_coordinate(post_processed_forecast).name())

        enforce_coordinate_ordering(
            post_processed_forecast, percentile_coord_name)
        enforce_coordinate_ordering(raw_forecast, "realization")
        raw_forecast = (
            self._recycle_raw_ensemble_realizations(
                post_processed_forecast, raw_forecast,
                percentile_coord_name))
        post_processed_forecast_realizations = self.rank_ecc(
            post_processed_forecast, raw_forecast,
            random_ordering=random_ordering,
            random_seed=random_seed)
        post_processed_forecast_realizations = (
            RebadgePercentilesAsRealizations.process(
                post_processed_forecast_realizations))

        enforce_coordinate_ordering(
            post_processed_forecast_realizations, "realization")
        return post_processed_forecast_realizations
