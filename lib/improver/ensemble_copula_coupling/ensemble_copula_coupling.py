# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
import numpy as np
from scipy.stats import norm


import iris
from iris.exceptions import CoordinateNotFoundError
import cf_units as unit

from improver.ensemble_calibration.ensemble_calibration_utilities import (
    concatenate_cubes, convert_cube_data_to_2d, rename_coordinate)
from improver.ensemble_copula_coupling.ensemble_copula_coupling_constants \
    import bounds_for_ecdf, units_of_bounds_for_ecdf
from improver.ensemble_copula_coupling.ensemble_copula_coupling_utilities \
    import (concatenate_2d_array_with_2d_array_endpoints,
            create_cube_with_percentiles, create_percentiles,
            get_bounds_of_distribution,
            insert_lower_and_upper_endpoint_to_1d_array,
            reshape_array_to_have_probabilistic_dimension_at_the_front)


class RebadgePercentilesAsMembers(object):
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

    def process(self, cube):
        """
        Rebadge percentiles as ensemble members. The ensemble member numbering
        will depend upon the number of percentiles in the input cube i.e.
        0, 1, 2, 3, ..., n-1, if there are n percentiles.

        Parameters
        ----------
        cube : Iris.cube.Cube
        Cube containing a percentile coordinate, which will be rebadged as
        ensemble member.

        """
        if not cube.coords("percentile"):
            msg = ("The percentile coordinate could not be found within"
                   "the input cube: {}.".format(cube))
            raise CoordinateNotFoundError(msg)

        plen = len(cube.coord("percentile").points)
        cube.coord("percentile").points = np.arange(plen)
        cube.coord("percentile").rename("realization")
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

    def _add_bounds_to_percentiles_and_forecast_at_percentiles(
            self, percentiles, forecast_at_percentiles, bounds_pairing):
        """
        Padding of the lower and upper bounds of the percentiles for a
        given phenomenon, and padding of forecast values using the
        constant lower and upper bounds.

        Parameters
        ----------
        percentiles : Numpy array
            Array of percentiles from a Cumulative Distribution Function.
        forecast_at_percentiles : Numpy array
            Array containing the underlying forecast values at each percentile.
        bounds_pairing : Tuple
            Lower and upper bound to be used as the ends of the
            empirical cumulative distribution function.
        Returns
        -------
        percentiles : Numpy array
            Array of percentiles from a Cumulative Distribution Function.
        forecast_at_percentiles : Numpy array
            Array containing the underlying forecast values at each percentile.
        """
        lower_bound, upper_bound = bounds_pairing
        percentiles = insert_lower_and_upper_endpoint_to_1d_array(
            percentiles, 0, 1)
        forecast_at_percentiles = concatenate_2d_array_with_2d_array_endpoints(
            forecast_at_percentiles, lower_bound, upper_bound)
        if np.any(np.diff(forecast_at_percentiles) < 0):
            msg = ("The end points added to the forecast at percentiles "
                   "values representing for each percentile must result in "
                   "an ascending order. "
                   "In this case, the forecast at percentile values {} "
                   "must be outside the allowable range given by the "
                   "bounds {}".format(
                       forecast_at_percentiles, bounds_pairing))
            raise ValueError(msg)
        if np.any(np.diff(percentiles) < 0):
            msg = ("The percentiles must be in ascending order."
                   "The input percentiles were {}".format(percentiles))
            raise ValueError(msg)
        return percentiles, forecast_at_percentiles

    def _sample_percentiles(
            self, forecast_at_percentiles, desired_percentiles,
            bounds_pairing):
        """
        Interpolation of forecast for a set of percentiles from an initial
        set of percentiles to a new set of percentiles. This is constructed
        by linearly interpolating between the original set of percentiles
        to a new set of percentiles.

        Parameters
        ----------
        forecast_at_percentiles : Iris CubeList or Iris Cube
            Cube or CubeList expected to contain a percentile coordinate.
        desired_percentiles : Numpy array
            Array of the desired percentiles.
        bounds_pairing : Tuple
            Lower and upper bound to be used as the ends of the
            empirical cumulative distribution function.

        Returns
        -------
        percentile_cube : Iris cube
            Cube containing values for the required diagnostic e.g.
            air_temperature at the required percentiles.

        """
        original_percentiles = (
            forecast_at_percentiles.coord("percentile").points)

        forecast_at_reshaped_percentiles = convert_cube_data_to_2d(
            forecast_at_percentiles, coord="percentile")

        original_percentiles, forecast_at_reshaped_percentiles = (
            self._add_bounds_to_percentiles_and_forecast_at_percentiles(
                original_percentiles, forecast_at_reshaped_percentiles,
                bounds_pairing))

        forecast_at_interpolated_percentiles = (
            np.empty(
                (forecast_at_reshaped_percentiles.shape[0],
                 len(desired_percentiles))))
        for index in range(forecast_at_reshaped_percentiles.shape[0]):
            forecast_at_interpolated_percentiles[index, :] = np.interp(
                desired_percentiles, original_percentiles,
                forecast_at_reshaped_percentiles[index, :])

        # Transpose data
        forecast_at_interpolated_percentiles = (
            forecast_at_interpolated_percentiles.T)

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        forecast_at_percentiles_data = (
            reshape_array_to_have_probabilistic_dimension_at_the_front(
                forecast_at_interpolated_percentiles, forecast_at_percentiles,
                "percentile", len(desired_percentiles)))

        for template_cube in forecast_at_percentiles.slices_over(
                "percentile"):
            template_cube.remove_coord("percentile")
            break
        percentile_cube = create_cube_with_percentiles(
            desired_percentiles, template_cube, forecast_at_percentiles_data)
        percentile_cube.cell_methods = {}
        return percentile_cube

    def process(self, forecast_at_percentiles, no_of_percentiles=None,
                sampling="quantile"):
        """
        1. Concatenates cubes with a percentile coordinate.
        2. Creates a list of percentiles.
        3. Accesses the lower and upper bound pair of the forecast values,
           in order to specify lower and upper bounds for the percentiles.
        4. Interpolate the percentile coordinate into an alternative
           set of percentiles using linear interpolation.

        Parameters
        ----------
        forecast_at_percentiles : Iris CubeList or Iris Cube
            Cube or CubeList expected to contain a percentile coordinate.
        no_of_percentiles : Integer or None
            Number of percentiles
            If None, the number of percentiles within the input
            forecast_at_percentiles cube is used as the number of percentiles.
        sampling : String
            Type of sampling of the distribution to produce a set of
            percentiles e.g. quantile or random.
            Accepted options for sampling are:
            Quantile: A regular set of equally-spaced percentiles aimed
                      at dividing a Cumulative Distribution Function into
                      blocks of equal probability.
            Random: A random set of ordered percentiles.

        Returns
        -------
        forecast_at_percentiles : Iris cube
            Cube with forecast values at the desired set of percentiles.

        """
        forecast_at_percentiles = concatenate_cubes(forecast_at_percentiles)

        if no_of_percentiles is None:
            no_of_percentiles = (
                len(forecast_at_percentiles.coord("percentile").points))

        percentiles = create_percentiles(
            no_of_percentiles, sampling=sampling)

        cube_units = forecast_at_percentiles.units
        bounds_pairing = (
            get_bounds_of_distribution(
                forecast_at_percentiles.name(), cube_units))

        forecast_at_percentiles = self._sample_percentiles(
            forecast_at_percentiles, percentiles, bounds_pairing)
        return forecast_at_percentiles


class GeneratePercentilesFromProbabilities(object):
    """
    Class for generating percentiles from probabilities.
    In combination with the Ensemble Reordering plugin, this is a variant
    Ensemble Copula Coupling.

    This class includes the ability to interpolate between probability
    thresholds in order to generate the percentiles, see Figure 1 from
    Flowerdew, 2014.

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

    def _add_bounds_to_thresholds_and_probabilities(
            self, threshold_points, probabilities_for_cdf, bounds_pairing):
        """
        Padding of the lower and upper bounds of the distribution for a
        given phenomenon for the threshold_points, and padding of
        probabilities of 0 and 1 to the forecast probabilities.

        Parameters
        ----------
        threshold_points : Numpy array
            Array of threshold values used to calculate the probabilities.
        probabilities_for_cdf : Numpy array
            Array containing the probabilities used for constructing an
            empirical cumulative distribution function i.e. probabilities
            below threshold.
        bounds_pairing : Tuple
            Lower and upper bound to be used as the ends of the
            empirical cumulative distribution function.
        Returns
        -------
        threshold_points : Numpy array
            Array of threshold values padded with the lower and upper bound
            of the distribution.
        probabilities_for_cdf : Numpy array
            Array containing the probabilities padded with 0 and 1 at each end.
        """
        lower_bound, upper_bound = bounds_pairing
        threshold_points = insert_lower_and_upper_endpoint_to_1d_array(
            threshold_points, lower_bound, upper_bound)
        probabilities_for_cdf = concatenate_2d_array_with_2d_array_endpoints(
            probabilities_for_cdf, 0, 1)
        if np.any(np.diff(threshold_points) < 0):
            msg = ("The end points added to the threshold values for "
                   "constructing the Cumulative Distribution Function (CDF) "
                   "must result in an ascending order. "
                   "In this case, the threshold points {} must be "
                   "outside the allowable range given by the "
                   "bounds {}".format(
                       threshold_points, bounds_pairing))
            raise ValueError(msg)
        return threshold_points, probabilities_for_cdf

    def _probabilities_to_percentiles(
            self, forecast_probabilities, percentiles, bounds_pairing):
        """
        Conversion of probabilities to percentiles through the construction
        of an empirical cumulative distribution function. This is effectively
        constructed by linear interpolation from the probabilities associated
        with each threshold to a set of percentiles.

        Parameters
        ----------
        forecast_probabilities : Iris cube
            Cube with a probability_above_threshold coordinate.
        percentiles : Numpy array
            Array of percentiles, at which the corresponding values will be
            calculated.
        bounds_pairing : Tuple
            Lower and upper bound to be used as the ends of the
            empirical cumulative distribution function.

        Returns
        -------
        percentile_cube : Iris cube
            Cube containing values for the required diagnostic e.g.
            air_temperature at the required percentiles.

        """
        threshold_points = (
            forecast_probabilities.coord("probability_above_threshold").points)

        prob_slices = convert_cube_data_to_2d(
            forecast_probabilities, coord="probability_above_threshold")

        # Invert probabilities
        probabilities_for_cdf = 1 - prob_slices

        if np.any(np.diff(probabilities_for_cdf) < 0):
            msg = ("The probability values used to construct the "
                   "Cumulative Distribution Function (CDF) "
                   "must be ascending i.e. in order to yield "
                   "a monotonically increasing CDF."
                   "The probabilities are {}".format(probabilities_for_cdf))
            raise ValueError(msg)

        threshold_points, probabilities_for_cdf = (
            self._add_bounds_to_thresholds_and_probabilities(
                threshold_points, probabilities_for_cdf, bounds_pairing))

        forecast_at_percentiles = (
            np.empty((probabilities_for_cdf.shape[0], len(percentiles))))
        for index in range(probabilities_for_cdf.shape[0]):
            forecast_at_percentiles[index, :] = np.interp(
                percentiles, probabilities_for_cdf[index, :],
                threshold_points)

        # Transpose data
        forecast_at_percentiles = forecast_at_percentiles.T

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        forecast_at_percentiles = (
            reshape_array_to_have_probabilistic_dimension_at_the_front(
                forecast_at_percentiles, forecast_probabilities,
                "probability_above_threshold", len(percentiles)))

        for template_cube in forecast_probabilities.slices_over(
                "probability_above_threshold"):
            template_cube.remove_coord("probability_above_threshold")
            break
        percentile_cube = create_cube_with_percentiles(
            percentiles, template_cube, forecast_at_percentiles)
        percentile_cube.cell_methods = {}
        return percentile_cube

    def process(self, forecast_probabilities, no_of_percentiles=None,
                sampling="quantile"):
        """
        1. Concatenates cubes with a probability_above_threshold coordinate.
        2. Creates a list of percentiles.
        3. Accesses the lower and upper bound pair to find the ends of the
           empirical cumulative distribution function.
        4. Convert the probability_above_threshold coordinate into
           values at a set of percentiles using linear interpolation,
           see Figure 1 from Flowerdew, 2014.

        Parameters
        ----------
        forecast_probabilities : Iris CubeList or Iris Cube
            Cube or CubeList expected to contain a probability_above_threshold
            coordinate.
        no_of_percentiles : Integer or None
            Number of percentiles
            If None, the number of thresholds within the input
            forecast_probabilities cube is used as the number of percentiles.
        sampling : String
            Type of sampling of the distribution to produce a set of
            percentiles e.g. quantile or random.
            Accepted options for sampling are:
            Quantile: A regular set of equally-spaced percentiles aimed
                      at dividing a Cumulative Distribution Function into
                      blocks of equal probability.
            Random: A random set of ordered percentiles.

        Returns
        -------
        forecast_at_percentiles : Iris cube
            Cube with forecast values at the desired set of percentiles.

        """
        forecast_probabilities = concatenate_cubes(forecast_probabilities)

        if no_of_percentiles is None:
            no_of_percentiles = (
                len(forecast_probabilities.coord(
                    "probability_above_threshold").points))

        percentiles = create_percentiles(
            no_of_percentiles, sampling=sampling)

        cube_units = (
            forecast_probabilities.coord("probability_above_threshold").units)
        bounds_pairing = (
            get_bounds_of_distribution(
                forecast_probabilities.name(), cube_units))

        forecast_at_percentiles = self._probabilities_to_percentiles(
            forecast_probabilities, percentiles, bounds_pairing)
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

    def _mean_and_variance_to_percentiles(
            self, calibrated_forecast_predictor, calibrated_forecast_variance,
            percentiles):
        """
        Function returning percentiles based on the supplied
        mean and variance. The percentiles are created by assuming a
        Gaussian distribution and calculating the value of the phenomenon at
        specific points within the distribution.

        Parameters
        ----------
        calibrated_forecast_predictor : cube
            Predictor for the calibrated forecast i.e. the mean.
        calibrated_forecast_variance : cube
            Variance for the calibrated forecast.
        percentiles : List
            Percentiles at which to calculate the value of the phenomenon at.

        Returns
        -------
        percentile_cube : Iris cube
            Cube containing the values for the phenomenon at each of the
            percentiles requested.

        """
        if not calibrated_forecast_predictor.coord_dims("time"):
            calibrated_forecast_predictor = iris.util.new_axis(
                calibrated_forecast_predictor, "time")
        if not calibrated_forecast_variance.coord_dims("time"):
            calibrated_forecast_variance = iris.util.new_axis(
                calibrated_forecast_variance, "time")

        calibrated_forecast_predictor_data = (
            calibrated_forecast_predictor.data.flatten())
        calibrated_forecast_variance_data = (
            calibrated_forecast_variance.data.flatten())

        result = np.zeros((calibrated_forecast_predictor_data.shape[0],
                           len(percentiles)))

        # Loop over percentiles, and use a normal distribution with the mean
        # and variance to calculate the values at each percentile.
        for index, percentile in enumerate(percentiles):
            percentile_list = np.repeat(
                percentile, len(calibrated_forecast_predictor_data))
            result[:, index] = norm.ppf(
                percentile_list, loc=calibrated_forecast_predictor_data,
                scale=np.sqrt(calibrated_forecast_variance_data))
            # If percent point function (PPF) returns NaNs, fill in
            # mean instead of NaN values. NaN will only be generated if the
            # variance is zero. Therefore, if the variance is zero, the mean
            # value is used for all gridpoints with a NaN.
            if np.any(calibrated_forecast_variance_data == 0):
                nan_index = np.argwhere(np.isnan(result[:, index]))
                result[nan_index, index] = (
                    calibrated_forecast_predictor_data[nan_index])
            if np.any(np.isnan(result)):
                msg = ("NaNs are present within the result for the {} "
                       "percentile. Unable to calculate the percent point "
                       "function.")
                raise ValueError(msg)

        # Transpose data
        result = result.T

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        result = (
            reshape_array_to_have_probabilistic_dimension_at_the_front(
                result, calibrated_forecast_predictor,
                "realization", len(percentiles)))

        for template_cube in calibrated_forecast_predictor.slices_over(
                "realization"):
            template_cube.remove_coord("realization")
            break
        percentile_cube = create_cube_with_percentiles(
            percentiles, template_cube, result)

        percentile_cube.cell_methods = {}
        return percentile_cube

    def process(self, calibrated_forecast_predictor_and_variance,
                raw_forecast):
        """
        Generate ensemble percentiles from the mean and variance.

        Parameters
        ----------
        calibrated_forecast_predictor_and_variance : Iris CubeList
            CubeList containing the calibrated forecast predictor and
            calibrated forecast variance.
        raw_forecast : Iris Cube or CubeList
            Cube or CubeList that is expected to be the raw
            (uncalibrated) forecast.

        Returns
        -------
        calibrated_forecast_percentiles : Iris cube
            Cube for calibrated percentiles.

        """
        (calibrated_forecast_predictor, calibrated_forecast_variance) = (
            calibrated_forecast_predictor_and_variance)

        calibrated_forecast_predictor = concatenate_cubes(
            calibrated_forecast_predictor)
        calibrated_forecast_variance = concatenate_cubes(
            calibrated_forecast_variance)
        rename_coordinate(
            raw_forecast, "ensemble_member_id", "realization")
        raw_forecast_members = concatenate_cubes(raw_forecast)

        no_of_percentiles = len(
            raw_forecast_members.coord("realization").points)

        percentiles = create_percentiles(no_of_percentiles)
        calibrated_forecast_percentiles = (
            self._mean_and_variance_to_percentiles(
                calibrated_forecast_predictor,
                calibrated_forecast_variance,
                percentiles))

        return calibrated_forecast_percentiles


class EnsembleReordering(object):
    """
    Plugin for applying the reordering step of Ensemble Copula Coupling,
    in order to generate ensemble members from percentiles.
    The percentiles are assumed to be in ascending order.

    Reference:
    Schefzik, R., Thorarinsdottir, T.L. & Gneiting, T., 2013.
    Uncertainty Quantification in Complex Simulation Models Using Ensemble
    Copula Coupling.
    Statistical Science, 28(4), pp.616-640.

    """
    def __init__(self):
        """Initialise the class"""
        pass

    def mismatch_between_length_of_raw_members_and_percentiles(
            self, post_processed_forecast_percentiles, raw_forecast_members):
        """
        Function to determine whether there is a mismatch between the number
        of percentiles and the number of raw forecast members. If more
        percentiles are requested than ensemble members, then the ensemble
        members are recycled. If fewer percentiles are requested than
        ensemble members, then only the first n ensemble members are used.

        Parameters
        ----------
        post_processed_forecast_percentiles : cube
            Cube for post-processed percentiles. The percentiles are assumed
            to be in ascending order.
        raw_forecast_members : cube
            Cube containing the raw (not post-processed) forecasts.

        Returns
        -------
        Iris cube
            Cube for the raw ensemble forecast, where the raw ensemble members
            have either been recycled or constrained, depending upon the
            number of percentiles present in the post-processed forecast cube.

        """
        plen = len(
            post_processed_forecast_percentiles.coord("percentile").points)
        mlen = len(raw_forecast_members.coord("realization").points)
        if plen == mlen:
            pass
        else:
            raw_forecast_members_extended = iris.cube.CubeList()
            realization_list = []
            mpoints = raw_forecast_members.coord("realization").points
            # Loop over the number of percentiles and finding the
            # corresponding ensemble member number. The ensemble member
            # numbers are recycled e.g. 1, 2, 3, 1, 2, 3, etc.
            for index in range(plen):
                realization_list.append(mpoints[index % len(mpoints)])
            # Extract the members required in the realization_list from
            # the raw_forecast_members. Edit the member number as appropriate
            # and append to a cubelist containing rebadged raw ensemble
            # members.
            for realization, index in zip(realization_list, range(plen)):
                constr = iris.Constraint(realization=realization)
                raw_forecast_member = raw_forecast_members.extract(constr)
                raw_forecast_member.coord("realization").points = index
                raw_forecast_members_extended.append(raw_forecast_member)
            raw_forecast_members = (
                concatenate_cubes(raw_forecast_members_extended))
        return raw_forecast_members

    def rank_ecc(
            self, post_processed_forecast_percentiles, raw_forecast_members,
            random_ordering=False):
        """
        Function to apply Ensemble Copula Coupling. This ranks the
        post-processed forecast members based on a ranking determined from
        the raw forecast members.

        Parameters
        ----------
        post_processed_forecast_percentiles : cube
            Cube for post-processed percentiles. The percentiles are assumed
            to be in ascending order.
        raw_forecast_members : cube
            Cube containing the raw (not post-processed) forecasts.
        random_ordering : Logical
            If random_ordering is True, the post-processed forecasts are
            reordered randomly, rather than using the ordering of the
            raw ensemble.

        Returns
        -------
        Iris cube
            Cube for post-processed members where at a particular grid point,
            the ranking of the values within the ensemble matches the ranking
            from the raw ensemble.

        """
        results = iris.cube.CubeList([])
        for rawfc, calfc in zip(
                raw_forecast_members.slices_over("time"),
                post_processed_forecast_percentiles.slices_over("time")):
            random_data = np.random.random(rawfc.data.shape)
            # Lexsort returns the indices sorted firstly by the
            # primary key, the raw forecast data (unless random_ordering
            # is enabled), and secondly by the secondary key, an array of
            # random data, in order to split tied values randomly.
            if random_ordering:
                fake_rawfc_data = np.random.random(rawfc.data.shape)
                sorting_index = (
                    np.lexsort((random_data, fake_rawfc_data), axis=0))
            else:
                sorting_index = np.lexsort((random_data, rawfc.data), axis=0)
            # Returns the indices that would sort the array.
            ranking = np.argsort(sorting_index, axis=0)
            # Index the post-processed forecast data using the ranking array.
            # np.choose allows indexing of a 3d array using a 3d array,
            calfc.data = np.choose(ranking, calfc.data)
            results.append(calfc)
        return concatenate_cubes(results)

    def process(
            self, post_processed_forecast, raw_forecast,
            random_ordering=False):
        """
        Reorder post-processed forecast using the ordering of the
        raw ensemble.

        Parameters
        ----------
        post_processed_forecast : Iris Cube or CubeList
            The cube or cubelist containing the post-processed
            forecast members.
        raw_forecast : Iris Cube or CubeList
            The cube or cubelist containing the raw (not post-processed)
            forecast.
        random_ordering : Logical
            If random_ordering is True, the post-processed forecasts are
            reordered randomly, rather than using the ordering of the
            raw ensemble.

        Returns
        -------
        post-processed_forecast_members : cube
            Cube for a new ensemble member where all points within the dataset
            are representative of a specified probability threshold across the
            whole domain.
        """
        rename_coordinate(
            raw_forecast, "ensemble_member_id", "realization")
        post_processed_forecast_percentiles = concatenate_cubes(
            post_processed_forecast,
            coords_to_slice_over=["percentile", "time"])
        raw_forecast_members = concatenate_cubes(raw_forecast)
        raw_forecast_members = (
            self.mismatch_between_length_of_raw_members_and_percentiles(
                post_processed_forecast_percentiles, raw_forecast_members))
        post_processed_forecast_members = self.rank_ecc(
            post_processed_forecast_percentiles, raw_forecast_members,
            random_ordering=random_ordering)
        rename_coordinate(
            post_processed_forecast_members, "percentile", "realization")
        return post_processed_forecast_members
