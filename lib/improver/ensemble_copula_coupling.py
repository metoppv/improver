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
import copy
import numpy as np
import random
from scipy.stats import norm

import cf_units as unit
import iris

from ensemble_calibration.ensemble_calibration_utilities import (
    concatenate_cubes, convert_cube_data_to_2d, rename_coordinate)
from ensemble_copula_coupling_constants import bounds_for_ecdf


class EnsembleCopulaCouplingUtilities(object):
    """
    Class containing utilities used to enable Ensemble Copula Coupling.
    """
    @staticmethod
    def create_percentiles(
            no_of_percentiles, sampling="quantile"):
        """
        Function to create percentiles.

        Parameters
        ----------
        no_of_percentiles : Int
            Number of percentiles.
        sampling : String
            Type of sampling of the distribution to produce a set of
            percentiles e.g. quantile or random.
            Accepted options for sampling are:
            Quantile: A regular set of equally-spaced percentiles aimed
                      at dividing a Cumulative Distribution Function into
                      blocks of equal probability.
            Random: A random set of ordered percentiles.

        For further details, Flowerdew, J., 2014.
        Calibrating ensemble reliability whilst preserving spatial structure.
        Tellus, Series A: Dynamic Meteorology and Oceanography, 66(1), pp.1-20.
        Schefzik, R., Thorarinsdottir, T.L. & Gneiting, T., 2013.
        Uncertainty Quantification in Complex Simulation Models Using Ensemble
        Copula Coupling.
        Statistical Science, 28(4), pp.616-640.

        Returns
        -------
        percentiles : List
            Percentiles calculated using the sampling technique specified.

        """
        if sampling in ["quantile"]:
            percentiles = np.linspace(
                1/float(1+no_of_percentiles),
                no_of_percentiles/float(1+no_of_percentiles),
                no_of_percentiles).tolist()
        elif sampling in ["random"]:
            percentiles = []
            for _ in range(no_of_percentiles):
                percentiles.append(
                    random.uniform(
                        1/float(1+no_of_percentiles),
                        no_of_percentiles/float(1+no_of_percentiles)))
            percentiles = sorted(percentiles)
        else:
            msg = "The {} sampling option is not yet implemented.".format(
                sampling)
            raise ValueError(msg)
        return percentiles

    @staticmethod
    def create_cube_with_percentiles(
            percentiles, template_cube, cube_data):
        """
        Create a cube with a percentile coordinate based on a template cube.

        Parameters
        ----------
        percentiles : List
            Ensemble percentiles.
        template_cube : Iris cube
            Cube to copy majority of coordinate definitions from.
        cube_data : Numpy array
            Data to insert into the template cube.
            The data is expected to have the shape of
            percentiles (0th dimension), time (1st dimension),
            y_coord (2nd dimension), x_coord (3rd dimension).

        Returns
        -------
        String
            Coordinate name of the matched coordinate.

        """
        percentile_coord = iris.coords.DimCoord(
            np.float32(percentiles), long_name="percentile",
            units=unit.Unit("1"), var_name="percentile")

        time_coord = template_cube.coord("time")
        y_coord = template_cube.coord(axis="y")
        x_coord = template_cube.coord(axis="x")

        dim_coords_and_dims = [
            (percentile_coord, 0), (time_coord, 1),
            (y_coord, 2), (x_coord, 3)]

        frt_coord = template_cube.coord("forecast_reference_time")
        fp_coord = template_cube.coord("forecast_period")
        aux_coords_and_dims = [(frt_coord, 1), (fp_coord, 1)]

        metadata_dict = copy.deepcopy(template_cube.metadata._asdict())

        cube = iris.cube.Cube(
            cube_data, dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims, **metadata_dict)
        cube.attributes = template_cube.attributes
        cube.cell_methods = template_cube.cell_methods
        return cube


class GeneratePercentilesFromProbabilities(object):
    """
    Class for generating percentiles from probabilities.
    In combination with the Ensemble Reordering plugin, this is Ensemble
    Copula Coupling.
    """

    def __init__(self):
        """
        Initialise the class.
        """
        pass

    def _add_bounds_to_thresholds_and_probabilities(
            self, threshold_points, probabilities_for_cdf, bounds_pairing):
        """
        Padding of the lower and upper bounds for a given phenomenon for the
        threshold_points, and padding of probabilities of 0 and 1 to the
        forecast probabilities.

        Parameters
        ----------
        threshold_points : Numpy array
            Array of threshold values used to calculate the probabilities.
        probabilities_for_cdf : Numpy array
            Array containing the probabilities used for constructing an
            empirical cumulative distribution function i.e. probabilities
            below threshold.

        Returns
        -------
        threshold_points : Numpy array
            Array of threshold values padded with the lower and upper bound.
        probabilities_for_cdf : Numpy array
            Array containing the probabilities padded with 0 and 1 at each end.
        bounds_pairing : Tuple
            Lower and upper bound to be used as the ends of the
            empirical cumulative distribution function.

        """
        lower_bound, upper_bound = bounds_pairing
        threshold_points = np.insert(threshold_points, 0, lower_bound)
        threshold_points = np.append(threshold_points, upper_bound)
        zeroes_array = np.zeros((probabilities_for_cdf.shape[0], 1))
        ones_array = np.ones((probabilities_for_cdf.shape[0], 1))
        probabilities_for_cdf = np.concatenate(
            (zeroes_array, probabilities_for_cdf, ones_array), axis=1)
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
            Cube with probabilities at the required percentiles.

        """
        threshold_points = (
            forecast_probabilities.coord("probability_above_threshold").points)

        prob_slices = convert_cube_data_to_2d(
            forecast_probabilities, coord="probability_above_threshold")

        # Invert probabilities
        probabilities_for_cdf = 1 - prob_slices

        threshold_points, probabilities_for_cdf = (
            self._add_bounds_to_thresholds_and_probabilities(
                threshold_points, probabilities_for_cdf, bounds_pairing))

        forecast_at_percentiles = (
            np.empty((probabilities_for_cdf.shape[0], len(percentiles))))
        for index in range(probabilities_for_cdf.shape[0]):
            forecast_at_percentiles[index, :] = np.interp(
                percentiles, probabilities_for_cdf[index, :],
                threshold_points)

        t_coord = forecast_probabilities.coord("time")
        y_coord = forecast_probabilities.coord(axis="y")
        x_coord = forecast_probabilities.coord(axis="x")

        forecast_at_percentiles = forecast_at_percentiles.reshape(
            len(percentiles), len(t_coord.points), len(y_coord.points),
            len(x_coord.points))
        percentile_cube = (
            EnsembleCopulaCouplingUtilities.create_cube_with_percentiles(
                percentiles, forecast_probabilities, forecast_at_percentiles))
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
           values at a set of percentiles.

        Parameters
        ----------
        forecast_probabilities : Iris CubeList or Iris Cube
            Cube or CubeList expected to contain a probability_above_threshold
            coordinate.
        no_of_percentiles : Integer
            Number of percentiles
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

        percentiles = EnsembleCopulaCouplingUtilities.create_percentiles(
            no_of_percentiles, sampling=sampling)

        # Extract bounds from dictionary of constants.
        bounds_pairing = bounds_for_ecdf[forecast_probabilities.name()]

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

        result = result.T

        t_coord = calibrated_forecast_predictor.coord("time")
        y_coord = calibrated_forecast_predictor.coord(axis="y")
        x_coord = calibrated_forecast_predictor.coord(axis="x")

        result = result.reshape(
            len(percentiles), len(t_coord.points), len(y_coord.points),
            len(x_coord.points))
        percentile_cube = (
            EnsembleCopulaCouplingUtilities.create_cube_with_percentiles(
                percentiles, calibrated_forecast_predictor, result))

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

        percentiles = EnsembleCopulaCouplingUtilities.create_percentiles(
            no_of_percentiles)
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
            Cube for post-processed members where at a particular grid point,
            the ranking of the values within the ensemble matches the ranking
            from the raw ensemble.

        """
        plen = len(
            post_processed_forecast_percentiles.coord("percentile").points)
        mlen = len(raw_forecast_members.coord("realization").points)
        if plen == mlen:
            pass
        elif plen > mlen or plen < mlen:
            raw_forecast_members_extended = iris.cube.CubeList()
            realization_list = []
            mpoints = raw_forecast_members.coord("realization").points
            for index in range(plen):
                realization_list.append(mpoints[index % len(mpoints)])
            for realization, index in zip(realization_list, range(plen)):
                constr = iris.Constraint(realization=realization)
                raw_forecast_member = raw_forecast_members.extract(constr)
                raw_forecast_member.coord("realization").points = index
                raw_forecast_members_extended.append(raw_forecast_member)
            raw_forecast_members = (
                concatenate_cubes(raw_forecast_members_extended))
        return post_processed_forecast_percentiles, raw_forecast_members

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
        post_processed_forecast_percentiles, raw_forecast_members = (
            self.mismatch_between_length_of_raw_members_and_percentiles(
                post_processed_forecast_percentiles, raw_forecast_members))
        post_processed_forecast_members = self.rank_ecc(
            post_processed_forecast_percentiles, raw_forecast_members)
        rename_coordinate(
            post_processed_forecast_members, "percentile", "realization")
        return post_processed_forecast_members
