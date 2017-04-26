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

from ensemble_calibration_utilities import (
    concatenate_cubes, rename_coordinate)


class GeneratePercentilesFromMeanAndVariance(object):
    """
    Plugin focussing on generating percentiles from mean and variance.
    In combination with the EnsembleReordering plugin, this is Ensemble
    Copula Coupling.
    """

    def __init__(self, calibrated_forecast_predictor_and_variance,
                 raw_forecast):
        """
        Initialise the class.

        Parameters
        ----------
        calibrated_forecast_predictor_and_variance : Iris CubeList
            CubeList containing the calibrated forecast predictor and
            calibrated forecast variance.
        raw_forecast : Iris Cube or CubeList
            Cube or CubeList that is expected to be the raw
            (uncalibrated) forecast.

        """
        (self.calibrated_forecast_predictor,
         self.calibrated_forecast_variance) = (
             calibrated_forecast_predictor_and_variance)
        self.raw_forecast = raw_forecast

    def _create_cube_with_percentiles(
            self, percentiles, template_cube, cube_data):
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
        percentile_cube = self._create_cube_with_percentiles(
            percentiles, calibrated_forecast_predictor, result)

        percentile_cube.cell_methods = {}
        return percentile_cube

    def _create_percentiles(
            self, no_of_percentiles, sampling="quantile"):
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

    def process(self):
        """
        Generate ensemble percentiles from the mean and variance.

        Returns
        -------
        calibrated_forecast_percentiles : Iris cube
            Cube for calibrated percentiles.

        """
        raw_forecast = self.raw_forecast

        calibrated_forecast_predictor = concatenate_cubes(
            self.calibrated_forecast_predictor)
        calibrated_forecast_variance = concatenate_cubes(
            self.calibrated_forecast_variance)
        rename_coordinate(
            self.raw_forecast, "ensemble_member_id", "realization")
        raw_forecast_members = concatenate_cubes(self.raw_forecast)

        no_of_percentiles = len(
            raw_forecast_members.coord("realization").points)

        percentiles = self._create_percentiles(no_of_percentiles)
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
    def __init__(self, calibrated_forecast, raw_forecast):
        """
        Parameters
        ----------
        calibrated_forecast : Iris Cube or CubeList
            The cube or cubelist containing the calibrated forecast members.
        raw_forecast : Iris Cube or CubeList
            The cube or cubelist containing the raw (uncalibrated) forecast.

        """
        self.calibrated_forecast = calibrated_forecast
        self.raw_forecast = raw_forecast

    def rank_ecc(self, calibrated_forecast_percentiles, raw_forecast_members):
        """
        Function to apply Ensemble Copula Coupling. This ranks the calibrated
        forecast members based on a ranking determined from the raw forecast
        members.

        Parameters
        ----------
        calibrated_forecast_percentiles : cube
            Cube for calibrated percentiles. The percentiles are assumed to be
            in ascending order.
        raw_forecast_members : cube
            Cube containing the raw (uncalibrated) forecasts.

        Returns
        -------
        Iris cube
            Cube for calibrated members where at a particular grid point,
            the ranking of the values within the ensemble matches the ranking
            from the raw ensemble.

        """
        results = iris.cube.CubeList([])
        for rawfc, calfc in zip(
                raw_forecast_members.slices_over("time"),
                calibrated_forecast_percentiles.slices_over("time")):
            random_data = np.random.random(rawfc.data.shape)
            # Lexsort returns the indices sorted firstly by the primary key,
            # the raw forecast data, and secondly by the secondary key, an
            # array of random data, in order to split tied values randomly.
            sorting_index = np.lexsort((random_data, rawfc.data), axis=0)
            # Returns the indices that would sort the array.
            ranking = np.argsort(sorting_index, axis=0)
            # Index the calibrated forecast data using the ranking array.
            # np.choose allows indexing of a 3d array using a 3d array,
            calfc.data = np.choose(ranking, calfc.data)
            results.append(calfc)
        return concatenate_cubes(results)

    def process(self):
        """
        Returns
        -------
        calibrated_forecast_members : cube
            Cube for a new ensemble member where all points within the dataset
            are representative of a specified probability threshold across the
            whole domain.
        """
        rename_coordinate(
            self.raw_forecast, "ensemble_member_id", "realization")
        calibrated_forecast_percentiles = concatenate_cubes(
            self.calibrated_forecast,
            coords_to_slice_over=["percentile", "time"])
        raw_forecast_members = concatenate_cubes(self.raw_forecast)
        calibrated_forecast_members = self.rank_ecc(
            calibrated_forecast_percentiles, raw_forecast_members)
        rename_coordinate(
            calibrated_forecast_members, "percentile", "realization")
        return calibrated_forecast_members
