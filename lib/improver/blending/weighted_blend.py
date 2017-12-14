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
"""Module containing classes for doing weighted blending by collapsing a
   whole dimension."""
import warnings

import numpy as np
import iris
from iris.analysis import Aggregator
from iris.exceptions import CoordinateNotFoundError

from improver.utilities.cube_manipulation import add_renamed_cell_method
from improver.utilities.cube_checker import find_percentile_coordinate
from improver.utilities.temporal import (
    cycletime_to_number, forecast_period_coord)


def conform_metadata(
        cube, cube_orig, coord, cycletime=None,
        coords_for_bounds_removal=None):
    """Ensure that the metadata conforms after blending together across
    the chosen coordinate.

    The metadata adjustments are:
        - Forecast reference time: If a cycletime is not present, the
          most recent available forecast_reference_time is used.
          If a cycletime is present, the cycletime is used as the
          forecast_reference_time instead.
        - Forecast period: If a forecast_period coordinate is present,
          and cycletime is not present, the lowest forecast_period is
          used. If a forecast_period coordinate is present, and the
          cycletime is present, forecast_periods are forceably calculated
          from the time and forecast_reference_time coordinate. This is
          because, if the cycletime is present, then the
          forecast_reference_time will also have been just re-calculated, so
          the forecast_period coordinate needs to be reset to match the
          newly calculated forecast_reference_time.
        - Forecast reference time and time: If forecast_reference_time and
          time coordinates are present, then a forecast_period coordinate is
          calculated and added to the cube.
        - Model_id, model_realization and realization coordinates are removed.
        - Remove bounds from the scalar coordinates, if the coordinates
          are specified within the coords_for_bounds_removal argument.

    Args:
        cube (iris.cube.Cube):
            Cube containing the metadata to be adjusted.
        cube_orig (iris.cube.Cube):
            Cube containing metadata that may be useful for adjusting
            metadata on the `cube` variable.
        coord (str):
            Coordinate that has been blended. This allows specific metadata
            changes to be limited to whichever coordinate is being blended.

    Keyword Args:
        cycletime (str):
            The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z.
        coords_for_bounds_removal (None or list):
            List of coordinates that are scalar and should have their bounds
            removed.

    Returns:
        cube (iris.cube.Cube):
            Cube containing the adjusted metadata.

    """
    if coord in ["forecast_reference_time", "model"]:
        if cube.coords("forecast_reference_time"):
            if cycletime is None:
                new_cycletime = (
                    np.max(cube_orig.coord("forecast_reference_time").points))
            else:
                cycletime_units = (
                    cube_orig.coord("forecast_reference_time").units.origin)
                cycletime_calendar = (
                    cube.coord("forecast_reference_time").units.calendar)
                new_cycletime = cycletime_to_number(
                    cycletime, time_unit=cycletime_units,
                    calendar=cycletime_calendar)
            cube.coord("forecast_reference_time").points = new_cycletime
            cube.coord("forecast_reference_time").bounds = None

        if cube.coords("forecast_period"):
            forecast_period = (
                forecast_period_coord(cube,
                                      force_lead_time_calculation=True))
            forecast_period.bounds = None
            forecast_period.convert_units(cube.coord("forecast_period").units)
            forecast_period.var_name = cube.coord("forecast_period").var_name
            cube.replace_coord(forecast_period)
        elif cube.coords("forecast_reference_time") and cube.coords("time"):
            forecast_period = (
                forecast_period_coord(cube))
            ndim = cube.coord_dims("time")
            cube.add_aux_coord(forecast_period, data_dims=ndim)

    for coord in ["model_id", "model_realization", "realization"]:
        if cube.coords(coord) and cube.coord(coord).shape == (1,):
            cube.remove_coord(coord)

    if coords_for_bounds_removal is None:
        coords_for_bounds_removal = []

    for coord in cube.coords():
        if coord.name() in coords_for_bounds_removal:
            if coord.shape == (1,) and coord.has_bounds():
                coord.bounds = None
    return cube


class PercentileBlendingAggregator(object):
    """Class for the percentile blending aggregator

       This class implements the method described by Combining Probabilities
       by Caroline Jones, 2017. This method implements blending in probability
       space.

       The steps are:
           1. At each geographic point in the cube we take the percentile
              threshold's values across the percentile dimensional coordinate.
              We recalculate, using linear interpolation, their probabilities
              in the pdf of the other points across the coordinate we are
              blending over. Thus at each point we have a set of thresholds
              and their corresponding probability values in each of the
              probability spaces across the blending coordinate.
           2. We do a weighted blend across all the probability spaces,
              combining all the thresholds in all the points in the coordinate
              we are blending over. This gives us an array of thresholds and an
              array of blended probailities for each of the grid points.
           3. We convert back to the original percentile values, again using
              linear interpolation, resulting in blended values at each of the
              original percentiles.

       References:
            Combining Probabilities by Caroline Jones, 2017:
            https://github.com/metoppv/improver/files/1128018/
            Combining_Probabilities.pdf
    """

    def __init__(self):
        """
        Initialise class.
        """
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<PercentileBlendingAggregator>')
        return result

    @staticmethod
    def aggregate(data, axis, arr_percent, arr_weights, perc_dim):
        """ Blend percentile aggregate function to blend percentile data
            along a given axis of a cube.

        Args:
            data (np.array):
                   Array containing the data to blend
            axis (int):
                   The index of the coordinate dimension in the cube. This
                   dimension will be aggregated over.
            arr_percent(np.array):
                     Array of percentile values e.g
                     [0, 20.0, 50.0, 70.0, 100.0],
                     same size as the percentile dimension of data.
            arr_weights(np.array):
                     Array of weights, same size as the axis dimension of data.
            perc_dim (int):
                     The index of the percentile coordinate
            (Note percent and weights have special meaning in Aggregator
             hence the rename.)

        Returns:
            result (np.array):
                     containing the weighted percentile blend data across
                     the chosen coord. The dimension associated with axis
                     has been collapsed, and the rest of the dimensions remain.
        """
        # Iris aggregators support indexing from the end of the array.
        if axis < 0:
            axis += data.ndim
        # Firstly ensure axis coordinate and percentile coordinate
        # are indexed as the first and second values in the data array
        data = np.moveaxis(data, [perc_dim, axis], [1, 0])

        # Determine the rest of the shape
        shape = data.shape[2:]
        input_shape = [data.shape[0],
                       data.shape[1],
                       np.prod(shape, dtype=int)]
        # Flatten the data that is not percentile or coord data
        data = data.reshape(input_shape)
        # Create the resulting data array, which is the shape of the original
        # data without dimension we are collapsing over
        result = np.zeros(input_shape[1:])
        # Loop over the flattened data, i.e. across all the data points in
        # each slice of the coordinate we are collapsing over, finding the
        # blended percentile values at each point.
        for i in range(data.shape[-1]):
            result[:, i] = (
                PercentileBlendingAggregator.blend_percentiles(
                    data[:, :, i], arr_percent, arr_weights))
        # Reshape the data and put the percentile dimension
        # back in the right place
        shape = arr_percent.shape + shape
        result = result.reshape(shape)
        # Percentile is now the leading dimension in the result. This needs
        # to move back to where it was in the input data. The result has
        # one less dimension than the original data as we have collapsed
        # one dimension.
        # If we have collapsed a dimension that was before the percentile
        # dimension in the input data, the percentile dimension moves forwards
        # one place compared to the original percentile dimension.
        if axis < perc_dim:
            result = np.moveaxis(result, 0, perc_dim-1)
        # Else we move the percentile dimension back to where it was in the
        # input data, as we have collapsed along a dimension that came after
        # it in the input cube.
        else:
            result = np.moveaxis(result, 0, perc_dim)
        return result

    @staticmethod
    def blend_percentiles(perc_values, percentiles, weights):
        """ Blend percentiles function, to calculate the weighted blend across
            a given axis of percentile data for a single grid point.

        Args:
            perc_values (np.array):
                    Array containing the percentile values to blend, with
                    shape: (length of coord to blend, num of percentiles)
            percentiles (np.array):
                    Array of percentile values e.g
                    [0, 20.0, 50.0, 70.0, 100.0],
                    same size as the percentile dimension of data.
            weights (np.array):
                    Array of weights, same size as the axis dimension of data,
                    that we will blend over.

        Returns:
            result (np.array):
                    containing the weighted percentile blend data
                    across the chosen coord
        """
        # Find the size of the dimension we want to blend over.
        num = perc_values.shape[0]
        # Create an array to store the weighted blending pdf
        combined_pdf = np.zeros((num, len(percentiles)))
        # Loop over the axis we are blending over finding the values for the
        # probability at each threshold in the pdf, for each of the other
        # points in the axis we are blending over. Use the values from the
        # percentiles if we are at the same point, otherwise use linear
        # interpolation.
        # Then add the probabilities multiplied by the correct weight to the
        # running total.
        for i in range(0, num):
            for j in range(0, num):
                if i == j:
                    recalc_values_in_pdf = percentiles
                else:
                    recalc_values_in_pdf = np.interp(perc_values[i],
                                                     perc_values[j],
                                                     percentiles)
                # Add the resulting probabilities multiplied by the right
                # weight to the running total for the combined pdf.
                combined_pdf[i] += recalc_values_in_pdf*weights[j]

        # Combine and sort the threshold values for all the points
        # we are blending.
        combined_perc_thres_data = np.sort(perc_values.flatten())

        # Combine and sort blended probability values.
        combined_perc_values = np.sort(combined_pdf.flatten())

        # Find the percentile values from this combined data by interpolating
        # back from probability values to the original percentiles.
        new_combined_perc = np.interp(percentiles,
                                      combined_perc_values,
                                      combined_perc_thres_data)
        return new_combined_perc


class MaxProbabilityAggregator(object):
    """Class for the Aggregator used to calculate the maximum weighted
       probability.

       1. Find the weighted probabilities for each point in the dimension of
          interest by multiplying each probability by the corresponding weight.
       2. Find the maximum weighted probability and return the array with one
          less dimension than the input array.
    """

    def __init__(self):
        """
        Initialise class.
        """
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<MaxProbabilityAggregator>')
        return result

    @staticmethod
    def aggregate(data, axis, arr_weights):
        """ Max probability aggregator method. Used to find the maximum
            weighted probability along a given axis.

        Args:
            data (np.array):
                   Array containing the data to blend
            axis (int):
                   The index of the coordinate dimension in the cube. This
                   dimension will be aggregated over.
            arr_weights (np.array):
                   Array of weights, same size as the axis dimension of data.


        Returns:
            result (np.array):
                     The data collapsed along the axis dimension, containing
                     the maximum weighted probability.
        """
        # Iris aggregators support indexing from the end of the array.
        if axis < 0:
            axis += data.ndim

        arr_weights = np.array(arr_weights)
        # Reshape the weights to match the shape of the data.
        shape = [len(arr_weights) if i == axis else 1
                 for i in range(data.ndim)]
        arr_weights = arr_weights.reshape(tuple(shape))
        # Calculate the weighted probabilities
        weighted_probs = data*arr_weights
        # Find the maximum along the axis of interest
        result = np.max(weighted_probs, axis=axis)

        return result


class WeightedBlendAcrossWholeDimension(object):
    """Apply a Weighted blend to a cube, collapsing across the whole
       dimension. Uses one of two methods, either weighted average, or
       the maximum of the weighted probabilities."""

    def __init__(self, coord, weighting_mode, coord_adjust=None,
                 cycletime=None, coords_for_bounds_removal=None):
        """Set up for a Weighted Blending plugin

        Args:
            coord (string):
                The name of a coordinate dimension in the cube.
            weighting_mode (string):
                One of 'weighted_maximum' or 'weighted_mean':
                 - Weighted mean: a normal weighted average over the coordinate
                   of interest.
                 - Weighted_maximum: the points in the coordinate of interest
                   are multiplied by the weights and then the maximum is taken.

        Keyword Args:
            coord_adjust (function):
                Function to apply to the coordinate after collapsing the cube
                to correct the values, for example for time windowing and
                cycle averaging the follow function would adjust the time
                coordinates.
                    e.g. coord_adjust = lambda pnts: pnts[len(pnts)/2]
            cycletime (str):
                The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z.
            coords_for_bounds_removal (None or list):
                List of coordinates that are scalar and should have their
                bounds removed.

        Raises:
            ValueError : If an invalid weighting_mode is given.
        """
        self.coord = coord
        if weighting_mode not in ['weighted_maximum', 'weighted_mean']:
            msg = ("weighting_mode: {} is not recognised, must be either "
                   "weighted_maximum or weighted_mean").format(weighting_mode)
            raise ValueError(msg)
        self.mode = weighting_mode
        self.coord_adjust = coord_adjust
        self.cycletime = cycletime
        self.coords_for_bounds_removal = coords_for_bounds_removal

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        description = ('<WeightedBlendAcrossWholeDimension:'
                       ' coord = {0:s}, weighting_mode = {1:s},'
                       ' coord_adjust = {2:s}>')
        return description.format(self.coord, self.mode, self.coord_adjust)

    def process(self, cube, weights=None):
        """Calculate weighted blend across the chosen coord, for either
           probabilistic or percentile data. If there is a percentile
           coordinate on the cube, it will blend using the
           PercentileBlendingAggregator but the percentile coordinate must
           have at least two points.

        Args:
            cube (iris.cube.Cube):
                   Cube to blend across the coord.
            weights (Optional list or np.array of weights):
                     or None (equivalent to equal weights).

        Returns:
            result (iris.cube.Cube):
                     containing the weighted blend across the chosen coord.

        Raises:
            TypeError : If the first argument not a cube.
            ValueError : If there is a percentile coord and it is not a
                           dimension coord in the cube.
            ValueError : If there is a percentile dimension with only one
                            point, we need at least two points in order to do
                            the blending.
            ValueError : If there are more than one percentile coords
                           in the cube.
            ValueError : If there is a percentile dimension on the cube and the
                         mode for blending is 'weighted_maximum'
            ValueError : If the weights shape do not match the dimension
                           of the coord we are blending over.
        Warns:
            Warning : If trying to blend across a scalar coordinate with only
                        one value. Returns the original cube in this case.

        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}.'.format(type(cube)))
            raise TypeError(msg)

        # Check that the points within the time coordinate are equal
        # if the coordinate being blended is forecast_reference_time.
        if self.coord == "forecast_reference_time":
            if cube.coords("time"):
                time_points = cube.coord("time").points
                if len(np.unique(time_points)) > 1:
                    msg = ("For blending using the forecast_reference_time "
                           "coordinate, the points within the time coordinate "
                           "need to be the same. The time points within the "
                           "input cube are {}".format(time_points))
                    raise ValueError(msg)

        # Check to see if the data is percentile data
        try:
            perc_coord = find_percentile_coordinate(cube)
            perc_dim = cube.coord_dims(perc_coord.name())
            if not perc_dim:
                msg = ('The percentile coord must be a dimension '
                       'of the cube.')
                raise ValueError(msg)
            # Check the percentile coordinate has more than one point,
            # otherwise raise an error as we won't be able to blend.
            if len(perc_coord.points) < 2.0:
                msg = ('Percentile coordinate does not have enough points'
                       ' in order to blend. Must have at least 2 percentiles.')
                raise ValueError(msg)
        except CoordinateNotFoundError:
            perc_coord = None
            perc_dim = None

        # If we have a percentile dimension and the mode is 'max' raise an
        # exception.
        if perc_coord and self.mode == 'weighted_maximum':
            msg = ('The "weighted_maximum" mode cannot be used with'
                   ' percentile data.')
            raise ValueError(msg)

        # check weights array matches coordinate shape if not None
        if weights is not None:
            if np.array(weights).shape != cube.coord(self.coord).points.shape:
                msg = ('The weights array must match the shape '
                       'of the coordinate in the input cube; '
                       'weight shape is '
                       '{0:s}'.format(np.array(weights).shape) +
                       ', cube shape is '
                       '{0:s}'.format(cube.coord(self.coord).points.shape))
                raise ValueError(msg)

        # If coord to blend over is a scalar_coord warn
        # and return original cube.
        coord_dim = cube.coord_dims(self.coord)
        if not coord_dim:
            msg = ('Trying to blend across a scalar coordinate with only one'
                   ' value. Returning original cube')
            warnings.warn(msg)
            result = cube
        else:
            try:
                cube.coord('threshold')
            except iris.exceptions.CoordinateNotFoundError:
                slices_over_threshold = [cube]
            else:
                if self.coord != 'threshold':
                    slices_over_threshold = cube.slices_over('threshold')
                else:
                    slices_over_threshold = [cube]

            cubelist = iris.cube.CubeList([])
            for cube_thres in slices_over_threshold:
                # Blend the cube across the coordinate
                # Use percentile Aggregator if required
                if perc_coord and self.mode == "weighted_mean":
                    percentiles = np.array(perc_coord.points, dtype=float)
                    perc_dim, = cube_thres.coord_dims(perc_coord.name())
                    # Set equal weights if none are provided
                    if weights is None:
                        num = len(cube_thres.coord(self.coord).points)
                        weights = np.ones(num) / float(num)
                    # Set up aggregator
                    PERCENTILE_BLEND = (Aggregator(
                        'weighted_mean',
                        PercentileBlendingAggregator.aggregate))

                    cube_new = cube_thres.collapsed(self.coord,
                                                    PERCENTILE_BLEND,
                                                    arr_percent=percentiles,
                                                    arr_weights=weights,
                                                    perc_dim=perc_dim)

                # Else do a simple weighted average
                elif self.mode == "weighted_mean":
                    # Equal weights are used as default.
                    weights_array = None
                    # Else broadcast the weights to be used by the aggregator.
                    coord_dim_thres = cube_thres.coord_dims(self.coord)
                    if weights is not None:
                        weights_array = (
                            iris.util.broadcast_to_shape(np.array(weights),
                                                         cube_thres.shape,
                                                         coord_dim_thres))
                    orig_cell_methods = cube_thres.cell_methods
                    # Calculate the weighted average.
                    cube_new = cube_thres.collapsed(self.coord,
                                                    iris.analysis.MEAN,
                                                    weights=weights_array)
                    # Update the name of the cell_method created by Iris to
                    # 'weighted_mean' to be consistent.
                    new_cell_methods = cube_new.cell_methods
                    extra_cm = (set(new_cell_methods) -
                                set(orig_cell_methods)).pop()
                    add_renamed_cell_method(cube_new,
                                            extra_cm,
                                            'weighted_mean')

                # Else use the maximum probability aggregator.
                elif self.mode == "weighted_maximum":
                    # Set equal weights if none are provided
                    if weights is None:
                        num = len(cube_thres.coord(self.coord).points)
                        weights = np.ones(num) / float(num)
                    # Set up aggregator
                    MAX_PROBABILITY = (Aggregator(
                        'weighted_maximum',
                        MaxProbabilityAggregator.aggregate))

                    cube_new = cube_thres.collapsed(self.coord,
                                                    MAX_PROBABILITY,
                                                    arr_weights=weights)
                cube_new = conform_metadata(
                    cube_new, cube_thres, coord=self.coord,
                    cycletime=self.cycletime,
                    coords_for_bounds_removal=self.coords_for_bounds_removal)
                cubelist.append(cube_new)
            result = cubelist.merge_cube()
            if isinstance(cubelist[0].data, np.ma.core.MaskedArray):
                result.data = np.ma.array(result.data)
        # If set adjust values of collapsed coordinates.
        if self.coord_adjust is not None:
            for crd in result.coords():
                if cube.coord_dims(crd.name()) == coord_dim:
                    pnts = cube.coord(crd.name()).points
                    crd.points = np.array(self.coord_adjust(pnts),
                                          dtype=crd.points.dtype)

        return result
