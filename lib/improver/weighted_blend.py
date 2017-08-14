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
"""Module containing Weighted Blend classes."""
import warnings

import numpy as np
import iris
from iris.analysis import Aggregator


class PercentileBlendingAggregator(object):
    """Class for the percentile blending aggregator

       This class implements the method described by Combining Probabilities
       by Caroline Jones, 2017. This method implements blending in probability
       space, rather than combining the percentiles directly.
       
       The steps are:
           1. We convert the values at percentiles to probabilities at
              the thresholds in the input cube, using linear interpolatin
              if required. This is calculated for each point in the cube. Each
              point in the coordinate we are blending over represents a new
              probability space, so for each point the probabilities are
              calculated in the probability space of all the other points.
           2. We do a weighted blend across all the probability spaces,
              combining all the thresholds in all the points in the coordinate
              we are blending over. This gives us an array of thresholds and an
              array of blended probailities for each of the thresholds.
           3. We convert back to the original percentile values, again using
              linear interpolation, resulting in blended values at each of the
              original percentiles.
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
            data : np.array
                   Array containing the data to blend
            axis : integer
                   The index of the coordinate dimension in the cube. This
                   dimension will be aggregated over.
            arr_percent: np.array
                     Array of percentile values e.g
                     [0, 20.0, 50.0, 70.0, 100.0],
                     same size as the percentile dimension of data.
            arr_weights: np.array
                     Array of weights, same size as the axis dimension of data.
            perc_dim : integer
                     The index of the percentile coordinate
            (Note percent and weights have special meaning in Aggregator
             hence the rename.)

        Returns:
            result : np.array
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
        result = None
        input_shape = [data.shape[0],
                        data.shape[1],
                        np.prod(shape)]
        # Flatten the data that is not percentile or coord data
        data = data.reshape(input_shape)
        # Create the resulting data array, which is the shape of the original
        # data without dimension we are collapsing over
        result = np.zeros(input_shape[1:])
        # Loop over the flatten data, i.e. acrosss all the data points in each
        # slice of the coordinate we are collapsing over, finding the blended
        # percentile values at each point.
        for i in range(data.shape[-1]):
            result[:, i] = (
                PercentileBlendingAggregator.blend_percentiles(
                    data[:, :, i], arr_percent, arr_weights))
        # Reshape the data and put the percentile dimension
        # back in the right place
        if arr_percent.shape > (1,):
            shape = arr_percent.shape + shape
            result = result.reshape(shape)
            if axis < perc_dim:
                if perc_dim != 1:
                    result = np.moveaxis(result, 0, perc_dim-1)
            else:
                if perc_dim != 0:
                    result = np.moveaxis(result, 0, perc_dim)
        return result

    @staticmethod
    def blend_percentiles(perc_values, percentiles, weights):
        """ Blend percentiles function, to calculate the weighted blend across
            a given axis of percentile data for a single grid point.

        Args:
            perc_values : np.array
                   Array containing the percentile values to blend, with
                   shape: (length of coord to blend, num of percentiles)
            percentiles: np.array
                         Array of percentile values e.g
                         [0, 20.0, 50.0, 70.0, 100.0],
                         same size as the percentile dimension of data.
            weights: np.array
                     Array of weights, same size as the axis dimension of data,
                     that we will blend over.

        Returns:
            result : np.array
                     containing the weighted percentile blend data
                     across the chosen coord
        """
        # Find the size of the dimension we want to blend over.
        num = perc_values.shape[0]
        # Create an array to store the weighted blending pdf
        combined_pdf = np.zeros((num, len(percentiles)))
        # Loop over the axis we are blending over finding the values for the
        # probability at each threshold, in the pdf for each of the other
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


class WeightedBlend(object):
    """Apply a Weighted blend to a cube."""

    def __init__(self, coord, coord_adjust=None):
        """Set up for a Weighted Blending plugin

        Args:
            coord : string
                     The name of a coordinate dimension in the cube.
            coord_adjust : Function to apply to the coordinate after
                           collapsing the cube to correct the values,
                           for example for time windowing and
                           cycle averaging the follow function would
                           adjust the time coordinates.
            e.g. coord_adjust = lambda pnts: pnts[len(pnts)/2]
        """
        self.coord = coord
        self.coord_adjust = coord_adjust

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return (
            '<WeightedBlend: coord = {0:s}>').format(self.coord)

    def process(self, cube, weights=None):
        """Calculate weighted blend across the chosen coord, for either
           probabilistic or percentile data.

        Args:
            cube : iris.cube.Cube
                   Cube to blend across the coord.
            weights: Optional list or np.array of weights
                     or None (equivalent to equal weights).

        Returns:
            result : iris.cube.Cube
                     containing the weighted blend across the chosen coord.

        Raises:
            ValueError 1 : first argument not a cube.
            ValueError 2 : The coord is not a coord with the cube.
            ValueError 3 : If there is a percentile coord it is not a
                           dimension coord in the cube.
            ValueError 4 : If there are more than one percentile coords
                           in the cube.
            ValueError 5 : weights shape do not match the dimension
                           of the coord.
            Warning : coord not a dimension within the cube.

        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}.'.format(type(cube)))
            raise ValueError(msg)
        if not cube.coords(self.coord):
            msg = ('The coord for this plugin must be '
                   'an existing coordinate in the input cube.')
            raise ValueError(msg)

        # Check to see if the data is percentile data
        perc_coord = None
        perc_dim = None
        perc_found = 0
        for coord in cube.coords():
            if coord.name().find('percentile') >= 0:
                perc_found += 1
                perc_coord = coord
        if perc_found == 1:
            perc_dim = cube.coord_dims(perc_coord.name())
            if not perc_dim:
                msg = ('The percentile coord must be a dimension '
                       'of the cube.')
                raise ValueError(msg)
        elif perc_found > 1:
            msg = ('There should only be one percentile coord'
                   'on the cube.')
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
                   'value. Returning original cube')
            warnings.warn(msg)
            result = cube

        # Blend the cube across the coordinate
        # Use percentile Aggregator if required
        elif perc_coord is not None:
            percentiles = np.array(perc_coord.points, dtype=float)
            perc_dim, = cube.coord_dims(perc_coord.name())
            # Set equal weights if none are provided
            if weights is None:
                num = len(cube.coord(self.coord).points)
                weights = np.ones(num) / float(num)
            # Set up aggregator
            PERCENTILE_BLEND = (Aggregator('percentile_blend',
                                PercentileBlendingAggregator.aggregate))

            result = cube.collapsed(self.coord,
                                    PERCENTILE_BLEND,
                                    arr_percent=percentiles,
                                    arr_weights=weights,
                                    perc_dim=perc_dim)

        # Else do a simple weighted average
        else:
            # Equal weights are used as default.
            weights_array = None
            # Else broadcast the weights to be used by the aggregator.
            if weights is not None:
                weights_array = iris.util.broadcast_to_shape(np.array(weights),
                                                             cube.shape,
                                                             coord_dim)
            # Calculate the weighted average.
            result = cube.collapsed(self.coord,
                                    iris.analysis.MEAN, weights=weights_array)

        # If set adjust values of collapsed coordinates.
        if self.coord_adjust is not None:
            for crd in result.coords():
                if cube.coord_dims(crd.name()) == coord_dim:
                    pnts = cube.coord(crd.name()).points
                    crd.points = np.array(self.coord_adjust(pnts),
                                          dtype=crd.points.dtype)

        return result
