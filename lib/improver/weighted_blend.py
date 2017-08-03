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


class BlendingUtilities(object):
    """Class for blending utilities functions"""

    def __init__(self):
        """
        Initialise class.
        """
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<BlendingUtilities>')
        return result

    @staticmethod
    def basic_weighted_average(cube, weights, coord, coord_dim):
        """Calculate weighted mean across the chosen coord

        Args:
            cube : iris.cube.Cube
                   Cube to blend across the coord.
            weights: Optional list or np.array of weights
                     or None (equivalent to equal weights)
            coord : string
                     The name of a coordinate dimension in the cube.
            coord_dim : tuple
                     The index of the coordinate dimension in the cube.

        Returns:
            result : iris.cube.Cube
                     containing the weighted mean across the chosen coord
        """
        # Supply weights as an array of weights whose shape matches the cube.
        weights_array = None
        if weights is not None:
            weights_array = iris.util.broadcast_to_shape(np.array(weights),
                                                         cube.shape,
                                                         coord_dim)
        # Calculate the weighted average.
        result = cube.collapsed(coord,
                                iris.analysis.MEAN, weights=weights_array)
        return result

    @staticmethod
    def blend_percentile_cube(cube, weights, coord, coord_dim, perc_coord):
        """ Blend together percentile cube

        Args:
            cube : iris.cube.Cube
                   Cube to blend across the coord.
            weights: Optional list or np.array of weights
                     or None (equivalent to equal weights)
            coord : string
                     The name of a coordinate dimension in the cube.
            coord_dim : tuple
                     The index of the coordinate dimension in the cube.
            perc_coord : iris.cube.DimCoord
                     The perecentile coordinate

        Returns:
            result : iris.cube.Cube
                     containing the weighted percentile blend
                     across the chosen coord
        """

        percentiles = np.array(perc_coord.points, dtype=float)
        num = cube.data.shape[coord_dim[0]]
        if weights is None:
            weights = np.ones(num) / float(num)
        PERCENTILE_BLEND = (Aggregator('percentile_blend',
                            BlendingUtilities.blend_percentile_aggregate))
        perc_dim, = cube.coord_dims(perc_coord.name())

        result = cube.collapsed(coord,
                                PERCENTILE_BLEND,
                                arr_percent=percentiles,
                                arr_weights=weights,
                                perc_dim=perc_dim)
        return result

    @staticmethod
    def blend_percentile_aggregate(data, axis,
                                   arr_percent, arr_weights, perc_dim):
        """ Blend percentile aggregate function

        Args:
            data : np.array
                   Array containing the data to blend
            axis : integer
                   The index of the coordinate dimension in the cube.
            arr_percent: np.array
                     Array of percentile values e.g
                     [0, 20.0, 50.0, 70.0, 100.0],
                     same size as the percentile dimension of data.
            arr_weights: np.array
                     Array of weights, same size as the axis dimension of data.
            perc_dim : integer
                     The index of the perecentile coordinate
            (Note percent and weights have special meaning in Aggregator
             hence the rename.)

        Returns:
            result : np.array
                     containing the weighted percentile blend data
                     across the chosen coord
        """
        if axis < 0:
            axis += data.ndim
        # Firstly ensure axis coordinate and percentile coordinate
        # are indexed as the first and second values in the data array

        data = np.moveaxis(data, [perc_dim, axis], [1, 0])

        # Determine the rest of the shape
        shape = data.shape[2:]
        result = None
        if shape:
            input_shape = [data.shape[0],
                           data.shape[1],
                           np.prod(shape)]
            # Flatten the data that is not percentile or coord data
            data = data.reshape(input_shape)
            # Create the resulting data array
            result = np.zeros(input_shape[1:])
            # Loop over the flatten data
            for i in range(data.shape[-1]):
                result[:, i] = (
                    BlendingUtilities.blend_percentiles(data[:, :, i],
                                                        arr_percent,
                                                        arr_weights))
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
        """ Blend percentiles function

        Args:
            perc_values : np.array
                   Array containing the percentile values to blend
                   shape (length of coord to blend, num of percentiles)
            percentiles: np.array
                         Array of percentile values e.g
                         [0, 20.0, 50.0, 70.0, 100.0],
                         same size as the percentile dimension of data.
            weights: np.array
                     Array of weights, same size as the axis dimension of data.
            perc_dim : integer
                     The index of the perecentile coordinate

        Returns:
            result : np.array
                     containing the weighted percentile blend data
                     across the chosen coord
        """
        num = perc_values.shape[0]
        recalc_values_in_pdf = np.zeros((num, num, len(percentiles)))
        for i in range(0, num):
            for j in range(0, num):
                if i == j:
                    recalc_values_in_pdf[i][j] = percentiles
                else:
                    recalc_values_in_pdf[i][j] = np.interp(perc_values[i],
                                                           perc_values[j],
                                                           percentiles)

        combined_pdf = np.zeros((num, len(percentiles)))
        for i in range(0, num):
            for j in range(0, num):
                combined_pdf[i] += recalc_values_in_pdf[i][j]*weights[j]

        # Combine and sort model1 and model 2 threshold values.
        combined_perc_thres_data = np.sort(perc_values.flatten())

        # Combine and sort blended probability values.
        combined_perc_values = np.sort(combined_pdf.flatten())

        # Find the percentile values from this combined data.
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
        """Calculate weighted blend across the chosen coord

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

        # Find the coords dimension.
        # If coord is a scalar_coord try adding it.
        collapse_dim = cube.coord_dims(self.coord)
        if not collapse_dim:
            msg = ('Could not find collapse dimension, '
                   'will try adding it')
            warnings.warn(msg)
            cube = iris.util.new_axis(cube, self.coord)
            collapse_dim = cube.coord_dims(self.coord)

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

        # Blend the cube across the coordinate
        if perc_coord is not None:
            result = BlendingUtilities.blend_percentile_cube(cube,
                                                             weights,
                                                             self.coord,
                                                             collapse_dim,
                                                             perc_coord)
        else:
            result = BlendingUtilities.basic_weighted_average(cube,
                                                              weights,
                                                              self.coord,
                                                              collapse_dim)

        # If set adjust values of collapsed coordinates.
        if self.coord_adjust is not None:
            for crd in result.coords():
                if cube.coord_dims(crd.name()) == collapse_dim:
                    pnts = cube.coord(crd.name()).points
                    crd.points = np.array(self.coord_adjust(pnts),
                                          dtype=crd.points.dtype)

        return result
