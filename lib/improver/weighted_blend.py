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


import iris
import numpy as np


class BasicWeightedAverage(object):
    """Apply a Basic Weighted Average to a cube.

    """

    def __init__(self, coord, coord_adjust=None):
        """Set up for a Basic Weighted Average Blending plugin

        Args:
            coord : string
                     The name/s of a coordinate dimension/s in the cube

            coord_adjust : Function to apply to the coordinate after
                           collapsing the cube to correct the values
                           for example for time windowing and
                           cycle averaging the follow function would
                           adjust the time coordinates
            e.g. coord_adjust = lambda pnts: pnts[len(pnts)/2]
        """
        self.coord = coord
        self.coord_adjust = coord_adjust

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return (
            '<BasicWeightedAverage: coord = {0:s}>').format(self.coord)

    def process(self, cube, weights=None):
        """Calculated weighted mean across the chosen coord

        Args:
            cube : iris.cube.Cube
                   Cube to blend across the coord.

            weights: Optional list or np.array of weights
                     or None (equivalent to equal weights)

        Returns:
            result : iris.cube.Cube

        """
        if not isinstance(cube, iris.cube.Cube):
            raise ValueError('the first argument must be an instance of ' +
                             'iris.cube.Cube')
        if not cube.coords(self.coord):
            raise ValueError('the coord for this plugin must be ' +
                             'an existing coordinate in the input cube')
        # Find the coords dimension.
        # If coord is a scalar_coord try adding it
        collapse_dim = cube.coord_dims(self.coord)
        if not collapse_dim:
            print 'Warning: Could not find collapse dimension ' + \
                'will try adding it'
            cube = iris.util.new_axis(cube, self.coord)
            collapse_dim = cube.coord_dims(self.coord)
        # supply weights as an array of weights whose shape matches the cube
        weights_array = None
        if weights is not None:
            if np.array(weights).shape != cube.coord(self.coord).points.shape:
                raise ValueError('the weights array must match the shape ' +
                                 'of the coordinate in the input cube')
            weights_array = iris.util.broadcast_to_shape(np.array(weights),
                                                         cube.shape,
                                                         collapse_dim)
        # Calculate the weighted average
        result = cube.collapsed(self.coord,
                                iris.analysis.MEAN, weights=weights_array)
        # if set adjust values of collapsed coordinates
        if self.coord_adjust is not None:
            for crd in result.coords():
                if cube.coord_dims(crd.name()) == collapse_dim:
                    pnts = cube.coord(crd.name()).points
                    crd.points = np.array(self.coord_adjust(pnts),
                                          dtype=crd.points.dtype)

        return result
