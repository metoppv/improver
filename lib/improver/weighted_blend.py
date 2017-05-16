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


class BasicWeightedAverage(object):
    """Apply a Basic Weighted Average to a cube.

    """

    def __init__(self, coord, coord_adjust=None):
        """Set up for processing an in-or-out of threshold binary field.

        Parameters
        ----------

        coord : string
            The name of a coordinate dimension in the cube

        coord_adjust :

        """
        self.coord = coord
        self.coord_adjust = coord_adjust

    def __str__(self):
        """Represent the configured plugin instance as a string."""
        return (
            '<BasicWeightedAverage: coord {0:s}>').format(self.coord)

    def process(self, cube, weights=None):
        """Convert each point to a fuzzy truth value based on threshold.

        Parameters
        ----------

        cube : iris.cube.Cube
            Cube to blend across the coord.

        weights: array of weights

        """
        if not isinstance(cube, iris.cube.Cube):
            raise ValueError('the first argument must be an instance of ' +
                             'iris.cube.Cube')
        if not cube.coords(self.coord):
            raise ValueError('the second argument must be ' +
                             'an existing coordinate in the input cube')
        collapse_dim = cube.coord_dims(self.coord)
        if not collapse_dim:
            cube = iris.util.new_axis(cube, self.coord)
            collapse_dim = cube.coord_dims(self.coord)
        if weights is not None:
            weights = iris.util.broadcast_to_shape(np.array(weights),
                                                   cube.shape, collapse_dim)
        result = cube.collapsed(coord, iris.analysis.MEAN, weights=weights)
        if self.coord_adjust is not None:
            # adjust values of collapsed coordinates
            for crd in result.coords():
                if cube.coord_dims(crd.name()) == collapse_dim:
                    pnts = cube.coord(crd.name()).points
                    crd.points = np.array(self.coord_adjust(pnts),
                                          dtype=crd.points.dtype)
        return result
