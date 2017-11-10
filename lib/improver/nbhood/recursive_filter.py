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
"""Module to apply a recursive filter to neighbourhooded data."""

import iris
import numpy as np

from improver.nbhood.square_kernel import SquareNeighbourhood


class RecursiveFilter(object):

    """
    Apply a recursive filter to the input cube.
    """

    def __init__(self, alpha_x=None, alpha_y=None, iterations=5, edge_width=1):
        """
        Initialise the class.

        Args:

        """
        if alpha_x is not None:
            self.alpha_x = alpha_x
        if alpha_y is not None:
            self.alpha_y = alpha_y
        self.iterations = iterations
        self.edge_width = edge_width


    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<RecursiveFilter: alpha_x: {}, alpha_y: {}, iterations: {},'
                  ' edge_width: {}')
        return result.format(self.alpha_x, self.alpha_y, self.iterations,
                             self.edge_width)

    @staticmethod
    def recurse_forward_x(grid, alphas):
        nx, ny = grid.shape
        for i in range(1, nx):
            grid[i, :] = ((1. - alphas[i, :]) * grid[i, :] +
                          alphas[i, :] * grid[i-1, :])
        return grid

    @staticmethod
    def recurse_backwards_x(grid, alphas):
        nx, ny = grid.shape
        for i in range(nx-2, -1, -1):
            grid[i, :] = ((1. - alphas[i, :]) * grid[i, :] +
                          alphas[i, :] * grid[i+1, :])
        return grid

    @staticmethod
    def recurse_forward_y(grid, alphas):
        nx, ny = grid.shape
        for i in range(1, ny):
            grid[:, i] = ((1. - alphas[:, i]) * grid[:, i] +
                          alphas[:, i] * grid[:, i-1])
        return grid

    @staticmethod
    def recurse_backwards_y(grid, alphas):
        nx, ny = grid.shape
        for i in range(ny-2, -1, -1):
            grid[:, i] = ((1. - alphas[:, i]) * grid[:, i] +
                          alphas[:, i] * grid[:, i+1])
        return grid

    @staticmethod
    def run_recursion(cube, alphas_x, alphas_y, iterations):
        output = cube.data
        for i in range(iterations):
            output = RecursiveFilter.recurse_forward_x(output, alphas_x.data)
            output = RecursiveFilter.recurse_backwards_x(output, alphas_x.data)
            output = RecursiveFilter.recurse_forward_y(output, alphas_y.data)
            output = RecursiveFilter.recurse_backwards_y(output, alphas_y.data)

        cube.data = output
        return cube



    def process(self, cube, alphas_x=None, alphas_y=None):
        """
        Set up the alpha parameters and run the recursive filter.

        """
        padded_cube = SquareNeighbourhood().pad_cube_with_halo(
            cube, self.edge_width, self.edge_width)

        if alphas_x is None:
            alphas_x = padded_cube.copy(data=np.ones(padded_cube.data.shape) *
                                        self.alpha_x)
        else:
            alphas_x = SquareNeighbourhood().pad_cube_with_halo(
                alphas_x, self.edge_width, self.edge_width)

        if alphas_y is None:
            alphas_y = padded_cube.copy(data=np.ones(padded_cube.data.shape) *
                                        self.alpha_y)
        else:
            alphas_y = SquareNeighbourhood().pad_cube_with_halo(
                alphas_y, self.edge_width, self.edge_width)

        cube = self.run_recursion(padded_cube, alphas_x, alphas_y,
                                  self.iterations)
        cube = SquareNeighbourhood().remove_halo_from_cube(
            cube, self.edge_width, self.edge_width)

        return cube
