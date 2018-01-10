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
"""Module to contain statistical operations."""

import numpy as np
import iris
from iris.exceptions import CoordinateNotFoundError


class ProbabilitiesFromPercentiles2D(object):
    """
    Generate a 2-dimensional field of probabilties by interpolating a
    percentiled cube of data to required points.

    Examples:

        Given a reference field of values against a percentile coordinate, an
        interpolation is performed using another field of values of the same
        type (e.g. height). This returns the percentile with which these
        heights would be associated in the reference field. This effectively
        uses the field of values as a 2-dimensional set of thresholds, and the
        percentiles looked up correspond to the probabilities of these
        thresholds being reached.

        Snow-fall level::

            Reference field: Percentiled snow fall level (m ASL)
            Other field: Orography (m ASL)

            300m ----------------- 30th Percentile snow fall level
            200m ----_------------ 20th Percentile snow fall level
            100m ---/-\----------- 10th Percentile snow fall level
            000m --/---\----------  0th Percentile snow fall level
            ______/     \_________ Orogaphy

        The orography heights are compared against the heights that
        correspond with percentile values to find the band in which they
        fall; this diagram hides the 2-dimensional variability of the snow
        fall level. The percentile values are then interpolated to the
        height of the point being considered. This constructs a
        2-dimensional field of probabilities that snow will be falling at
        each point in the orography field.
    """

    def __init__(self, reference_cube, output_name, inverse_ordering=False):
        """
        Initialise class.

        Args:
            reference_cube (iris.cube.Cube):
                The percentiled field from which probabilities will be obtained
                using the input cube. This cube should contain a percentiles
                dimension, with fields of values that correspond to these
                percentiles. The cube passed to the process method will contain
                values of the same diagnostic (e.g. height) as this reference
                cube.
            output_name (str):
                The name of the cube being created,
                e.g.'probability_of_snowfall'.
        Keyword Args:
            inverse_ordering (bool):
                Set True if the percentiled data increases in the opposite
                sense to the percentile coordinate.
                e.g.  0th Percentile - Value = 10
                     10th Percentile - Value = 5
                     20th Percenitle - Value = 0
        """
        self.reference_cube = reference_cube
        self.output_name = output_name
        self.inverse_ordering = inverse_ordering

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ProbabilitiesFromPercentiles2D: reference_cube: {}, '
                  'output_name: {}, inverse_ordering: {}'.format(
                      self.reference_cube, self.output_name,
                      self.inverse_ordering))
        return result

    def create_probability_cube(self, cube):
        """
        Create a probability cube in which to store the calculated
        probabilities.

        Args:
            cube (iris.cube.Cube):
                The input cube to be used as a template.
        Returns:
            probability_cube (iris.cube.Cube):
                The new cube with suitable metadata.
        """
        cube_format = next(cube.slices([cube.coord(axis='y'),
                                        cube.coord(axis='x')]))
        probabilities = cube_format.copy(data=np.full(cube_format.shape, 1.,
                                                      dtype=float))
        percentile_coordinate, = [coord.name() for coord in
                                  cube_format.coords()
                                  if 'percentile' in coord.name()]
        if percentile_coordinate:
            probabilities.remove_coord(percentile_coordinate)
        probabilities.units = 1
        probabilities.rename(self.output_name)
        return probabilities

    def percentile_interpolation(self, cube, reference_cube):
        """
        Perform the percentile interpolation to construct the probability
        field.

        Args:
            cube (iris.cube.Cube):
                A cube of values, that effectively behave as thresholds, for
                which it is desired to obtain probability values from a
                percentiled reference cube.
        Returns:
            probabilities (iris.cube.Cube):
                A cube of probabilities obtained by interpolating the
                percentile values.
        """
        probabilities = self.create_probability_cube(reference_cube)

        array_shape = list(cube.shape)
        array_shape.insert(0, 2)
        array_shape = tuple(array_shape)
        percentile_bounds = np.full(array_shape, -1, dtype=float)
        height_bounds = np.full(array_shape, -1001., dtype=float)
        height_bounds[1] = -1.

        percentile_coordinate, = [coord for coord in reference_cube.coords()
                                  if 'percentiles' in coord.name()]
        percentiles = percentile_coordinate.points

        for index, pslice in enumerate(reference_cube.slices_over(
                percentile_coordinate)):
            indices = (cube.data < pslice.data if self.inverse_ordering else
                       cube.data > pslice.data)
            percentile_bounds[0, indices] = percentiles[index]
            height_bounds[0, indices] = pslice.data[indices]
            try:
                # Usual behaviour where the orography falls between heights
                # corresponding to percentiles.
                percentile_bounds[1, indices] = percentiles[index + 1]
                height_bounds[1, indices] = reference_cube[index+1].data[
                    indices]
            except IndexError:
                # Invoked if we have reached the top of the available heights.
                percentile_bounds[1, indices] = percentiles[index]
                height_bounds[1, indices] = pslice.data[indices]

        with np.errstate(divide='ignore'):
            interpolants, = ((cube.data - height_bounds[0]) /
                             np.diff(height_bounds, n=1, axis=0))

        with np.errstate(invalid='ignore'):
            probabilities.data, = (percentile_bounds[0] + interpolants *
                                   np.diff(percentile_bounds, n=1, axis=0))
        probabilities.data = probabilities.data/100.

        above_top_band = np.isinf(interpolants)
        below_bottom_band = height_bounds[0] < -1000
        probabilities.data[below_bottom_band] = 0.
        probabilities.data[above_top_band] = 1.

        return probabilities

    def process(self, cube):
        """
        Slice the cube over realization if present and call the percentile
        interpolation method.

        Args:
            cube (iris.cube.Cube):
                A cube of values, that effectively behave as thresholds, for
                which it is desired to obtain probability values from a
                percentiled reference cube.
        Returns:
            output_cubes (iris.cube.Cube):
                A cube of probabilities obtained by interpolating the
                percentile values.
        """
        try:
            self.reference_cube.coord_dims('realization')
        except CoordinateNotFoundError:
            cube_slices = [self.reference_cube]
        else:
            cube_slices = self.reference_cube.slices_over('realization')

        output_cubes = iris.cube.CubeList()
        for cube_slice in cube_slices:
            output_cube = self.percentile_interpolation(cube, cube_slice)
            output_cubes.append(output_cube)
        if len(output_cubes) > 1:
            output_cubes = output_cubes.merge_cube()
        else:
            output_cubes = output_cubes[0]

        return output_cubes
