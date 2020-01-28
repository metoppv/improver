# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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

import warnings

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin
from improver.metadata.probabilistic import find_percentile_coordinate
from improver.utilities.cube_checker import check_cube_coordinates


class ProbabilitiesFromPercentiles2D(BasePlugin):
    r"""
    Generate a 2-dimensional field of probabilities by interpolating a
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

    def __init__(self, percentiles_cube, output_name):
        """
        Initialise class. Sets an inverse_ordering (bool) switch to true for
        cases where the percentiled data increases in the opposite sense to the
        percentile coordinate:

                e.g.  0th Percentile - Value = 10
                     10th Percentile - Value = 5
                     20th Percentile - Value = 0

        Args:
            percentiles_cube (iris.cube.Cube):
                The percentiled field from which probabilities will be obtained
                using the input cube. This cube should contain a percentiles
                dimension, with fields of values that correspond to these
                percentiles. The cube passed to the process method will contain
                values of the same diagnostic (e.g. height) as this reference
                cube.
            output_name (str):
                The name of the cube being created,
                e.g.'probability_of_snow_falling_level_below_ground_level'
        """
        self.percentile_coordinate = find_percentile_coordinate(
            percentiles_cube)
        if self.percentile_coordinate.points.shape[0] < 2:
            msg = ("Percentile coordinate has only one value. Interpolation "
                   "using ProbabilitiesFromPercentiles2D requires multiple "
                   "values are provided.")
            raise ValueError(msg)
        self.percentiles_cube = percentiles_cube
        self.output_name = output_name

        # Set inverse_ordering switch
        percentile_slices = percentiles_cube.slices_over(
            self.percentile_coordinate)
        self.inverse_ordering = False
        first_percentile = next(percentile_slices).data
        for percentile_values in percentile_slices:
            last_percentile = percentile_values.data
        if (first_percentile - last_percentile >= 0).all():
            self.inverse_ordering = True

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ProbabilitiesFromPercentiles2D: percentiles_cube: {}, '
                  'output_name: {}, inverse_ordering: {}'.format(
                      self.percentiles_cube, self.output_name,
                      self.inverse_ordering))
        return result

    def create_probability_cube(self, cube, threshold_cube):
        """
        Create a 2-dimensional probability cube in which to store the
        calculated probabilities.

        Args:
            cube (iris.cube.Cube):
                Template for the output probability cube. This is a slice
                created in process, containing a percentile coordinate as well
                as x and y coordinates. We keep all the metadata from this cube
                but dispose of the percentile coordinate as we will be filling
                the cube with probabilities.
            threshold_cube (iris.cube.Cube):
                A 2-dimensional cube of "threshold" values containing metadata
                required to construct a probability cube.

        Returns:
            iris.cube.Cube:
                A new 2-dimensional probability cube with suitable metadata.
        """
        cube_format = next(cube.slices([cube.coord(axis='y'),
                                        cube.coord(axis='x')]))
        probabilities = cube_format.copy(data=np.full(cube_format.shape,
                                                      np.nan, dtype=float))
        try:
            probabilities.remove_coord(self.percentile_coordinate)
        except CoordinateNotFoundError:
            pass

        probabilities.units = 1
        probabilities.rename(self.output_name)
        probabilities.attributes['thresholded_using'] = threshold_cube.name()
        probabilities.attributes['relative_to_threshold'] = 'below'
        if self.inverse_ordering is True:
            probabilities.attributes['relative_to_threshold'] = 'above'
        return probabilities

    def percentile_interpolation(self, threshold_cube, percentiles_cube):
        """
        Using a percentiles_cube containing a distinct percentile distribution
        for each point on a 2-dimensional grid, we can interpolate through each
        distribution to obtain a probability. The point to which we interpolate
        is defined by the threshold_cube.
        Note that the current implementation assumes that in cases of a
        degenerate percentile distribution, the right most bin in which a
        threshold value is found is chosen.

        e.g.
        ::

            Percentile: 0 10 20 30 40 50 ...
            Height (m): 0 0 0 15 30 40 ...

        A height of 0m will be associated with a probabilty of 20%. This is
        not correct, but nor is the approach of taking 0%. The percentile
        approach is not suitable with these degenerate distributions, so be
        wary of the returned probabilities.

        Examples:
            This simple linear interpolator works in the following way.

            percentiles_cube::

                [ [[2.0, 2.0, 2.0],
                   [2.0, 2.0, 2.0],
                   [2.0, 2.0, 2.0]],

                  [[4.0, 4.0, 4.0],
                   [4.0, 4.0, 4.0],
                   [4.0, 4.0, 4.0]] ]

            threshold_cube::

                [ [1.0, 1.0, 1.0],
                  [3.0, 3.0, 3.0],
                  [5.0, 5.0, 5.0] ]

            value_bounds::

                [ [[np.nan, np.nan, np.nan],
                   [np.nan, np.nan, np.nan],
                   [np.nan, np.nan, np.nan]],

                  [[np.nan, np.nan, np.nan],
                   [np.nan, np.nan, np.nan],
                   [np.nan, np.nan, np.nan]] ]

            percentile_bounds::

                [ [[-1, -1, -1],
                   [-1, -1, -1],
                   [-1, -1, -1]],

                  [[-1, -1, -1],
                   [-1, -1, -1],
                   [-1, -1, -1]] ]

            1. Create slices over each percentile, and using the correct
               inequality (as determined by inverse_ordering) compare the
               threshold values to the percentiles slice; here we assume
               inverse_ordering is False, so we use >=. We then populate the
               value_bounds and percentile_bounds arrays.

               Slice 0 - 0th Percentile::

                   [[1.0 >= 2.0, 1.0 >= 2.0, 1.0 >= 2.0],
                    [3.0 >= 2.0, 3.0 >= 2.0, 3.0 >= 2.0],
                    [5.0 >= 2.0, 5.0 >= 2.0, 5.0 >= 2.0]]

                   [[False, False, False],
                    [True, True, True],
                    [True, True, True]]

               The value_bounds array has a leading dimensions with 2 indices
               to be associated with the lower [0] and upper bounds [1] about
               the threshold being considered. The [0] index is populated with
               the values in the slice of percentiles_cube at every True index.
               The [1] index is populated with the values in the next slice of
               percentiles_cube.
               ::

                   [ [[np.nan, np.nan, np.nan],
                      [2.0, 2.0, 2.0],
                      [2.0, 2.0, 2.0]],

                     [[np.nan, np.nan, np.nan],
                      [4.0, 4.0, 4.0],
                      [4.0, 4.0, 4.0]] ]

               The percentile_bounds array is also contains a leading dimension
               associated with lower and upper bounds about the thresholds. The
               lower bound array is populated at every True index with the
               current percentile value (0 in this first slice), whilst the
               upper bound array takes the percentile value from the next
               slice.
               ::

                   [ [[-1, -1, -1],
                      [0, 0, 0],
                      [0, 0, 0]],

                     [[-1, -1, -1],
                      [50, 50, 50],
                      [50, 50, 50]] ]

               After the same process is applied to the next slice, the 50th
               percentile, we end up with value_bounds::

                   [ [[np.nan, np.nan, np.nan],
                      [2.0, 2.0, 2.0],
                      [4.0, 4.0, 4.0]],

                     [[np.nan, np.nan, np.nan],
                      [4.0, 4.0, 4.0],
                      [4.0, 4.0, 4.0]] ]

               And percentile bounds::

                   [ [[-1, -1, -1],
                      [0, 0, 0],
                      [50, 50, 50]],

                     [[-1, -1, -1],
                      [50, 50, 50],
                      [50, 50, 50]] ]

               Note that where there is no availble +1 index in the
               percentiles_cube the upper bound is set to be the same as the
               lower_bound.

            2. When all slices have been interated over, the interpolants are
               calculated using the threshold values and the values_bounds.
               ::

                   (threshold_cube.data - lower_bound) /
                   (upper_bound - lower_bound)

               If the upper_bound and lower_bound are the same this leads to
               a divide by 0 calculation, resulting in np.inf as the output.

            3. The interpolants are used to calculate the percentile value at
               each point in the array using the percentile_bounds.
               ::

                   lower_percentile_bound + interpolants *
                   (upper_percentile_bounds - lower_percentile_bounds)

               The percentiles are divided by 100 to give a fractional
               probability.

            5. Any probabilities that are calculated to be np.inf indicate that
               the associated point has a threshold value that is above
               the top percentile band. These points are given a probability
               value of 1.

            4. Any points for which the calculated probability is np.nan had
               threshold values that were never found to fall within a
               percentile band, and so must be below the lowest band. These
               points are given a probability value of 0.

        Args:
            threshold_cube (iris.cube.Cube):
                A 2-dimensional cube of "threshold" values for which it is
                desired to obtain probability values from the percentiled
                reference cube. This cube should have the same x and y
                dimensions as percentiles_cube.
            percentiles_cube (iris.cube.Cube):
                A 3-dimensional cube, 1 dimension describing the percentile
                distributions, and 2-dimensions shared with the threshold_cube,
                typically x and y.

        Returns:
            iris.cube.Cube:
                A 2-dimensional cube of probabilities obtained by interpolating
                between percentile values.

        """
        percentiles = self.percentile_coordinate.points
        probabilities = self.create_probability_cube(percentiles_cube,
                                                     threshold_cube)

        # Create array with additional 2 dimensions to contain upper and lower
        # bounds.
        array_shape = [2] + list(threshold_cube.shape)
        percentile_bounds = np.full(array_shape, -1, dtype=np.float32)
        value_bounds = np.full(array_shape, np.nan, dtype=np.float32)

        for index, pslice in enumerate(percentiles_cube.slices_over(
                self.percentile_coordinate)):
            # Change to use < & > to force degenerate percentile distributions
            # to use the first percentile band that the threshold falls within.
            indices = (threshold_cube.data <= pslice.data
                       if self.inverse_ordering else
                       threshold_cube.data >= pslice.data)
            percentile_bounds[0, indices] = percentiles[index]
            value_bounds[0, indices] = pslice.data[indices]
            try:
                # Usual behaviour where the threshold value falls between
                # values corresponding to percentiles.
                percentile_bounds[1, indices] = percentiles[index + 1]
                value_bounds[1, indices] = percentiles_cube[index+1].data[
                    indices]
            except IndexError:
                # Invoked if we have reached the top of the available values.
                percentile_bounds[1, indices] = percentiles[index]
                value_bounds[1, indices] = pslice.data[indices]

        with np.errstate(divide='ignore', invalid='ignore'):
            numerator = (threshold_cube.data - value_bounds[0])
            denominator = np.diff(value_bounds, n=1, axis=0)[0]
            interpolants = numerator/denominator
            interpolants[denominator == 0] = np.inf

        with np.errstate(invalid='ignore'):
            probabilities.data, = (percentile_bounds[0] + interpolants *
                                   np.diff(percentile_bounds, n=1, axis=0))
        probabilities.data = probabilities.data/np.float32(100.)

        above_top_band = np.isinf(interpolants)
        below_bottom_band = np.isnan(value_bounds[0])
        probabilities.data[below_bottom_band] = 0.
        probabilities.data[above_top_band] = 1.

        return probabilities

    def process(self, threshold_cube):
        """
        Slice the percentiles cube over any non-spatial coordinates
        (realization, time, etc) if present, and call the percentile
        interpolation method for each resulting cube.

        Args:
            threshold_cube (iris.cube.Cube):
                A cube of values, that effectively behave as thresholds, for
                which it is desired to obtain probability values from a
                percentiled reference cube.
        Returns:
            iris.cube.Cube:
                A cube of probabilities obtained by interpolating between
                percentile values at the "threshold" level.
        """
        cube_slices = self.percentiles_cube.slices(
            [self.percentile_coordinate, self.percentiles_cube.coord(axis='y'),
             self.percentiles_cube.coord(axis='x')])

        if threshold_cube.ndim != 2:
            msg = ('threshold cube has too many ({} > 2) dimensions - slicing '
                   'to x-y grid'.format(threshold_cube.ndim))
            warnings.warn(msg)
            threshold_cube = next(threshold_cube.slices([
                threshold_cube.coord(axis='y'),
                threshold_cube.coord(axis='x')]))

        if threshold_cube.units != self.percentiles_cube.units:
            threshold_cube.convert_units(self.percentiles_cube.units)

        output_cubes = iris.cube.CubeList()
        for cube_slice in cube_slices:
            output_cube = self.percentile_interpolation(threshold_cube,
                                                        cube_slice)
            output_cubes.append(output_cube)

        probability_cube = output_cubes.merge_cube()

        reference_cube = next(self.percentiles_cube.slices_over(
            self.percentile_coordinate))

        probability_cube = check_cube_coordinates(reference_cube,
                                                  probability_cube)
        return probability_cube
