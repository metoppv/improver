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
"""Plugin to calculate probabilities of occurrence between specified thresholds
"""

import iris
import numpy as np

from iris.exceptions import CoordinateNotFoundError

from improver.utilities.cube_checker import find_threshold_coordinate
from improver.utilities.cube_metadata import extract_diagnostic_name


class OccurrenceBetweenThresholds(object):
    """Calculate the probability of occurrence between thresholds"""

    def __init__(self, threshold_ranges):
        """
        Initialise the class

        Args:
            threshold_ranges (list):
                List of 2-item iterables specifying thresholds between which
                probabilities should be calculated
        """
        self.threshold_ranges = threshold_ranges

    @staticmethod
    def _get_multiplier(thresh_coord):
        """
        Check whether the cube contains "above" or "below" threshold
        probabilities.  For "above", the probability of occurrence between
        thresholds is the difference between probabilities at the higher
        and lower thresholds: P(higher) - P(lower).  For "below" it is the
        inverse of this: P(lower) - P(higher), which is implemented by
        multiplying the difference by -1.

        Args:
            thresh_coord (iris.coords.DimCoord):
                Threshold-type coordinate from the input cube

        Returns:
            multiplier (float):
                1. or -1.

        Raises:
            ValueError: If the spp__relative_to_threshold attribute is
                not recognised
        """
        if thresh_coord.attributes['spp__relative_to_threshold'] == 'above':
            multiplier = 1.
        elif thresh_coord.attributes['spp__relative_to_threshold'] == 'below':
            multiplier = -1.
        else:
            raise ValueError('Input cube must contain probabilities of '
                             'occurrence above or below threshold')
        return multiplier

    def _slice_cube(self, cube):
        """
        Extract required slices from input cube

        Args:
            cube (iris.cube.Cube):
                Input cube

        Returns:
            cubes (list):
                List of 2-item lists containing lower and upper
                threshold cubes
        """
        thresh_coord = find_threshold_coordinate(cube)
        cubes = []
        for t_range in self.threshold_ranges:
            t_range.sort()
            lower_constraint = iris.Constraint(coord_values={
                thresh_coord: lambda t: np.isclose(t.point, t_range[0])})
            lower_cube = cube.extract(lower_constraint)
            upper_constraint = iris.Constraint(coord_values={
                thresh_coord: lambda t: np.isclose(t.point, t_range[1])})
            upper_cube = cube.extract(upper_constraint)
            cubes.append([lower_cube, upper_cube])

        return cubes

    def process(self, cube):
        """
        Calculate probabilities between thresholds for the input cube

        Args:
            cube (iris.cube.Cube):
                Probability cube containing thresholded data (above or below)

        Returns:
            output_cube (iris.cube.Cube):
                Cube containing probability of occurrence between thresholds
        """
        # if cube has no threshold-type coordinate, raise an error
        try:
            thresh_coord = find_threshold_coordinate(cube)
        except CoordinateNotFoundError:
            raise ValueError('Input cube has no threshold-type coordinate')

        # if cube contains below threshold probabilities, need to multiply
        # difference by -1
        multiplier = self._get_multiplier(thresh_coord)

        # extract suitable cube slices
        cube_slices = self._slice_cube(cube)

        # generate "between thresholds" fields
        cubelist = iris.cube.CubeList([])
        for (lower_cube, upper_cube) in cube_slices:
            # construct difference cube
            between_thresholds_data = (
                upper_cube.data-lower_cube.data)*multiplier
            between_thresholds_cube = upper_cube.copy(between_thresholds_data)

            # add threshold coordinate bounds
            lower_threshold = lower_cube.coord(thresh_coord.name()).points[0]
            upper_threshold = upper_cube.coord(thresh_coord.name()).points[0]
            between_thresholds_cube.coord(thresh_coord.name()).bounds = (
                [lower_threshold, upper_threshold])

            cubelist.append(between_thresholds_cube)

        output_cube = cubelist.merge_cube()
        output_cube.rename(
            'probability_of_{}_between_thresholds'.format(
                extract_diagnostic_name(cube)))

        return output_cube
