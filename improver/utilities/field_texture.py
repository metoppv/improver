# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
""" Module containing plugin to calculate the texture in a field considering edge transitions."""

import numpy as np
import iris

from improver import BasePlugin
from improver.threshold import BasicThreshold
from improver.utilities.cube_manipulation import collapsed
from iris.exceptions import CoordinateNotFoundError
from improver.nbhood.square_kernel import SquareNeighbourhood
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS, MANDATORY_ATTRIBUTES
from improver.metadata.utilities import create_new_diagnostic_cube


class FieldTexture(BasePlugin):
    """Plugin to calculate the texture in a binary field considering edge transitions

    Code methodology:

    1) Takes a binary field that has been thresholded at a cloud cover fraction 
       of interest and looks for the transitions/edges in the field that mark 
       out a transition from cloudy to non-cloudy grid cell. 
    2) The transition calculation is then fed into the neighbourhooding code 
       (_calculate_ratio) to calculate a ratio for each cell.
    3) The new cube of ratios is then thresholded and the realization coordinates
       are collapsed to generate a mean of the thresholded ratios.
    """

    def __init__(self, nbhood_radius, ratio_threshold):
        """
        The class is called with the default constraints for the processing code.

        Args:
            nbhood_radius (float):
                A neighbourhood radius of sufficient size to capture the region
                and all actual transitions, in metres.

            ratio_threshold (float):
                A threshold to re-normalise values about a sensible value.

        """
        self.nbhood_radius = nbhood_radius
        self.ratio_threshold = ratio_threshold

    def _calculate_ratio(self, cube, radius):
        """
        Calculates the ratio of each cell using neighbourhooding.

        Args:
            cube (cube):
                Input data in cube format containing a single realization.

            radius (float):
                Average radius for neighbourhood in metres.

        Returns:
            ratio (cube):
                A ratio between 0 and 1 of actual transitions over potential transitions.
        """

        data = self._calculate_transitions(cube.data)
        result_cube = create_new_diagnostic_cube('cell_edge_differences', 1, cube, MANDATORY_ATTRIBUTE_DEFAULTS, data=data)
        potential_transitions = SquareNeighbourhood(sum_or_fraction='sum').run(cube, radius=radius)
        potential_transitions.data = np.where(cube.data > 0, potential_transitions.data * 4, 0)

        # Note that where there is no cloud the value is forced to 1. So high values
        # correspond to scattered cloud or no cloud.

        actual_transitions = SquareNeighbourhood(sum_or_fraction='sum').run(result_cube, radius=radius)
        actual_transitions.data = np.where(cube.data > 0, actual_transitions.data, 0)
        ratio = actual_transitions.copy(data=np.ones_like(actual_transitions.data))
        np.divide(actual_transitions.data, potential_transitions.data, out=ratio.data, where=(potential_transitions.data > 0))
        return ratio

    def _calculate_transitions(self, data):
        padded_data = np.pad(data, 1, mode='edge')
        diff_x = np.abs(np.diff(padded_data, axis=1))
        diff_y = np.abs(np.diff(padded_data, axis=0))
        cell_sum_x = diff_x[:, 0:-1] + diff_x[:, 1:]
        cell_sum_y = diff_y[0:-1, :] + diff_y[1:, :]
        cell_sum = cell_sum_x[1:-1, :] + cell_sum_y[:, 1:-1]
        cell_sum = np.where(data > 0, cell_sum, 0)
        print(cell_sum)
        return cell_sum

    def _calculate_clumpiness(self, input_cube):
        """
        Calculates a field of cloud texture to use in differentiating solid and
        more scattered cloud.

        Args:
            input_cube (cube):
                Input data in cube format with multiple-realizations

        Returns:
            clumpiness (cube):
                A cube of binary data, where 1 represents sunlight and 0
                represents cloud
        """
        ratios = iris.cube.CubeList()

        try:
            cslices = input_cube.slices_over('realization')
        except CoordinateNotFoundError:
            cslices = [input_cube]
        for cslice in cslices:
            ratios.append(self._calculate_ratio(cslice, self.nbhood_radius))
        ratios = ratios.merge_cube()
        thresholded = BasicThreshold(self.ratio_threshold).process(ratios)
        thresholded = iris.util.squeeze(collapsed(thresholded, 'realization', iris.analysis.MEAN))
        return thresholded

    def process(self, input_cube):
        """ 
        Calculates the ratio of actual to potential transitions over a binary
        field and then produces threshold texture outputs.

        Args:
            input_cube (cube):
                Input data in cube formate with multiple-realizations.

        Returns:
            threshold (cube)
                Mean of the thresholded ratios in cube format.
        """

        thresholded = self._calculate_clumpiness(input_cube)
        return thresholded
