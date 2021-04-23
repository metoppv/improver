# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
""" Module containing plugin to calculate whether or not the input field texture
    exceeds a given threshold."""

from typing import Optional

import iris
import numpy as np
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.nbhood.square_kernel import SquareNeighbourhood
from improver.threshold import BasicThreshold
from improver.utilities.cube_manipulation import collapse_realizations


class FieldTexture(BasePlugin):
    """Plugin to calculate whether or not the input field texture exceeds a
       given threshold.

    1) Takes a binary field that has been thresholded and looks for the transitions/edges
       in the field that mark out a transition.
    2) The transition calculation is then fed into the neighbourhooding code
       (_calculate_ratio) to calculate a ratio for each cell. This is the texture
       of the input field.
    3) The new cube of ratios is then thresholded and the realization coordinates
       are collapsed to generate a mean of the thresholded ratios. This gives a binary
       indication of whether a field is rough (texture values close to 1) or smooth
       (texture values close to zero).
    """

    def __init__(
        self,
        nbhood_radius: float,
        textural_threshold: float,
        diagnostic_threshold: float,
        model_id_attr: Optional[str] = None,
    ) -> None:
        """
        Args:
            nbhood_radius:
                The neighbourhood radius in metres within which the number of potential
                transitions should be calculated. This forms the denominator in the
                calculation of the ratio of actual to potential transitions that indicates a
                field's texture. A larger radius should be used for diagnosing larger-scale
                textural features.

            textural_threshold:
                A unit-less threshold value that defines the ratio value above which
                the field is considered rough and below which the field is considered
                smoother.

            diagnostic_threshold:
                A user defined threshold value related either to cloud or precipitation,
                used to extract the corresponding dimensional cube with assumed units of 1.

            model_id_attr:
                Name of the attribute used to identify the source model for
                blending.
        """
        self.nbhood_radius = nbhood_radius
        self.textural_threshold = textural_threshold
        self.diagnostic_threshold = diagnostic_threshold
        self.model_id_attr = model_id_attr

    def _calculate_ratio(self, cube: Cube, cube_name: str, radius: float) -> Cube:
        """
        Calculates the ratio of actual to potential value transitions in a
        neighbourhood about each cell.

        The process is as follows:

            1. For each grid cell find the number of cells of value 1 in a surrounding
               neighbourhood of a size defined by the arg radius. The potential
               transitions within that neighbourhood are defined as the number of
               orthogonal neighbours (up, down, left, right) about cells of value 1.
               This is 4 times the number of cells of value 1.
            2. Calculate the number of actual transitions within the neighbourhood,
               that is the number of cells of value 0 that orthogonally abut cells
               of value 1.
            3. Calculate the ratio of actual to potential transitions.

        Ratios approaching 1 indicate that there are many transitions, so the field
        is highly textured (rough). Ratios close to 0 indicate a smoother field.

        A neighbourhood full of cells of value 1 will return ratios of 0; the
        diagnostic that has been thresholded to produce the binary field is found
        everywhere within that neighbourhood, giving a smooth field. At the other
        extreme, in neighbourhoods in which there are no cells of value 1 the ratio
        is set to 1.

        Args:
            cube:
                Input data in cube format containing a two-dimensional field
                of binary data.
            cube_name:
                Name of input data cube, used for determining output texture cube name.
            radius:
                Radius for neighbourhood in metres.

        Returns:
            A ratio between 0 and 1 of actual transitions over potential transitions.
        """
        # Calculate the potential transitions within neighbourhoods.
        potential_transitions = SquareNeighbourhood(sum_or_fraction="sum").run(
            cube, radius=radius
        )
        potential_transitions.data = 4 * potential_transitions.data

        # Calculate the actual transitions for each grid cell of value 1 and
        # store them in a cube.
        actual_transitions = potential_transitions.copy(
            data=self._calculate_transitions(cube.data)
        )

        # Sum the number of actual transitions within the neighbourhood.
        actual_transitions = SquareNeighbourhood(sum_or_fraction="sum").run(
            actual_transitions, radius=radius
        )

        # Calculate the ratio of actual to potential transitions in areas where the
        # original diagnostic value was greater than zero. Where the original value
        # was zero, set ratio value to one.

        ratio = np.ones_like(actual_transitions.data)
        ratio[cube.data > 0] = (
            actual_transitions.data[cube.data > 0]
            / potential_transitions.data[cube.data > 0]
        )

        # Create a new cube to contain the resulting ratio data.
        ratio = create_new_diagnostic_cube(
            "texture_of_{}".format(cube_name),
            "1",
            cube,
            mandatory_attributes=generate_mandatory_attributes(
                [cube], model_id_attr=self.model_id_attr
            ),
            data=ratio,
        )
        return ratio

    @staticmethod
    def _calculate_transitions(data: ndarray) -> ndarray:
        """
        Identifies actual transitions present in a binary field. These transitions
        are defined as the number of cells of value zero that directly neighbour
        cells of value 1. The number of transitions is calculated for all cells
        of value 1 whilst avoiding double-counting transitions. The number of
        transitions for cells of value 0 is set to 0.

        Args:
            data:
                A NumPy array of the input cube for data manipulation.

        Returns:
            A NumPy array containing the transitions for ratio calculation.
        """
        padded_data = np.pad(data, 1, mode="edge")
        diff_x = np.abs(np.diff(padded_data, axis=1))
        diff_y = np.abs(np.diff(padded_data, axis=0))
        cell_sum_x = diff_x[:, 0:-1] + diff_x[:, 1:]
        cell_sum_y = diff_y[0:-1, :] + diff_y[1:, :]
        cell_sum = cell_sum_x[1:-1, :] + cell_sum_y[:, 1:-1]
        cell_sum = np.where(data > 0, cell_sum, 0)
        return cell_sum

    def process(self, input_cube: Cube) -> Cube:
        """
        Calculates a field of texture to use in differentiating solid and
        more scattered features.

        Args:
            input_cube:
                Input data in cube format containing the field for which the
                texture is to be assessed.

        Returns:
            A cube containing either the mean across realization of the
            thresholded ratios to give the field texture, if a realization
            coordinate is present, or the thresholded ratios directly, if
            no realization coordinate is present.
        """

        values = np.unique(input_cube.data)
        non_binary = np.where((values != 0) & (values != 1), True, False)
        if non_binary.any():
            raise ValueError("Incorrect input. Cube should hold binary data only")

        # Create new cube name for _calculate_ratio method.
        cube_name = find_threshold_coordinate(input_cube).name()
        # Extract threshold from input data to work with, taking into account floating
        # point comparisons.
        cube = input_cube.extract(
            iris.Constraint(
                coord_values={
                    cube_name: lambda cell: np.isclose(
                        cell.point, self.diagnostic_threshold
                    )
                }
            )
        )
        try:
            cube.remove_coord(cube_name)
        except AttributeError:
            msg = "Threshold {} is not present on coordinate with values {} {}"
            raise ValueError(
                msg.format(
                    self.diagnostic_threshold,
                    input_cube.coord(cube_name).points,
                    input_cube.coord(cube_name).units,
                )
            )
        ratios = iris.cube.CubeList()

        try:
            cslices = cube.slices_over("realization")
        except CoordinateNotFoundError:
            cslices = [cube]

        for cslice in cslices:
            ratios.append(self._calculate_ratio(cslice, cube_name, self.nbhood_radius))

        ratios = ratios.merge_cube()
        thresholded = BasicThreshold(self.textural_threshold).process(ratios)

        # Squeeze scalar threshold coordinate.
        try:
            field_texture = iris.util.squeeze(collapse_realizations(thresholded))
        except CoordinateNotFoundError:
            field_texture = iris.util.squeeze(thresholded)

        return field_texture
