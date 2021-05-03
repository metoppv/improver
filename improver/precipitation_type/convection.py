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
"""Module containing convection diagnosis utilities."""

from typing import List, Optional, Union

import iris
import numpy as np
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.nbhood.nbhood import NeighbourhoodProcessing
from improver.threshold import BasicThreshold
from improver.utilities.spatial import DifferenceBetweenAdjacentGridSquares


class DiagnoseConvectivePrecipitation(BasePlugin):
    """
    Diagnose the convective precipitation ratio by using differences between
    adjacent grid squares to help distinguish convective and stratiform
    precipitation. Convective precipitation features would be anticipated
    to have sharp features compared with broader (less sharp) features for
    stratiform precipitation.
    """

    def __init__(
        self,
        lower_threshold: float,
        higher_threshold: float,
        neighbourhood_method: str,
        radii: Union[float, List[float]],
        fuzzy_factor: Optional[float] = None,
        comparison_operator: str = ">",
        lead_times: Optional[List[float]] = None,
        weighted_mode: bool = True,
        use_adjacent_grid_square_differences: bool = True,
    ) -> None:
        """
        Args:
            lower_threshold:
                The threshold point for 'significant' datapoints to define the
                lower threshold e.g. 0 mm/hr.
            higher_threshold:
                The threshold point for 'significant' datapoints to define the
                higher threshold e.g. 5 mm/hr.
            neighbourhood_method:
                Name of the neighbourhood method to use. Options: 'circular',
                'square'.
            radii:
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.
            fuzzy_factor:
                Percentage above or below threshold for fuzzy membership value.
                If None, no fuzzy_factor is applied.
            comparison_operator:
                Indicates the comparison_operator to use with the threshold.
                e.g. 'ge' or '>=' to evaluate data >= threshold or '<' to
                evaluate data < threshold. When using fuzzy_factor, there
                is no difference between < and <= or > and >=.
                Valid choices: > >= < <= gt ge lt le.
            lead_times:
                List of lead times or forecast periods, at which the radii
                within radii are defined. The lead times are expected
                in hours.
            weighted_mode:
                If True, use a circle for neighbourhood kernel with
                weighting decreasing with radius.
                If False, use a circle with constant weighting.
            use_adjacent_grid_square_differences:
                If True, use the differences between adjacent grid squares
                to diagnose convective precipitation.
                If False, use the raw field without calculating differences to
                diagnose convective precipitation.
        """
        self.lower_threshold = lower_threshold
        self.higher_threshold = higher_threshold
        self.neighbourhood_method = neighbourhood_method
        self.radii = radii
        self.fuzzy_factor = fuzzy_factor
        self.comparison_operator = comparison_operator
        self.lead_times = lead_times
        self.weighted_mode = weighted_mode
        self.use_adjacent_grid_square_differences = use_adjacent_grid_square_differences

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = (
            "<DiagnoseConvectivePrecipitation: lower_threshold {}; "
            "higher_threshold {}; neighbourhood_method: {}; "
            "radii: {}; fuzzy_factor {}; "
            "comparison_operator: {}; lead_times: {}; "
            "weighted_mode: {};"
            "use_adjacent_grid_square_differences: {}>"
        )
        return result.format(
            self.lower_threshold,
            self.higher_threshold,
            self.neighbourhood_method,
            self.radii,
            self.fuzzy_factor,
            self.comparison_operator,
            self.lead_times,
            self.weighted_mode,
            self.use_adjacent_grid_square_differences,
        )

    def _calculate_convective_ratio(
        self, cubelist: CubeList, threshold_list: List[float]
    ) -> ndarray:
        """
        Calculate the convective ratio by:

        1. Apply neighbourhood processing to cubes that have been thresholded
           using an upper and lower threshold.
        2. Calculate the convective ratio by:
           higher_threshold_cube / lower_threshold_cube.
           For example, the higher_threshold might be 5 mm/hr, whilst the
           lower_threshold might be 0.1 mm/hr.

        The convective ratio can have the following values:
            * A non-zero fractional value, indicating that both the higher
              and lower thresholds were exceeded.
            * A zero value, if the lower threshold was exceeded, whilst the
              higher threshold was not exceeded.
            * A NaN value (np.nan), if neither the higher or lower thresholds
              were exceeded, such that the convective ratio was 0/0.

        Args:
            cube:
                Cubelist containing cubes from which the convective ratio
                will be calculated. The cube should have been thresholded,
                so that values within cube.data are between 0.0 and 1.0.
            threshold_list:
                The list of thresholds.

        Returns:
            Array of convective ratio.

        Raises:
            ValueError: If a value of infinity or a value greater than 1.0
                        are found within the convective ratio.
        """
        neighbourhooded_cube_dict = {}
        for cube, threshold in zip(cubelist, threshold_list):
            neighbourhooded_cube = NeighbourhoodProcessing(
                self.neighbourhood_method,
                self.radii,
                lead_times=self.lead_times,
                weighted_mode=self.weighted_mode,
            )(cube)
            neighbourhooded_cube_dict[threshold] = neighbourhooded_cube

        # Ignore runtime warnings from divide by 0 errors.
        with np.errstate(invalid="ignore", divide="ignore"):
            convective_ratio = np.divide(
                neighbourhooded_cube_dict[self.higher_threshold].data,
                neighbourhooded_cube_dict[self.lower_threshold].data,
            )

        infinity_condition = np.sum(np.isinf(convective_ratio)) > 0.0
        with np.errstate(invalid="ignore"):
            greater_than_1_condition = np.sum(convective_ratio > 1.0) > 0.0

        if infinity_condition or greater_than_1_condition:
            if infinity_condition:
                start_msg = (
                    "A value of infinity was found for the " "convective ratio: {}."
                ).format(convective_ratio)
            elif greater_than_1_condition:
                start_msg = (
                    "A value of greater than 1.0 was found for the "
                    "convective ratio: {}."
                ).format(convective_ratio)
            msg = (
                "{}\nThis value is not plausible as the fraction above the "
                "higher threshold must be less than the fraction "
                "above the lower threshold."
            ).format(start_msg)
            raise ValueError(msg)

        return convective_ratio

    @staticmethod
    def absolute_differences_between_adjacent_grid_squares(cube: Cube) -> CubeList:
        """
        Compute the absolute differences between grid squares and put the
        resulting cubes into a cubelist.

        Args:
            cube:
                The cube from which adjacent grid square differences will be
                calculated.

        Returns:
            Cubelist containing cubes with the absolute difference
            between adjacent grid squares along x and y, respectively.
        """
        diff_along_x_cube, diff_along_y_cube = DifferenceBetweenAdjacentGridSquares()(
            cube
        )
        # Compute the absolute values of the differences to ensure that
        # negative differences are included.
        diff_along_x_cube.data = np.absolute(diff_along_x_cube.data)
        diff_along_y_cube.data = np.absolute(diff_along_y_cube.data)
        cubelist = iris.cube.CubeList([diff_along_x_cube, diff_along_y_cube])
        return cubelist

    def iterate_over_threshold(self, cubelist: CubeList, threshold: float) -> CubeList:
        """
        Iterate over the application of thresholding to multiple cubes.

        Args:
            cubelist:
                Cubelist containing cubes to be thresholded.
            threshold:
                The threshold that will be applied.

        Returns:
            Cubelist after thresholding each cube.
        """
        cubes = iris.cube.CubeList([])
        for cube in cubelist:
            threshold_cube = BasicThreshold(
                threshold,
                fuzzy_factor=self.fuzzy_factor,
                comparison_operator=self.comparison_operator,
            )(cube.copy())
            # Will only ever contain one slice on threshold
            for cube_slice in threshold_cube.slices_over(
                find_threshold_coordinate(threshold_cube)
            ):
                threshold_cube = cube_slice

            cubes.append(threshold_cube)
        return cubes

    @staticmethod
    def sum_differences_between_adjacent_grid_squares(
        cube: Cube, thresholded_cubes: CubeList
    ) -> Cube:
        """
        Put the differences back onto the original grid by summing together
        the array with offsets. This covers the fact that the difference
        cubes will result in output on a staggered grid compared with the
        input cube.

        Args:
            cube:
                The cube with the original grid.
            thresholded_cubes:
                Cubelist containing differences between adjacent grid squares
                along x and differences between adjacent grid squares along y,
                which have been thresholded.

        Returns:
            Cube on the original grid with the values from the thresholded
            adjacent grid square difference cubes inserted. The resulting
            values have been restricted to be between 0 and 1.
        """
        threshold_cube_x, threshold_cube_y = thresholded_cubes
        cube_on_orig_grid = cube.copy()
        cube_on_orig_grid.data = np.zeros(cube_on_orig_grid.shape)
        cube_on_orig_grid.data[..., :-1, :] += threshold_cube_y.data
        cube_on_orig_grid.data[..., 1:, :] += threshold_cube_y.data
        cube_on_orig_grid.data[..., :, :-1] += threshold_cube_x.data
        cube_on_orig_grid.data[..., :, 1:] += threshold_cube_x.data
        return cube_on_orig_grid

    def process(self, cube: Cube) -> Cube:
        """
        Calculate the convective ratio either for the underlying field e.g.
        precipitation rate, or using the differences between adjacent grid
        squares.

        If the difference between adjacent grid squares is used, firstly the
        absolute differences are calculated, and then the difference cubes are
        thresholded using a high and low threshold. The thresholded difference
        cubes are then summed in order to put these cubes back onto the grid
        of the original cube. The convective ratio is then calculated by
        applying neighbourhood processing to the resulting cubes by dividing
        the high threshold cube by the low threshold cube.

        Args:
            cube:
                The cube from which the convective ratio will be calculated.

        Returns:
            Cube containing the convective ratio defined as the ratio
            between a cube with a high threshold applied and a cube with a
            low threshold applied.
        """
        cubelist = iris.cube.CubeList([])
        threshold_list = [self.lower_threshold, self.higher_threshold]
        if self.use_adjacent_grid_square_differences:
            for threshold in threshold_list:
                diff_cubelist = self.absolute_differences_between_adjacent_grid_squares(
                    cube
                )
                thresholded_cubes = self.iterate_over_threshold(
                    diff_cubelist, threshold
                )
                cubelist.append(
                    self.sum_differences_between_adjacent_grid_squares(
                        cube, thresholded_cubes
                    )
                )
        else:
            for threshold in threshold_list:
                cubelist.extend(self.iterate_over_threshold([cube], threshold))

        convective_ratios = self._calculate_convective_ratio(cubelist, threshold_list)

        attributes = generate_mandatory_attributes([cube])
        output_cube = create_new_diagnostic_cube(
            "convective_ratio", "1", cube, attributes, data=convective_ratios
        )

        return output_cube


class ConvectionRatioFromComponents(BasePlugin):
    """
    Diagnose the convective precipitation ratio by using differences between
    convective and dynamic components.
    """

    def __init__(self) -> None:
        self.convective = None
        self.dynamic = None

    def _split_input(self, cubes: Union[CubeList, List[Cube]]) -> None:
        """
        Extracts convective and dynamic components from the list as objects on the class
        and ensures units are m s-1
        """
        if not isinstance(cubes, iris.cube.CubeList):
            cubes = iris.cube.CubeList(cubes)
        self.convective = self._get_cube(cubes, "lwe_convective_precipitation_rate")
        self.dynamic = self._get_cube(cubes, "lwe_stratiform_precipitation_rate")

    @staticmethod
    def _get_cube(cubes: CubeList, name: str) -> Cube:
        """
        Get one cube named "name" from the list of cubes and set its units to m s-1.

        Args:
            cubes:
            name:

        Returns:
            Cube with units set
        """
        try:
            (cube,) = cubes.extract(name)
        except ValueError:
            raise ValueError(
                f"Cannot find a cube named '{name}' in {[c.name() for c in cubes]}"
            )
        if cube.units != "m s-1":
            cube = cube.copy()
            try:
                cube.convert_units("m s-1")
            except ValueError:
                raise ValueError(
                    f"Input {name} cube cannot be converted to 'm s-1' from {cube.units}"
                )
        return cube

    def _convective_ratio(self) -> ndarray:
        """
        Calculates the convective ratio from the convective and dynamic precipitation
        rate components, masking data where both are zero. The tolerance for comparing
        with zero is 1e-9 m s-1.
        """
        precipitation = self.convective + self.dynamic
        with np.errstate(divide="ignore", invalid="ignore"):
            convective_ratios = np.ma.masked_where(
                np.isclose(precipitation.data, 0.0, atol=1e-9),
                self.convective.data / precipitation.data,
            )
        return convective_ratios

    def process(self, cubes: List[Cube], model_id_attr: Optional[str] = None) -> Cube:
        """
        Calculate the convective ratio from the convective and dynamic components as:
            convective_ratio = convective / (convective + dynamic)

        If convective + dynamic is zero, then the resulting point is masked.

        Args:
            cubes:
                Both the convective and dynamic components as iris.cube.Cube in a list
                with names 'lwe_convective_precipitation_rate' and
                'lwe_stratiform_precipitation_rate'
            model_id_attr:
                Name of the attribute used to identify the source model for
                blending. This is inherited from the input temperature cube.

        Returns:
            Cube containing the convective ratio.
        """

        self._split_input(cubes)

        attributes = generate_mandatory_attributes(
            [self.convective], model_id_attr=model_id_attr
        )
        output_cube = create_new_diagnostic_cube(
            "convective_ratio",
            "1",
            self.convective,
            attributes,
            data=self._convective_ratio(),
        )

        return output_cube
