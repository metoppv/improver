# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin to calculate probabilities of occurrence between specified thresholds"""

from typing import List

import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver import PostProcessingPlugin
from improver.metadata.probabilistic import (
    find_threshold_coordinate,
    probability_is_above_or_below,
)


class OccurrenceBetweenThresholds(PostProcessingPlugin):
    """Calculate the probability of occurrence between thresholds"""

    def __init__(
        self, threshold_ranges: List[List[float]], threshold_units: str
    ) -> None:
        """
        Initialise the class.  Threshold ranges must be specified in a unit
        that is NOT sensitive to differences at the 1e-5 (float32) precision
        level.

        Args:
            threshold_ranges:
                List of 2-item iterables specifying thresholds between which
                probabilities should be calculated
            threshold_units:
                Units in which the thresholds are specified

        Raises:
            ValueError:
                If any of the specified thresholds are indistinguishable at the
                1e-5 (float32) precision level
        """
        threshold_diffs = np.diff(threshold_ranges)
        if any(diff < 1e-5 for diff in threshold_diffs):
            raise ValueError(
                "Plugin cannot distinguish between thresholds at {} {}".format(
                    threshold_ranges, threshold_units
                )
            )
        self.threshold_ranges = threshold_ranges
        self.threshold_units = threshold_units

    def _slice_cube(self) -> List[List[Cube]]:
        """
        Extract required slices from input cube

        Returns:
            List of 2-item lists containing lower and upper
            threshold cubes

        Raises:
            ValueError:
                If any of the required constraints returns None
        """
        thresh_coord = self.cube.coord(self.thresh_coord.name())
        error_string = (
            f"{thresh_coord.name()} threshold {{}} {self.threshold_units} "
            "is not available\n"
        )
        error_msg = ""

        cubes = []
        for t_range in self.threshold_ranges:
            t_range.sort()
            lower_constraint = iris.Constraint(
                coord_values={
                    thresh_coord: lambda t: np.isclose(t.point, t_range[0], atol=1e-5)
                }
            )
            lower_cube = self.cube.extract(lower_constraint)
            if lower_cube is None:
                error_msg += error_string.format(t_range[0])
            upper_constraint = iris.Constraint(
                coord_values={
                    thresh_coord: lambda t: np.isclose(t.point, t_range[1], atol=1e-5)
                }
            )
            upper_cube = self.cube.extract(upper_constraint)
            if upper_cube is None:
                error_msg += error_string.format(t_range[1])
            cubes.append([lower_cube, upper_cube])

        if error_msg:
            # if any thresholds were unavailable, raise errors together here
            raise ValueError(error_msg)

        return cubes

    def _get_multiplier(self) -> np.float32:
        """
        Check whether the cube contains "above" or "below" threshold
        probabilities.  For "above", the probability of occurrence between
        thresholds is the difference between probabilities at the lower
        and higher thresholds: P(lower) - P(higher).  For "below" it is the
        inverse of this: P(higher) - P(lower), which is implemented by
        multiplying the difference by -1.

        Returns:
            1. or -1.

        Raises:
            ValueError: If the spp__relative_to_threshold attribute is
                not recognised
        """
        relative_to_threshold = probability_is_above_or_below(self.cube)
        if relative_to_threshold == "above":
            multiplier = 1.0
        elif relative_to_threshold == "below":
            multiplier = -1.0
        else:
            raise ValueError(
                "Input cube must contain probabilities of "
                "occurrence above or below threshold"
            )
        return np.float32(multiplier)

    def _calculate_probabilities(self) -> Cube:
        """
        Calculate between_threshold probabilities cube

        Returns:
            Merged cube containing recalculated probabilities
        """
        multiplier = self._get_multiplier()
        thresh_name = self.thresh_coord.name()

        cubelist = iris.cube.CubeList([])
        for lower_cube, upper_cube in self.cube_slices:
            # construct difference cube
            between_thresholds_data = (lower_cube.data - upper_cube.data) * multiplier
            between_thresholds_cube = upper_cube.copy(between_thresholds_data)

            # add threshold coordinate bounds
            lower_threshold = lower_cube.coord(thresh_name).points[0]
            upper_threshold = upper_cube.coord(thresh_name).points[0]
            between_thresholds_cube.coord(thresh_name).bounds = [
                lower_threshold,
                upper_threshold,
            ]

            cubelist.append(between_thresholds_cube)

        return cubelist.merge_cube()

    def _update_metadata(self, output_cube: Cube, original_units: Unit) -> None:
        """
        Update output cube name and threshold coordinate

        Args:
            output_cube:
                Cube containing new "between_thresholds" probabilities
            original_units:
                Required threshold-type coordinate units
        """
        new_name = self.cube.name().replace(
            "{}_threshold".format(probability_is_above_or_below(self.cube)),
            "between_thresholds",
        )
        output_cube.rename(new_name)

        new_thresh_coord = output_cube.coord(self.thresh_coord.name())
        new_thresh_coord.convert_units(original_units)
        new_thresh_coord.attributes["spp__relative_to_threshold"] = "between_thresholds"

    def process(self, cube: Cube) -> Cube:
        """
        Calculate probabilities between thresholds for the input cube

        Args:
            cube:
                Probability cube containing thresholded data (above or below)

        Returns:
            Cube containing probability of occurrence between thresholds
        """
        # if cube has no threshold-type coordinate, raise an error
        try:
            self.thresh_coord = find_threshold_coordinate(cube)
        except CoordinateNotFoundError:
            raise ValueError(
                "Input is not a probability cube (has no threshold-type coordinate)"
            )
        self.cube = cube.copy()

        # check input cube units and convert if needed
        original_units = self.thresh_coord.units
        if original_units != self.threshold_units:
            self.cube.coord(self.thresh_coord).convert_units(self.threshold_units)

        # extract suitable cube slices
        self.cube_slices = self._slice_cube()

        # generate "between thresholds" probabilities
        output_cube = self._calculate_probabilities()
        self._update_metadata(output_cube, original_units)
        return output_cube
