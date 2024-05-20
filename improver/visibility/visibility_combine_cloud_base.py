# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module for combining a visibility forecast with a scaled cloud base at ground level forecast"""

from typing import Tuple

import numpy as np
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.cube_combiner import Combine
from improver.metadata.probabilistic import (
    find_threshold_coordinate,
    is_probability,
    probability_is_above_or_below,
)
from improver.utilities.flatten import flatten
from improver.utilities.probability_manipulation import invert_probabilities


class VisibilityCombineCloudBase(PostProcessingPlugin):
    """
    Combines the probability of visibility relative to a threshold with
    the probability of cloud base at ground level.

    The probability of cloud base at ground level is used as proxy for
    low visibility at a grid square. By combining these diagnostics we
    add additional detail to the visibility diagnostic by capturing
    low cloud over higher areas of orography not resolved in the underlying
    visibility diagnostic.

    The probability of cloud base at ground level is multiplied by a scalar
    before combining. The scalar is dependent on which visibility threshold
    it is combined with. This is done to produce variation across the final
    visibility thresholds following combining.

    The maximum probability is then taken of the scaled cloud base at ground
    level and the visibility forecast. This is used as the outputted visibility
    forecast.
    """

    def __init__(
        self, initial_scaling_value: float, first_unscaled_threshold: float
    ) -> None:
        """Initialize plugin and define constants in the scaling distribution

        Args:
            initial_scaling_value:
                Defines the scaling value used when combining with a visibility
                threshold of 0m.
            first_unscaled_threshold:
                Defines the first threshold that will have a scaling value of 1.0.
                All thresholds greater than this will also have a scaling value of 1.0.
        """
        self.initial_scaling_value = initial_scaling_value
        self.first_unscaled_threshold = first_unscaled_threshold

    def separate_input_cubes(self, cubes: CubeList) -> Tuple[Cube, Cube]:
        """Separate cubelist into a visibility cube and a cloud base at ground level cube.

        Args:
            cubes:
                A cubelist only containing a cube of the probability of visibility
                relative to thresholds and a cube of the probability of cloud base at
                ground level
        Returns:
            A tuple containing a cube of the probability of visibility relative to threshold
            and a cube of the probability of cloud base at ground level.
        Raises:
            ValueError:
                If the input cubelist does not have exactly two cubes
            ValueError:
                If input cubelist doesn't contain a visibility cube and a cloud base at ground
                level cube
        """
        visibility_cube = None
        cloud_base_ground_cube = None

        cubes = flatten(cubes)
        cube_names = [cube.name() for cube in cubes]
        if len(cubes) != 2:
            raise ValueError(
                f"""Exactly two cubes should be provided; one for visibility and one for
                cloud base at ground level. Provided cubes are {cube_names}"""
            )

        for cube in cubes:
            if "visibility_in_air" in cube.name():
                visibility_cube = cube
            elif "cloud_base" in cube.name():
                cloud_base_ground_cube = cube
        if not visibility_cube or not cloud_base_ground_cube:
            raise ValueError(
                f"""A visibility and cloud base at ground level cube must be provided.
                The provided cubes are {cube_names}"""
            )

        return (visibility_cube, cloud_base_ground_cube)

    def get_scaling_factors(self, vis_thresholds: np.ndarray) -> np.ndarray:
        """Calculates a scaling factor for every visibility threshold. The scaling factor
        is determined differently depending on the threshold:

        1) If the threshold is greater than or equal to first_unscaled_threshold then the
        scaling factor is always 1.0.
        2) If the threshold is less than first_unscaled_threshold then a scaling factor
        is calculated by inputting the threshold into an inverted fourth level polynomial.
        The constants in this curve have been defined such that a threshold equal to
        first_unscaled_threshold gives a scaling factor of 1.0 and a threshold of 0m gives
        a scaling factor equal to initial_scaling_value.

        This distribution has been determined by experimentation and chosen for combining
        visibility with a cloud base at ground level of 4.5 oktas or greater.

        Args:
            vis_thresholds:
                An array of visibility thresholds
        Returns:
            An array of scaling factors. This will be the same length as vis_thresholds
            and the scaling factor will have the same index as the corresponding threshold
            in vis_thresholds.
        """
        scaling_factors = (1 - self.initial_scaling_value) * (
            -1 / self.first_unscaled_threshold ** 4
        ) * (vis_thresholds - self.first_unscaled_threshold) ** 4 + 1

        scaling_factors = np.where(
            vis_thresholds > self.first_unscaled_threshold, 1, scaling_factors
        )
        return scaling_factors

    def process(self, cubes: CubeList) -> Cube:
        """
        Combines the visibility cube with the cloud base at ground level cube. This is
        done for every visibility threshold independently. Before combining, the cloud
        base at ground level cube is multiplied by a scaling factor which either reduces
        or has no impact on the probabilities in the cube. The scaling factor is calculated
        based upon which visibility threshold the cloud base will be combined with.
        The maximum probability is then taken between the scaled cloud base cube and the
        corresponding visibility threshold.

        Args:

            cubes:
                containing:
                    visibility:
                        Cube of probability of visibility relative to thresholds
                    cloud base at ground level:
                        Cube of probability of cloud base at ground level. This cube should only
                        have spatial dimensions (e.g. spot_index or x,y coordinates).
        Returns:
            Cube of visibility data that has been combined with the scaled cloud base at ground
            level cube.

        Raises:
            ValueError:
                If visibility_cube is not a probability cube
            ValueError:
                If cloud_base_ground_cube is not a probability cube
        """

        visibility_cube, cloud_base_ground_cube = self.separate_input_cubes(cubes)

        for cube in [visibility_cube, cloud_base_ground_cube]:
            if not is_probability(cube):
                raise ValueError(f"Cube {cube.name()} must be a probability cube")

        # Ensure all probabilities are probability of being below threshold and if not invert
        # the probabilities.
        vis_inverted = False
        if probability_is_above_or_below(visibility_cube) == "above":
            visibility_cube = invert_probabilities(visibility_cube)
            vis_inverted = True

        # Calculate the scaling factors for each visibility threshold
        visibility_threshold_coord = find_threshold_coordinate(visibility_cube).name()
        visibility_thresholds = visibility_cube.coord(visibility_threshold_coord).points
        cloud_base_scaling_factor = self.get_scaling_factors(visibility_thresholds)

        # Combine each visibility threshold with the scaled cloud base at ground level cube
        vis_combined_list = CubeList()
        for vis_slice, scaling_factor in zip(
            visibility_cube.slices_over(visibility_threshold_coord),
            cloud_base_scaling_factor,
        ):
            scaled_cloud_base = cloud_base_ground_cube * scaling_factor
            vis_combined_list.append(
                Combine(operation="max", expand_bound=False)(
                    CubeList([vis_slice, scaled_cloud_base])
                )
            )

        vis_cube = vis_combined_list.merge_cube()

        if vis_inverted:
            vis_cube = invert_probabilities(vis_cube)
        return vis_cube
