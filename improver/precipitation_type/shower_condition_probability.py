# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin to construct a shower conditions probability"""

from typing import Optional, Tuple

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.threshold import Threshold
from improver.utilities.cube_manipulation import collapse_realizations

from .utilities import make_shower_condition_cube


class ShowerConditionProbability(PostProcessingPlugin):
    """Plugin to calculate the probability that conditions are such that
    precipitation, should it be present, will be showery, based on input cloud
    amounts and the convective ratio."""

    def __init__(
        self,
        cloud_threshold: float = 0.8125,
        convection_threshold: float = 0.8,
        model_id_attr: Optional[str] = None,
    ) -> None:
        """
        Args:
            cloud_threshold:
                The fractional cloud coverage value at which to threshold the
                cloud data.
            convection_threshold:
                The convective ratio value at which to threshold the convective
                ratio data.
            model_id_attr:
                Name of the attribute used to identify the source model for
                blending.
        """
        self.cloud_threshold = cloud_threshold
        self.convection_threshold = convection_threshold
        self.model_id_attr = model_id_attr
        self.cloud_constraint = iris.Constraint(
            cube_func=lambda cube: "cloud_area_fraction" in cube.name()
        )
        self.convection_constraint = iris.Constraint(
            cube_func=lambda cube: "convective_ratio" in cube.name()
        )

    def _create_shower_condition_cube(self, data: ndarray, cube: Cube) -> Cube:
        """
        Returns a shower condition cube, with coordinates and mandatory
        attributes based upon the provided cube. The threshold coordinate is
        modified to describe shower conditions, such that the probabilities
        describe the likelihood of conditions being showery. The arbitrary
        threshold value is 1.

        Args:
            data:
                The shower condition probabilities to populate the new cube.
            cube:
                The cube to use as a template, and from which to extract
                attributes for use in the new diagnostic cube.

        Returns:
            A probability of shower conditions cube.
        """
        template = make_shower_condition_cube(cube)
        attributes = generate_mandatory_attributes(
            [cube], model_id_attr=self.model_id_attr
        )

        result = create_new_diagnostic_cube(
            template.name(), "1", template, mandatory_attributes=attributes, data=data,
        )

        return result

    def _extract_inputs(self, cubes: CubeList) -> Tuple[Cube, Cube]:
        """
        Extract the required input cubes from the input cubelist and check
        they are as required.

        Args:
            cubes:
                A cubelist containing a cube of cloud fraction and one of
                convective ratio.

        Returns:
            The cloud and convection cubes extracted from the cubelist.

        Raises:
            ValueError: If the expected cubes are not within the cubelist.
            ValueError: If the input cubes have different shapes, perhaps due
                        to a missing realization in one and not the other.
        """

        try:
            (cloud,) = cubes.extract(self.cloud_constraint)
            (convection,) = cubes.extract(self.convection_constraint)
        except ValueError:
            input_cubes = ", ".join([cube.name() for cube in cubes])
            msg = (
                "A cloud area fraction and convective ratio are required, "
                f"but the inputs were: {input_cubes}"
            )
            raise ValueError(msg)

        if cloud.shape != convection.shape:
            msg = (
                "The cloud area fraction and convective ratio cubes are not "
                "the same shape and cannot be combined to generate a shower"
                " probability"
            )
            raise ValueError(msg)
        return cloud, convection

    def process(self, cubes: CubeList) -> Cube:
        """
        Create a shower condition probability from cloud fraction and convective
        ratio fields. This plugin thresholds the two input diagnostics,
        creates a hybrid probability field from the resulting binary fields,
        and then collapses the realizations to give a non-binary probability
        field that represents the likelihood of conditions being showery.

        Args:
            cubes:
                A cubelist containing a cube of cloud fraction and one of
                convective ratio.

        Returns:
            Probability of any precipitation, if present, being classified as
            showery
        """
        cloud, convection = self._extract_inputs(cubes)

        # Threshold cubes
        cloud_thresholded = Threshold(
            threshold_values=self.cloud_threshold, comparison_operator="<="
        ).process(cloud)
        convection_thresholded = Threshold(
            threshold_values=self.convection_threshold
        ).process(convection)

        # Fill any missing data in the convective ratio field with zeroes.
        if np.ma.is_masked(convection_thresholded.data):
            convection_thresholded.data = convection_thresholded.data.filled(0)

        # Create a combined field taking the maximum of each input
        shower_probability = np.maximum(
            cloud_thresholded.data, convection_thresholded.data
        ).astype(FLOAT_DTYPE)

        result = self._create_shower_condition_cube(
            shower_probability, convection_thresholded
        )

        try:
            shower_conditions = collapse_realizations(result)
        except CoordinateNotFoundError:
            shower_conditions = result

        return iris.util.squeeze(shower_conditions)
