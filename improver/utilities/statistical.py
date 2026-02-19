# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain methods for handling statistical operations."""

from typing import List, Optional, Union

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.metadata.utilities import create_new_diagnostic_cube


class DistributionalParameters(BasePlugin):
    """
    Class for estimating distributional parameters given some statistics.
    """

    def __init__(
        self,
        distribution: str = "norm",
        truncation_points: Optional[list[float]] = None,
    ):
        """
        Initialize class for estimating distributional parameters.

        Args:
            distribution:
                The distribution for which parameters are to be estimated. The default
                is a normal distribution.
            truncation_points:
                List containing the lower and upper truncation points for a truncated
                normal distribution.
        """
        self.distribution_dict = {
            "norm": self._normal_parameters,
            "truncnorm": self._truncated_normal_parameters,
            "gamma": self._gamma_parameters,
        }

        if distribution in self.distribution_dict.keys():
            self.distribution = distribution
        else:
            raise ValueError(
                f"Distribution '{distribution}' is not supported. "
                f"Supported distributions are: {list(self.distribution_dict.keys())}"
            )

        if self.distribution != "truncnorm" and truncation_points is not None:
            raise ValueError(
                "Truncation points should not be provided for non-truncated normal "
                f"distributions. The following distribution was chosen: {distribution}."
            )
        else:
            self.truncation_points = truncation_points

    @staticmethod
    def _normal_parameters(
        mean: np.array, sd: np.array
    ) -> tuple[None, np.array, np.array]:
        """
        Estimate parameters for a normal distribution given mean and standard deviation
        cubes.

        Args:
            mean:
                Array of mean values.
            sd:
                Array of standard deviation values.

        Returns:
            Arrays containing location and scale parameters of a normal distribution.
        """
        shape = None
        loc = mean
        scale = sd
        return shape, loc, scale

    def _truncated_normal_parameters(
        self, mean: np.array, sd: np.array
    ) -> tuple[List[np.array], np.array, np.array]:
        """
        Estimate parameters for a truncated normal distribution given mean and standard
        deviation cubes.

        Args:
            mean:
                Array of mean values.
            sd:
                Array of standard deviation values.

        Returns:
            Arrays containing location and scale parameters of a truncated normal
            distribution.

        Raises:
            ValueError:
                If truncation points are not provided or if the number of truncation
                points is not equal to two.
        """
        if self.truncation_points is None or len(self.truncation_points) != 2:
            raise ValueError(
                "Upper and lower truncation points must be provided for truncated "
                "normal distribution. The following truncation points were provided: "
                f"{self.truncation_points}."
            )

        shape = [
            np.full_like(mean, self.truncation_points[0]),
            np.full_like(mean, self.truncation_points[1]),
        ]
        loc = mean
        scale = sd
        return shape, loc, scale

    @staticmethod
    def _gamma_parameters(
        mean: np.array, sd: np.array
    ) -> tuple[np.array, np.array, np.array]:
        """
        Estimate parameters for a gamma distribution given mean and standard deviation
        cubes.

        Args:
            mean:
                Array of mean values.
            sd:
                Array of standard deviation values.

        Returns:
            Arrays containing shape and scale parameters of a gamma distribution.
        """
        shape = (mean / sd) ** 2
        loc = np.full_like(mean, 0)
        scale = sd**2 / mean
        return shape, loc, scale

    def process(
        self,
        mean_cube: Cube,
        sd_cube: Cube,
    ) -> tuple[Union[CubeList, Cube, None], Union[Cube, None], Union[Cube, None]]:
        """
        Estimate distributional parameters given mean and variance cubes.

        Args:
            mean_cube:
                Cube containing the mean values.
            sd_cube:
                Cube containing the standard deviation values.

        Returns:
            The shape, location and scale parameter cubes. The shape parameter(s) may
            be a cubelist if multiple shape parameters are returned. Any of the
            parameters may be None if not applicable for the chosen distribution.
        """
        shape, loc, scale = self.distribution_dict.get(self.distribution)(
            mean_cube.data, sd_cube.data
        )

        shape_parameter = location_parameter = scale_parameter = None

        if shape is not None:
            shape = shape if isinstance(shape, list) else [shape]
            shape_parameter = CubeList()
            for arr in shape:
                shape_parameter.append(
                    create_new_diagnostic_cube(
                        "shape_parameter",
                        mean_cube.units,
                        mean_cube,
                        mean_cube.attributes,
                        data=arr,
                    )
                )
            shape_parameter = (
                shape_parameter[0] if len(shape_parameter) == 1 else shape_parameter
            )

        if loc is not None:
            location_parameter = create_new_diagnostic_cube(
                "location_parameter",
                mean_cube.units,
                mean_cube,
                mean_cube.attributes,
                data=loc,
            )
        if scale is not None:
            scale_parameter = create_new_diagnostic_cube(
                "scale_parameter",
                mean_cube.units,
                mean_cube,
                mean_cube.attributes,
                data=scale,
            )

        return shape_parameter, location_parameter, scale_parameter
