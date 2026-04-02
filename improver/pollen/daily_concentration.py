# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculations to produce Pollen Daily Concentration values."""

import warnings
from copy import deepcopy

import numpy as np
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.metadata.constants import FLOAT_DTYPE


class PollenDailyConcentration(PostProcessingPlugin):
    """Plugin to calculate the Pollen Daily Concentration.

    The input cube for this plugin comes from the output of the
    PollenHourlyConcentration plugin. This plugin calculates a daily
    mean from 24 hours of hourly data. If 24 hours of data is not
    available then output values from this plugin are set as NaN.
    """

    # The output cube is calculated from the input CubeList and then manipulated in place
    _output_cube = None

    def _calculate(self, cubes: tuple[Cube, ...] | CubeList):
        """Perform calculations on input cubes.

        For each grid point, calculate the mean pollen concentration across all hours,
        and use this as the daily pollen concentration for that grid point. If there are
        not enough hours of data, set the output values to NaN and issue a warning.

        Args:
            cubes:
                Input cubes for hourly pollen concentrations
        """
        cube_count = len(cubes)
        if cube_count >= 23:
            # Stack the cubes along a new time dimension and calculate the mean across that dimension
            stacked_data = np.stack([cube.data for cube in cubes], axis=0)
            self._output_cube.data = np.mean(stacked_data, axis=0).astype(FLOAT_DTYPE)
        else:
            # Warning message if not at least 23 cubes, and set average data values to NaN
            self._output_cube.data = np.full_like(
                cubes[0].data, np.nan, dtype=FLOAT_DTYPE
            )
            warnings.warn(
                f"Expected at least 23 cubes for hourly data, but got {cube_count}. Output values set to NaN.",
                UserWarning,
            )

    def _metadata(self, taxa: str):
        """Change the cube name and other metadata.
        Args:
            taxa:
                The pollen taxa being processed, used to update the cube name and metadata
        """
        self._output_cube.rename(f"{taxa.lower()}_concentration")

    def process(self, cubes: tuple[Cube, ...] | CubeList) -> Cube:
        """Calculate the Pollen Daily Concentration.

        Args:
            cubes:
                Input cubes for hourly pollen concentrations

        Returns:
            The calculated output cube.

        Warns:
            UserWarning:
                If output values fall outside typical expected ranges
        """
        # Create output_cube ready to take data from calculations, using the first cube as a template
        template_cube = cubes[0]
        self._output_cube = deepcopy(template_cube)
        taxa = template_cube.attributes.get("taxa").lower()

        self._calculate(cubes)
        self._metadata(taxa)
        return self._output_cube
