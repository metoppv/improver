# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculations to produce Pollen Daily Concentration values."""

from copy import deepcopy

import numpy as np
from iris.cube import Cube, CubeList


class PollenDailyConcentration:
    # The output cube is calculated from the input CubeList and then manipulated in place
    _output_cube = None

    def _calculate(self, cubes: tuple[Cube, ...] | CubeList):
        """Perform calculations on input cubes.

        Applies the scaling factor to the raw data for the relevant pollen species,
        and converts from g/m3 to grains/m3 using pollen diameter and density.
        """
        cube_count = len(cubes)
        if cube_count == 24:
            # Stack the cubes along a new time dimension and calculate the mean and apply the scaling factor
            stacked_data = np.stack([cube.data for cube in cubes], axis=0)
            self._output_cube.data = np.mean(stacked_data, axis=0)
        else:
            # Warning message if not 24 cubes, and set average data values to NaN
            self._output_cube.data = np.full_like(cubes[0].data, np.nan)
            UserWarning(
                f"Expected 24 cubes for hourly data, but got {cube_count}. Output values set to NaN."
            )

    def _metadata(self, species: str):
        """Change the cube name and other metadata.
        Args:
            species:
                The pollen species being processed, used to update the cube name and metadata
        """
        self._output_cube.rename(f"{species.lower()}_1day_concentration")

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
        species = template_cube.attributes.get("species").lower()

        self._calculate(cubes)
        self._metadata(species)
        return self._output_cube
