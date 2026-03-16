# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculations to produce Pollen Hourly Index values."""

from iris.cube import Cube, CubeList


class PollenHourlyIndex:
    def process(self, cubes: tuple[Cube, ...] | CubeList) -> Cube:
        """Calculate the Pollen Index.

        Args:
            cubes:
                Input cubes for all pollen types

        Returns:
            The calculated output cube.

        Warns:
            UserWarning:
                If output values fall outside typical expected ranges
        """
        self._load_input_cubes(cubes)
        self._apply_scaling_factors_per_species()
        self._convert_to_grains_per_cubic_meter()
        self._calculate_daily_mean_concentrations()
        self._calculate_hourly_pollen_values()
        self._calculate_daily_pollen_values()
        output_data = self._calculate()
        output_cube = self._make_output_cube(output_data)
        return output_cube
