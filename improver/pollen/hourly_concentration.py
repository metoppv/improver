# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculations to produce Pollen Hourly Concentration values."""

import numpy as np
from iris.cube import Cube


class PollenHourlyConcentration:
    # The names of pollen types that are expected by this class
    _POLLEN_NAMES = [
        "grass_pollen",
        "birch_pollen",
        "oak_pollen",
        "hazel_pollen",
        "alder_pollen",
        "nettle_pollen",
        "ash_pollen",
        "plane_pollen",
    ]
    # Diameter of pollen grains for each pollen type in metres
    _POLLEN_DIAMETER = {
        "grass_pollen": 35e-6,
        "birch_pollen": 22e-6,
        "oak_pollen": 29e-6,
        "hazel_pollen": 28e-6,
        "alder_pollen": 25e-6,
        "nettle_pollen": 13e-6,
        "ash_pollen": 23e-6,
        "plane_pollen": 19e-6,
    }  # meters
    # Density of each pollen type in kg per cubic metre
    _POLLEN_DENSITY = {
        "grass_pollen": 1000.0,
        "birch_pollen": 800.0,
        "oak_pollen": 800.0,
        "hazel_pollen": 800.0,
        "alder_pollen": 800.0,
        "nettle_pollen": 1000.0,
        "ash_pollen": 800.0,
        "plane_pollen": 920.0,
    }  # kg/m3

    # Scaling factors can change, so need to be passed in
    _scaling_factors_dict = {}

    # The input cube will load from a CubeList and be manipulated in place
    hourly_concentrations_cube = None

    def _load_input_cube(self, cube: Cube):
        """Loads the required input cube for the calculation. This is stored
        internally as a Cube object.

        Args:
            cube:
                Input Cube containing the necessary data.

        Raises:
            ValueError:
                If the number of cubes does not match the expected number.
        """
        self.hourly_concentrations_cube = cube

        # Check that the pollen species is one that is handled by the class
        species = self.hourly_concentrations_cube.attributes.get("species").lower()
        if species not in self._POLLEN_NAMES:
            raise ValueError(f"Pollen species {species} not handled")

    def _calculate(self):
        """Perform calculations on input cube.

        Applies the scaling factor to the raw data for the relevant pollen species,
        and converts from g/m3 to grains/m3 using pollen diameter and density.
        """
        species = self.hourly_concentrations_cube.attributes.get("species").lower()
        scaling_factor = self._scaling_factors_dict[species][1]
        diameter = self._POLLEN_DIAMETER[species]
        density = self._POLLEN_DENSITY[species]
        # print(f"Converting {species} from g/m3 to grains/m3 using diameter {diameter} m and density {density} kg/m3")
        volume = (4 / 3) * np.pi * (diameter / 2) ** 3
        mass_per_grain = volume * density

        self.hourly_concentrations_cube.data = (
            self.hourly_concentrations_cube.data * scaling_factor / mass_per_grain
        )

    def _metadata(self):
        """Change the cube name and other metadata."""
        self.hourly_concentrations_cube.rename(
            self.hourly_concentrations_cube.name().lower()
        )
        self.hourly_concentrations_cube.convert_units("grains / m3")

    def process(
        self,
        cube: Cube,
        scaling_factors_dict: dict,
    ) -> Cube:
        """Calculate the Pollen Concentrations.

        Args:
            cube:
                Input cube for any pollen type handled by the class
            scaling_factors_dict:
                Scaling factors to user per pollen type

        Returns:
            The calculated output cube.
        """
        self._scaling_factors_dict = scaling_factors_dict
        self._load_input_cube(cube)
        self._calculate()
        self._metadata()
        return self.hourly_concentrations_cube
