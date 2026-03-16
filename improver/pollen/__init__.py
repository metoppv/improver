# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Pollen calculation components."""

from copy import deepcopy

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin

_DAILY_SCALING_FACTOR = 1.008

# Mapping of pollen short names to their Latin names for use in metadata.
_POLLEN_SHORTNAME_2_LATIN = {
    "grass_pollen": "Poaceae",
    # Trees
    "birch_pollen": "Betula",
    "oak_pollen": "Quercus",
    "hazel_pollen": "Corylus",
    "alder_pollen": "Alnus",
    "willow_pollen": "Salix",
    "ash_pollen": "Fraxinus",
    "plane_pollen": "Platanus",
    "cypress_pollen": "Cupressaceae",
    "elm_pollen": "Ulmus",
    # Weeds
    "nettle_pollen": "Urtica",
    "mugwort_pollen": "Artemisia",
    "ragweed_pollen": "Ambrosia",
    "plantain_pollen": "Plantago",
    "goosefoot_pollen": "Chenopodium",
}

#: Threshold index levels - minimum value (grains/m3) for each index.
_POLLEN_INDEX = {  # 0=No pollen, 1=Low, 2=Moderate, 3=High, 4=Very High
    # (5=extra level just for contour levels)
    "DPI": np.array([0, 1, 2, 3, 4, 5]),
    "grass_pollen": np.array([0.0, 0.01, 30.0, 50.0, 150.0, 5000.0]),
    # Trees
    "birch_pollen": np.array([0.0, 0.01, 40.0, 80.0, 200.0, 5000.0]),
    "oak_pollen": np.array([0.0, 0.01, 30.0, 50.0, 200.0, 5000.0]),
    "hazel_pollen": np.array([0.0, 0.01, 30.0, 50.0, 80.0, 5000.0]),
    "alder_pollen": np.array([0.0, 0.01, 30.0, 50.0, 80.0, 5000.0]),
    "ash_pollen": np.array([0.0, 0.01, 30.0, 50.0, 200.0, 5000.0]),
    "plane_pollen": np.array([0.0, 0.01, 30.0, 50.0, 200.0, 5000.0]),
    # Weeds
    "nettle_pollen": np.array([0.0, 0.01, 40.0, 80.0, 200.0, 5000.0]),
    "mugwort_pollen": np.array([0.0, 0.01, 10.0, 20.0, 40.0, 5000.0]),
    "cypress_pollen": np.array([0.0, 0.01, 40.0, 80.0, 200.0, 5000.0]),
    "plantain_pollen": np.array([0.0, 0.01, 10.0, 20.0, 40.0, 5000.0]),
    "goosefoot_pollen": np.array([0.0, 0.01, 10.0, 20.0, 40.0, 5000.0]),
}


class PollenBase(BasePlugin):
    """
    Class for Pollen calculations.

    This class provides common functionality for all pollen
    components, including:

    - Apply a scaling factor to the raw data for each species (this may change regularly)
    - Convert from g/m3 to grains/m3 using the pollen diameter and density (see below for calculation) - this gets hourly pollen concentrations by species
    - Calculate a mean over 24 hours (9am local that day, to 9am local next day) and then apply a scaling factor of 1.008 to get daily pollen concentrations by species

    - Load the baseline cube for the pollen species being calculated, and use it to
      set the metadata for the output cube.

    - Provide a common interface for calculating pollen concentrations, which can be
      implemented by subclasses.
    """

    _POLLEN_SHORTNAME_2_LONGNAME = {
        k: "grain_concentration_of_" + v.lower() + "_pollen_in_air"
        for k, v in _POLLEN_SHORTNAME_2_LATIN.items()
    }

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

    #: Threshold index levels - minimum value (grains/m3) for each index.
    _POLLEN_INDEX = {  # 0=No pollen, 1=Low, 2=Moderate, 3=High, 4=Very High
        # (5=extra level just for contour levels)
        "DPI": np.array([0, 1, 2, 3, 4, 5]),
        "grass_pollen": np.array([0.0, 0.01, 30.0, 50.0, 150.0, 5000.0]),
        # Trees
        "birch_pollen": np.array([0.0, 0.01, 40.0, 80.0, 200.0, 5000.0]),
        "oak_pollen": np.array([0.0, 0.01, 30.0, 50.0, 200.0, 5000.0]),
        "hazel_pollen": np.array([0.0, 0.01, 30.0, 50.0, 80.0, 5000.0]),
        "alder_pollen": np.array([0.0, 0.01, 30.0, 50.0, 80.0, 5000.0]),
        "ash_pollen": np.array([0.0, 0.01, 30.0, 50.0, 200.0, 5000.0]),
        "plane_pollen": np.array([0.0, 0.01, 30.0, 50.0, 200.0, 5000.0]),
        # Weeds
        "nettle_pollen": np.array([0.0, 0.01, 40.0, 80.0, 200.0, 5000.0]),
        "mugwort_pollen": np.array([0.0, 0.01, 10.0, 20.0, 40.0, 5000.0]),
        "cypress_pollen": np.array([0.0, 0.01, 40.0, 80.0, 200.0, 5000.0]),
        "plantain_pollen": np.array([0.0, 0.01, 10.0, 20.0, 40.0, 5000.0]),
        "goosefoot_pollen": np.array([0.0, 0.01, 10.0, 20.0, 40.0, 5000.0]),
    }

    scaling_factors_dict = {
        "hazel_pollen": [1.0, 2.54],
        "alder_pollen": [1.0, 14.6],
        "ash_pollen": [1.0, 132.0],
        "birch_pollen": [1.0, 3.68],
        "plane_pollen": [1.0, 11.4],
        "oak_pollen": [1.0, 1.70],
        "grass_pollen": [1.0, 0.0226],
        "nettle_pollen": [1.0, 213.0],
        "daily_mean_hazel_pollen": [1.0, 2.54],
        "daily_mean_alder_pollen": [1.0, 14.6],
        "daily_mean_ash_pollen": [1.0, 132.0],
        "daily_mean_birch_pollen": [1.0, 3.68],
        "daily_mean_plane_pollen": [1.0, 11.4],
        "daily_mean_oak_pollen": [1.0, 1.70],
        "daily_mean_grass_pollen": [1.0, 0.0226],
        "daily_mean_nettle_pollen": [1.0, 213.0],
    }

    _expected_number_of_hourly_cubes = 24 * len(
        _POLLEN_DENSITY
    )  # 24 hours of data for each pollen species
    _hourly_concentrations_cubes: tuple[Cube, ...] | CubeList
    _daily_mean_concentration_cubes = {}
    _hourly_pollen_values_cubes = {}
    _daily_pollen_values_cubes = {}

    # def __init__(self, config: dict) -> None:
    #     """
    #     Initialise the plugin.

    #     Args:
    #         config:
    #             A dictionary containing configuration parameters for the plugin. This should include:
    #             - "pollen_species": A list of pollen species to calculate (e.g., ["grass", "tree", "weed"]).
    #             - "scaling_factors": A dictionary mapping each pollen species to its scaling factor (e.g., {"grass": 1.008, "tree": 1.008, "weed": 1.008}).
    #             - "baseline_cube_paths": A dictionary mapping each pollen species to the file path of its baseline cube (e.g., {"grass": "path/to/grass_baseline_cube.nc", "tree": "path/to/tree_baseline_cube.nc", "weed": "path/to/weed_baseline_cube.nc"}).
    #     """
    #     self.config = config

    def _load_input_cubes(self, cubes: tuple[Cube, ...] | CubeList):
        """Loads the required input cubes for the calculation. These are stored
        internally as Cube objects.

        Args:
            cubes:
                Input cubes containing the necessary data.

        Raises:
            ValueError:
                If the number of cubes does not match the expected number.
        """
        if len(cubes) != self._expected_number_of_hourly_cubes:
            raise ValueError(
                f"Expected {self._expected_number_of_hourly_cubes} cubes, found {len(cubes)}"
            )
        self._hourly_concentrations_cubes = cubes

    def _apply_scaling_factors_per_species(self):
        """Applies the scaling factors to the raw data for each pollen species as specified in the config."""
        for cube in self._hourly_concentrations_cubes:
            species = cube.attributes.get("species").lower()
            # print(f"Applying scaling factor for species: {species}")
            if species in self.scaling_factors_dict.keys():
                # scaling_factor = self.scaling_factors_dict[species][1]
                # print(f"Scaling factor for {species}: {scaling_factor}")
                cube.data = cube.data * self.scaling_factors_dict[species][1]

    def _convert_to_grains_per_cubic_meter(self):
        """Converts the data from g/m3 to grains/m3 using the pollen diameter and density."""
        for cube in self._hourly_concentrations_cubes:
            species = cube.attributes.get("species").lower()
            if species in self._POLLEN_DIAMETER and species in self._POLLEN_DENSITY:
                diameter = self._POLLEN_DIAMETER[species]
                density = self._POLLEN_DENSITY[species]
                # print(f"Converting {species} from g/m3 to grains/m3 using diameter {diameter} m and density {density} kg/m3")
                volume = (4 / 3) * np.pi * (diameter / 2) ** 3
                mass_per_grain = volume * density
                cube.data = cube.data / mass_per_grain

    def _calculate_daily_mean_concentrations(self):
        """Calculates the daily mean for each pollen species over a 24-hour period (9am local to 9am local next day)."""
        # Group cubes by species into a dict keyed on species name
        print(
            "NOT YET DOING THE 9am to 9am grouping, just taking the mean of all 24 cubes for each species!"
        )
        species_to_cubes = {}
        for cube in self._hourly_concentrations_cubes:
            species = cube.attributes.get("species").lower()
            if species not in species_to_cubes:
                species_to_cubes[species] = []
            species_to_cubes[species].append(cube)
        print(
            f"Grouped cubes by species: { {species: len(cubes) for species, cubes in species_to_cubes.items()} }"
        )

        # Calculate daily mean for each species and add it to the _daily_mean_concentration_cubes dict
        for species, cubes in species_to_cubes.items():
            if len(cubes) != 24:
                raise ValueError(
                    f"Expected 24 cubes for species {species}, found {len(cubes)}"
                )
            # Stack the cubes along a new time dimension and calculate the mean
            stacked_data = np.stack([cube.data for cube in cubes], axis=0)
            daily_mean_data = np.mean(stacked_data, axis=0)
            print(f"daily_mean.shape for {species}: {daily_mean_data.shape}")
            # Create a new cube copying the first input cube so that metadata is
            # included, and apply _DAILY_SCALING_FACTOR
            self._daily_mean_concentration_cubes[species] = deepcopy(cubes[0])
            self._daily_mean_concentration_cubes[species].data = (
                daily_mean_data * _DAILY_SCALING_FACTOR
            )
            print(
                f"Calculated daily mean for species {species}\n{self._daily_mean_concentration_cubes[species]}"
            )

    def _calculate(self) -> np.ndarray:
        """Performs the core calculation for the pollen index. This method should be
        implemented by subclasses to perform the specific calculation for each pollen type.

        Returns:
            A numpy array containing the calculated pollen index values.
        """
        raise NotImplementedError("Subclasses must implement this method")

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
