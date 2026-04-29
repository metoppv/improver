# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculations to produce Pollen Hourly Concentration values."""

import numpy as np
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.metadata.constants import FLOAT_DTYPE

from .utilities import build_output_cube_with_new_units


class PollenHourlyConcentration(PostProcessingPlugin):
    """Plugin to calculate the Pollen Hourly Concentration.

    The input cube for this plugin comes from the output of the
    Numerical Atmospheric dispersion Modelling Environment (NAME).
    It is 2D gridded data of pollen concentrations given as g/m3 and
    the plugin converts this to concentrations in grains/m3 using
    pollen diameter and density.
    """

    # The names of pollen types that are expected by this class
    _POLLEN_NAMES = [
        "grass",
        "birch",
        "oak",
        "hazel",
        "alder",
        "nettle",
        "ash",
        "plane",
        "weed",
    ]
    # Diameter of pollen grains for each pollen type in metres
    _POLLEN_DIAMETER = {
        "grass": 35e-6,
        "birch": 22e-6,
        "oak": 29e-6,
        "hazel": 28e-6,
        "alder": 25e-6,
        "nettle": 13e-6,
        "ash": 23e-6,
        "plane": 19e-6,
        "weed": 13e-6,
    }  # meters
    # Density of each pollen type in kg per cubic metre
    _POLLEN_DENSITY = {
        "grass": 1000.0,
        "birch": 800.0,
        "oak": 800.0,
        "hazel": 800.0,
        "alder": 800.0,
        "nettle": 1000.0,
        "ash": 800.0,
        "plane": 920.0,
        "weed": 1000.0,
    }  # kg/m3
    _POLLEN_SHORTNAME_2_LATIN = {
        "grass": "Poaceae",
        # Trees
        "birch": "Betula",
        "oak": "Quercus",
        "hazel": "Corylus",
        "alder": "Alnus",
        "ash": "Fraxinus",
        "plane": "Platanus",
        # Weeds
        "nettle": "Urticaceae",
        "weed": "Urticaceae",
    }
    _POLLEN_SHORTNAME_2_LONGNAME = {
        k: "number_concentration_of_" + v.lower() + "_pollen_grains_in_air"
        for k, v in _POLLEN_SHORTNAME_2_LATIN.items()
    }

    # Scaling factors can change, so need to be passed in
    _scaling_factors_dict = None

    # The output cube is a deepcopy of the input cube (to keep metadata) and is then manipulated in place
    _output_cube = None

    def __init__(self, scaling_factors_dict: dict = None) -> None:
        """Initialise class.

        Args:
            scaling_factors_dict:
                Optional scaling factors to use per pollen type
        """
        self._scaling_factors_dict = scaling_factors_dict

    def _calculate(self, taxa: str):
        """Perform calculations on input cube.

        Applies the scaling factor to the raw data for the relevant pollen taxa,
        and converts from g/m3 to grains/m3 using pollen diameter and density.

        Args:
            taxa:
                The pollen taxa being processed, used to update the cube name and metadata
        """
        if self._scaling_factors_dict is not None:
            self._scaling_factor = self._scaling_factors_dict[taxa][1]
        else:
            self._scaling_factor = 1.0
        diameter = self._POLLEN_DIAMETER[taxa]
        density = self._POLLEN_DENSITY[taxa]
        volume = (4 / 3) * np.pi * (diameter / 2) ** 3
        mass_per_grain = volume * density

        # Data is in g/m3, so convert to kg/m3 by dividing by 1000,
        # (then apply scaling factor)
        # and convert to grains/m3
        new_data = (
            (self._output_cube.data / 1000.0) * self._scaling_factor
        ) / mass_per_grain
        self._output_cube.data = new_data.astype(FLOAT_DTYPE)

    def _metadata(self, taxa: str):
        """Change the cube name and other metadata.
        Args:
            taxa:
                The pollen taxa being processed, used to update the cube name and metadata
        """
        self._output_cube.attributes["biological_taxon_name"] = (
            self._POLLEN_SHORTNAME_2_LATIN[taxa]
        )
        self._output_cube.attributes["forecast_period"] = np.int32(3600)
        self._output_cube.attributes["scaling_factor"] = self._scaling_factor
        self._output_cube.rename(self._POLLEN_SHORTNAME_2_LONGNAME[taxa])

    def process(
        self,
        cube: Cube,
    ) -> Cube:
        """Calculate the Pollen Concentrations.

        Args:
            cube:
                Input cube for any pollen type handled by the class
            scaling_factors_dict:
                Optional scaling factors to use per pollen type

        Returns:
            The calculated output cube.
        """
        self._output_cube = build_output_cube_with_new_units(self, cube, "m-3")

        # Check that the pollen taxa is one that is handled by the class
        taxa = self._output_cube.attributes.get("taxa").lower()
        # Remove "_pollen" from the taxa name if it is present, to match the keys in the dictionaries
        taxa = taxa.lower().replace("_pollen", "")
        if taxa not in self._POLLEN_NAMES:
            raise ValueError(f"Pollen taxa {taxa} not handled")

        self._calculate(taxa)
        self._metadata(taxa)
        return self._output_cube
