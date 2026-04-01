# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculations to produce Pollen Indexes for a period (Hourly or Daily)."""

import numpy as np
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.pollen import build_output_cube_with_new_units


class PollenIndexForPeriod(PostProcessingPlugin):
    """Plugin to calculate a Pollen Index cube for either Daily or Hourly.

    Pollen Concentration values in the input cube are compared with threshold
    values appropriate for the pollen species represented by the cube, and
    categorized as indexes 0 to 4 for each grid point.
    """

    #: Threshold index levels - minimum value (grains/m3) for each index.
    _POLLEN_INDEX = {  # 0=No pollen, 1=Low, 2=Moderate, 3=High, 4=Very High
        # (5=extra level just for contour levels)
        "index": np.array([0, 1, 2, 3, 4, 5]),
        "grass_pollen": np.array([0.0, 0.01, 30.0, 50.0, 150.0, 5000.0]),
        "birch_pollen": np.array([0.0, 0.01, 40.0, 80.0, 200.0, 5000.0]),
        "oak_pollen": np.array([0.0, 0.01, 30.0, 50.0, 200.0, 5000.0]),
        "hazel_pollen": np.array([0.0, 0.01, 30.0, 50.0, 80.0, 5000.0]),
        "alder_pollen": np.array([0.0, 0.01, 30.0, 50.0, 80.0, 5000.0]),
        "ash_pollen": np.array([0.0, 0.01, 30.0, 50.0, 200.0, 5000.0]),
        "plane_pollen": np.array([0.0, 0.01, 30.0, 50.0, 200.0, 5000.0]),
        "nettle_pollen": np.array([0.0, 0.01, 40.0, 80.0, 200.0, 5000.0]),
        "weed_pollen": np.array([0.0, 0.01, 40.0, 80.0, 200.0, 5000.0]),
    }

    # The output cube is a deepcopy of the input cube (to keep metadata) and is then manipulated in place
    _output_cube = None

    def _calculate(self, species: str):
        """Calculate the Pollen Index.

        Use values in _POLLEN_INDEX to determine the pollen index for each grid point.

        Args:
            species:
                The pollen species being processed, used to update the cube name and metadata
        """
        if species not in self._POLLEN_INDEX:
            raise ValueError(f"Pollen species {species} not handled")
        thresholds = self._POLLEN_INDEX[species]
        # Use np.digitize to find the index of the first threshold that is greater than the data value
        self._output_cube.data = (
            np.digitize(self._output_cube.data, thresholds) - 1
        )  # Subtract 1 to get 0-based index

    def _metadata(self, species: str):
        """Change the cube name and other metadata.
        Args:
            species:
                The pollen species being processed, used to update the cube name and metadata
        """
        # The 1-hour (PT01H) or 1-day (PT24H) period to use in the new cube name
        period = self._output_cube.name()[-5:]
        self._output_cube.rename(f"{species}_index_{period}")

        cube_attrbutes = self._output_cube.attributes
        # Change the following Attributes in the output cube if the key and old value
        # match, then change the value to the new value specified in the dictionary:
        attr_to_change_dict = {
            # key: [old value, new value]
            "quantity": ["Concentration", "Pollen Index"],
        }
        for attr, (old_value, new_value) in attr_to_change_dict.items():
            if attr in cube_attrbutes and cube_attrbutes[attr] == old_value:
                cube_attrbutes[attr] = new_value

    def process(self, cube: Cube) -> Cube:
        """Calculate the Pollen Index.

        Use values in _POLLEN_INDEX to determine the pollen index for each grid point,
        based on the pollen concentration values in the input cube.

        Args:
            cube:
                Input cube of hourly or daily pollen concentrations for a specific pollen type

        Returns:
            The calculated output cube.

        Warns:
            UserWarning:
                If output values fall outside typical expected ranges
        """
        species = cube.attributes.get("species").lower()
        self._output_cube = build_output_cube_with_new_units(self, cube, 1)
        self._calculate(species)
        self._metadata(species)
        return self._output_cube
