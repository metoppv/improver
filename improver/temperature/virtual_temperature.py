# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculate the virtual temperature from temperature and humidity mixing ratio."""

from typing import Union

from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.utilities.common_input_handle import as_cubelist


class VirtualTemperature(BasePlugin):
    """Plugin class to handle virtual temperature calculations."""

    @staticmethod
    def get_virtual_temperature(temperature: Cube, humidity_mixing_ratio: Cube) -> Cube:
        """
        Calculate the virtual temperature from temperature and humidity mixing ratio.

        Args:
            temperature:
                Cube of temperature.
            humidity_mixing_ratio:
                Cube of humidity mixing ratio.

        Returns:
            Cube of virtual_temperature.
        """
        # Calculate the virtual temperature
        virtual_temperature = temperature * (1 + 0.61 * humidity_mixing_ratio)
        # Workaround as cf-units id not correctly pickleable:
        # https://github.com/SciTools/iris/issues/6378
        # The units get lost when being calculated as part of running a graph
        # using the multiprocessing scheduler in dagrunner and so need to be
        # added back after the calculation on line 33.
        virtual_temperature.units = str(virtual_temperature.units)

        # Update the cube metadata
        virtual_temperature.rename("virtual_temperature")
        virtual_temperature.attributes["units_metadata"] = "on-scale"

        return virtual_temperature

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """
        Main entry point for this class.

        Args:
            cubes:
                air_temperature:
                    Cube of temperature.
                humidity_mixing_ratio:
                    Cube of humidity mixing ratios.

        Returns:
            Cube of virtual_temperature.
        """

        # Get the cubes into the correct format and extract the relevant cubes
        cubes = as_cubelist(*cubes)
        (self.temperature, self.humidity_mixing_ratio) = cubes.extract_cubes(
            ["air_temperature", "humidity_mixing_ratio"]
        )

        # Calculate the virtual temperature
        return self.get_virtual_temperature(
            self.temperature, self.humidity_mixing_ratio
        )
