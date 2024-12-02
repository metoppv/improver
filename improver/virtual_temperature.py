# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculate the gradient between two vertical levels."""

from typing import Union

from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.utilities.common_input_handle import as_cubelist


class VirtualTemperature(BasePlugin):
    """Calculates the virtual temperature from temperature and specific humidity."""

    @staticmethod
    def get_virtual_temperature(temperature: Cube, specific_humidity: Cube) -> Cube:
        """
        Calculate the virtual temperature from temperature and specific humidity.

        Args:
            temperature:
                Cube of temperature
            specific_humidity:
                Cube of specific humidity

        Returns:
            Cube of virtual temperature
        """
        # Calculate the virtual temperature
        virtual_temperature = temperature * (1 + 0.61 * specific_humidity)
        virtual_temperature.rename("air_temperature")

        return virtual_temperature

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """
        Main entry point for this class.

        Args:
            cubes:
                air_temperature:
                    Cube of temperature on pressure levels
                specific_humidity:
                    Cube of specific humidity on pressure levels.

        Returns:
            Cube of virtual temperature.
        """

        cubes = as_cubelist(*cubes)
        temperature, specific_humidity = cubes.extract(
            ["air_temperature", "specific_humidity"]
        )

        return self.get_virtual_temperature(temperature, specific_humidity)
