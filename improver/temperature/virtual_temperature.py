# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculate the virtual temperature."""

from typing import Union

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.constants import EARTH_REPSILON
from improver.utilities.common_input_handle import as_cubelist


class VirtualTemperature(BasePlugin):
    """Plugin class to handle virtual temperature calculations from humidity mixing ratio."""

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
        virtual_temperature.attributes["units_metadata"] = "on_scale"

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


class VirtualTemperatureFromSpecificHumidity(BasePlugin):
    """Plugin class to handle virtual temperature calculations from specific humidity.

    This calculates virtual temperature using the specific humidity output from StaGE
    as an input, which is desirable in the calculation of Air Density.

    This virtual temperature (Tv) calculation also uses condensates, if provided.

    Condensed water (rain, snow and liquid cloud droplets) add weight to a volume of air
    without contributing to the pressure.
    Their effect can be included in the calculation of virtual temperature.

    The data on the mixing ratios of liquid water (qcl) and ice (qcf) are available
    so a slightly improved estimate of Tv, accounting for this weight, is:

    Tv = T [(q / epsilon) + (1 - q - qcl - qcf - qR) ].

    This is only significant lower down in the atmosphere, and where there is cloud,
    and in most cases a very small adjustment. Given the densest clouds have condensate
    specific humidities typically no higher than 5 g kg-1, the adjustment is unlikely
    to ever be more than 0.5%.

    """

    @staticmethod
    def get_virtual_temperature_specific_humidity(
        temperature: Cube,
        specific_humidity: Cube,
        cloud_water_mixing_ratio: Cube = None,
        cloud_ice_mixing_ratio: Cube = None,
    ):
        """
        Calculate the virtual temperature from temperature,
        specific humidity and condensates, if provided.

        Args:
            Required:
                air_temperature:
                    Cube of temperature.
                specific_humidity:
                    Cube of specific humidity on pressure levels.
            Optional:
                cloud_water_mixing_ratio_on_pressure_levels:
                    Cube of cloud water mixing ratio on pressure levels.
                cloud_ice_mixing_ratio_on_pressure_levels:
                    Cube of cloud ice mixing ratio on pressure levels.

        Returns:
            Cube of virtual_temperature (K).
        """
        condensates = 0
        if cloud_water_mixing_ratio and cloud_ice_mixing_ratio:
            condensates = cloud_water_mixing_ratio - cloud_ice_mixing_ratio
        elif cloud_water_mixing_ratio:
            condensates = cloud_water_mixing_ratio
        elif cloud_ice_mixing_ratio:
            condensates = cloud_ice_mixing_ratio
        water_vapour_in_air = specific_humidity - condensates
        ratio_of_water_vapour_in_air = 1 - water_vapour_in_air
        ratio_of_gas_constants_of_dry_to_moist_air = specific_humidity / EARTH_REPSILON
        virtual_temperature = temperature * (
            ratio_of_gas_constants_of_dry_to_moist_air + ratio_of_water_vapour_in_air
        )
        virtual_temperature.data.astype(np.float32)
        # Workaround as cf-units id not correctly pickleable:
        # https://github.com/SciTools/iris/issues/6378
        # The units get lost when being calculated as part of running a graph
        # using the multiprocessing scheduler in dagrunner and so need to be
        # added back after the calculation on line 33.
        virtual_temperature.units = str(virtual_temperature.units)

        # Update the cube metadata
        virtual_temperature.rename("virtual_temperature")
        virtual_temperature.attributes["units_metadata"] = "on_scale"

        return virtual_temperature

    def __init__(self):
        self.temperature = None
        self.specific_humidity = None
        self.cloud_water_mixing_ratio = None
        self.cloud_ice_mixing_ratio = None

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """Main entry point for this class.
        Use Specific Humidity and optional condensate cubes to calculate
          the Virtual Temperature cube.

        Args:
            Required:
                cube:
                    air_temperature
                cube:
                    specific_humidity
            Optional:
                cube:
                    cloud_liquid_water_mixing_ratio
                cube:
                    cloud_ice_mixing_ratio

        Returns:
            Cube: virtual_temperature (K).
        """
        # Get the cubes into the correct format and extract the relevant cubes,
        cubes = as_cubelist(*cubes)

        # Extract the temperature cube.
        self.temperature = cubes.extract_cube("air_temperature")

        # Ensure air_temperature is in Kelvin.
        self.temperature.convert_units("K")

        # Extract the specific humidity cube.
        self.specific_humidity = cubes.extract_cube("specific_humidity")

        # If condensates have been given, extract them.
        for cube in cubes:
            if "cloud_liquid_water_mixing_ratio" in cube.name():
                self.cloud_water_mixing_ratio = cubes.extract_cube(
                    "cloud_liquid_water_mixing_ratio"
                )
            if "cloud_ice_mixing_ratio" in cube.name():
                self.cloud_ice_mixing_ratio = cubes.extract_cube(
                    "cloud_ice_mixing_ratio"
                )

        # Calculate the Virtual Temperature using the given inputs.
        return self.get_virtual_temperature_specific_humidity(
            self.temperature,
            self.specific_humidity,
            cloud_water_mixing_ratio=self.cloud_water_mixing_ratio,
            cloud_ice_mixing_ratio=self.cloud_ice_mixing_ratio,
        )
