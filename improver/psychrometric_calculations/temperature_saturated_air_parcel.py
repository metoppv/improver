# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""module to calculate the temperature of a saturated air parcel that has
ascended adiabatically from the cloud condensation level (CCL) to a pressure
level."""

from typing import Union

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.psychrometric_calculations.cloud_condensation_level import (
    CloudCondensationLevel,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    HumidityMixingRatio,
    adjust_for_latent_heat,
    dry_adiabatic_temperature,
    saturated_humidity,
)
from improver.utilities.common_input_handle import as_cubelist


class TemperatureSaturatedAirParcel(BasePlugin):
    """
    Plugin to calculate the temperature of a saturated air parcel as it rises
    adiabatically from the cloud condensation level (CCL) to a pressure level.
    The default is set at 500 hPa for the Lifted Index (LI) calculations.
    """

    def __init__(self, pressure_level: float = 50000.0):
        """
        Set up class

        Args:
            pressure_level:
                The pressure level that the air parcel is lifted to adiabatically
                from the cloud condensation level. (Pa)
        """
        self.pressure_level = pressure_level
        self.temperature: Cube = None
        self.pressure: Cube = None

    def parcel_temp_after_ascent(self) -> np.array:
        """Calculates the temperature of a saturated air parcel when it has been lifted
        from the CCL to a pressure level. This has been set at 500 hPa for the easy
        calculation of Lifted Index (LI).

        Returns:
            Array of temperature of an air parcel at a pressure level (K)
        """
        relative_humidity_cube = self.make_saturated_relative_humidity_cube()
        humidity_cube = HumidityMixingRatio()(
            [self.temperature, self.pressure, relative_humidity_cube]
        )
        ccl_temperature_cube, ccl_pressure_cube = CloudCondensationLevel()(
            [self.temperature, self.pressure, humidity_cube]
        )
        humidity_mixing_ratio_at_ccl = saturated_humidity(
            ccl_temperature_cube.data, ccl_pressure_cube.data
        )
        dry_parcel_temperature_after_ascent = dry_adiabatic_temperature(
            ccl_temperature_cube.data, ccl_pressure_cube.data, self.pressure_level
        )
        parcel_temperature_after_ascent, _ = adjust_for_latent_heat(
            dry_parcel_temperature_after_ascent,
            humidity_mixing_ratio_at_ccl,
            self.pressure_level,
        )
        return parcel_temperature_after_ascent

    def make_saturated_relative_humidity_cube(self) -> Cube:
        """Creates a cube of relative humidity at the cloud condensation level (CCL)
        with a value of 1.0, as by definition the relative humidity is 100
        percent at the CCL.

        The temperature cube is used as a template for the metadata of the relative humidity cube.
        Only the name and units will be replaced.

        Returns:
            A cube of relative humidity at the CCL with a value of 1.0.
        """
        relative_humidity_cube = self.temperature.copy(
            np.ones_like(self.temperature.data)
        )
        relative_humidity_cube.rename("relative_humidity")
        relative_humidity_cube.units = "1"
        return relative_humidity_cube

    def make_temperature_cube(self, temp_after_saturated_ascent: np.ndarray) -> Cube:
        """Puts the temperature information into a cube with appropriate metadata.
        Uses self.temperature as a template for the metadata of the temperature cube.
        Only the name and units will be replaced.

        Args:
            temp_after_saturated_ascent:
                An n dimensional array of the temperature after a saturated air parcel has
                been lifted adiabatically from the cloud condensation level (CCL) to a
                pressure level (K).

        Returns:
            A cube of the temperature of a saturated air parcel when it has been lifted
            adiabatically from the CCL to another pressure level (K)
        """
        temp_cube = create_new_diagnostic_cube(
            name="parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level",
            units="K",
            template_cube=self.temperature,
            mandatory_attributes=generate_mandatory_attributes([self.temperature]),
            data=temp_after_saturated_ascent,
        )
        temp_cube.add_aux_coord(
            DimCoord(
                self.pressure_level,
                long_name="pressure",
                units="Pa",
                attributes={"positive": "down"},
            )
        )
        return temp_cube

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """
        Calculates the temperature of a saturated air parcel that has risen adiabatically
        from the cloud condensation level to a pressure level.

        Args:
            cubes:
                Cubes of temperature (K), pressure (Pa) at the cloud condensation level.


        Returns:
            Cube of parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level

        """
        cubes = as_cubelist(cubes)
        # Look for temperature and pressure cubes in the input list, and rename
        # them to air_temperature and surface_air_pressure since these names are
        # expected by the psychrometric calculations.
        for cube in cubes:
            cube_standard_name = cube.standard_name
            cube_name = cube.name()
            if ((cube_standard_name is not None and cube_standard_name.find("temperature") != -1) or
            (cube_name is not None and cube_name.find("temperature") != -1)):
                if cube.units != "K":
                    cube.convert_units("K")
                cube.rename("air_temperature")
                self.temperature = cube
            if ((cube_standard_name is not None and cube_standard_name.find("pressure") != -1) or
            (cube_name is not None and cube_name.find("pressure") != -1)):
                if cube.units != "Pa":
                    cube.convert_units("Pa")
                cube.rename("surface_air_pressure")
                self.pressure = cube

        if self.temperature is None:
            raise ValueError("Cube with 'temperature' in its name is required")
        if self.pressure is None:
            raise ValueError("Cube with 'pressure' in its name is required")

        parcel_temp_at_pressure_level = self.parcel_temp_after_ascent()
        temp_cube = self.make_temperature_cube(parcel_temp_at_pressure_level)
        return temp_cube
