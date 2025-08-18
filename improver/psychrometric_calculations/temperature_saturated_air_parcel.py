# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""module to calculate the temperature of a saturated air parcel that has
ascended adiabatically from the cloud condensation level (CCL) to a pressure
level."""

from typing import Tuple, Union

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
        self.temperature = (None,)
        self.pressure = (None,)
        self.relative_humidity = (None,)

    def parcel_temp_after_ascent(
        self, temperature: Cube, pressure: Cube, relative_humidity: Cube
    ) -> Tuple[np.array, Cube]:
        """Calculates the temperature of a saturated air parcel when it has been lifted
        from the CCL to a pressure level. This has been set at 500 hPa for the easy
        calculation of Lifted Index (LI).

        Args:
            temperature:
                Cube of screen temperature
            pressure:
                Cube of air pressure at the surface
            relative_humidity:
                Cube of relative humidity at the surface
        Returns:
            Tuple of temperature of an air parcel at a pressure level (K) and temperature
            of CCL (K)
        """
        humidity = HumidityMixingRatio()([temperature, pressure, relative_humidity])
        CCL_temp, CCL_pressure = CloudCondensationLevel()(
            [temperature, pressure, humidity]
        )
        humidity_mixing_ratio_at_ccl = saturated_humidity(
            CCL_temp.data, CCL_pressure.data
        )
        t_dry = dry_adiabatic_temperature(
            CCL_temp.data, CCL_pressure.data, self.pressure_level
        )
        t_2, _ = adjust_for_latent_heat(
            t_dry, humidity_mixing_ratio_at_ccl, self.pressure_level
        )
        return t_2, CCL_temp

    def make_temperature_cube(
        self, temp_after_saturated_ascent: np.ndarray, ccl_temp: Cube
    ) -> Cube:
        """Puts the temperature information into a cube with appropriate metadata.

        Args:
            temp_after_saturated_ascent:
                An n dimensional array of the temperature after a saturated air parcel has
                been lifted adiabatically from the cloud condensation level (CCL) to a
                pressure level (K).
            ccl_temp:
                Cube of cloud condensation level temperature

        Returns:
            A cube of the temperature of a saturated air parcel when it has been lifted
            adiabatically from the CCL to another pressure level (K)
        """
        temp_cube = create_new_diagnostic_cube(
            name="parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level",
            units="K",
            template_cube=ccl_temp,
            mandatory_attributes=generate_mandatory_attributes([ccl_temp]),
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
        from the CCL to a pressure level.

        Args:
            cubes:
                Cubes of temperature (K), pressure (Pa)
                and relative humidity (fraction)

        Returns:
            Cube of parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level

        """
        cubes = as_cubelist(cubes)
        (self.temperature, self.pressure, self.relative_humidity) = CubeList(
            cubes
        ).extract(["air_temperature", "surface_air_pressure", "relative_humidity"])
        parcel_temp_at_pressure_level, ccl_temp = self.parcel_temp_after_ascent(
            self.temperature,
            self.pressure,
            self.relative_humidity,
        )
        temp_cube = self.make_temperature_cube(
            parcel_temp_at_pressure_level,
            ccl_temp,
        )
        return temp_cube
