# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""module to calculate the temperature of a saturated air parcel that has
ascended adiabatically from the cloud condensation level (CCL) to a pressure
level."""

from typing import Tuple, Union

import numpy as np
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

    def __init__(self):
        """
        Set up class
        """
        self.temperature = (None,)
        self.pressure = (None,)
        self.RH = (None,)

    @staticmethod
    def parcel_temp_after_ascent(
        temperature: Cube,
        pressure: Cube,
        RH: Cube,
        pressure_level: np.float32 = 50000.0,
    ) -> Tuple[np.array, Cube]:
        """Calculates the temperature of a saturated air parcel when it has been lifted
        from the CCL to a pressure level. This has been set at 500 hPa for the easy
        calculation of Lifted Index (LI).

        Args:
            temperature:
                Cube of screen temperature
            pressure:
                Cube of air pressure at the surface
            RH:
                Cube of relative humidity at the surface
            pressure_level:
                The pressure level to which you want the air parcel ascend. The default
                is 500 hPa (for the Lifted Index)
        Returns:
            Tuple of temperature of an air parcel at a pressure level (K) and temperature
            of CCL (K)
        """
        humidity = HumidityMixingRatio()([temperature, pressure, RH])
        CCL_temp, CCL_pressure = CloudCondensationLevel()(
            [temperature, pressure, humidity]
        )
        humidity_mixing_ratio_at_ccl = saturated_humidity(
            CCL_temp.data, CCL_pressure.data
        )
        t_dry = dry_adiabatic_temperature(
            CCL_temp.data, CCL_pressure.data, pressure_level
        )
        t_2, _ = adjust_for_latent_heat(
            t_dry, humidity_mixing_ratio_at_ccl, pressure_level
        )
        return t_2, CCL_temp

    @staticmethod
    def make_temperature_cube(
        temp_after_saturated_ascent: np.ndarray,
        CCL_temp: Cube,
    ) -> Cube:
        """Puts the temperature information into a cube with appropriate metadata.

        Args:
            temp_after_saturated_ascent:
                An n dimensional array of the temperature after a saturated air parcel has
                been lifted adiabatically from the cloud condensation level (CCL) to a
                pressure level (K).
            CCL_temp:
                Cube of cloud condensation level temperature

        Returns:
            A cube of the temperature of a saturated air parcel when it has been lifted
            adiabatically from the CCL to another pressure level (K)
        """
        temp_cube = create_new_diagnostic_cube(
            name="parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level",
            units="K",
            template_cube=CCL_temp,
            mandatory_attributes=generate_mandatory_attributes([CCL_temp]),
            data=temp_after_saturated_ascent,
        )
        return temp_cube

    def process(
        self,
        *cubes: Union[Cube, CubeList],
        pressure_level: float = 50000.0,
    ) -> Cube:
        """
        Calculates the temperature of a saturated air parcel that has risen adiabatically
        from the CCL to a pressure level.

        Args:
            cubes:
                Cubes of temperature (K), pressure (Pa)
                and relative humidity
            pressure_level:
                The pressure level that the air parcel is lifted to adiabatically
                from the cloud condensation level.

        Returns:
            Cube of parcel_temperature_after_saturated_ascent_from_ccl_to_pressure_level

        """
        cubes = as_cubelist(cubes)
        (self.temperature, self.pressure, self.RH) = CubeList(cubes).extract(
            ["air_temperature", "surface_air_pressure", "relative_humidity"]
        )
        parcel_temp_at_pressure_level, CCL_temp = self.parcel_temp_after_ascent(
            self.temperature,
            self.pressure,
            self.RH,
            pressure_level,
        )
        temp_cube = self.make_temperature_cube(
            parcel_temp_at_pressure_level,
            CCL_temp,
        )
        return temp_cube
