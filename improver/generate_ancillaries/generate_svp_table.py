# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""A module for creating a saturated vapour pressure table"""

import warnings

import iris
import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.constants import TRIPLE_PT_WATER


class SaturatedVapourPressureTable(BasePlugin):
    """
    Plugin to create a saturated vapour pressure lookup table.
    """

    cube_name = "saturated_vapour_pressure"
    svp_units = "hPa"
    svp_si_units = "Pa"
    MAX_VALID_TEMPERATURE_WATER = 373.0
    MAX_VALID_TEMPERATURE_ICE = 273.15
    MIN_VALID_TEMPERATURE_WATER = 223.0
    MIN_VALID_TEMPERATURE_ICE = 173.0
    constants = {
        1: 10.79574,
        2: 5.028,
        3: 1.50475e-4,
        4: -8.2969,
        5: 0.42873e-3,
        6: 4.76955,
        7: 0.78614,
        8: -9.09685,
        9: 3.56654,
        10: 0.87682,
        11: 0.78614,
    }

    def __init__(
        self,
        t_min: float = 183.15,
        t_max: float = 338.25,
        t_increment: float = 0.1,
        water_only: bool = False,
        ice_only: bool = False,
    ) -> None:
        """
        Create a table of saturated vapour pressures that can be interpolated
        through to obtain an SVP value for any temperature within the range
        t_min --> (t_max - t_increment).

        The default min/max values create a table that provides SVP values
        covering the temperature range -90C to +65.1C. Note that the last
        bin is not used, so the SVP value corresponding to +65C is the highest
        that will be used.

        Args:
            t_min:
                The minimum temperature for the range, in Kelvin.
            t_max:
                The maximum temperature for the range, in Kelvin.
            t_increment:
                The temperature increment at which to create values for the
                saturated vapour pressure between t_min and t_max.
            water_only:
                The table will only create values for the saturated vapour
                pressure with respect to water.
            ice_only:
                The table will only create values for the saturated vapour
                pressure with respect to ice.

        """
        self.t_min = t_min
        self.t_max = t_max
        self.t_increment = t_increment
        self.water_only = water_only
        self.ice_only = ice_only

        if self.water_only and self.ice_only:
            raise ValueError(
                "'water_only' and 'ice_only' flags cannot both be set to True"
            )

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = (
            f"<{self.__class__.__name__}: t_min: {self.t_min}; t_max: {self.t_max}; "
            f"t_increment: {self.t_increment}>"
        )
        return result

    def saturation_vapour_pressure_goff_gratch(self, temperature: ndarray) -> ndarray:
        """
        Saturation Vapour pressure in a water vapour system calculated using
        the Goff-Gratch Equation (WMO standard method).

        Args:
            temperature:
                Temperature values in Kelvin. Valid from 173 K to 373 K
                (173 K < T < 273.15 K for ice, 223 K < T < 373 K for water).

        Returns:
            Corresponding values of saturation vapour pressure for a pure
            water vapour system, in hPa.

        References:
            Numerical data and functional relationships in science and
            technology. New series. Group V. Volume 4. Meteorology.
            Subvolume b. Physical and chemical properties of the air, P35.
        """
        self._check_temperature_limits(temperature)

        # Iterate over the temperature array, updating the temperature values
        # with newly calculated saturation vapour pressure values.
        svp = temperature.copy()
        with np.nditer(svp, op_flags=["readwrite"]) as it:
            for cell in it:
                if (cell > TRIPLE_PT_WATER or self.water_only) and not self.ice_only:
                    n0 = self.constants[1] * (1.0 - TRIPLE_PT_WATER / cell)
                    n1 = self.constants[2] * np.log10(cell / TRIPLE_PT_WATER)
                    n2 = self.constants[3] * (
                        1.0
                        - np.power(
                            10.0, (self.constants[4] * (cell / TRIPLE_PT_WATER - 1.0))
                        )
                    )
                    n3 = self.constants[5] * (
                        np.power(
                            10.0, (self.constants[6] * (1.0 - TRIPLE_PT_WATER / cell))
                        )
                        - 1.0
                    )
                    log_es = n0 - n1 + n2 + n3 + self.constants[7]
                    cell[...] = np.power(10.0, log_es)
                else:
                    n0 = self.constants[8] * ((TRIPLE_PT_WATER / cell) - 1.0)
                    n1 = self.constants[9] * np.log10(TRIPLE_PT_WATER / cell)
                    n2 = self.constants[10] * (1.0 - (cell / TRIPLE_PT_WATER))
                    log_es = n0 - n1 + n2 + self.constants[11]
                    cell[...] = np.power(10.0, log_es)

        return svp

    def _check_temperature_limits(self, temperature: ndarray):
        """
        Raise exception if temperature values fall outside the range for which the
        method is considered valid (see reference).

        Args:
            temperature (ndarray):
                Array of temperature values to be validated.

        Raises:
            UserWarning:
                If any temperature value is outside the valid range defined by
                self.MIN_VALID_TEMPERATURE_ICE and self.MAX_VALID_TEMPERATURE_WATER,
                a warning is issued.

                If either self.water_only or self.ice_only has been set to True, the
                warning will use the corresponding minimum and maximum temperature
                values.

        Returns:
            None
        """
        if (
            temperature.max() > self.MAX_VALID_TEMPERATURE_WATER
            or temperature.min() < self.MIN_VALID_TEMPERATURE_ICE
        ) and not (self.water_only or self.ice_only):
            msg = "Temperatures out of SVP table range: min {}, max {}"
            warnings.warn(msg.format(temperature.min(), temperature.max()))
        elif (
            temperature.max() > self.MAX_VALID_TEMPERATURE_WATER
            or temperature.min() < self.MIN_VALID_TEMPERATURE_WATER
        ) and self.water_only:
            msg = "Temperatures out of SVP table range for water: min {}, max {}"
            warnings.warn(msg.format(temperature.min(), temperature.max()))
        elif (
            temperature.max() > self.MAX_VALID_TEMPERATURE_ICE
            or temperature.min() < self.MIN_VALID_TEMPERATURE_ICE
        ) and self.ice_only:
            msg = "Temperatures out of SVP table range for ice: min {}, max {}"
            warnings.warn(msg.format(temperature.min(), temperature.max()))

    def as_cube(self, svp_data: np.ndarray, temperatures: np.ndarray) -> Cube:
        """
        Converts saturation vapor pressure data and corresponding temperatures into an Iris Cube.

        Args:
            svp_data (np.ndarray):
                Array containing saturation vapor pressure values.
            temperatures (np.ndarray):
                Array of temperature values (in Kelvin) corresponding to the svp_data.

        Returns
            Cube:
                An Iris Cube containing the saturation vapor pressure data with temperature as a dimension coordinate.
                The cube is converted to SI units for vapor pressure and includes attributes for the minimum temperature,
                maximum temperature, and temperature increment used in the data.
        """
        temperature_coord = iris.coords.DimCoord(
            temperatures, "air_temperature", units="K"
        )
        # Output of the Goff-Gratch is in hPa, but we want to return in Pa.
        svp = iris.cube.Cube(
            svp_data,
            long_name=self.cube_name,
            units=self.svp_units,
            dim_coords_and_dims=[(temperature_coord, 0)],
        )
        svp.convert_units(self.svp_si_units)
        svp.attributes["minimum_temperature"] = self.t_min
        svp.attributes["maximum_temperature"] = self.t_max
        svp.attributes["temperature_increment"] = self.t_increment
        return svp

    def process(self) -> Cube:
        """
        Creates a lookup table of saturation vapour pressure in a pure water vapour system for a specified temperature range.

        Args:
            self.t_min (float):
                Minimum temperature (inclusive) for the lookup table.
            self.t_max (float):
                Maximum temperature (inclusive) for the lookup table.
            self.t_increment (float):
                Temperature increment for the lookup table.
            self.saturation_vapour_pressure_goff_gratch (callable):
                Method to calculate saturation vapour pressure for given temperatures.
            self.as_cube (callable):
                Method to convert data and temperatures into a Cube object.

        Returns:
            Cube:
                A cube of saturated vapour pressure values at temperature points defined by t_min, t_max, and t_increment.
        """
        temperatures = np.arange(
            self.t_min, self.t_max + 0.5 * self.t_increment, self.t_increment
        )
        svp_data = self.saturation_vapour_pressure_goff_gratch(temperatures)

        svp = self.as_cube(svp_data, temperatures)

        return svp
