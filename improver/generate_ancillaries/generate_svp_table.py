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

    MAX_VALID_TEMPERATURE = 373.0
    MIN_VALID_TEMPERATURE = 173.0

    def __init__(
        self, t_min: float = 183.15, t_max: float = 338.25, t_increment: float = 0.1
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
        """
        self.t_min = t_min
        self.t_max = t_max
        self.t_increment = t_increment

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = (
            "<SaturatedVapourPressureTable: t_min: {}; t_max: {}; "
            "t_increment: {}>".format(self.t_min, self.t_max, self.t_increment)
        )
        return result

    def saturation_vapour_pressure_goff_gratch(self, temperature: ndarray) -> ndarray:
        """
        Saturation Vapour pressure in a water vapour system calculated using
        the Goff-Gratch Equation (WMO standard method).

        Args:
            temperature:
                Temperature values in Kelvin. Valid from 173K to 373K

        Returns:
            Corresponding values of saturation vapour pressure for a pure
            water vapour system, in hPa.

        References:
            Numerical data and functional relationships in science and
            technology. New series. Group V. Volume 4. Meteorology.
            Subvolume b. Physical and chemical properties of the air, P35.
        """
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
        triple_pt = TRIPLE_PT_WATER

        # Values for which method is considered valid (see reference).
        # WetBulbTemperature.check_range(temperature.data, 173., 373.)
        if (
            temperature.max() > self.MAX_VALID_TEMPERATURE
            or temperature.min() < self.MIN_VALID_TEMPERATURE
        ):
            msg = "Temperatures out of SVP table range: min {}, max {}"
            warnings.warn(msg.format(temperature.min(), temperature.max()))

        svp = temperature.copy()
        for cell in np.nditer(svp, op_flags=["readwrite"]):
            if cell > triple_pt:
                n0 = constants[1] * (1.0 - triple_pt / cell)
                n1 = constants[2] * np.log10(cell / triple_pt)
                n2 = constants[3] * (
                    1.0 - np.power(10.0, (constants[4] * (cell / triple_pt - 1.0)))
                )
                n3 = constants[5] * (
                    np.power(10.0, (constants[6] * (1.0 - triple_pt / cell))) - 1.0
                )
                log_es = n0 - n1 + n2 + n3 + constants[7]
                cell[...] = np.power(10.0, log_es)
            else:
                n0 = constants[8] * ((triple_pt / cell) - 1.0)
                n1 = constants[9] * np.log10(triple_pt / cell)
                n2 = constants[10] * (1.0 - (cell / triple_pt))
                log_es = n0 - n1 + n2 + constants[11]
                cell[...] = np.power(10.0, log_es)

        return svp

    def process(self) -> Cube:
        """
        Create a lookup table of saturation vapour pressure in a pure water
        vapour system for the range of required temperatures.

        Returns:
           A cube of saturated vapour pressure values at temperature
           points defined by t_min, t_max, and t_increment (defined above).
        """
        temperatures = np.arange(
            self.t_min, self.t_max + 0.5 * self.t_increment, self.t_increment
        )
        svp_data = self.saturation_vapour_pressure_goff_gratch(temperatures)

        temperature_coord = iris.coords.DimCoord(
            temperatures, "air_temperature", units="K"
        )

        # Output of the Goff-Gratch is in hPa, but we want to return in Pa.
        svp = iris.cube.Cube(
            svp_data,
            long_name="saturated_vapour_pressure",
            units="hPa",
            dim_coords_and_dims=[(temperature_coord, 0)],
        )
        svp.convert_units("Pa")
        svp.attributes["minimum_temperature"] = self.t_min
        svp.attributes["maximum_temperature"] = self.t_max
        svp.attributes["temperature_increment"] = self.t_increment

        return svp
