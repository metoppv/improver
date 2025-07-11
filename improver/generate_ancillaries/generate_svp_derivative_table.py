# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""A module for creating a table of the first derivative of saturated vapour pressure"""

import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver.constants import TRIPLE_PT_WATER
from improver.generate_ancillaries.generate_svp_table import (
    SaturatedVapourPressureTable,
)


class SaturatedVapourPressureTableDerivative(SaturatedVapourPressureTable):
    """
    Plugin to create a first derivative saturated vapour pressure lookup table,
    which is only valid for temperatures between 173K and 373K.

    .. Further information is available in:
    .. include:: extended_documentation/generate_ancillaries/
       generate_svp_derivative_table.rst
    """

    cube_name = "saturated_vapour_pressure_derivative"
    svp_units = "hPa/K"
    svp_si_units = "Pa/K"

    def derivative_saturation_vapour_pressure_goff_gratch(
        self, temperature: ndarray
    ) -> ndarray:
        """
        Saturation vapour pressure first derivative in a water vapour system
        calculated using the Goff-Gratch Equation (WMO standard method).

        Args:
            temperature:
                Temperature values in Kelvin. Valid from 173K to 373K

        Returns:
            Corresponding values of saturation vapour pressure first derivative
            for a pure water vapour system, in hPa.

        References:
            Numerical data and functional relationships in science and
            technology. New series. Group V. Volume 4. Meteorology.
            Subvolume b. Physical and chemical properties of the air, P35.
        """
        self._check_temperature_limits(temperature)

        # Iterate over the temperature and original saturation vapour pressure
        # arrays simultaneously, updating the temperature values with newly calculated
        # first derivative saturation vapour pressure values.
        svp_original = self.saturation_vapour_pressure_goff_gratch(temperature)
        svp_derivative = temperature.copy()
        with np.nditer(
            [svp_derivative, svp_original], op_flags=["readwrite"]
        ) as it:
            for cell, svp_original_cell_val in it:
                if cell > TRIPLE_PT_WATER:
                    n0 = (self.constants[1] * TRIPLE_PT_WATER) / (cell**2)
                    n1 = self.constants[2] / (cell * np.log(10))
                    n2 = (
                        np.log(10)
                        * ((self.constants[3] * self.constants[4]) / TRIPLE_PT_WATER)
                        * np.power(10, (self.constants[4] * ((cell / TRIPLE_PT_WATER) - 1.0)))
                    )
                    n3 = (
                        np.log(10)
                        * ((self.constants[5] * self.constants[6] * TRIPLE_PT_WATER) / (cell**2))
                        * np.power(10, (self.constants[6] * (1.0 - (TRIPLE_PT_WATER / cell))))
                    )
                    cell[...] = np.log(10) * (n0 - n1 - n2 + n3) * svp_original_cell_val
                else:
                    n0 = self.constants[8] * (TRIPLE_PT_WATER / (cell**2))
                    n1 = self.constants[9] / (cell * np.log(10))
                    n2 = self.constants[10] / TRIPLE_PT_WATER
                    cell[...] = np.log(10) * (-n0 + n1 - n2) * svp_original_cell_val

        return svp_derivative

    def process(self) -> Cube:
        """
        Create a lookup table of saturation vapour pressure first derivative
        in a pure water vapour system for the range of required temperatures.

        Args:
            self.t_min (float):
                The minimum temperature (in Kelvin or Celsius, as appropriate) for the lookup table.
            self.t_max (float):
                The maximum temperature (in Kelvin or Celsius, as appropriate) for the lookup table.
            self.t_increment (float):
                The increment between temperature points in the lookup table.
            self.derivative_saturation_vapour_pressure_goff_gratch (callable):
                Method to calculate saturation vapour pressure first derivative for given temperatures.
            self.as_cube (callable):
                Method to convert data and temperatures into a Cube object.

        Returns:
           Cube:
               A cube of saturated vapour pressure derivative values at temperature
               points defined by t_min, t_max, and t_increment (defined above).
        """
        temperatures = np.arange(
            self.t_min, self.t_max + 0.5 * self.t_increment, self.t_increment
        )

        svp_derivative_data = self.derivative_saturation_vapour_pressure_goff_gratch(temperatures)

        svp_derivative = self.as_cube(svp_derivative_data, temperatures)

        return svp_derivative
