# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Plugins to create ancillary data."""

import iris
import numpy as np
import improver.constants as cc
from improver.psychrometric_calculations.psychrometric_calculations import (
    WetBulbTemperature)


class SaturatedVapourPressureTable(object):

    """
    Plugin to create a saturated vapour pressure lookup table.
    """

    def __init__(self, T_min=183.15, T_max=338.15, T_increment=0.1):
        """
        Initialise class.

        Args:
            T_min : float
                The minimum temperature for the range.
            T_max : float
                The maximum temperature for the range.
            T_increment : float
                The temperature increment at which to create values for the
                saturated vapour pressure between T_min and T_max.
        """
        self.T_min = T_min
        self.T_max = T_max
        self.T_increment = T_increment

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<SaturatedVapourPressureTable: T_min: {}; T_max: {}; '
                  'T_increment: {}>'.format(self.T_min, self.T_max,
                                            self.T_increment))
        return result

    def saturation_vapour_pressure_goff_gratch(self, temperature):
        """
        Saturation Vapour pressure in a water vapour system calculated using
        the Goff-Gratch Equation (WMO standard method).

        Args:
            temperature : iris.cube.Cube
                Cube of temperature which will be converted to Kelvin
                prior to calculation. Valid from 173K to 373K

        Returns:
            svp : iris.cube.Cube
                Cube containing the saturation vapour pressure of a pure
                water vapour system. A correction must be applied to the data
                when used to convert this to the SVP in air; see the
                WetBulbTemperature._pressure_correct_svp function.

        References:
            Numerical data and functional relationships in science and
            technology. New series. Group V. Volume 4. Meteorology.
            Subvolume b. Physical and chemical properties of the air, P35.
        """
        constants = {1: 10.79574,
                     2: 5.028,
                     3: 1.50475E-4,
                     4: -8.2969,
                     5: 0.42873E-3,
                     6: 4.76955,
                     7: 0.78614,
                     8: -9.09685,
                     9: 3.56654,
                     10: 0.87682,
                     11: 0.78614}
        triple_pt = cc.TRIPLE_PT_WATER

        WetBulbTemperature._check_range(temperature, 173., 373.)

        data = temperature.data
        for cell in np.nditer(data, op_flags=['readwrite']):
            if cell > triple_pt:
                n0 = constants[1] * (1. - triple_pt / cell)
                n1 = constants[2] * np.log10(cell / triple_pt)
                n2 = constants[3] * (1. - np.power(10.,
                                                   (constants[4] *
                                                    (cell / triple_pt - 1.))))
                n3 = constants[5] * (np.power(10., (constants[6] *
                                                    (1. - triple_pt / cell))) -
                                     1.)
                log_es = n0 - n1 + n2 + n3 + constants[7]
                cell[...] = (np.power(10., log_es))
            else:
                n0 = constants[8] * ((triple_pt / cell) - 1.)
                n1 = constants[9] * np.log10(triple_pt / cell)
                n2 = constants[10] * (1. - (cell / triple_pt))
                log_es = n0 - n1 + n2 + constants[11]
                cell[...] = (np.power(10., log_es))

        # Create SVP cube
        temperature_coord = iris.coords.DimCoord(temperature.data,
                                                 'air_temperature', units='K')
        svp = iris.cube.Cube(
            data, long_name='saturated_vapour_pressure', units='hPa',
            dim_coords_and_dims=[(temperature_coord, 0)])
        svp.attributes['minimum_temperature'] = self.T_min
        svp.attributes['maximum_temperature'] = self.T_max
        svp.attributes['temperature_increment'] = self.T_increment
        # Output of the Goff-Gratch is in hPa, but we want to return in Pa.
        svp.convert_units('Pa')
        return svp

    def process(self):
        """
        Create a saturated vapour pressure lookup table.

        Returns:
            svp : iris.cube.Cube
               A cube of saturated vapour pressure values at temperature
               points defined by T_min, T_max, and T_increment (defined above).
        """
        n_points = (self.T_max - self.T_min) / self.T_increment
        temperatures = np.linspace(self.T_min, self.T_max, n_points)
        temperature = iris.cube.Cube(temperatures, 'air_temperature',
                                     units='K')
        svp = self.saturation_vapour_pressure_goff_gratch(temperature)
        return svp
