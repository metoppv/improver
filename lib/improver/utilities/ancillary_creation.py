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
    Utilities)


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

    def process(self):
        """
        Create a saturated vapour pressure lookup table by calling the
        Utilities.saturation_vapour_pressure_goff_gratch function in
        psychrometric_calculations.Utilities.

        Returns:
            svp : iris.cube.Cube
               A cube of saturated vapour pressure values at temperature
               points defined by T_min, T_max, and T_increment (defined above).
        """
        temperatures = np.arange(self.T_min, self.T_max + 0.5*self.T_increment,
                                 self.T_increment)
        temperature = iris.cube.Cube(temperatures, 'air_temperature',
                                     units='K')

        svp = Utilities.saturation_vapour_pressure_goff_gratch(temperature)

        temperature_coord = iris.coords.DimCoord(
            temperature.data, 'air_temperature', units='K')

        svp.add_dim_coord(temperature_coord, 0)
        svp.attributes['minimum_temperature'] = self.T_min
        svp.attributes['maximum_temperature'] = self.T_max
        svp.attributes['temperature_increment'] = self.T_increment

        return svp
