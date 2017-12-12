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
from improver.psychrometric_calculations.psychrometric_calculations import (
    Utilities)


class SaturatedVapourPressureTable(object):

    """
    Plugin to create a saturated vapour pressure lookup table.
    """

    def __init__(self, t_min=183.15, t_max=338.15, t_increment=0.1):
        """
        Initialise class.

        Args:
            t_min : float
                The minimum temperature for the range.
            t_max : float
                The maximum temperature for the range.
            t_increment : float
                The temperature increment at which to create values for the
                saturated vapour pressure between t_min and t_max.
        """
        self.t_min = t_min
        self.t_max = t_max
        self.t_increment = t_increment

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<SaturatedVapourPressureTable: t_min: {}; t_max: {}; '
                  't_increment: {}>'.format(self.t_min, self.t_max,
                                            self.t_increment))
        return result

    def process(self):
        """
        Create a saturated vapour pressure lookup table by calling the
        Utilities.saturation_vapour_pressure_goff_gratch function in
        psychrometric_calculations.Utilities.

        Returns:
            svp : iris.cube.Cube
               A cube of saturated vapour pressure values at temperature
               points defined by t_min, t_max, and t_increment (defined above).
        """
        temperatures = np.arange(self.t_min, self.t_max + 0.5*self.t_increment,
                                 self.t_increment)
        temperature = iris.cube.Cube(temperatures, 'air_temperature',
                                     units='K')

        svp = Utilities.saturation_vapour_pressure_goff_gratch(temperature)

        temperature_coord = iris.coords.DimCoord(
            temperature.data, 'air_temperature', units='K')

        svp.add_dim_coord(temperature_coord, 0)
        svp.attributes['minimum_temperature'] = self.t_min
        svp.attributes['maximum_temperature'] = self.t_max
        svp.attributes['temperature_increment'] = self.t_increment

        return svp
