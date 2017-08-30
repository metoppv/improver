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
"""Module to contain plugins for use with psychrometric calculations"""

import iris


from improver.psychrometric_calculations.psychrometric_calculations import (
    wet_bulb)


class WetBulb(object):
    """Calculate the wet bulb temperature given cubes of temperature,
    relative humidity and pressure"""

    def __init__(self):
        """Initialise class."""
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<WetBulbTemperature>')
        return result

    @staticmethod
    def process(temperature, rel_humidity, pressure):
        """Calculate the Wet Bulb Temperature

        Parameters
        ----------
        temp: cube
            Cube of temperature which will be converted to Celsius
            prior to calculation
            Valid from -100C to 200 C
        rel_humidity: cube
            Cube of relative humidity
        pressure: cube
            Cube of pressure which will be converted to kilopPascals
            prior to calculation

        Returns
        -------
        saturation : cube
            Cube containing the saturation vapour pressure of the
            air in Pa

        """
        # Check that inputs are of correct type
        if not isinstance(temperature, iris.cube.Cube):
            emsg = "Temperature is not a cube, but {}"
            raise TypeError(emsg.format(type(temperature)))
        if not isinstance(rel_humidity, iris.cube.Cube):
            emsg = "Temperature is not a cube, but {}"
            raise TypeError(emsg.format(type(rel_humidity)))
        if not isinstance(pressure, iris.cube.Cube):
            emsg = "Pressure is not a cube, but {}"
            raise TypeError(emsg.format(type(pressure)))

        # Check that cubes are all the same shape
        if not temperature.shape == pressure.shape == rel_humidity.shape:
            emsg = ("input cubes must have the same shapes. Your"
                    " input cubes have the following shapes:\n"
                    "Temperature: {}\n"
                    "Relative Humidity: {}\n"
                    "Pressure: {} \n")
            raise TypeError(emsg.format(temperature.shape,
                                        rel_humidity.shape,
                                        pressure.shape))

        # Wrap the wetbulb function and return it.
        return wet_bulb(temperature, rel_humidity, pressure)
