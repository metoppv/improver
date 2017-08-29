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

# import numpy as np
import iris
# import cf_units

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
    def process(temperature, rel_humidity, pressure=None):
        """Calculate the Wet Bulb Temperature

        Parameters
        ----------
        temp: cube
            Cube of temperature which will be converted to Celsius
            prior to calculation
            Valid from -100C to 200 C
        rel_humidity: cube
            Cube of relative humidity in %
        pressure: cube or None
            Cube of pressure which will be converted to kilopPascals
            prior to calculation
            In many cases the user may wish to use cubes temperature and
            relative humidity on 


        Returns
        -------
        saturation : cube
            Cube containing the saturation vapour pressure of the
            air in Pa

        """
        
        # Check that inputs are of correct type
        if not isinstance(temperature, iris.cube.Cube):
            emsg = "Temperature cube is not a cube, but {}"
            raise ValueError(emsg.format(type(temperature)))
        if not isinstance(rel_humidity, iris.cube.Cube):
            emsg = "Temperature cube is not a cube, but {}"
            raise ValueError(emsg.format(type(rel_humidity)))
#         if not isinstance(pressure, (cube, float)):
#             emsg = "Pressure is not a cube or a float, but {}"
#             raise ValueError(emsg.format(type(pressure)))

        two_d_coords = [temperature.coord(axis=ax).name() for ax in ('x', 'y')]
        match = list(set([coord.name() for coord in temperature.coords()]) - set(two_d_coords))
        
        if pressure is None:
            ls = []
        # Use case: I have cubes of rh and temperature on pressure levels
            for t in temperature.slices(two_d_coords):
                for rh in rel_humidity.slices(two_d_coords):
                    if rh.coords() == t.coords():
                        wb = wet_bulb(t, rh, t.coord('pressure').points[0])
                        ls.append(wb)
            ls = iris.cube.CubeList(ls)
            return ls.merge_cube()

        elif isinstance(pressure, iris.cube.Cube):
        # Use case: I have cubes of "surface" pressure, rh and temperature
            if not isinstance(pressure, iris.cube.Cube):
                emsg = "Pressure is not a cube, but {}"
                raise ValueError(emsg.format(type(pressure)))
            
            # Check that each input cube is flat in the z dim
            for cube in temperature, rel_humidity, pressure:
                # replace this with isinstance?
                try:
                    if  len(cube.coord(axis='z').points) != 1:
                        emsg = ("In cases where pressure is given "
                                "all cubes should have either no z dimension"
                                " or a scalar value.\n"
                                "Your cube of {} has dimension {}"
                                " with length {}")
                        emsg.format(cube.name, cube.coord(axis='z'),
                                    len(cube.coord(axis='z').points))
                        raise TypeError(emsg)
                except:
                    pass
            ls = []
            for t in temperature.slices(two_d_coords):
                for rh in rel_humidity.slices(two_d_coords):
                    for p in p.slices(two_d_coords):
                        if rh.coords() == t.coords():
                            wb =(wet_bulb(t,
                                          rh,
                                          p))
                            ls.append(wb)
            ls = iris.cube.CubeList(ls)
            return ls.merge_cube()
    
    
