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
"""This module defines the utilities required for wxcode plugin """

WXCODE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
WXMEANING = ['Clear Night',
             'Sunny Day',
             'Partly Cloudy Night',
             'Partly Cloudy Day',
             'Dust',
             'Mist',
             'Fog',
             'Cloudy',
             'Overcast',
             'Light Shower Night',
             'Light Shower Day',
             'Drizzle',
             'Light Rain',
             'Heavy Shower Night',
             'Heavy Shower Day',
             'Heavy Rain',
             'Sleet Shower Night',
             'Sleet Shower Day',
             'Sleet',
             'Hail Shower Night',
             'Hail Shower Day',
             'Hail',
             'Light Snow Shower Night',
             'Light Snow Shower Day',
             'Light Snow',
             'Heavy Snow Shower Night',
             'Heavy Snow Shower Day',
             'Heavy Snow',
             'Thunder Shower Night',
             'Thunder Shower Day',
             'Thunder']


def add_wxcode_metadata(cube):
    """ Add weather code metadata to a cube
    Args:
        cube: Iris.cube.Cube
            Cube which needs weather code metadata added.
    Returns:
        cube: Iris.cube.Cube
            Cube with weather code metadata added.
    """
    cube.long_name = "weather_code"
    cube.standard_name = None
    cube.var_name = None
    cube.units = "1"
    cube.attributes.update({'weather_code': WXCODE})
    cube.attributes.update({'weather_code_meaning': WXMEANING})
    return cube
