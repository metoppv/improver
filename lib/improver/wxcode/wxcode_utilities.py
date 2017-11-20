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

from collections import OrderedDict
import numpy as np

_WX_DICT_IN = {0: 'Clear_Night',
               1: 'Sunny_Day',
               2: 'Partly_Cloudy_Night',
               3: 'Partly_Cloudy_Day',
               4: 'Dust',
               5: 'Mist',
               6: 'Fog',
               7: 'Cloudy',
               8: 'Overcast',
               9: 'Light_Shower_Night',
               10: 'Light_Shower_Day',
               11: 'Drizzle',
               12: 'Light_Rain',
               13: 'Heavy_Shower_Night',
               14: 'Heavy_Shower_Day',
               15: 'Heavy_Rain',
               16: 'Sleet_Shower_Night',
               17: 'Sleet_Shower_Day',
               18: 'Sleet',
               19: 'Hail_Shower_Night',
               20: 'Hail_Shower_Day',
               21: 'Hail',
               22: 'Light_Snow_Shower_Night',
               23: 'Light_Snow_Shower_Day',
               24: 'Light_Snow',
               25: 'Heavy_Snow_Shower_Night',
               26: 'Heavy_Snow_Shower_Day',
               27: 'Heavy_Snow',
               28: 'Thunder_Shower_Night',
               29: 'Thunder_Shower_Day',
               30: 'Thunder'}

WX_DICT = OrderedDict(sorted(_WX_DICT_IN.items(), key=lambda t: t[0]))


def add_wxcode_metadata(cube):
    """ Add weather code metadata to a cube
    Args:
        cube (Iris.cube.Cube):
            Cube which needs weather code metadata added.
    Returns:
        cube (Iris.cube.Cube):
            Cube with weather code metadata added.
    """
    cube.long_name = "weather_code"
    cube.standard_name = None
    cube.var_name = None
    cube.units = "1"
    wx_keys = np.array(WX_DICT.keys())
    cube.attributes.update({'weather_code': wx_keys})
    wxstring = " ".join(WX_DICT.values())
    cube.attributes.update({'weather_code_meaning': wxstring})
    return cube


def expand_nested_lists(query, key):
    """
    Produce flat lists from list and nested lists.

    Args:
        query (dict):
            A single query from the decision tree.
        key (string):
            A string denoting the field to be taken from the dict.

    Returns:
        items (list):
            A 1D list containing all the values for a given key.
    """
    items = []
    for item in query[key]:
        if isinstance(item, list):
            items.extend(item)
        else:
            items.extend([item])
    return items
