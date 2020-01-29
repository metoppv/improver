# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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

from improver.wxcode.wxcode_decision_tree import wxcode_decision_tree
from improver.wxcode.wxcode_decision_tree_global import (
    wxcode_decision_tree_global)

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

DAYNIGHT_CODES = [1, 3, 10, 14, 17, 20, 23, 26, 29]


def weather_code_attributes():
    """
    Returns:
        dict:
            Attributes defining weather code meanings.
    """
    import numpy as np
    attributes = {}
    wx_keys = np.array(list(WX_DICT.keys()))
    attributes.update({'weather_code': wx_keys})
    wxstring = " ".join(WX_DICT.values())
    attributes.update({'weather_code_meaning': wxstring})
    return attributes


def expand_nested_lists(query, key):
    """
    Produce flat lists from list and nested lists.

    Args:
        query (dict):
            A single query from the decision tree.
        key (str):
            A string denoting the field to be taken from the dict.

    Returns:
        list:
            A 1D list containing all the values for a given key.
    """
    items = []
    for item in query[key]:
        if isinstance(item, list):
            items.extend(item)
        else:
            items.extend([item])
    return items


def update_daynight(cubewx):
    """ Update weather cube depending on whether it is day or night

    Args:
        cubewx(iris.cube.Cube):
            Cube containing only daytime weather symbols.

    Returns:
        iris.cube.Cube:
            Cube containing day and night weather symbols

    Raises:
        CoordinateNotFoundError : cube must have time coordinate.

    """
    import iris
    import numpy as np
    from iris.exceptions import CoordinateNotFoundError
    import improver.utilities.solar as solar

    if not cubewx.coords("time"):
        msg = ("cube must have time coordinate ")
        raise CoordinateNotFoundError(msg)
    time_dim = cubewx.coord_dims('time')
    if not time_dim:
        cubewx_daynight = iris.util.new_axis(cubewx.copy(), 'time')
    else:
        cubewx_daynight = cubewx.copy()
    daynightplugin = solar.DayNightMask()
    daynight_mask = daynightplugin.process(cubewx_daynight)

    # Loop over the codes which decrease by 1 if a night time value
    # e.g. 1 - sunny day becomes 0 - clear night.
    for val in DAYNIGHT_CODES:
        index = np.where(cubewx_daynight.data == val)
        # Where day leave as is, where night correct weather
        # code to value  - 1.
        cubewx_daynight.data[index] = np.where(
            daynight_mask.data[index] == daynightplugin.day,
            cubewx_daynight.data[index],
            cubewx_daynight.data[index] - 1)

    if not time_dim:
        cubewx_daynight = iris.util.squeeze(cubewx_daynight)
    return cubewx_daynight


def interrogate_decision_tree(wxtree):
    """
    Obtain a list of necessary inputs from the decision tree as it is currently
    defined. Return a formatted string that contains the diagnostic names, the
    thresholds needed, and whether they are thresholded above or below these
    values. This output is used to create the CLI help, informing the user of
    the necessary inputs.

    Args:
        wxtree (str):
            The weather symbol tree that is to be interrogated.

    Returns:
        list of str:
            Returns a formatted string descring the diagnostics required,
            including threshold details.
    """

    # Get current weather symbol decision tree and populate a list of
    # required inputs for printing.
    if wxtree == 'high_resolution':
        queries = wxcode_decision_tree()
    elif wxtree == 'global':
        queries = wxcode_decision_tree_global()
    else:
        raise ValueError('Unknown decision tree name provided.')

    # Diagnostic names and threshold values.
    requirements = {}
    for query in queries.values():
        diagnostics = expand_nested_lists(query, 'diagnostic_fields')
        thresholds = expand_nested_lists(query, 'diagnostic_thresholds')
        for diagnostic, threshold in zip(diagnostics, thresholds):
            requirements.setdefault(diagnostic, set()).add(threshold)

    # Create a list of formatted strings that will be printed as part of the
    # CLI help.
    output = []
    for requirement, uniq_thresh in sorted(requirements.items()):
        units, = set(u for (_, u) in uniq_thresh)  # enforces same units
        thresh_str = ', '.join(map(str, sorted(v for (v, _) in uniq_thresh)))
        output.append('{} ({}): {}'.format(requirement, units, thresh_str))

    n_files = len(output)
    formatted_string = ('{}\n'*n_files)
    formatted_output = formatted_string.format(*output)

    return formatted_output
