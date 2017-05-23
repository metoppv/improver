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

"""
A module that loads and makes accessible shared data such as orography and
land masks.

"""

from improver.spotdata.read_input import Load


def get_ancillary_data(diagnostics, ancillary_path):
    '''
    Takes in a list of desired diagnostics and determines which ancillary
    (i.e. non-time dependent) fields are required given their neighbour
    finding or data extraction methods.

    Args:
    -----
    diagnostics: dictionary containing each diagnostic to be processed with
                 associated options for how they should be produced, e.g.
                 method of neighbour selection, method of data extraction etc.

    Returns:
    --------
    ancillary_data:
                 dictionary containing named ancillary data; the key gives the
                 name and the item is the iris.cube.Cube of data.

    '''
    ancillary_data = {}

    orography = Load('single_file').process(
        ancillary_path + '/orography.nc', 'surface_altitude')

    ancillary_data.update({'orography': orography})

    # Check if the land mask is used for any diagnostics.
    if any([('land' in diagnostics[key]['neighbour_finding'])
            for key in diagnostics.keys()]):

        land = Load('single_file').process(
            ancillary_path + '/land_mask.nc', 'land_binary_mask')

        ancillary_data.update({'land': land})

    return ancillary_data


# Function that checks the presence of ancillary data when it is used and
# raises an exception if it is missing.

def data_from_ancillary(ancillary_data, key):
    '''
    Check for an iris.cube.Cube of <key> information in the ancillary data
    dictionary.

    Args:
    -----
    ancillary_data : ancillary_data dictionary defined by get_ancillary_data
                     function.
    key            : name of ancillary field requested.

    Returns:
    --------
    iris.cube.Cube.data from the <key>.

    Raises:
    -------
    Exception if the <key> cube has not been loaded.

    '''

    if ancillary_data is not None and ancillary_data[key]:
        return ancillary_data[key].data
    else:
        raise Exception('Ancillary data {} has not been loaded.'.format(key))
