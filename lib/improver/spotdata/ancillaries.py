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
    """
    Takes in a list of desired diagnostics and determines which ancillary
    (i.e. non-time dependent) fields are required given their neighbour
    finding or data extraction methods.

    Args:
        diagnostics (dict):
            Dictionary containing each diagnostic to be processed with
            associated options for how they should be produced, e.g. method of
            neighbour selection, method of data extraction etc.

        ancillary_path (string):
            String giving the path of ancillary files to be used.

    Returns:
        ancillary_data (dict):
            Dictionary containing named ancillary data; the key gives the name
            and the item is the iris.cube.Cube of data.

    Raises:
        IOError if required input files are not found.

    """
    ancillary_data = {}

    try:
        orography = Load('single_file').process(
            ancillary_path + '/highres_orog.nc', diagnostic='surface_altitude')
    except:
        raise IOError('Orography file not found.')

    ancillary_data['orography'] = orography

    # Check if the land mask is used for any diagnostics.
    if any([(diagnostics[key]['neighbour_finding']['land_constraint'])
            for key in diagnostics.keys()]):
        try:
            land = Load('single_file').process(
                ancillary_path + '/land_mask.nc',
                diagnostic='land_binary_mask')
        except:
            raise IOError('Land mask file not found.')

        ancillary_data['land_mask'] = land

    return ancillary_data
