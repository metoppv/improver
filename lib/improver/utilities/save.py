# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Module for saving netcdf cubes with desired attribute types."""

import iris


def append_metadata_cube(cubelist, global_keys):
    """ Create a metadata cube associated with statistical
        post-processing attributes of the input cube list.

    Args:
        cubelist (iris.cube.CubeList):
            List of cubes to be saved
        global_keys (list):
            List of attributes to be treated as global across cubes and within
            any netCDF files produced using these cubes.
    Returns:
        iris.cube.Cubelist with appended metadata cube
    """
    keys_for_global_attr = {}

    # Collect keys from each cubes attributes that match with global_keys
    for cube in cubelist:
        keys = cube.attributes
        keys_for_global_attr = {k for k in keys.keys() if k in global_keys}

    # Set up a basic prefix cube
    prefix_cube = iris.cube.Cube(0, long_name='prefixes',
                                 var_name='prefix_list')

    # Attributes have to appear on all cubes in a cubelist for Iris 2 to save
    # these attributes as global in a resulting netCDF file, so add all of the
    # global attributes to the prefix cube (otherwise they will be made
    # variables in the netCDF file).
    for key in keys_for_global_attr:
        prefix_cube.attributes[key] = cube.attributes[key]

    # Add metadata prefix attributes to the prefix cube
    prefix_cube.attributes['spp__'] = \
        'http://reference.metoffice.gov.uk/statistical-process/properties/'
    prefix_cube.attributes['spv__'] = \
        'http://reference.metoffice.gov.uk/statistical-process/values/'
    prefix_cube.attributes['spd__'] = \
        'http://reference.metoffice.gov.uk/statistical-process/def/'
    prefix_cube.attributes['rdf__'] = \
        'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
    prefix_cube.attributes['bald__'] = 'http://binary-array-ld.net/latest/'

    cubelist.append(prefix_cube)
    # bald__isPrefixedBy should be an attribute on all the cubes
    for cube in cubelist:
        cube.attributes['bald__isPrefixedBy'] = 'prefix_list'

    return cubelist


def save_netcdf(cubelist, filename):
    """Save the input Cube or CubeList as a NetCDF file.

    Uses the functionality provided by iris.fileformats.netcdf.save with
    local_keys to record non-global attributes as data attributes rather than
    global attributes.

    Args:
        cubelist (iris.cube.Cube or iris.cube.CubeList):
            Cube or list of cubes to be saved
        filename (str):
            Filename to save input cube(s)
    """
    if isinstance(cubelist, iris.cube.Cube):
        cubelist = [cubelist]

    global_keys = ['title', 'um_version', 'grid_id', 'source', 'Conventions',
                   'institution', 'history', 'bald__isPrefixedBy']
    local_keys = {key for cube in cubelist
                  for key in cube.attributes.keys()
                  if key not in global_keys}

    cubelist = append_metadata_cube(cubelist, global_keys)

    iris.fileformats.netcdf.save(cubelist, filename, local_keys=local_keys)
