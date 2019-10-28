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
"""General IMPROVER metadata utilities"""

import hashlib
import json


def generate_hash(data_in):
    """
    Generate a hash from the data_in that can be used to uniquely identify
    equivalent data_in.

    Args:
        data_in (any):
            The data from which a hash is to be generated. This can be of any
            type that can be dumped to JSON (dict, list, str, float, etc).
    Returns:
        hash (string):
            A hexadecimal string which is a hash hexdigest of the data as a
            string.
    """
    json_bytestring = json.dumps(data_in, sort_keys=True).encode('utf-8')
    return hashlib.sha256(json_bytestring).hexdigest()


def create_coordinate_hash(cube):
    """
    Generate a hash based on the input cube's x and y coordinates. This
    acts as a unique identifier for the grid which can be used to allow two
    grids to be compared.

    Args:
        cube (iris.cube.Cube):
            The cube from which x and y coordinates will be used to
            generate a hash.
    Returns:
        coordinate_hash (str):
            A hash created using the x and y coordinates of the input cube.
    """
    hashable_data = []
    for axis in ('x', 'y'):
        coord = cube.coord(axis=axis)
        hashable_data.extend([
            list(coord.points),
            list(coord.bounds) if isinstance(coord.bounds, list) else None,
            coord.standard_name,
            coord.long_name,
            coord.coord_system,
            coord.units
        ])
    return generate_hash(hashable_data)
