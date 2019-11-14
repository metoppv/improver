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
import pprint

import iris
import dask.array as da
import numpy as np


def create_new_diagnostic_cube(
        name, units, coordinate_template, attributes=None, data=None,
        dtype=np.float32):
    """
    Creates a template for a new diagnostic cube with suitable metadata.

    Args:
        name (str):
            Standard or long name for output cube
        units (str or cf_units.Unit):
            Units for output cube
        coordinate_template (iris.cube.Cube):
            Cube from which to copy dimensional and auxiliary coordinates
        attributes (dict or None):
            Dictionary of attribute names and values
        data (numpy.ndarray or None):
            Data array.  If not set, cube is filled with zeros using a lazy
            data object, as this will be overwritten later by the caller
            routine.
        dtype (numpy.dtype):
            Datatype for dummy cube data if "data" argument is None.

    Returns:
        iris.cube.Cube:
            Cube with correct metadata to accommodate new diagnostic field
    """
    if data is None:
        data = da.zeros_like(coordinate_template.core_data(), dtype=dtype)

    aux_coords_and_dims, dim_coords_and_dims = [
        [(coord, coordinate_template.coord_dims(coord))
         for coord in getattr(coordinate_template, coord_type)]
        for coord_type in ('aux_coords', 'dim_coords')]

    cube = iris.cube.Cube(
        data, units=units, attributes=attributes,
        dim_coords_and_dims=dim_coords_and_dims,
        aux_coords_and_dims=aux_coords_and_dims)
    cube.rename(name)

    return cube


def generate_hash(data_in):
    """
    Generate a hash from the data_in that can be used to uniquely identify
    equivalent data_in.

    Args:
        data_in (any):
            The data from which a hash is to be generated. This can be of any
            type that can be pretty printed.
    Returns:
        str:
            A hexadecimal string which is a hash hexdigest of the data as a
            string.
    """
    bytestring = pprint.pformat(data_in).encode('utf-8')
    return hashlib.sha256(bytestring).hexdigest()


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
        str:
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
