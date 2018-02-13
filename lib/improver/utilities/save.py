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
"""Module for saving netcdf cubes with desired attribute types."""

import iris

iris.FUTURE.netcdf_promote = True
iris.FUTURE.netcdf_no_unlimited = True


def save_netcdf(cube, filename, unlimited_dimensions=None):
    """Save the cube provided as a NetCDF file.

    Uses the functionality provided by iris.fileformats.netcdf.save with
    local_keys to record shared attributes as data attributes rather than
    global attributes.

    NOTE current wrapper is a placeholder replicating the existing iris.save
    functionality.

    Args:
        cube (iris.cube.Cube):
            Ouptut cube
        filename (str):
            Filename to save output cube

    Kwargs:
        unlimited_dimensions (iterable of strings and/or iris.coords.Coord):
            See iris.fileformats.netcdf.save
    """

    local_keys = None
    # TODO perform appropriate cube manipulation here to obtain local_keys

    iris.fileformats.netcdf.save(cube, filename, local_keys=local_keys,
                                 unlimited_dimensions=unlimited_dimensions)
