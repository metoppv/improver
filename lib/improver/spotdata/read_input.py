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
Plugins written for the Improver site specific process chain.
For reading data files from UM output and site specification input.

"""

from iris import load_cube, load
from iris import FUTURE
from iris.cube import CubeList

FUTURE.netcdf_promote = True


class Load(object):

    """Plugin for loading data."""

    def __init__(self, method):
        """
        Simple function that currently takes a filename and loads a netCDF
        file.

        """
        self.method = method

    def process(self, filepath, diagnostic):
        """
        Simple wrapper for using iris load on a supplied netCDF file.

        Returns
        -------
        Cube
        A cube containing the data from the netCDF file.

        """
        function = getattr(self, self.method)
        return function(filepath, diagnostic)

    @staticmethod
    def single_file(filepath, diagnostic):
        """ Load and return a single iris.cube.Cube """
        return load_cube(filepath, diagnostic)

    @staticmethod
    def multi_file(filepath, diagnostic):
        """ Load multiple cubes and return a iris.cube.CubeList """
        return load(filepath, diagnostic)


def get_additional_diagnostics(diagnostic_name, diagnostic_data_path,
                               time_extract=None):
    """
    Load additional diagnostics needed for particular spot data processes.

    Args
    ----
    diagnostic_name : The name of the diagnostic to be loaded. Used to find
                      the relevant file.
    time_extract    : An iris constraint to extract and return only data from
                      the desired time.

    Returns
    -------
    cube            : An iris.cube.CubeList containing the desired diagnostic
                      data, with a single entry is time_extract is provided.

    """
    with FUTURE.context(cell_datetime_objects=True):
        cubes = Load('multi_file').process(
            diagnostic_data_path + '/*/*' + diagnostic_name + '*',
            None)
        if time_extract is not None:
            cube = cubes.extract(time_extract)
            cubes = CubeList()
            cubes.append(cube)
        return cubes
