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
"""Plugins written for the Improver site specific process chain."""

import os
import iris
from iris import FUTURE

FUTURE.netcdf_no_unlimited = True


class WriteOutput(object):
    """ Writes diagnostic cube data in a format determined by the method."""

    def __init__(self, method, dir_path=None, filename=None):
        """
        Select the method (format) for writing out the data cubes.

        Args:
            method (string):
                Method of writing data. Currently supported methods are:
                - as_netcdf : writes out a netcdf file.

            path (string):
                Optional string setting the output path for the file. If unset
                files are written to current working directory.

        """
        self.method = method
        self.dir_path = dir_path
        if dir_path is None:
            self.dir_path = os.getcwd()
        self.filename = filename

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<WriteOutput: method: {}, dir_path: {}>')
        return result.format(self.method, self.dir_path)

    def process(self, cube):
        """Call the required method"""
        try:
            function = getattr(self, self.method)
        except:
            raise AttributeError('Unknown method "{}" passed to '
                                 'WriteOutput.'.format(self.method))
        function(cube)

    def as_netcdf(self, cube):
        """
        Writes iris.cube.Cube data out to netCDF format files.

        Args:
            cube (iris.cube.Cube):
                Diagnostic data cube.

        Returns:
            Nil. Writes out file to filepath or working directory.

        """
        if self.filename is None:
            self.filename = cube.name()
        iris.save(
            cube, '{}.nc'.format(os.path.join(self.dir_path, self.filename)))
