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
"""
Functions to assist with interface to pysteps library
"""

import numpy as np
from numpy.ma import MaskedArray

import iris
from improver.utilities.load import load_cube


class PystepsImporter(object):
    """
    Class containing methods to import from IMPROVER NetCDF file or cube
    and return data and metadata as required by the pysteps plotting functions.

    Reference:
        https://pysteps.readthedocs.io/en/latest/pysteps_reference/
        io.html#pysteps-io-importers
    """
    def __init__(self):
        """
        Set up some universally required metadata
        """
        self.metadata = {}
        self.metadata['unit'] = 'mm/h'      # define pysteps-acceptable units
        self.metadata['accutime'] = 0       # TODO required?
        self.metadata['transform'] = None   # TODO required?

    @staticmethod
    def _extract_coord_properties(coord):
        """Get grid spacing and coordinate extremes
        Assumes points in ascending order"""
        coord.convert_units('m')
        spacing = np.diff(coord.points)[0]
        min_point = coord.points[0] - 0.5*spacing   # TODO points or bounds?
        max_point = coord.points[-1] + 0.5*spacing  # TODO points or bounds?
        return spacing, min_point, max_point

    def _get_x_metadata(self):
        """Get metadata from x-coordinate"""
        xspacing, min_x, max_x = self.extract_coord_properties(
            self.cube.coord(axis='x'))
        self.metadata['xpixelsize'] = xspacing
        self.metadata['x1'] = min_x
        self.metadata['x2'] = max_x

    def _get_y_metadata(self):
        """Get metadata from y-coordinate"""
        yspacing, min_y, max_y = self.extract_coord_properties(
            self.cube.coord(axis='y'))
        if yspacing < 0:
            # y-data indexed from top
            self.metadata['ypixelsize'] = -1*yspacing
            self.metadata['y1'] = max_y
            self.metadata['y2'] = min_y
            self.metadata['yorigin'] = 'upper'
        else:
            # y-data indexed from bottom
            self.metadata['ypixelsize'] = yspacing
            self.metadata['y1'] = min_y
            self.metadata['y2'] = max_y
            self.metadata['yorigin'] = 'lower'

    def _import_geodata(self):
        """
        Return projection and coordinate information from IMPROVER cube
        """
        projdef = ""
        for k,v in self.cube.coord_system().as_cartopy_crs().proj4_params.items():
            try:
                projdef += "+{}={:.3f} ".format(k, float(v))
            except ValueError:
                projdef += "+{}={} ".format(k, v)
        ellps = "WGS84"
        projdef += "+ellps={}".format(ellps)
        self.metadata["projection"] = projdef
        self._get_x_metadata()
        self._get_y_metadata()

    def process_cube(self, precip_cube):
        """
        Read required data and metadata from cube

        Args:
            precip_cube (iris.cube.Cube):
                Input precipitation rate cube with IMPROVER metadata

        Returns:
            precip_rate (np.ndarray):
                2D array of precipitation rates in mm/h
            metadata (dict):
                Dictionary of metadata required by pysteps algorithms
        """
        # check cube contains rates
        if 'rate' not in precip_cube.name():
            msg = '{} is not a precipitation rate cube'
            raise ValueError(msg.format(precip_cube.name()))

        # extract unmasked data in required units
        self.cube = precip_cube.copy()
        self.cube.convert_units(self.metadata['unit'])
        precip_rate = np.ma.filled(self.cube.data, np.nan)

        # populate metadata dictionary
        self.metadata['institution'] = self.cube.attributes['institution']
        self._import_geodata()

        return precip_rate, self.metadata

    def process_netcdf(self, filepath):
        """
        Read required data and metadata from file

        Args:
            filepath (str):
                Full path to precipitation rate file

        Returns:
            precip_rate (np.ndarray):
                2D array of precipitation rates in mm/h
            metadata (dict):
                Dictionary of metadata required by pysteps algorithms
        """
        cube = load_cube(filepath)
        return self.process_cube(cube)
