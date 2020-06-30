# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
Functions to assist with interface to pysteps plotting routines.

This module is for plotting only and should NOT be merged into the main
IMPROVER master.
"""

import numpy as np

from improver.utilities.load import load_cube
from improver.utilities.spatial import (
    calculate_grid_spacing,
    check_if_grid_is_equal_area,
)


class PystepsImporter(object):
    """
    Class containing methods to import from IMPROVER NetCDF file or cube
    and return data and metadata.  This is to allow use of the pysteps
    plotting functions in investigations only, and should not be required for
    any of the pysteps processing routines.

    Reference:
        https://pysteps.readthedocs.io/en/latest/pysteps_reference/
        io.html#pysteps-io-importers
    """

    def __init__(self):
        """
        Set up some universally required metadata
        """
        self.metadata = {}
        self.metadata["unit"] = "mm/h"  # pysteps-acceptable units
        self.metadata["accutime"] = 0  # accumulation period (0 for rates)
        self.metadata["transform"] = None  # data transform

    @staticmethod
    def _extract_coord_limits(coord):
        """Get coordinate extremes in metres"""
        n_coord = coord.copy()
        n_coord.convert_units("m")
        return min(n_coord.points), max(n_coord.points)

    def _set_coord_metadata(self):
        """Extract metadata from x- and y-coordinates and set in dict"""
        min_x, max_x = self._extract_coord_limits(self.cube.coord(axis="x"))
        self.metadata["xpixelsize"] = calculate_grid_spacing(
            self.cube, "metres", axis="x"
        )
        self.metadata["x1"] = min_x
        self.metadata["x2"] = max_x
        min_y, max_y = self._extract_coord_limits(self.cube.coord(axis="y"))
        self.metadata["ypixelsize"] = calculate_grid_spacing(
            self.cube, "metres", axis="y"
        )
        self.metadata["y1"] = min_y
        self.metadata["y2"] = max_y
        self.metadata["yorigin"] = "lower"

    def _set_geodata(self):
        """
        Set projection and coordinate information in metadata dictionary
        """
        projection = ""
        crd_sys = self.cube.coord_system()
        for k, v in crd_sys.as_cartopy_crs().proj4_params.items():
            try:
                projection += "+{}={:.3f} ".format(k, float(v))
            except ValueError:
                projection += "+{}={} ".format(k, v)
        projection += "+ellps=WGS84"
        self.metadata["projection"] = projection
        self._set_coord_metadata()

    def process_cube(self, precip_cube):
        """
        Generate pysteps metadata dictionary and data array from cube

        Args:
            precip_cube (iris.cube.Cube):
                Input precipitation rate cube with IMPROVER metadata

        Returns:
            (tuple) : tuple containing:
                **precip_rate** (np.ndarray):
                    2D array of precipitation rates in mm/h
                **metadata** (dict):
                    Dictionary of metadata required by pysteps algorithms

        Raises:
            ValueError: if input cube does not contain some variety of
                precipitation rates
        """
        if "rate" not in precip_cube.name():
            msg = "{} is not a precipitation rate cube"
            raise ValueError(msg.format(precip_cube.name()))
        check_if_grid_is_equal_area(precip_cube)

        # extract unmasked data in required units
        self.cube = precip_cube.copy()
        self.cube.convert_units(self.metadata["unit"])
        precip_rate = np.ma.filled(self.cube.data, np.nan)

        # populate metadata dictionary
        try:
            self.metadata["institution"] = self.cube.attributes["institution"]
        except KeyError:
            self.metadata["institution"] = "unknown"
        self._set_geodata()
        return precip_rate, self.metadata

    def process_netcdf(self, filepath):
        """
        Generate pysteps metadata dictionary and data array from file

        Args:
            filepath (str):
                Full path to precipitation rate file

        Returns:
            (tuple) : tuple containing:
                **precip_rate** (np.ndarray):
                    2D array of precipitation rates in mm/h
                **metadata** (dict):
                    Dictionary of metadata required by pysteps algorithms
        """
        cube = load_cube(filepath)
        return self.process_cube(cube)
