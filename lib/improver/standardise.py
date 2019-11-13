#!/usr/bin/env python
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
"""Plugin to regrid cube data and standardise metadata"""

import warnings
from iris.analysis import Nearest, Linear

from improver.metadata.amend import amend_attributes
from improver.metadata.check_datatypes import check_cube_not_float64
from improver.utilities.spatial import RegridLandSea


class StandardiseGridAndMetadata:

    """Plugin to regrid cube data and standardise metadata"""

    def __init__(self, regrid_mode='bilinear', extrapolation_mode='nanmask',
                 landmask=None, landmask_vicinity=25000,
                 regrid_attributes=None):
        """
        Initialise regridding parameters

        Args:
            regrid_mode (str):
                Mode of interpolation in regridding.
            extrapolation_mode (str):
                Mode to fill regions outside the domain in regridding.
            landmask (iris.cube.Cube or None):
                Land-sea mask, required for "nearest-with-mask" regrid option.
            landmask_vicinity (float):
                Radius of vicinity to search for a coastline, in metres
            regrid_attributes (list of str or None):
                List of attribute names to inherit from the target grid cube,
                eg mosg__model_configuration, that describe the new grid. If
                None, a list of Met Office-specific attributes is used.
        """
        if not landmask and "nearest-with-mask" in regrid_mode:
            msg = ("An argument has been specified that requires an input "
                   "landmask cube but none has been provided")
            raise ValueError(msg)
        self.landmask = landmask
        self.regrid_mode = regrid_mode
        self.extrapolation_mode = extrapolation_mode
        self.landmask_vicinity = landmask_vicinity

        self.regrid_attributes = regrid_attributes
        if self.regrid_attributes is None:
            self.regrid_attributes = [
                'mosg__grid_version', 'mosg__grid_domain', 'mosg__grid_type',
                'mosg__model_configuration', 'institution']

    def _regrid_landsea(self, cube, target_grid):
        """
        Apply land-sea masking to the regridded cube. Raise warnings if
        landmask metadata is not as expected.

        Returns:
            iris.cube.Cube: Regridded cube
        """
        if "land_binary_mask" not in self.landmask.name():
            msg = ("Expected land_binary_mask in input_landmask cube "
                   "but found {}".format(repr(self.landmask)))
            warnings.warn(msg)

        if "land_binary_mask" not in target_grid.name():
            msg = ("Expected land_binary_mask in target_grid cube "
                   "but found {}".format(repr(target_grid)))
            warnings.warn(msg)

        return RegridLandSea(vicinity_radius=self.landmask_vicinity).process(
            cube, self.landmask, target_grid)

    def _regrid_to_target(self, cube, target_grid):
        """
        Regrid cube to target_grid and inherit appropriate grid attributes

        Returns:
            iris.cube.Cube: Regridded cube with updated attributes
        """
        regridder = Linear(extrapolation_mode=self.extrapolation_mode)
        if self.regrid_mode in ["nearest", "nearest-with-mask"]:
            regridder = Nearest(extrapolation_mode=self.extrapolation_mode)
        cube = cube.regrid(target_grid, regridder)

        if self.regrid_mode in ["nearest-with-mask"]:
            cube = self._regrid_landsea(cube, target_grid)

        attributes_to_inherit = (
            {k: v for (k, v) in target_grid.attributes.items()
            if k in self.regrid_attributes})
        amend_attributes(cube, attributes_to_inherit)

        return cube

    def process(self, cube, target_grid=None, attributes_dict=None,
                fix_float64=False):
        """
        Perform regridding and metadata adjustments

        Args:
            cube (iris.cube.Cube):
                Input cube to be standardised
            target_grid (iris.cube.Cube or None):
                Cube on the required grid
            attributes_dict (dict or None):
                Dictionary of required attribute updates. Keys are
                attribute names, and values are the required value or "remove".
            fix_float64 (bool):
                Flag to de-escalate float64 precision

        Returns:
            iris.cube.Cube
        """
        if target_grid:
            cube = self._regrid_to_target(cube, target_grid)

        if attributes_dict:
            amend_attributes(cube, attributes_dict)

        check_cube_not_float64(cube, fix=fix_float64)

        return cube

