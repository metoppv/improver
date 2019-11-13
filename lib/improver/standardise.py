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

from improver.metadata.amend import amend_metadata
from improver.metadata.check_datatypes import check_cube_not_float64
from improver.utilities.spatial import RegridLandSea


class StandardiseGridAndMetadata:

    """Plugin to regrid cube data and standardise metadata"""

    def __init__(self, regrid_mode='bilinear', extrapolation_mode='nanmask',
                 landmask=None, landmask_vicinity=25000):
        """
        Initialise regridding parameters
        """
        if not landmask and "nearest-with-mask" in regrid_mode:
            msg = ("An argument has been specified that requires an input "
                   "landmask cube but none has been provided")
            raise ValueError(msg)
        self.landmask = landmask
        self.regrid_mode = regrid_mode
        self.extrapolation_mode = extrapolation_mode
        self.landmask_vicinity = landmask_vicinity

    def _regrid_landsea(self, cube, target_grid):
        """
        Apply land-sea masking to the regridded cube. Raise warnings if
        landmask metadata is not as expected.
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
        Regrid cube to target_grid

        Args:
            cube (iris.cube.Cube):
                Input cube to be standardised
            target_grid (iris.cube.Cube):
                Cube on the required grid

        Returns:
            iris.cube.Cube
        """
        regridder = Linear(extrapolation_mode=self.extrapolation_mode)
        if self.regrid_mode in ["nearest", "nearest-with-mask"]:
            regridder = Nearest(extrapolation_mode=self.extrapolation_mode)
        cube = cube.regrid(target_grid, regridder)

        if self.regrid_mode in ["nearest-with-mask"]:
            cube = self._regrid_landsea(cube, target_grid)

        target_grid_attributes = (
            {k: v for (k, v) in target_grid.attributes.items()
            if 'mosg__' in k or 'institution' in k})
        cube = amend_metadata(cube, attributes=target_grid_attributes)

        return cube

    def process(self, cube, target_grid=None, metadata_dict=None,
                fix_float64=False):
        """
        Perform regridding and metadata adjustments

        Args:
            cube (iris.cube.Cube):
                Input cube to be standardised
            target_grid (iris.cube.Cube or None):
                Cube on the required grid
            metadata_dict (dict or None):
                Dictionary of required metadata updates
            fix_float64 (bool):
                Flag to de-escalate float64 precision

        Returns:
            iris.cube.Cube
        """

        if target_grid:
            cube = self._regrid_to_target(cube, target_grid)

        if metadata_dict:
            cube = amend_metadata(cube, **metadata_dict)

        check_cube_not_float64(cube, fix=fix_float64)

        return cube

