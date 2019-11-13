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

    def __init__(self):
        pass

    def process(self, cube, target_grid=None, source_landsea=None,
                metadata_dict=None, regrid_mode='bilinear',
                extrapolation_mode='nanmask', landmask_vicinity=25000,
                fix_float64=False):

        # if a target grid file has been specified, then regrid optionally
        # applying float64 data check, metadata change, Iris nearest and
        # extrapolation mode as required.
        if target_grid:
            regridder = Linear(extrapolation_mode=extrapolation_mode)

            if regrid_mode in ["nearest", "nearest-with-mask"]:
                regridder = Nearest(extrapolation_mode=extrapolation_mode)

            cube = cube.regrid(target_grid, regridder)

            if regrid_mode in ["nearest-with-mask"]:
                if not source_landsea:
                    msg = ("An argument has been specified that requires an input "
                           "landmask cube but none has been provided")
                    raise ValueError(msg)

                if "land_binary_mask" not in source_landsea.name():
                    msg = ("Expected land_binary_mask in input_landmask cube "
                           "but found {}".format(repr(source_landsea)))
                    warnings.warn(msg)

                if "land_binary_mask" not in target_grid.name():
                    msg = ("Expected land_binary_mask in target_grid cube "
                           "but found {}".format(repr(target_grid)))
                    warnings.warn(msg)

                cube = RegridLandSea(
                    vicinity_radius=landmask_vicinity).process(
                    cube, source_landsea, target_grid)

            target_grid_attributes = (
                {k: v for (k, v) in target_grid.attributes.items()
                 if 'mosg__' in k or 'institution' in k})
            cube = amend_metadata(
                cube, attributes=target_grid_attributes)

        # Change metadata only option:
        # if output file path and json metadata file specified,
        # change the metadata
        if metadata_dict:
            cube = amend_metadata(cube, **metadata_dict)

        check_cube_not_float64(cube, fix=fix_float64)

        return cube

