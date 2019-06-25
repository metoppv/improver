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
"""Plugin to calculate blend weights and blend data across a dimension"""

from cf_units import Unit
import numpy as np
import iris

from improver.utilities.cube_manipulation import sort_coord_in_cube
from improver.blending.weights import (
    ChooseWeightsLinear, ChooseDefaultWeightsLinear,
    ChooseDefaultWeightsNonLinear)
from improver.blending.weighted_blend import (
    MergeCubesForWeightedBlending, WeightedBlendAcrossWholeDimension)


def calculate_blending_weights(cube, blend_coord, method,
                               blend_coord_unit=None,
                               weighting_coord=None, wts_dict=None,
                               y0val=None, ynval=None, cval=None):
    """
    Wrapper for plugins to calculate blending weights using the command line
    options specified.

    Args:
        cube (iris.cube.Cube):
            Cube of input data to be blended
        blend_coord (str):
            Coordinate over which blending will be performed (eg "model" for
            grid blending)
        method (str):
            Weights calculation method ("linear", "nonlinear", "dict" or
            "mask")

    Kwargs:
        blend_coord_unit (str or cf_units.Unit):
            Unit of blending coordinate (for default weights plugins)
        weighting_coord (str):
            Coordinate over which linear weights should be calculated from dict
        wts_dict (dict):
            Dictionary containing parameters for linear weights calculation
        y0val (float):
            Intercept parameter for default linear weights plugin
        ynval (float):
            Gradient parameter for default linear weights plugin
        cval (float):
            Parameter for default non-linear weights plugin

    Returns:
        weights (iris.cube.Cube):
            Cube containing 1D array of weights for blending
    """
    # sort input cube by blending coordinate
    cube = sort_coord_in_cube(cube, blend_coord, order="ascending")

    # set blending coordinate units
    if "time" in blend_coord:
        coord_unit = Unit(blend_coord_unit, "gregorian")
    elif blend_coord_unit != 'hours since 1970-01-01 00:00:00.':
        coord_unit = blend_coord_unit
    else:
        coord_unit = 'no_unit'

    # calculate blending weights
    if method == "dict":
        # get dictionary access
        if "model" in blend_coord:
            config_coord = "model_configuration"
        else:
            config_coord = blend_coord

        # calculate linear weights from dictionary
        weights_cube = ChooseWeightsLinear(
            weighting_coord, wts_dict,
            config_coord_name=config_coord).process(cube)

        # sort weights cube by blending coordinate
        # TODO do we need this now?  Check / match order in blending plugin...
        weights = sort_coord_in_cube(
            weights_cube, blend_coord, order="ascending")

    elif method == "linear":
        weights = ChooseDefaultWeightsLinear(
            y0val=y0val, ynval=ynval).process(
                cube, blend_coord, coord_unit=coord_unit)

    elif method == "nonlinear":
        # this is set here rather than in the CLI arguments in order to check
        # for invalid argument combinations
        cvalue = cval if cval else 0.85
        weights = ChooseDefaultWeightsNonLinear(cvalue).process(
            cube, blend_coord, coord_unit=coord_unit)

    return weights



