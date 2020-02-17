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

"""Script to run weighted blending."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube,
            coordinate,
            weighting_method='linear',
            weighting_coord='forecast_period',
            weighting_config: cli.inputjson = None,
            attributes_config: cli.inputjson = None,
            cycletime: str = None,
            y0val: float = None,
            ynval: float = None,
            cval: float = None,
            model_id_attr: str = None,
            spatial_weights_from_mask=False,
            fuzzy_length=20000.0):
    """Runs weighted blending.

    Check for inconsistent arguments, then calculate a weighted blend
    of input cube data using the options specified.

    Args:
        cubes (iris.cube.CubeList):
            Cubelist of cubes to be blended.
        coordinate (str):
            The coordinate over which the blending will be applied.
        weighting_method (str):
            Method to use to calculate weights used in blending.
            "linear" (default): calculate linearly varying blending weights.
            "nonlinear": calculate blending weights that decrease
            exponentially with increasing blending coordinates.
            "dict": calculate weights using a dictionary passed in.
        weighting_coord (str):
            Name of coordinate over which linear weights should be scaled.
            This coordinate must be available in the weights dictionary.
        weighting_config (dict or None):
            Dictionary from which to calculate blending weights. Dictionary
            format is as specified in
            improver.blending.weights.ChoosingWeightsLinear
        attributes_config (dict):
            Dictionary describing required changes to attributes after blending
        cycletime (str):
            The forecast reference time to be used after blending has been
            applied, in the format YYYYMMDDTHHMMZ. If not provided, the
            blended file takes the latest available forecast reference time
            from the input datasets supplied.
        y0val (float):
            The relative value of the weighting start point (lowest value of
            blend coord) for choosing default linear weights.
            If used this must be a positive float or 0.
        ynval (float):
            The relative value of the weighting end point (highest value of
            blend coord) for choosing default linear weights. This must be a
            positive float or 0.
            Note that if blending over forecast reference time, ynval >= y0val
            would normally be expected (to give greater weight to the more
            recent forecast).
        cval (float):
            Factor used to determine how skewed the non-linear weights will be.
            A value of 1 implies equal weighting.
        model_id_attr (str):
            The name of the dataset attribute to be used to identify the source
            model when blending data from different models.
        spatial_weights_from_mask (bool):
            If True, this option will result in the generation of spatially
            varying weights based on the masks of the data we are blending.
            The one dimensional weights are first calculated using the chosen
            weights calculation method, but the weights will then be adjusted
            spatially based on where there is masked data in the data we are
            blending. The spatial weights are calculated using the
            SpatiallyVaryingWeightsFromMask plugin.
        fuzzy_length (float):
            When calculating spatially varying weights we can smooth the
            weights so that areas close to areas that are masked have lower
            weights than those further away. This fuzzy length controls the
            scale over which the weights are smoothed. The fuzzy length is in
            terms of m, the default is 20km. This distance is then converted
            into a number of grid squares, which does not have to be an
            integer. Assumes the grid spacing is the same in the x and y
            directions and raises an error if this is not true. See
            SpatiallyVaryingWeightsFromMask for more details.

    Returns:
        iris.cube.Cube:
            Merged and blended Cube.

    Raises:
        RuntimeError:
            If calc_method is linear and cval is not None.
        RuntimeError:
            If calc_method is nonlinear and either y0val and ynval is not None.
        RuntimeError:
            If calc_method is dict and weights_dict is None.
    """
    from improver.blending.calculate_weights_and_blend import WeightAndBlend

    if (weighting_method == "linear") and cval:
        raise RuntimeError('Method: linear does not accept arguments: cval')
    if (weighting_method == "nonlinear") and any([y0val, ynval]):
        raise RuntimeError('Method: non-linear does not accept arguments:'
                           ' y0val, ynval')
    if (weighting_method == "dict") and weighting_config is None:
        raise RuntimeError('Dictionary is required if wts_calc_method="dict"')
    if "model" in coordinate and model_id_attr is None:
        raise RuntimeError('model_id_attr must be specified for '
                           'model blending')

    plugin = WeightAndBlend(
        coordinate, weighting_method,
        weighting_coord=weighting_coord, wts_dict=weighting_config,
        y0val=y0val, ynval=ynval, cval=cval)
    result = plugin.process(
        cubes, cycletime=cycletime, model_id_attr=model_id_attr,
        spatial_weights=spatial_weights_from_mask, fuzzy_length=fuzzy_length,
        attributes_dict=attributes_config)
    return result
