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
"""Script to run neighbourhooding processing when iterating over a coordinate
defining a series of masks."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            mask: cli.inputcube,
            weights: cli.inputcube = None,
            *,
            coord_for_masking,
            radii: cli.comma_separated_list,
            lead_times: cli.comma_separated_list = None,
            area_sum=False,
            remask=False,
            collapse_dimension=False):
    """Runs neighbourhooding processing iterating over a coordinate by mask.

    Apply the requested neighbourhood method via the
    ApplyNeighbourhoodProcessingWithMask plugin to a file with one diagnostic
    dataset in combination with a cube containing one or more masks.
    The mask dataset may have an extra dimension compared to the input
    diagnostic. In this case, the user specifies the name of the extra
    coordinate and this coordinate is iterated over so each mask is applied
    to separate slices over the input cube. These intermediate masked datasets
    are then concatenated, resulting in a dataset that has been processed
    using multiple masks and has gained an extra dimension from the masking.
    There is also an option to re-mask the output dataset, so that after
    neighbourhood processing non-zero values are only present for unmasked
    grid points.
    There is an alternative option of collapsing the dimension that we gain
    using this processing using a weighted average.

    Args:
        cube (iris.cube.Cube):
            Cube to be processed.
        mask (iris.cube.Cube):
            Cube to act as a mask.
        weights (iris.cube.Cube, Optional):
            Cube containing the weights which are used for collapsing the
            dimension gained through masking. (Optional).
        coord_for_masking (str):
            String matching the name of the coordinate that will be used
            for masking.
        radii (list of float):
            The radius or a list of radii in metres of the neighbourhood to
            apply.
            If it is a list, it must be the same length as lead_times, which
            defines at which lead time to use which nbhood radius. The radius
            will be interpolated for intermediate lead times.
        lead_times (list of int):
            The lead times in hours that correspond to the radii to be used.
            If lead_times are set, radii must be a list the same length as
            lead_times. Lead times must be given as integer values.
        area_sum (bool):
            Return sum rather than fraction over the neighbourhood area.
        remask (bool):
            Include this option to apply the original un-neighbourhood
            processed mask to the neighbourhood processed cube.
            Otherwise the original un-neighbourhood processed mask
            is not applied. Therefore, the neighbourhood processing may result
            in values being present in area that were originally masked.
        collapse_dimension (bool):
            Include this option to collapse the dimension from the mask, by
            doing a weighted mean using the weights provided. This is only
            suitable when the result is left unmasked, so there is data to
            weight between the points in the coordinate we are collapsing.

    Returns:
        iris.cube.Cube:
            A cube after being fully processed.
    """
    from improver.nbhood import radius_by_lead_time
    from improver.nbhood.use_nbhood import (
        ApplyNeighbourhoodProcessingWithAMask,
        CollapseMaskedNeighbourhoodCoordinate,
    )

    sum_or_fraction = 'sum' if area_sum else 'fraction'

    radius_or_radii, lead_times = radius_by_lead_time(radii, lead_times)

    result = ApplyNeighbourhoodProcessingWithAMask(
        coord_for_masking, radius_or_radii, lead_times=lead_times,
        sum_or_fraction=sum_or_fraction,
        re_mask=remask)(cube, mask)

    # Collapse with the masking dimension.
    if collapse_dimension:
        result = CollapseMaskedNeighbourhoodCoordinate(
            coord_for_masking, weights).process(result)
    return result
