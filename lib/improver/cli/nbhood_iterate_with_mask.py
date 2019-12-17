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
@cli.with_intermediate_output
def process(cube: cli.inputcube,
            mask: cli.inputcube,
            weights: cli.inputcube = None,
            *,
            coord_for_masking, radius: float = None,
            radii_by_lead_time=None,
            sum_or_fraction="fraction",
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
            dimension gained through masking.
        coord_for_masking (str):
            String matching the name of the coordinate that will be used
            for masking.
        radius (float):
            The radius in metres of the neighbourhood to apply.
            Rounded up to convert into integer number of grid points east and
            north, based on the characteristic spacing at the zero indices of
            the cube projection-x and y coordinates.
        radii_by_lead_time (float or list of float):
            A list with the radius in metres at [0] and the lead_time at [1]
            Lead time is a List of lead times or forecast periods, at which
            the radii within 'radii' are defined. The lead times are expected
            in hours.
        sum_or_fraction (str):
            Identifier for whether sum or fraction should be returned from
            neighbourhooding.
            Sum represents the sum of the neighbourhood.
            Fraction represents the sum of the neighbourhood divided by the
            neighbourhood area.
        remask (bool):
            If True, the original un-neighbourhood processed mask
            is applied to mask out the neighbourhood processed cube.
            If False, the original un-neighbourhood processed mask is not
            applied.
            Therefore, the neighbourhood processing may result in
            values being present in areas that were originally masked.
        collapse_dimension (bool):
            Collapse the dimension from the mask, by doing a weighted mean
            using the weights provided.  This is only suitable when the result
            is left unmasked, so there is data to weight between the points
            in the coordinate we are collapsing.

    Returns:
        (tuple): tuple containing:
            **result** (iris.cube.Cube):
                A cube after being fully processed.
            **intermediate_cube** (iris.cube.Cube):
                A cube before it is collapsed, if 'collapse_dimension' is True.

    """
    from improver.nbhood.use_nbhood import (
        ApplyNeighbourhoodProcessingWithAMask,
        CollapseMaskedNeighbourhoodCoordinate,
    )
    from improver.utilities.cli_utilities import radius_or_radii_and_lead

    radius_or_radii, lead_times = radius_or_radii_and_lead(radius,
                                                           radii_by_lead_time)

    result = ApplyNeighbourhoodProcessingWithAMask(
        coord_for_masking, radius_or_radii, lead_times=lead_times,
        sum_or_fraction=sum_or_fraction,
        re_mask=remask).process(cube, mask)
    intermediate_cube = None

    # Collapse with the masking dimension.
    if collapse_dimension:
        intermediate_cube = result.copy()
        result = CollapseMaskedNeighbourhoodCoordinate(
            coord_for_masking, weights).process(result)
    return result, intermediate_cube
