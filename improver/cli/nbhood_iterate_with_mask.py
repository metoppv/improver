#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run neighbourhooding processing when iterating over a coordinate
defining a series of masks."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    mask: cli.inputcube,
    weights: cli.inputcube = None,
    *,
    coord_for_masking,
    neighbourhood_shape="square",
    radii: cli.comma_separated_list,
    lead_times: cli.comma_separated_list = None,
    area_sum=False,
):
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
    If weights are given the masking dimension that we gain will be collapsed
    using a weighted average.

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
        neighbourhood_shape (str):
            Name of the neighbourhood method to use.
            Options: "circular", "square".
            Default: "square".
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


    Returns:
        iris.cube.Cube:
            A cube after being fully processed.
    """
    from improver.nbhood import radius_by_lead_time
    from improver.nbhood.use_nbhood import ApplyNeighbourhoodProcessingWithAMask

    radius_or_radii, lead_times = radius_by_lead_time(radii, lead_times)

    result = ApplyNeighbourhoodProcessingWithAMask(
        coord_for_masking,
        neighbourhood_shape,
        radius_or_radii,
        lead_times=lead_times,
        collapse_weights=weights,
        sum_only=area_sum,
    )(cube, mask)

    return result
