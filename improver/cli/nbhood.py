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
"""Script to run neighbourhood processing."""

from improver import cli
from improver.constants import DEFAULT_PERCENTILES


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            mask: cli.inputcube = None,
            *,
            neighbourhood_output,
            neighbourhood_shape,
            radii: cli.comma_separated_list,
            lead_times: cli.comma_separated_list = None,
            degrees_as_complex=False,
            weighted_mode=False,
            area_sum=False,
            remask=False,
            percentiles: cli.comma_separated_list = DEFAULT_PERCENTILES,
            halo_radius: float = None):
    """Runs neighbourhood processing.

    Apply the requested neighbourhood method via the
    NeighbourhoodProcessing plugin to a Cube.

    Args:
        cube (iris.cube.Cube):
            The Cube to be processed.
        mask (iris.cube.Cube):
            A cube to mask the input cube. The data should contain 1 for
            usable points and 0 for discarded points.
            Only supported with square neighbourhoods. (Optional)
        neighbourhood_output (str):
            The form of the results generated using neighbourhood processing.
            If "probabilities" is selected, the mean probability with a
            neighbourhood is calculated. If "percentiles" is selected, then
            the percentiles are calculated with a neighbourhood. Calculating
            percentiles from a neighbourhood is only supported for a circular
            neighbourhood.
            Options: "probabilities", "percentiles".
        neighbourhood_shape (str):
            Name of the neighbourhood method to use. Only a "circular"
            neighbourhood shape is applicable for calculating "percentiles"
            output.
            Options: "circular", "square".
        radii (list of float):
            The radius or a list of radii in metres of the neighbourhood to
            apply.
            If it is a list, it must be the same length as lead_times, which
            defines at which lead time to use which nbhood radius. The radius
            will be interpolated for intermediate lead times.
        lead_times (list of int):
            The lead times in hours that correspond to the radii to be used.
            If lead_times are set, radii must be a list the same length as
            lead_times.
        degrees_as_complex (bool):
            Include this option to process angles as complex numbers.
            Not compatible with circular kernel or percentiles.
        weighted_mode (bool):
            Include this option to set the weighting to decrease with radius.
            Otherwise a constant weighting is assumed.
            weighted_mode is only applicable for calculating "probability"
            neighbourhood output using the circular kernel.
        area_sum (bool):
            Return sum rather than fraction over the neighbourhood area.
        remask (bool):
            Include this option to apply the original un-neighbourhood
            processed mask to the neighbourhood processed cube.
            Otherwise the original un-neighbourhood processed mask
            is not applied. Therefore, the neighbourhood processing may result
            in values being present in area that were originally masked.
        percentiles (float):
            Calculates value at the specified percentiles from the
            neighbourhood surrounding each grid point. This argument has no
            effect if the output is probabilities.
        halo_radius (float):
            Set this radius in metres to define the excess halo to clip. Used
            where a larger grid was defined than the standard grid and we want
            to clip the grid back to the standard grid. Otherwise no clipping
            is applied.

    Returns:
        iris.cube.Cube:
            A processed Cube.

    Raises:
        RuntimeError:
            If weighted_mode is used with the wrong neighbourhood_output.
        RuntimeError:
            If degree_as_complex is used with
            neighbourhood_output='percentiles'.
        RuntimeError:
            If degree_as_complex is used with neighbourhood_shape='circular'.
    """
    from improver.nbhood import radius_by_lead_time
    from improver.nbhood.nbhood import (
        GeneratePercentilesFromANeighbourhood, NeighbourhoodProcessing)
    from improver.utilities.pad_spatial import remove_cube_halo
    from improver.wind_calculations.wind_direction import WindDirection

    sum_or_fraction = 'sum' if area_sum else 'fraction'

    if neighbourhood_output == "percentiles":
        if weighted_mode:
            raise RuntimeError('weighted_mode cannot be used with'
                               'neighbourhood_output="percentiles"')
        if degrees_as_complex:
            raise RuntimeError('Cannot generate percentiles from complex '
                               'numbers')

    if neighbourhood_shape == "circular":
        if degrees_as_complex:
            raise RuntimeError(
                'Cannot process complex numbers with circular neighbourhoods')

    if degrees_as_complex:
        # convert cube data into complex numbers
        cube.data = WindDirection.deg_to_complex(cube.data)

    radius_or_radii, lead_times = radius_by_lead_time(radii, lead_times)

    if neighbourhood_output == "probabilities":
        result = NeighbourhoodProcessing(
            neighbourhood_shape, radius_or_radii,
            lead_times=lead_times,
            weighted_mode=weighted_mode,
            sum_or_fraction=sum_or_fraction,
            re_mask=remask)(cube, mask_cube=mask)
    elif neighbourhood_output == "percentiles":
        result = GeneratePercentilesFromANeighbourhood(
            neighbourhood_shape, radius_or_radii,
            lead_times=lead_times,
            percentiles=percentiles)(cube)

    if degrees_as_complex:
        # convert neighbourhooded cube back to degrees
        result.data = WindDirection.complex_to_deg(result.data)
    if halo_radius is not None:
        result = remove_cube_halo(result, halo_radius)
    return result
