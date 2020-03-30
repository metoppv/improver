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
"""Script to run neighbourhooding processing over areas of land and sea
separately before combining them to return unified fields. Topographic zones
may also be employed, with the sea area being treated as a distinct zone."""
from improver import cli


@cli.clizefy
@cli.with_output
@cli.with_intermediate_output
def process(cube: cli.inputcube,
            mask: cli.inputcube,
            weights: cli.inputcube = None,
            *,
            radii: cli.comma_separated_list,
            lead_times: cli.comma_separated_list = None,
            area_sum=False,
            return_intermediate=False):
    """ Module to process land and sea separately before combining them.

    Neighbourhood the input dataset over two distinct regions of land and sea.
    If performed as a single level neighbourhood, a land-sea mask should be
    provided. If instead topographic_zone neighbourhooding is being employed,
    the mask should be one of topographic zones. In the latter case a weights
    array is also needed to collapse the topographic_zone coordinate. These
    weights are created with the improver generate-topography-bands-weights
    CLI and should be made using a land-sea mask, which will then be employed
    within this code to draw the distinction between the two surface types.

    Args:
        cube (iris.cube.Cube):
            A cube to be processed.
        mask (iris.cube.Cube):
            A cube containing either a mask of topographic zones over land or
            a land-sea mask. If this is a land-sea mask, land points should be
            set to one and sea points set to zero.
        weights (iris.cube.Cube):
            A cube containing the weights which are used for collapsing the
            dimension gained through masking. These weights must have been
            created using a land-sea mask. (Optional).
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
        return_intermediate (bool):
            Include this option to return a cube with results following
            topographic masked neighbourhood processing of land points and
            prior to collapsing the topographic_zone coordinate. If no
            topographic masked neighbourhooding occurs, there will be no
            intermediate cube and a warning.

    Returns:
        (tuple): tuple containing:
            **result** (iris.cube.Cube):
                A cube of the processed data.
            **intermediate_cube** (iris.cube.Cube):
                A cube of the intermediate data, before collapsing.

    Raises:
        ValueError:
            If the topographic zone mask has the attribute
            topographic_zones_include_seapoints.
        IOError:
            if a weights cube isn't given and a topographic_zone mask is given.
        ValueError:
            If the weights cube has the attribute
            topographic_zones_include_seapoints.
        RuntimeError:
            If lead times are not None and has a different length to radii.
        TypeError:
            A weights cube has been provided but no topographic zone.

    """
    import warnings

    import numpy as np

    from improver.nbhood.nbhood import NeighbourhoodProcessing
    from improver.nbhood.use_nbhood import (
        ApplyNeighbourhoodProcessingWithAMask,
        CollapseMaskedNeighbourhoodCoordinate)

    sum_or_fraction = 'sum' if area_sum else 'fraction'

    masking_coordinate = intermediate_cube = None
    if any(['topographic_zone' in coord.name()
            for coord in mask.coords(dim_coords=True)]):

        if mask.attributes['topographic_zones_include_seapoints'] == 'True':
            raise ValueError('The topographic zones mask cube must have been '
                             'masked to exclude sea points, but '
                             'topographic_zones_include_seapoints = True')

        if not weights:
            raise TypeError('A weights cube must be provided if using a mask '
                            'of topographic zones to collapse the resulting '
                            'vertical dimension.')

        if weights.attributes['topographic_zones_include_seapoints'] == 'True':
            raise ValueError('The weights cube must be masked to exclude sea '
                             'points, but topographic_zones_include_seapoints '
                             '= True')

        masking_coordinate = 'topographic_zone'
        land_sea_mask = weights[0].copy(data=weights[0].data.mask)
        land_sea_mask.rename('land_binary_mask')
        land_sea_mask.remove_coord(masking_coordinate)
        # Create land and sea masks in IMPROVER format (inverse of
        # numpy standard) 1 - include this region, 0 - exclude this region.
        land_only = land_sea_mask.copy(
            data=np.logical_not(land_sea_mask.data).astype(int))
        sea_only = land_sea_mask.copy(data=land_sea_mask.data.astype(int))

    else:
        if weights is not None:
            raise TypeError('A weights cube has been provided but will not be '
                            'used')
        land_sea_mask = mask
        # In this case the land is set to 1 and the sea is set to 0 in the
        # input mask.
        sea_only = land_sea_mask.copy(
            data=np.logical_not(land_sea_mask.data).astype(int))
        land_only = land_sea_mask.copy(data=land_sea_mask.data.astype(int))

    if lead_times is None:
        radius_or_radii = float(radii[0])
    else:
        if len(radii) != len(lead_times):
            raise RuntimeError("If leadtimes are supplied, it must be a list"
                               " of equal length to a list of radii.")
        radius_or_radii = [float(x) for x in radii]
        lead_times = [int(x) for x in lead_times]

    if return_intermediate is not None and masking_coordinate is None:
        warnings.warn('No topographic_zone coordinate found, so no '
                      'intermediate file will be saved.')

    # Section for neighbourhood processing land points.
    if land_only.data.max() > 0.0:
        if masking_coordinate is None:
            result_land = NeighbourhoodProcessing(
                'square', radius_or_radii, lead_times=lead_times,
                sum_or_fraction=sum_or_fraction,
                re_mask=True)(cube, land_only)
        else:
            result_land = ApplyNeighbourhoodProcessingWithAMask(
                masking_coordinate, radius_or_radii, lead_times=lead_times,
                sum_or_fraction=sum_or_fraction,
                re_mask=False)(cube, mask)

            if return_intermediate:
                intermediate_cube = result_land.copy()
            # Collapse the masking coordinate.
            result_land = CollapseMaskedNeighbourhoodCoordinate(
                masking_coordinate, weights=weights).process(result_land)
        result = result_land

    # Section for neighbourhood processing sea points.
    if sea_only.data.max() > 0.0:
        result_sea = NeighbourhoodProcessing(
            'square', radius_or_radii, lead_times=lead_times,
            sum_or_fraction=sum_or_fraction,
            re_mask=True)(cube, sea_only)
        result = result_sea

    # Section for combining land and sea points following land and sea points
    # being neighbourhood processed individually.
    if sea_only.data.max() > 0.0 and land_only.data.max() > 0.0:
        # Recombine cubes to be a single output.
        combined_data = result_land.data.filled(0) + result_sea.data.filled(0)
        result = result_land.copy(data=combined_data)

    return result, intermediate_cube
