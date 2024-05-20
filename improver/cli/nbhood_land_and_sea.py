#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run neighbourhooding processing over areas of land and sea
separately before combining them to return unified fields. Topographic zones
may also be employed, with the sea area being treated as a distinct zone."""
from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    mask: cli.inputcube,
    weights: cli.inputcube = None,
    *,
    neighbourhood_shape="square",
    radii: cli.comma_separated_list,
    lead_times: cli.comma_separated_list = None,
    area_sum=False,
):
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
        (tuple): tuple containing:
            **result** (iris.cube.Cube):
                A cube of the processed data.

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
    import numpy as np

    from improver.nbhood.nbhood import NeighbourhoodProcessing
    from improver.nbhood.use_nbhood import ApplyNeighbourhoodProcessingWithAMask

    masking_coordinate = None
    if any(
        "topographic_zone" in coord.name() for coord in mask.coords(dim_coords=True)
    ):

        if mask.attributes["topographic_zones_include_seapoints"] == "True":
            raise ValueError(
                "The topographic zones mask cube must have been "
                "masked to exclude sea points, but "
                "topographic_zones_include_seapoints = True"
            )

        if not weights:
            raise TypeError(
                "A weights cube must be provided if using a mask "
                "of topographic zones to collapse the resulting "
                "vertical dimension."
            )

        if weights.attributes["topographic_zones_include_seapoints"] == "True":
            raise ValueError(
                "The weights cube must be masked to exclude sea "
                "points, but topographic_zones_include_seapoints "
                "= True"
            )

        masking_coordinate = "topographic_zone"
        land_sea_mask = weights[0].copy(data=weights[0].data.mask)
        land_sea_mask.rename("land_binary_mask")
        land_sea_mask.remove_coord(masking_coordinate)
        # Create land and sea masks in IMPROVER format (inverse of
        # numpy standard) 1 - include this region, 0 - exclude this region.
        land_only = land_sea_mask.copy(
            data=np.logical_not(land_sea_mask.data).astype(int)
        )
        sea_only = land_sea_mask.copy(data=land_sea_mask.data.astype(int))

    else:
        if weights is not None:
            raise TypeError("A weights cube has been provided but will not be used")
        land_sea_mask = mask
        # In this case the land is set to 1 and the sea is set to 0 in the
        # input mask.
        sea_only = land_sea_mask.copy(
            data=np.logical_not(land_sea_mask.data).astype(int)
        )
        land_only = land_sea_mask.copy(data=land_sea_mask.data.astype(int))

    if lead_times is None:
        radius_or_radii = float(radii[0])
    else:
        if len(radii) != len(lead_times):
            raise RuntimeError(
                "If leadtimes are supplied, it must be a list"
                " of equal length to a list of radii."
            )
        radius_or_radii = [float(x) for x in radii]
        lead_times = [int(x) for x in lead_times]

    # Section for neighbourhood processing land points.
    if land_only.data.max() > 0.0:
        if masking_coordinate is None:
            result_land = NeighbourhoodProcessing(
                neighbourhood_shape,
                radius_or_radii,
                lead_times=lead_times,
                sum_only=area_sum,
                re_mask=True,
            )(cube, land_only)
        else:
            result_land = ApplyNeighbourhoodProcessingWithAMask(
                masking_coordinate,
                neighbourhood_shape,
                radius_or_radii,
                lead_times=lead_times,
                collapse_weights=weights,
                sum_only=area_sum,
            )(cube, mask)
        result = result_land

    # Section for neighbourhood processing sea points.
    if sea_only.data.max() > 0.0:
        result_sea = NeighbourhoodProcessing(
            neighbourhood_shape,
            radius_or_radii,
            lead_times=lead_times,
            sum_only=area_sum,
            re_mask=True,
        )(cube, sea_only)
        result = result_sea

    # Section for combining land and sea points following land and sea points
    # being neighbourhood processed individually.
    if sea_only.data.max() > 0.0 and land_only.data.max() > 0.0:
        # Recombine cubes to be a single output.
        combined_data = result_land.data.filled(0) + result_sea.data.filled(0)
        result = result_land.copy(data=combined_data)

    return result
