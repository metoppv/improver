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
"""Script to create neighbour cubes for extracting spot data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(orography: cli.inputcube,
            land_sea_mask: cli.inputcube,
            site_list: cli.inputjson,
            *,
            all_methods=False,
            land_constraint=False,
            similar_altitude=False,
            search_radius: float = None,
            node_limit: int = None,
            site_coordinate_system=None,
            site_coordinate_options=None,
            site_x_coordinate=None,
            site_y_coordinate=None):
    """Create neighbour cubes for extracting spot data.

    Determine grid point coordinates within the provided cubes that neighbour
    spot data sites defined within the provided JSON/Dictionary.
    If no options are set the returned cube will contain the nearest neighbour
    found for each site. Other constrained neighbour finding methods can be
    set with options below.
    1. Nearest neighbour.
    2. Nearest land point neighbour.
    3. Nearest neighbour with minimum height difference.
    4. Nearest land point neighbour with minimum height difference.

    Args:
        orography (iris.cube.Cube):
            Cube of model orography for the model grid on which neighbours are
            being found.
        land_sea_mask (iris.cube.Cube):
            Cube of model land mask for the model grid on which neighbours are
            being found, with land points set to one and sea points set to
            zero.
        site_list (dict):
            Dictionary that contains the spot sites for which neighbouring grid
            points are to be found.
        all_methods (bool):
            If True, this will return a cube containing the nearest grid point
            neighbours to spot sites as defined by each possible combination
            of constraints.
        land_constraint (bool):
            If True, this will return a cube containing the nearest grid point
            neighbours to spot sites that are also land points. May be used
            with the similar_altitude option.
        similar_altitude (bool):
            If True, this will return a cube containing the nearest grid point
            neighbour to each spot site that is found, within a given search
            radius, to minimise the height difference between the two. May be
            used with the land_constraint option.
        search_radius (float):
            The radius in metres about a spot site within which to search for
            a grid point neighbour that is land or which has a smaller height
            difference than the nearest.
        node_limit (int):
            When searching within the defined search_radius for suitable
            neighbours, a KDTree is constructed. This node_limit prevents the
            tree from becoming too large for large search radii. A default of
            36 will be set, which is to say the nearest 36 grid points will be
            considered. If the search radius is likely to contain more than
            36 points, this value should be increased to ensure all point
            are considered.
        site_coordinate_system (cartopy coordinate system):
            The coordinate system in which the site coordinates are provided
            within the site list. This must be provided as the name of a
            cartopy coordinate system. The Default will become PlateCarree.
        site_coordinate_options (str):
            JSON formatted string of options passed to the cartopy coordinate
            system given in site_coordinate_system. "globe" is handled as a
            special case to construct a cartopy Globe object.
        site_x_coordinate (str):
            The key that identifies site x coordinates in the provided site
            dictionary. Defaults to longitude.
        site_y_coordinate (str):
            The key that identifies site y coordinates in the provided site
            dictionary. Defaults to latitude.

    Returns:
        iris.cube.Cube:
            The processed Cube.

    Raises:
        ValueError:
            If all_methods is used with land_constraint or similar_altitude.

    """
    import json

    import numpy as np
    import cartopy.crs as ccrs
    import iris

    from improver.spotdata.neighbour_finding import NeighbourSelection
    from improver.utilities.cube_manipulation import (
        merge_cubes, enforce_coordinate_ordering)

    PROJECTION_LIST = [
        'AlbersEqualArea', 'AzimuthalEquidistant', 'EuroPP', 'Geocentric',
        'Geodetic', 'Geostationary', 'Globe', 'Gnomonic',
        'LambertAzimuthalEqualArea', 'LambertConformal', 'LambertCylindrical',
        'Mercator', 'Miller', 'Mollweide', 'NearsidePerspective',
        'NorthPolarStereo', 'OSGB', 'OSNI', 'Orthographic', 'PlateCarree',
        'Projection', 'Robinson', 'RotatedGeodetic', 'RotatedPole',
        'Sinusoidal', 'SouthPolarStereo', 'Stereographic',
        'TransverseMercator', 'UTM']

    # Check valid options have been selected.
    if all_methods is True and (land_constraint or similar_altitude):
        raise ValueError(
            'Cannot use all_methods option with other constraints.')

    # Filter kwargs for those expected by plugin and which are set.
    # This preserves the plugin defaults for unset options.
    args = {
        'land_constraint': land_constraint,
        'minimum_dz': similar_altitude,
        'search_radius': search_radius,
        'site_coordinate_system': site_coordinate_system,
        'site_coordinate_options': site_coordinate_options,
        'site_x_coordinate': site_x_coordinate,
        'node_limit': node_limit,
        'site_y_coordinate': site_y_coordinate
    }
    fargs = (site_list, orography, land_sea_mask)
    kwargs = {k: v for (k, v) in args.items() if v is not None}

    # Deal with coordinate systems for sites other than PlateCarree.
    if 'site_coordinate_system' in kwargs.keys():
        scrs = kwargs['site_coordinate_system']
        if scrs not in PROJECTION_LIST:
            raise ValueError('invalid projection {}'.format(scrs))
        site_crs = getattr(ccrs, scrs)
        scrs_opts = json.loads(kwargs.pop('site_coordinate_options', '{}'))
        if 'globe' in scrs_opts:
            crs_globe = ccrs.Globe(**scrs_opts['globe'])
            del scrs_opts['globe']
        else:
            crs_globe = ccrs.Globe()
        kwargs['site_coordinate_system'] = site_crs(
            globe=crs_globe, **scrs_opts)
    # Call plugin to generate neighbour cubes
    if all_methods:
        methods = [
            {**kwargs, 'land_constraint': False, 'minimum_dz': False},
            {**kwargs, 'land_constraint': True, 'minimum_dz': False},
            {**kwargs, 'land_constraint': False, 'minimum_dz': True},
            {**kwargs, 'land_constraint': True, 'minimum_dz': True}
        ]

        all_methods = iris.cube.CubeList([])
        for method in methods:
            all_methods.append(NeighbourSelection(**method).process(*fargs))

        squeezed_cubes = iris.cube.CubeList([])
        for index, cube in enumerate(all_methods):
            cube.coord('neighbour_selection_method').points = np.int32(index)
            squeezed_cubes.append(iris.util.squeeze(cube))

        result = merge_cubes(squeezed_cubes)
    else:
        result = NeighbourSelection(**kwargs).process(*fargs)

    enforce_coordinate_ordering(
        result,
        ['spot_index', 'neighbour_selection_method', 'grid_attributes'])

    return result
