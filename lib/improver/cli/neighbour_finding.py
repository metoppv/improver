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

import json

from argparse import RawDescriptionHelpFormatter
from textwrap import wrap

import iris
import cartopy.crs as ccrs
from improver.argparser import ArgParser, safe_eval
from improver.spotdata.neighbour_finding import NeighbourSelection
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.utilities.cube_manipulation import (merge_cubes,
                                                  enforce_coordinate_ordering)

PROJECTION_LIST = [
    'AlbersEqualArea', 'AzimuthalEquidistant', 'EuroPP', 'Geocentric',
    'Geodetic', 'Geostationary', 'Globe', 'Gnomonic',
    'LambertAzimuthalEqualArea', 'LambertConformal', 'LambertCylindrical',
    'Mercator', 'Miller', 'Mollweide', 'NearsidePerspective',
    'NorthPolarStereo', 'OSGB', 'OSNI', 'Orthographic', 'PlateCarree',
    'Projection', 'Robinson', 'RotatedGeodetic', 'RotatedPole', 'Sinusoidal',
    'SouthPolarStereo', 'Stereographic', 'TransverseMercator', 'UTM']


def main():
    """Load in arguments and get going."""
    description = (
        "Determine grid point coordinates within the provided cubes that "
        "neighbour spot data sites defined within the provided JSON "
        "file. If no options are set the returned netCDF file will contain the"
        " nearest neighbour found for each site. Other constrained neighbour "
        "finding methods can be set with options below.")
    options = ("\n\nThese methods are:\n\n 1. nearest neighbour\n"
               " 2. nearest land point neighbour\n"
               " 3. nearest neighbour with minimum height difference\n"
               " 4. nearest land point neighbour with minimum height "
               "difference")

    parser = ArgParser(
        description=('\n'.join(wrap(description, width=79)) + options),
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("site_list_filepath", metavar="SITE_LIST_FILEPATH",
                        help="Path to a JSON file that contains the spot sites"
                        " for which neighbouring grid points are to be found.")
    parser.add_argument("orography_filepath", metavar="OROGRAPHY_FILEPATH",
                        help="Path to a NetCDF file of model orography for the"
                        " model grid on which neighbours are being found.")
    parser.add_argument("landmask_filepath", metavar="LANDMASK_FILEPATH",
                        help="Path to a NetCDF file of model land mask for the"
                        " model grid on which neighbours are being found.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILEPATH",
                        help="The output path for the resulting NetCDF")

    parser.add_argument(
        "--all_methods", default=False, action='store_true',
        help="If set this will return a cube containing the nearest grid point"
        " neighbours to spot sites as defined by each possible combination of"
        " constraints.")

    group = parser.add_argument_group('Apply constraints to neighbour choice')
    group.add_argument(
        "--land_constraint", default=False, action='store_true',
        help="If set this will return a cube containing the nearest grid point"
        " neighbours to spot sites that are also land points. May be used with"
        " the minimum_dz option.")
    group.add_argument(
        "--minimum_dz", default=False, action='store_true',
        help="If set this will return a cube containing the nearest grid point"
        " neighbour to each spot site that is found, within a given search"
        " radius, to minimise the height difference between the two. May be"
        " used with the land_constraint option.")
    group.add_argument(
        "--search_radius", metavar="SEARCH_RADIUS", type=float,
        help="The radius in metres about a spot site within which to search"
        " for a grid point neighbour that is land or which has a smaller "
        " height difference than the nearest. The default value is 10000m "
        "(10km).")
    group.add_argument(
        "--node_limit", metavar="NODE_LIMIT", type=int,
        help="When searching within the defined search_radius for suitable "
        "neighbours, a KDTree is constructed. This node_limit prevents the "
        "tree from becoming too large for large search radii. A default of 36"
        " is set, which is to say the nearest 36 grid points will be "
        "considered. If the search_radius is likely to contain more than 36 "
        "points, this value should be increased to ensure all points are "
        "considered.")

    s_group = parser.add_argument_group('Site list options')
    s_group.add_argument(
        "--site_coordinate_system", metavar="SITE_COORDINATE_SYSTEM",
        help="The coordinate system in which the site coordinates are provided"
        " within the site list. This must be provided as the name of a cartopy"
        " coordinate system. The default is a PlateCarree system, with site"
        " coordinates given by latitude/longitude pairs. This can be a"
        " complete definition, including parameters required to modify a"
        " default system, e.g. Miller(central_longitude=90). If a globe is"
        " required this can be specified as e.g."
        " Globe(semimajor_axis=100, semiminor_axis=100).")
    s_group.add_argument(
        "--site_x_coordinate", metavar="SITE_X_COORDINATE",
        help="The x coordinate key within the JSON file. The plugin default is"
        " 'longitude', but can be changed using this option if required.")
    s_group.add_argument(
        "--site_y_coordinate", metavar="SITE_Y_COORDINATE",
        help="The y coordinate key within the JSON file. The plugin default is"
        " 'latitude', but can be changed using this option if required.")

    m_group = parser.add_argument_group('Metadata')
    m_group.add_argument(
        "--grid_metadata_identifier", metavar="GRID_METADATA_IDENTIFIER",
        help="A string to identify attributes from the netCDF files that"
        " should be copied onto the output cube. Attributes are compared"
        " for a partial match. The default is 'mosg' which corresponds"
        " to Met Office Standard Grid attributes which should be copied"
        " across.")

    args = parser.parse_args()

    # Open input files
    with open(args.site_list_filepath, 'r') as site_file:
        sitelist = json.load(site_file)
    orography = load_cube(args.orography_filepath)
    landmask = load_cube(args.landmask_filepath)
    fargs = (sitelist, orography, landmask)

    # Filter kwargs for those expected by plugin and which are set.
    # This preserves the plugin defaults for unset options.
    kwarg_list = ['land_constraint', 'minimum_dz', 'search_radius',
                  'site_coordinate_system', 'site_x_coordinate', 'node_limit',
                  'site_y_coordinate', 'grid_metadata_identifier']
    kwargs = {k: v for (k, v) in vars(args).items() if k in kwarg_list and
              v is not None}

    # Deal with coordinate systems for sites other than PlateCarree.
    if 'site_coordinate_system' in kwargs.keys():
        scrs = kwargs['site_coordinate_system']
        kwargs['site_coordinate_system'] = safe_eval(scrs, ccrs,
                                                     PROJECTION_LIST)

    # Check valid options have been selected.
    if args.all_methods is True and (kwargs['land_constraint'] is True or
                                     kwargs['minimum_dz'] is True):
        raise ValueError(
            'Cannot use all_methods option with other constraints.')

    # Call plugin to generate neighbour cubes
    if args.all_methods:
        methods = []
        methods.append({
            **kwargs, 'land_constraint': False, 'minimum_dz': False})
        methods.append({
            **kwargs, 'land_constraint': True, 'minimum_dz': False})
        methods.append({
            **kwargs, 'land_constraint': False, 'minimum_dz': True})
        methods.append({
            **kwargs, 'land_constraint': True, 'minimum_dz': True})

        all_methods = iris.cube.CubeList([])
        for method in methods:
            all_methods.append(NeighbourSelection(**method).process(*fargs))

        for index, cube in enumerate(all_methods):
            cube.coord('neighbour_selection_method').points = index
        result = merge_cubes(all_methods)
    else:
        result = NeighbourSelection(**kwargs).process(*fargs)

    result = enforce_coordinate_ordering(
        result,
        ['spot_index', 'neighbour_selection_method', 'grid_attributes'])

    # Save the neighbour cube
    save_netcdf(result, args.output_filepath)


if __name__ == "__main__":
    main()
