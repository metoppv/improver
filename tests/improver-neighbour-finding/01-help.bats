#!/usr/bin/env bats
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

@test "neighbour-finding -h" {
  run improver neighbour-finding -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver neighbour-finding [-h] [--profile]
                                  [--profile_file PROFILE_FILE]
                                  [--all_methods] [--land_constraint]
                                  [--minimum_dz]
                                  [--search_radius SEARCH_RADIUS]
                                  [--node_limit NODE_LIMIT]
                                  [--site_coordinate_system SITE_COORDINATE_SYSTEM]
                                  [--site_x_coordinate SITE_X_COORDINATE]
                                  [--site_y_coordinate SITE_Y_COORDINATE]
                                  [--grid_metadata_identifier GRID_METADATA_IDENTIFIER]
                                  SITE_LIST_FILEPATH OROGRAPHY_FILEPATH
                                  LANDMASK_FILEPATH OUTPUT_FILEPATH

Determine grid point coordinates within the provided cubes that neighbour spot
data sites defined within the provided JSON file. If no options are set the
returned netCDF file will contain the nearest neighbour found for each site.
Other constrained neighbour finding methods can be set with options below.

These methods are:

 1. nearest neighbour
 2. nearest land point neighbour
 3. nearest neighbour with minimum height difference
 4. nearest land point neighbour with minimum height difference

positional arguments:
  SITE_LIST_FILEPATH    Path to a JSON file that contains the spot sites for
                        which neighbouring grid points are to be found.
  OROGRAPHY_FILEPATH    Path to a NetCDF file of model orography for the model
                        grid on which neighbours are being found.
  LANDMASK_FILEPATH     Path to a NetCDF file of model land mask for the model
                        grid on which neighbours are being found.
  OUTPUT_FILEPATH       The output path for the resulting NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --all_methods         If set this will return a cube containing the nearest
                        grid point neighbours to spot sites as defined by each
                        possible combination of constraints.

Apply constraints to neighbour choice:
  --land_constraint     If set this will return a cube containing the nearest
                        grid point neighbours to spot sites that are also land
                        points. May be used with the minimum_dz option.
  --minimum_dz          If set this will return a cube containing the nearest
                        grid point neighbour to each spot site that is found,
                        within a given search radius, to minimise the height
                        difference between the two. May be used with the
                        land_constraint option.
  --search_radius SEARCH_RADIUS
                        The radius in metres about a spot site within which to
                        search for a grid point neighbour that is land or
                        which has a smaller height difference than the
                        nearest. The default value is 10000m (10km).
  --node_limit NODE_LIMIT
                        When searching within the defined search_radius for
                        suitable neighbours, a KDTree is constructed. This
                        node_limit prevents the tree from becoming too large
                        for large search radii. A default of 36 is set, which
                        is to say the nearest 36 grid points will be
                        considered. If the search_radius is likely to contain
                        more than 36 points, this value should be increased to
                        ensure all points are considered.

Site list options:
  --site_coordinate_system SITE_COORDINATE_SYSTEM
                        The coordinate system in which the site coordinates
                        are provided within the site list. This must be
                        provided as the name of a cartopy coordinate system.
                        The default is a PlateCarree system, with site
                        coordinates given by latitude/longitude pairs. This
                        can be a complete definition, including parameters
                        required to modify a default system, e.g.
                        Miller(central_longitude=90). If a globe is required
                        this can be specified as e.g.
                        Globe(semimajor_axis=100, semiminor_axis=100).
  --site_x_coordinate SITE_X_COORDINATE
                        The x coordinate key within the JSON file. The plugin
                        default is 'longitude', but can be changed using this
                        option if required.
  --site_y_coordinate SITE_Y_COORDINATE
                        The y coordinate key within the JSON file. The plugin
                        default is 'latitude', but can be changed using this
                        option if required.

Metadata:
  --grid_metadata_identifier GRID_METADATA_IDENTIFIER
                        A string to identify attributes from the netCDF files
                        that should be copied onto the output cube. Attributes
                        are compared for a partial match. The default is
                        'mosg' which corresponds to Met Office Standard Grid
                        attributes which should be copied across.
__HELP__
  [[ "$output" == "$expected" ]]
}
