# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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

"""Neighbour finding for the Improver site specific process chain."""

import numpy as np
import cartopy.crs as ccrs
from iris.analysis.trajectory import interpolate
from improver.spotdata.ancillaries import data_from_ancillary
from improver.spotdata.common_functions import (ConditionalListExtract,
                                                nearest_n_neighbours,
                                                get_nearest_coords,
                                                index_of_minimum_difference,
                                                list_entry_from_index,
                                                node_edge_test)


class PointSelection(object):
    """
    For the selection of source data from a grid for use in deriving
    conditions at an arbitrary coordinate.

    """

    def __init__(self, method='default'):
        """neighbour_list = find_nearest_neighbours(cube, spot_sites)
        The class is called with the desired method to be used in determining
        the grid point closest to sites of interest.

        """
        self.method = method

    def process(self, cube, sites, **kwargs):
        """
        Call the correct function to enact the method of PointSelection
        specified.

        """
        function = getattr(self, self.method)
        return function(cube, sites, **kwargs)

    @staticmethod
    def fast_nearest_neighbour(cube, sites, ancillary_data=None):
        '''
        Use iris coord.nearest_neighbour_index function to locate the nearest
        grid point to the given latitude/longitude pair.

        Performed on a 2D-surface; consider using the much slower
        iris.analysis.trajectory.interpolate method for a more correct nearest
        neighbour search with projection onto a spherical surface; this is
        typically much slower.

        Args:
        -----
        cube           : Iris cube of gridded data.
        sites          : Dictionary of site data, including lat/lon and
                         altitude information.
                         e.g. {<site_id>: {'latitude': 50, 'longitude': 0,
                                           'altitude': 10}}
        ancillary_data : A dictionary containing additional model data that
                         is needed. e.g. {'orography': <cube of orography>}

        Returns:
        --------
        neighbours: Numpy array of grid i,j coordinates that are nearest to
                    each site coordinate given. Includes height difference
                    between site and returned grid point if orography is
                    provided.

        '''
        if ancillary_data is not None and ancillary_data['orography']:
            calculate_dz = True
            orography = data_from_ancillary(ancillary_data, 'orography')
        else:
            calculate_dz = False

        neighbours = np.empty(len(sites), dtype=[('i', 'i8'),
                                                 ('j', 'i8'),
                                                 ('dz', 'f8'),
                                                 ('edge', 'bool_')])

        # Check cube coords are lat/lon, else transform lookup coordinates.
        trg_crs = xy_test(cube)

        imax = cube.coord(axis='y').shape[0]
        jmax = cube.coord(axis='x').shape[0]

        for i_site, site in enumerate(sites.itervalues()):
            latitude, longitude, altitude = (site['latitude'],
                                             site['longitude'],
                                             site['altitude'])

            longitude, latitude = xy_transform(trg_crs, latitude, longitude)
            i_latitude, j_longitude = get_nearest_coords(cube, latitude,
                                                         longitude)

            dz_site_grid = 0.
            if calculate_dz:
                dz_site_grid = altitude - orography[i_latitude, j_longitude]

            neighbours[i_site] = (int(i_latitude), int(j_longitude),
                                  dz_site_grid,
                                  (i_latitude == imax or j_longitude == jmax))

        return neighbours

#     @staticmethod
#     def nearest_neighbour(cube, sites, ancillary_data=None):
#         '''
#         Uses the
#         iris.analysis._interpolate_private._nearest_neighbour_indices_ndcoords
#         function to locate the nearest grid point to the given latitude/
#         longitude pair, taking into account the projection of the cube.
#
#         Method is equivalent to extracting data directly with
#         iris.analysis.trajectory.interpolate method, which calculates nearest
#         neighbours using great arcs on a spherical surface. Using the private
#         function we are able to get the list of indices for reuse by multiple
#         diagnostics.
#
#         Args:
#         -----
#         cube           : Iris cube of gridded data.
#         sites          : Dictionary of site data, including lat/lon and
#                          altitude information.
#         ancillary_data : A dictionary containing additional model data that
#                          is needed. e.g. {'orography': <cube of orography>}
#
#         Returns:
#         --------
#         neighbours: Numpy array of grid i,j coordinates that are nearest to
#                     each site coordinate given. Includes height difference
#                     between site and returned grid point if orography is
#                     provided.
#
#         '''
#         if ancillary_data is not None and ancillary_data['orography']:
#             calculate_dz = True
#             orography = data_from_ancillary(ancillary_data, 'orography')
#         else:
#             calculate_dz = False
#
#         neighbours = np.empty(len(sites), dtype=[('i', 'i8'),
#                                                  ('j', 'i8'),
#                                                  ('dz', 'f8')])
#
#         # Check cube coords are lat/lon, else transform lookup coordinates.
#         trg_crs = xy_test(cube)
#
#         spot_sites = [('latitude',
#                        [sites[key]['latitude'] for key in sites.keys()]),
#                       ('longitude',
#                        [sites[key]['longitude'] for key in sites.keys()])]
#
#         spot_orography = interpolate(cube, spot_sites, method='nearest')
#
#         cube_lats = cube.coord(axis='y').points
#         spot_lats = spot_orography.coord('latitude').points
#
#         cube_lons = cube.coord(axis='x').points
#         spot_lons = spot_orography.coord('longitude').points
#
#         int_ind_i = []
#         int_ind_j = []
#         for point in spot_lats:
#             indices_lat = (np.where(point == cube_lats)[0][0])
#             int_ind_i.append(indices_lat)
#         for point in spot_lons:
#             indices_lon = (np.where(point == cube_lons)[0][0])
#             int_ind_j.append(indices_lon)
#         i_indices = int_ind_i
#         j_indices = int_ind_j
#
#         # i_indices, j_indices = zip(*[(i, j) for _, i, j in neighbour_list])
#
#         dz = [0] * len(neighbour_list)
#         if calculate_dz:
#             altitudes = [sites[key]['altitude'] for key in sites.keys()]
#             dz = altitudes - orography[i_indices, j_indices]
#
#         neighbours['i'] = i_indices
#         neighbours['j'] = j_indices
#         neighbours['dz'] = dz
#
#         return neighbours

    def minimum_height_error_neighbour(self, cube, sites,
                                       default_neighbours=None,
                                       relative_z=None,
                                       land_constraint=False,
                                       ancillary_data=None):

        '''
        Find the horizontally nearest neighbour, then relax the conditions
        to find the neighbouring point in the 9 nearest nodes to the input
        coordinate that minimises the height difference. This is typically
        used for temperature, where vertical displacement can be much more
        important that horizontal displacement in determining the conditions.

        A vertical displacement bias may be applied with the relative_z
        keyword; whether to prefer grid points above or below the site, or
        neither.

        A land constraint may be applied that requires a land grid point be
        selected for a site that is over land. Currently this is established
        by checking that the nearest grid point barring any other conditions
        is a land point. If a site is a sea point it will use the nearest
        neighbour as there should be no vertical displacement difference with
        other sea points.

        Args:
        -----
        cube           : Iris cube of gridded data.
        sites          : Dictionary of site data, including lat/lon and
                         altitude information.
        relative_z     : Sets the preferred vertical displacement of the grid
                         point relative to the site; above/below/None.
        land_constraint: A boolean that determines if land sites should only
                         select from grid points also over land.
        ancillary_data : A dictionary containing additional model data that
                         is needed.
                         Must contain {'orography': <cube of orography>}.
                         Needs {'land': <cube of land mask>} if using land
                         constraint.

        Returns:
        --------
        neighbours: Numpy array of grid i,j coordinates that are nearest to
                    each site coordinate given. Includes height difference
                    between site and returned grid point.

        '''
        # Use the default nearest neighbour list as a starting point, and
        # if for some reason it is missing, recreate the list using the fast
        # method.
        if default_neighbours is None:
            neighbour_list = self.fast_nearest_neighbour(cube, sites,
                                                         ancillary_data)
        else:
            neighbour_list = default_neighbours

        orography = data_from_ancillary(ancillary_data, 'orography')
        if land_constraint:
            land = data_from_ancillary(ancillary_data, 'land')

        for i_site, site in enumerate(sites.itervalues()):
            altitude = site['altitude']

            i, j = neighbour_list['i'][i_site], neighbour_list['j'][i_site]
            edgecase = neighbour_list['edge'][i_site]

            node_list = nearest_n_neighbours(i, j, 9)
            if edgecase:
                node_list = node_edge_test(node_list, cube)

            if land_constraint:
                # Check that we are considering a land point and that at least
                # one neighbouring point is also land. If not no modification
                # is made to the nearest neighbour coordinates.

                exclude_self = nearest_n_neighbours(i, j, 9, exclude_self=True)
                if edgecase:
                    exclude_self = node_edge_test(exclude_self, cube)
                if not land[i, j] or not any(land[exclude_self]):
                    continue

                node_list = ConditionalListExtract('not_equal_to').process(
                    land, node_list, 0)

            dz_nearest = abs(altitude - orography[i, j])
            dzs = altitude - orography[node_list]

            dzs, dz_nearest, dz_subset = apply_bias(
                relative_z, dzs, dz_nearest, altitude, orography, i, j)

            ij_min = index_of_minimum_difference(dzs, subset_list=dz_subset)
            i_min, j_min = list_entry_from_index(node_list, ij_min)
            dz_min = abs(altitude - orography[i_min, j_min])

            if dz_min < dz_nearest:
                neighbour_list[i_site] = i_min, j_min, dzs[ij_min], edgecase

        return neighbour_list

# Wrapper routines to use the dz minimisation routine with various options.
# These can be called as methods and set in the diagnostic configs.
# It may be better to simply use the keyword options at a higher level,
# but that will make the config more complex.

    def min_dz_no_bias(self, cube, sites, **kwargs):
        ''' Return local grid neighbour with minimum vertical displacement'''
        return self.minimum_height_error_neighbour(cube, sites,
                                                   relative_z=None,
                                                   **kwargs)

    def min_dz_biased_above(self, cube, sites, **kwargs):
        '''
        Return local grid neighbour with minimum vertical displacement,
        biased to select grid points above the site altitude.

        '''
        return self.minimum_height_error_neighbour(cube, sites,
                                                   relative_z='above',
                                                   **kwargs)

    def min_dz_biased_below(self, cube, sites, **kwargs):
        '''
        Return local grid neighbour with minimum vertical displacement,
        biased to select grid points below the site altitude.

        '''
        return self.minimum_height_error_neighbour(cube, sites,
                                                   relative_z='below',
                                                   **kwargs)

    def min_dz_land_no_bias(self, cube, sites, **kwargs):
        '''
        Return local grid neighbour with minimum vertical displacement.
        Require land point neighbour if site is a land point.

        '''
        return self.minimum_height_error_neighbour(cube, sites,
                                                   relative_z=None,
                                                   land_constraint=True,
                                                   **kwargs)

    def min_dz_land_biased_above(self, cube, sites, **kwargs):
        '''
        Return local grid neighbour with minimum vertical displacement,
        biased to select grid points above the site altitude.
        Require land point neighbour if site is a land point.

        '''
        return self.minimum_height_error_neighbour(cube, sites,
                                                   relative_z='above',
                                                   land_constraint=True,
                                                   **kwargs)

    def min_dz_land_biased_below(self, cube, sites, **kwargs):
        '''
        Return local grid neighbour with minimum vertical displacement,
        biased to select grid points below the site altitude.
        Require land point neighbour if site is a land point.

        '''
        return self.minimum_height_error_neighbour(cube, sites,
                                                   relative_z='below',
                                                   land_constraint=True,
                                                   **kwargs)


def apply_bias(relative_z, dzs, dz_nearest, altitude, orography, i, j):
    '''
    Bias neighbour selection to look for grid points with an
    altitude that is above or below the site if relative_z is
    not None.

    '''
    if relative_z == 'above':
        dz_subset, = np.where(dzs <= 0)
        if dz_nearest > 0:
            dz_nearest = 1.E6
    elif relative_z == 'below':
        dz_subset, = np.where(dzs >= 0)
        if dz_nearest < 0:
            dz_nearest = 1.E6

    if relative_z is None or len(dz_subset) == 0 or len(dz_subset) == len(dzs):
        dz_subset = np.arange(len(dzs))
        dz_nearest = abs(altitude - orography[i, j])

    return dzs, dz_nearest, dz_subset


def xy_test(cube):
    '''
    Test whether a diagnostic cube is on a latitude/longitude grid or uses an
    alternative projection.

    Args:
    -----
    cube    : A diagnostic cube to test.

    Returns:
    --------
    trg_crs : None if the cube data is on a latitude/longitude grid. Otherwise
              trg_crs is the coordinate system in a cartopy format.
    '''
    trg_crs = None
    if (not cube.coord(axis='x').name() == 'longitude' or
            not cube.coord(axis='y').name() == 'latitude'):
        trg_crs = cube.coord_system().as_cartopy_crs()
    return trg_crs


def xy_transform(trg_crs, latitude, longitude):
    '''
    Transforms latitude/longitude coordinate pairs from a latitude/longitude
    grid into an alternative projection defined by trg_crs.

    Args:
    -----
    trg_crs   : Target coordinate system in cartopy format.
    latitude  : Latitude coordinate.
    longitude : Longitude coordinate.

    Returns:
    --------
    x, y : longitude and latitude transformed into the target coordinate
           system.

    '''
    if trg_crs is None:
        return longitude, latitude
    else:
        return trg_crs.transform_point(longitude, latitude,
                                       ccrs.PlateCarree())
