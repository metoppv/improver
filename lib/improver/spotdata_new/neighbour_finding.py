# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
from scipy import spatial

import cartopy.crs as ccrs

from improver.utilities.spatial import (
    get_nearest_coords, lat_lon_determine, lat_lon_transform, coordinate_transform)
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.spotdata.common_functions import (
    ConditionalListExtract, nearest_n_neighbours,
    index_of_minimum_difference, list_entry_from_index, node_edge_check,
    apply_bias)


class NeighbourSelection(object):
    """
    For the selection of a grid point near an arbitrary coordinate, where the
    selection may be the nearest point, or a point that fulfils other
    imposed constraints.

    Constraints available for determining the neighbours are:

    1. land_constraint which requires the selected point to be on land.
    2. minimum_dz which minimises the vertical displacement between the
       given coordinate (when an altitude is provided) and the grid point
       where its altitude is provided by the relevant model or high resolution
       orography.
    3. A combination of the above, where the land constraint is primary and out
       of available land points, the one with the minimal vertical displacement
       is chosen.
    """

    def __init__(self, land_constraint=False, minimum_dz=False,
                 search_radius=1.0E4,
                 site_coordinate_system=ccrs.PlateCarree()):
        """
        Args:
            land_constraint (bool):
                If True the selected neighbouring grid point must be on land,
                where this is determined using a land_mask.
            minimum_dz (bool):
                If True the selected neighbouring grid point must be chosen to
                minimise the vertical displacement compared to the site
                altitude.
            neighbourhood_size (int):
                The length/width of a square neighbourhood centred upon the
                nearest neighbour within which to find a neighbour that matches
                the applied constraints. This number should typically be small
                as we are looking for a nearby point.
            site_coordinate_system (cartopy coordinate system):
                The coordinate system of the sitelist coordinates that will be
                provided. This defaults to be a latitude/longitude grid, a
                PlateCarree projection.
        """
        self.minimum_dz = minimum_dz
        self.land_constraint = land_constraint
        self.search_radius = search_radius
        self.site_coordinate_system = site_coordinate_system
        self.site_x_axis = 'longitude'
        self.site_y_axis = 'latitude'
        self.site_altitude = 'altitude'
        self.geodectic_coordinate_system = False

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return ('<NeighbourSelection: land_constraint: {}, ' +
                'minimum_dz: {}>').format(self.land_constraint,
                                          self.minimum_dz)

    def _transform_sites_coordinate_system(self, sites, cube):
        """
        Function to convert coordinate pairs that specify spot sites into the
        coordinate system of the model from which data will be extracted.

        Args:
            sites (dict):
                A dictionary containing the information about spot sites.
            cube (iris.cube.Cube):
                A cube from the model from which data will be extracted. This
                provides the coordinate system onto which the spot site's
                coordinates should be remapped.

        Returns:
            np.array:
                An array containing the x and y coordinates of the spot sites
                in the target coordinate system, shaped as (n_sites, 2).
        """
        target_coordinate_system = cube.coord_system().as_cartopy_crs()
        x_points = np.array([site[self.site_x_axis] for site in sites])
        y_points = np.array([site[self.site_y_axis] for site in sites])

        return coordinate_transform(self.site_coordinate_system,
                                    target_coordinate_system,
                                    x_points, y_points)[:, 0:2]

    @staticmethod
    def get_nearest_indices(site_coords, cube):
        """
        Uses the iris cube method nearest_neighbour_index to find the nearest
        grid points to a site.

        Args:
            site_coords (np.array):
                An array of shape (n_sites, 2) that contains the x and y
                coordinates of the sites.
            cube (iris.cube.Cube):
                Cube containing a representative grid.
        Returns:
            nearest_indices (np.array):
                A list of shape (2, n_sites) that contains the x and y indices
                of the nearest grid points to the sites. Note that the shape of
                the returned array is reversed to ease use beyond this point.
        """
        nearest_indices = np.zeros((len(site_coords), 2)).astype(np.int)
        for index, (x_point, y_point) in enumerate(site_coords):
            nearest_indices[index, 0] = (
                cube.coord(axis='x').nearest_neighbour_index(x_point))
            nearest_indices[index, 1] = (
                cube.coord(axis='y').nearest_neighbour_index(y_point))
        return nearest_indices

    @staticmethod
    def geocentric_cartesian(cube, x_coords, y_coords):
        """
        A function to convert a geodetic (lat/lon) coordinate system into a
        geocentric (3D trignonometric) system. This function ignores orographic
        height differences between coordinates, giving a 2D projected
        neighbourhood akin to selecting a neighbourhood of grid points about a
        point without considering their vertical displacement.

        Args:
            cube (iris.cube.Cube):
                A cube from which is taken the globe for which the geocentric
                coordinates are being calculated.
            x_coords (np.array):
                An array of x coordinates that will represent one axis of the
                mesh of coordinates to be transformed.
            y_coords (np.array):
                An array of y coordinates that will represent one axis of the
                mesh of coordinates to be transformed.

        Returns:
            cartesian_nodes (np.array):
                An array of all the xyz combinations that describe the nodes of
                the grid, now in 3D geocentric cartesian coordinates. The shape
                of the array is (n_nodes, 3), order x[:, 0], y[:, 1], z[:, 2].
        """
        coordinate_system = cube.coord_system().as_cartopy_crs()
        cartesian_calculator = coordinate_system.as_geocentric()
        z_coords = np.zeros_like(x_coords)
        cartesian_nodes = cartesian_calculator.transform_points(
            coordinate_system, x_coords, y_coords, z_coords)
        return cartesian_nodes

    def build_KDTree(self, land_mask):

        if self.land_constraint:
            included_points = np.nonzero(land_mask.data)
        else:
            included_points = np.where(np.isfinite(land_mask.data.data))

        x_indices = included_points[0]
        y_indices = included_points[1]
        x_coords = land_mask.coord(axis='x').points[x_indices]
        y_coords = land_mask.coord(axis='y').points[y_indices]

        if self.geodetic_coordinate_system:
            nodes = self.geocentric_cartesian(land_mask, x_coords, y_coords)
        else:
            nodes = list(zip(x_coords, y_coords))

        index_nodes = np.array(list(zip(x_indices, y_indices)))

        return spatial.cKDTree(nodes), index_nodes

    def select_minimum_dz(self, orography, site_altitudes, index_nodes,
                          index, distance, indices, land_mask):

        # Values beyond the imposed search radius are set to inf,
        # these need to be excluded.
        valid_indices = np.where(np.isfinite(distance))

        # If no valid neighbours are available in the tree, return None.
        if valid_indices[0].shape[0] == 0:
            return None

        distance = distance[valid_indices]
        indices = indices[valid_indices]

        # Calculate the difference in height between the spot site
        # and grid point.
        grid_point_altitudes = orography.data[tuple(index_nodes[indices].T)]
        vertical_displacements = abs(grid_point_altitudes -
                                     site_altitudes[index])

        # The tree returns an ordered array, the first element
        # being the closest. We search the array for the first
        # element that matches the minimum vertical displacement
        # found, giving us the nearest such point.
        index_of_minimum = (
            np.argmax(vertical_displacements ==
                      vertical_displacements.min()))

        grid_point = index_nodes[indices][index_of_minimum]

        return grid_point


    def process(self, sites, orography, land_mask):

        index_nodes = []
        # Check if we are dealing with a global grid
        self.geodetic_coordinate_system = (
            orography.coord_system().as_cartopy_crs().is_geodetic())

        # Enforce x-y coordinate order for input cubes.
        orography = enforce_coordinate_ordering(
            orography, [orography.coord(axis='x').name(),
                        orography.coord(axis='y').name()])
        land_mask = enforce_coordinate_ordering(
            land_mask, [land_mask.coord(axis='x').name(),
                        land_mask.coord(axis='y').name()])

        # Remap site coordinates on to coordinate system of the model grid.
        site_coords = self._transform_sites_coordinate_system(sites, orography)

        # Find nearest neighbour point using quick iris method.
        nearest_indices = self.get_nearest_indices(site_coords, orography)

        # Create an array containing site altitudes, using the nearest point
        # orography height for any that are unset.
        site_altitudes = np.array([site.get(self.site_altitude, None)
                                  for site in sites])
        site_altitudes = np.where(np.isnan(site_altitudes.astype(float)),
                                  orography.data[tuple(nearest_indices.T)],
                                  site_altitudes)

        # If further constraints are being applied, build a KD Tree which
        # includes points filtered by constraint.
        if self.land_constraint or self.minimum_dz:
            tree, index_nodes = self.build_KDTree(land_mask)

            # Site coordinates made cartesian for global coordinate system
            if self.geodetic_coordinate_system:
                site_coords = self.geocentric_cartesian(
                    orography, site_coords[:, 0], site_coords[:, 1])

            if not self.minimum_dz:
                distances, node_indices = tree.query([site_coords])
                land_neighbour_indices, = index_nodes[node_indices]
                distances = np.array([distances[0], distances[0]]).T
                nearest_indices = np.where(distances < self.search_radius,
                                           land_neighbour_indices,
                                           nearest_indices)
            else:
                distances, node_indices = tree.query(
                    [site_coords], distance_upper_bound=self.search_radius,
                    k=36)

                for index, (distance, indices) in enumerate(zip(
                        distances[0], node_indices[0])):

                    grid_point = self.select_minimum_dz(
                        orography, site_altitudes, index_nodes, index,
                        distance, indices, land_mask)
                    if grid_point is not None:
                        nearest_indices[index] = grid_point

        vertical_displacements = (site_altitudes -
                                  orography.data[tuple(nearest_indices.T)])

        # Return cube of neighbours
#        print('Land?', land_mask.data[tuple(nearest_indices.T)])
#        notset = np.where(land_mask.data[tuple(nearest_indices.T)] == 0)
#        for item in notset[0]:
#            print('Still not land', sites[item])
#            print('{} {}'.format(sites[item]['latitude'], sites[item]['longitude']))

        return nearest_indices, vertical_displacements, site_coords
