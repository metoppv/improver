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

"""Neighbour finding for the Improver site specific process chain."""

import warnings

import cartopy.crs as ccrs
import numpy as np
from scipy.spatial import cKDTree

from improver import BasePlugin
from improver.metadata.utilities import create_coordinate_hash
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


class NeighbourSelection(BasePlugin):
    """
    For the selection of a grid point near an arbitrary coordinate, where the
    selection may be the nearest point, or a point that fulfils other
    imposed constraints.

    Constraints available for determining the neighbours are:

    1. land_constraint which requires the selected point to be on land.
    2. minimum_dz which minimises the vertical displacement between the
       given coordinate (when an altitude is provided) and the grid point
       where its altitude is provided by the relevant model or high resolution
       orography. Note that spot coordinates provided without an altitude are
       given the altitude of the nearest grid point taken from the orography
       cube.
    3. A combination of the above, where the land constraint is primary and out
       of available land points, the one with the minimal vertical displacement
       is chosen.
    """

    def __init__(self, land_constraint=False, minimum_dz=False,
                 search_radius=1.0E4,
                 site_coordinate_system=ccrs.PlateCarree(),
                 site_x_coordinate='longitude', site_y_coordinate='latitude',
                 node_limit=36):
        """
        Args:
            land_constraint (bool):
                If True the selected neighbouring grid point must be on land,
                where this is determined using a land_mask.
            minimum_dz (bool):
                If True the selected neighbouring grid point must be chosen to
                minimise the vertical displacement compared to the site
                altitude.
            search_radius (float):
                The radius in metres from a spot site within which to search
                for a grid point neighbour.
            site_coordinate_system (cartopy coordinate system):
                The coordinate system of the sitelist coordinates that will be
                provided. This defaults to be a latitude/longitude grid, a
                PlateCarree projection.
            site_x_coordinate (str):
                The key that identifies site x coordinates in the provided site
                dictionary. Defaults to longitude.
            site_y_coordinate (str):
                The key that identifies site y coordinates in the provided site
                dictionary. Defaults to latitude.
            node_limit (int):
                The upper limit for the number of nearest neighbours to return
                when querying the tree for a selection of neighbours from which
                one matching the minimum_dz constraint will be picked.
        """
        self.minimum_dz = minimum_dz
        self.land_constraint = land_constraint
        self.search_radius = search_radius
        self.site_coordinate_system = site_coordinate_system
        self.site_x_coordinate = site_x_coordinate
        self.site_y_coordinate = site_y_coordinate
        self.site_altitude = 'altitude'
        self.node_limit = node_limit
        self.global_coordinate_system = False

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return ('<NeighbourSelection: land_constraint: {}, ' +
                'minimum_dz: {}, search_radius: {}, site_coordinate_system'
                ': {}, site_x_coordinate:{}, site_y_coordinate: {}, '
                'node_limit: {}>').format(
                    self.land_constraint, self.minimum_dz, self.search_radius,
                    self.site_coordinate_system.__class__,
                    self.site_x_coordinate, self.site_y_coordinate,
                    self.node_limit)

    def neighbour_finding_method_name(self):
        """
        Create a name to describe the neighbour method based on the constraints
        provided.

        Returns:
            str:
                A string that describes the neighbour finding method employed.
                This is essentially a concatenation of the options.
        """
        method_name = '{}{}{}'.format('nearest',
                                      '_land' if self.land_constraint else '',
                                      '_minimum_dz' if self.minimum_dz else '')
        return method_name

    def _transform_sites_coordinate_system(self, x_points, y_points,
                                           target_crs):
        """
        Function to convert coordinate pairs that specify spot sites into the
        coordinate system of the model from which data will be extracted. Note
        that the cartopy functionality returns a z-coordinate which we do not
        want in this case, as such only the first two columns are returned.

        Args:
            x_points (numpy.ndarray):
                An array of x coordinates to be transformed in conjunction
                with the corresponding y coordinates.
            y_points (numpy.ndarray):
                An array of y coordinates to be transformed in conjunction
                with the corresponding x coordinates.
            target_crs (cartopy.crs):
                Coordinate system to which the site coordinates should be
                transformed. This should be the coordinate system of the model
                from which data will be spot extracted.
        Returns:
            numpy.ndarray:
                An array containing the x and y coordinates of the spot sites
                in the target coordinate system, shaped as (n_sites, 2). The
                z coordinate column is excluded from the return.
        """
        return target_crs.transform_points(
            self.site_coordinate_system, x_points, y_points)[:, 0:2]

    def check_sites_are_within_domain(self, sites, site_coords, site_x_coords,
                                      site_y_coords, cube):
        """
        A function to remove sites from consideration if they fall outside the
        domain of the provided model cube. A warning is raised and the details
        of each rejected site are printed.

        Args:
            sites (list of dict):
                A list of dictionaries defining the spot sites for which
                neighbours are to be found. e.g.:

                   [{'altitude': 11.0, 'latitude': 57.867000579833984,
                    'longitude': -5.632999897003174, 'wmo_id': 3034}]

            site_coords (numpy.ndarray):
                An array of shape (n_sites, 2) that contains the spot site
                coordinates in the coordinate system of the model cube.
            site_x_coords (numpy.ndarray):
                The x coordinates of the spot sites in their original
                coordinate system, from which invalid sites must be removed.
            site_y_coords (numpy.ndarray):
                The y coordinates of the spot sites in their original
                coordinate system, from which invalid sites must be removed.
            cube (iris.cube.Cube):
                A cube that is representative of the model/grid from which spot
                data will be extracted.

        Returns:
            (tuple): tuple containing:
                **sites** (numpy.ndarray):
                    The sites modified to filter out the sites falling outside
                    the grid domain of the cube.
                **site_coords** (numpy.ndarray):
                    The site_coords modified to filter out the sites falling
                    outside the grid domain of the cube.
                **site_x_coords** (numpy.ndarray):
                    The x_coords modified to filter out the sites falling
                    outside the grid domain of the cube.
                **site_y_coords** (numpy.ndarray):
                    The y_coords modified to filter out the sites falling
                    outside the grid domain of the cube.
        """
        # Get the grid domain limits
        x_min = cube.coord(axis='x').bounds.min()
        x_max = cube.coord(axis='x').bounds.max()
        y_min = cube.coord(axis='y').bounds.min()
        y_max = cube.coord(axis='y').bounds.max()

        if self.global_coordinate_system:
            domain_valid = np.where(
                (site_coords[:, 1] >= y_min) & (site_coords[:, 1] <= y_max))

            domain_invalid = np.where(
                (site_coords[:, 1] < y_min) | (site_coords[:, 1] > y_max))
        else:
            domain_valid = np.where(
                (site_coords[:, 0] >= x_min) & (site_coords[:, 0] <= x_max) &
                (site_coords[:, 1] >= y_min) & (site_coords[:, 1] <= y_max))

            domain_invalid = np.where(
                (site_coords[:, 0] < x_min) | (site_coords[:, 0] > x_max) |
                (site_coords[:, 1] < y_min) | (site_coords[:, 1] > y_max))

        num_invalid = len(domain_invalid[0])
        if num_invalid > 0:
            msg = ("{} spot sites fall outside the grid domain and will not be"
                   " processed. These sites are:\n".format(num_invalid))
            dyn_msg = '{}\n'
            for site in np.array(sites)[domain_invalid]:
                msg += dyn_msg.format(site)
            warnings.warn(msg)

        sites = np.array(sites)[domain_valid]
        site_coords = site_coords[domain_valid]
        site_x_coords = site_x_coords[domain_valid]
        site_y_coords = site_y_coords[domain_valid]
        return sites, site_coords, site_x_coords, site_y_coords

    @staticmethod
    def get_nearest_indices(site_coords, cube):
        """
        Uses the iris cube method nearest_neighbour_index to find the nearest
        grid points to a site.

        Args:
            site_coords (numpy.ndarray):
                An array of shape (n_sites, 2) that contains the x and y
                coordinates of the sites.
            cube (iris.cube.Cube):
                Cube containing a representative grid.
        Returns:
            numpy.ndarray:
                A list of shape (n_sites, 2) that contains the x and y indices
                of the nearest grid points to the sites.
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
        A function to convert a global (lat/lon) coordinate system into a
        geocentric (3D trignonometric) system. This function ignores orographic
        height differences between coordinates, giving a 2D projected
        neighbourhood akin to selecting a neighbourhood of grid points about a
        point without considering their vertical displacement.

        Args:
            cube (iris.cube.Cube):
                A cube from which is taken the globe for which the geocentric
                coordinates are being calculated.
            x_coords (numpy.ndarray):
                An array of x coordinates that will represent one axis of the
                mesh of coordinates to be transformed.
            y_coords (numpy.ndarray):
                An array of y coordinates that will represent one axis of the
                mesh of coordinates to be transformed.
        Returns:
            numpy.ndarray:
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
        """
        Build a KDTree for extracting the nearest point or points to a site.
        The tree can be built with a constrained set of grid points, e.g. only
        land points, if required.

        Args:
            land_mask (iris.cube.Cube):
                A land mask cube for the model/grid from which grid point
                neighbours are being selected.
        Returns:
            (tuple): tuple containing:
                **scipy.spatial.ckdtree.cKDTree**:
                    A KDTree containing the required nodes, built using the
                    scipy cKDTree method.
                **numpy.ndarray**:
                    An array of shape (n_nodes, 2) that contains the x and y
                    indices that correspond to the selected node,
                    e.g. node=100 -->  x_coord_index=10, y_coord_index=300,
                    index_nodes[100] = [10, 300]
        """
        if self.land_constraint:
            included_points = np.nonzero(land_mask.data)
        else:
            included_points = np.where(np.isfinite(land_mask.data.data))

        x_indices = included_points[0]
        y_indices = included_points[1]
        x_coords = land_mask.coord(axis='x').points[x_indices]
        y_coords = land_mask.coord(axis='y').points[y_indices]

        if self.global_coordinate_system:
            nodes = self.geocentric_cartesian(land_mask, x_coords, y_coords)
        else:
            nodes = list(zip(x_coords, y_coords))

        index_nodes = np.array(list(zip(x_indices, y_indices)))

        return cKDTree(nodes), index_nodes

    def select_minimum_dz(self, orography, site_altitude, index_nodes,
                          distance, indices):
        """
        Given a selection of nearest neighbours to a given site, this function
        calculates the absolute vertical displacement between the site and the
        neighbours. It then returns grid indices of the neighbour with the
        minimum vertical displacement (i.e. at the most similar altitude). The
        number of neighbours to consider is a maximum of node_limit, but these
        may be limited by the imposed search_radius, or this limit may be
        insufficient to reach the search radius, in which case a warning is
        raised.

        Args:
            orography (iris.cube.Cube):
                A cube of orography, used to obtain the grid point altitudes.
            site_altitude (float):
                The altitude of the spot site being considered.
            index_nodes (numpy.ndarray):
                An array of shape (n_nodes, 2) that contains the x and y
                indices that correspond to the selected node,
            distance (numpy.ndarray):
                An array that contains the distances from the spot site to each
                grid point neighbour being considered. The number maybe np.inf
                if the site is beyond the search_radius.
            indices (numpy.ndarray):
                An array of tree node indices identifying the neigbouring grid
                points, the list corresponding to the array of distances.
        Returns:
            numpy.ndarray or None:
                A 2-element array giving the x and y indices of the chosen grid
                point neighbour. Returns None if no valid neighbours were found
                in the tree query.
        """
        # Values beyond the imposed search radius are set to inf,
        # these need to be excluded.
        valid_indices = np.where(np.isfinite(distance))

        # If no valid neighbours are available in the tree, return None.
        if valid_indices[0].shape[0] == 0:
            return None

        # If the last distance is finite the number of tree nodes may not be
        # sufficient to fill the search radius, raise a warning.
        if np.isfinite(distance[-1]):
            msg = ('Limit on number of nearest neighbours to return, {}, may '
                   'not be sufficiently large to fill search_radius {}'.format(
                       self.node_limit, self.search_radius))
            warnings.warn(msg)

        indices = indices[valid_indices]

        # Calculate the difference in height between the spot site
        # and grid point.
        grid_point_altitudes = orography.data[tuple(index_nodes[indices].T)]
        vertical_displacements = abs(grid_point_altitudes - site_altitude)

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
        """
        Using the constraints provided, find the nearest grid point neighbours
        to the given spot sites for the model/grid given by the input cubes.
        Returned is a cube that contains the defining characteristics of the
        spot sites (e.g. x coordinate, y coordinate, altitude) and the indices
        of the selected grid point neighbour.

        Args:
            sites (list of dict):
                A list of dictionaries defining the spot sites for which
                neighbours are to be found. e.g.:

                   [{'altitude': 11.0, 'latitude': 57.867000579833984,
                    'longitude': -5.632999897003174, 'wmo_id': 3034}]

            orography (iris.cube.Cube):
                A cube of orography, used to obtain the grid point altitudes.
            land_mask (iris.cube.Cube):
                A land mask cube for the model/grid from which grid point
                neighbours are being selected, with land points set to one and
                sea points set to zero.
        Returns:
            iris.cube.Cube:
                A cube containing both the spot site information and for each
                the grid point indices of its nearest neighbour as per the
                imposed constraints.
        """
        # Check if we are dealing with a global grid.
        self.global_coordinate_system = orography.coord(axis='x').circular

        # Exclude regional grids with spatial dimensions other than metres.
        if not self.global_coordinate_system:
            if not orography.coord(axis='x').units == 'metres':
                msg = ('Cube spatial coordinates for regional grids must be'
                       'in metres to match the defined search_radius.')
                raise ValueError(msg)

        # Ensure land_mask and orography are on the same grid.
        if not orography.dim_coords == land_mask.dim_coords:
            msg = ('Orography and land_mask cubes are not on the same '
                   'grid.')
            raise ValueError(msg)

        # Enforce x-y coordinate order for input cubes.
        enforce_coordinate_ordering(
            orography, [orography.coord(axis='x').name(),
                        orography.coord(axis='y').name()])
        enforce_coordinate_ordering(
            land_mask, [land_mask.coord(axis='x').name(),
                        land_mask.coord(axis='y').name()])

        # Remap site coordinates on to coordinate system of the model grid.
        site_x_coords = np.array([site[self.site_x_coordinate]
                                  for site in sites])
        site_y_coords = np.array([site[self.site_y_coordinate]
                                  for site in sites])
        site_coords = self._transform_sites_coordinate_system(
            site_x_coords, site_y_coords,
            orography.coord_system().as_cartopy_crs())

        # Exclude any sites falling outside the domain given by the cube and
        # notify the user.
        sites, site_coords, site_x_coords, site_y_coords = (
            self.check_sites_are_within_domain(
                sites, site_coords, site_x_coords, site_y_coords,
                orography))

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
            # Build the KDTree, an internal test for the land_constraint checks
            # whether to exclude sea points from the tree.
            tree, index_nodes = self.build_KDTree(land_mask)

            # Site coordinates made cartesian for global coordinate system
            if self.global_coordinate_system:
                site_coords = self.geocentric_cartesian(
                    orography, site_coords[:, 0], site_coords[:, 1])

            if not self.minimum_dz:
                # Query the tree for the nearest neighbour, in this case a land
                # neighbour is returned along with the distance to it.
                distances, node_indices = tree.query([site_coords])
                # Look up the grid coordinates that correspond to the tree node
                land_neighbour_indices, = index_nodes[node_indices]
                # Use the found land neighbour if it is within the
                # search_radius, otherwise use the nearest neighbour.
                distances = np.array([distances[0], distances[0]]).T
                nearest_indices = np.where(distances < self.search_radius,
                                           land_neighbour_indices,
                                           nearest_indices)
            else:
                # Query the tree for self.node_limit nearby neighbours.
                distances, node_indices = tree.query(
                    [site_coords], distance_upper_bound=self.search_radius,
                    k=self.node_limit)
                # Loop over the sites and for each choose the returned
                # neighbour with the minimum vertical displacement.
                for index, (distance, indices) in enumerate(zip(
                        distances[0], node_indices[0])):
                    grid_point = self.select_minimum_dz(
                        orography, site_altitudes[index], index_nodes,
                        distance, indices)
                    # None is returned if the tree query returned no neighbours
                    # within the search radius.
                    if grid_point is not None:
                        nearest_indices[index] = grid_point

        # Calculate the vertical displacements between the chosen grid point
        # and the spot site.
        vertical_displacements = (site_altitudes -
                                  orography.data[tuple(nearest_indices.T)])

        # Create a list of WMO IDs if available. These are stored as strings
        # to accommodate the use of 'None' for unset IDs.
        wmo_ids = [str(site.get('wmo_id', None)) for site in sites]

        # Construct a name to describe the neighbour finding method employed
        method_name = self.neighbour_finding_method_name()

        # Create an array of indices and displacements to return
        data = np.stack((nearest_indices[:, 0], nearest_indices[:, 1],
                         vertical_displacements), axis=1)
        data = np.expand_dims(data, 1).astype(np.float32)

        # Regardless of input sitelist coordinate system, the site coordinates
        # are stored as latitudes and longitudes in the neighbour cube.
        if self.site_coordinate_system != ccrs.PlateCarree():
            lon_lats = self._transform_sites_coordinate_system(
                site_x_coords, site_y_coords, ccrs.PlateCarree())
            longitudes = lon_lats[:, 0]
            latitudes = lon_lats[:, 1]
        else:
            longitudes = site_x_coords
            latitudes = site_y_coords

        # Create a cube of neighbours
        neighbour_cube = build_spotdata_cube(
            data, 'grid_neighbours', 1, site_altitudes.astype(np.float32),
            latitudes.astype(np.float32), longitudes.astype(np.float32),
            wmo_ids, neighbour_methods=[method_name],
            grid_attributes=['x_index', 'y_index', 'vertical_displacement'])

        # Add a hash attribute based on the model grid to ensure the neighbour
        # cube is only used with a compatible grid.
        grid_hash = create_coordinate_hash(orography)
        neighbour_cube.attributes['model_grid_hash'] = grid_hash

        return neighbour_cube
