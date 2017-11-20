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
from improver.spotdata.common_functions import (
    ConditionalListExtract, nearest_n_neighbours, get_nearest_coords,
    index_of_minimum_difference, list_entry_from_index, node_edge_check,
    apply_bias, xy_determine, xy_transform)


class PointSelection(object):
    """
    For the selection of source data from a grid for use in deriving
    conditions at an arbitrary coordinate.

    Methods available for determining the neighbours are:

        fast_nearest_neighbour: Closest neighbouring grid point to spot site
                                calculated on a 2D plane (lat/lon).

        minimum_height_error_neighbour
                              : This method uses the nearest neighbour as a
                                starting point but then loosens this constraint
                                to minimise the vertical displacement between
                                the spot site and grid points.

    """

    def __init__(self, method='fast_nearest_neighbour',
                 vertical_bias=None, land_constraint=False):
        """
        The class is called with the desired method to be used in determining
        the grid points closest to sites of interest.

        Args:
            method (string):
                Name of the method of neighbour finding to be used.

            vertical_bias (string or None):
                Sets the preferred vertical displacement bias of the grid point
                relative to the site; above/below/None. If this criteria cannot
                be met (e.g. bias below, but all grid points above site) the
                smallest vertical displacment neighbour will be returned.

            land_constraint (boolean):
                If True spot data sites on land should only select neighbouring
                grid points also over land.

        """
        self.method = method
        self.vertical_bias = vertical_bias
        self.land_constraint = land_constraint

    def process(self, cube, sites, ancillary_data,
                default_neighbours=None, no_neighbours=9):
        """
        Call the selected method for determining neighbouring grid points
        after preparing the necessary diagnostics to be passed in.

        Args:
            cube (iris.cube.Cube):
                Cube of gridded data of a diagnostic; the diagnostic is
                unimportant as long as the grid is structured in the same way
                as those from which data will be extracted using the neighbour
                list.
            sites (OrderedDict):
                Site data, including latitude/longitude and altitude
                information.
                e.g.::

                  {<site_id>: {'latitude': 50, 'longitude': 0,
                               'altitude': 10}}
            ancillary_data (dict):
                Dictionary of ancillary (time invariant) model data that is
                needed.
                e.g.::

                  {'orography': <cube of orography>}
            default_neighbours/no_neighbours :
                see minimum_height_error_neighbour() below.

        Returns:
            neighbours (numpy.dtype):
                Array of grid i,j coordinates that are nearest to each site
                coordinate given. Includes vertical displacement between site
                and returned grid point if orography is provided. Edgepoint is
                a boolean that indicates if the chosen grid point neighbour is
                on the edge of the domain for a circular (e.g. global
                cylindrical) grid; (fields: i, j, dz, edgepoint).

        """
        if self.method == 'fast_nearest_neighbour':
            if 'orography' in ancillary_data.keys():
                orography = ancillary_data['orography'].data
            else:
                orography = None
            return self.fast_nearest_neighbour(cube, sites,
                                               orography=orography)
        elif self.method == 'minimum_height_error_neighbour':
            orography = ancillary_data['orography'].data

            land_mask = None
            if self.land_constraint:
                land_mask = ancillary_data['land_mask'].data

            return self.minimum_height_error_neighbour(
                cube, sites, orography, land_mask=land_mask,
                default_neighbours=default_neighbours,
                no_neighbours=no_neighbours)
        else:
            # Should not make it here unless an unknown method is passed in.
            raise AttributeError('Unknown method "{}" passed to {}.'.format(
                self.method, self.__class__.__name__))

    @staticmethod
    def fast_nearest_neighbour(cube, sites, orography=None):
        """
        Use iris coord.nearest_neighbour_index function to locate the nearest
        grid point to the given latitude/longitude pair.

        Performed on a 2D-surface; consider using the much slower
        iris.analysis.trajectory.interpolate method for a more correct nearest
        neighbour search with projection onto a spherical surface; this is
        typically much slower.


        Args:
            cube/sites : See process() above.

            orography (numpy.array):
                Array of orography data extracted from an iris.cube.Cube that
                corresponds to the grids on which all other input diagnostics
                will be provided (iris.cube.Cube.data).

        Returns:
            neighbours (numpy.array):
                See process() above.

        """
        neighbours = np.empty(len(sites), dtype=[('i', 'i8'),
                                                 ('j', 'i8'),
                                                 ('dz', 'f8'),
                                                 ('edgepoint', 'bool_')])

        # Check cube coords are lat/lon, else transform lookup coordinates.
        trg_crs = xy_determine(cube)

        imax = cube.coord(axis='y').shape[0]
        jmax = cube.coord(axis='x').shape[0]
        iname = cube.coord(axis='y').name()
        jname = cube.coord(axis='x').name()

        for i_site, site in enumerate(sites.itervalues()):
            latitude, longitude, altitude = (site['latitude'],
                                             site['longitude'],
                                             site['altitude'])

            longitude, latitude = xy_transform(trg_crs, latitude, longitude)
            i_latitude, j_longitude = get_nearest_coords(
                cube, latitude, longitude, iname, jname)
            dz_site_grid = 0.

            # Calculate SpotData site vertical displacement from model
            # orography. If site altitude set with np.nan or orography data
            # is unavailable, assume site is at equivalent altitude to nearest
            # neighbour.
            if orography is not None and altitude != np.nan:
                dz_site_grid = altitude - orography[i_latitude, j_longitude]
            else:
                dz_site_grid = 0.

            neighbours[i_site] = (int(i_latitude), int(j_longitude),
                                  dz_site_grid,
                                  (i_latitude == imax or j_longitude == jmax))

        return neighbours

    def minimum_height_error_neighbour(self, cube, sites, orography,
                                       land_mask=None,
                                       default_neighbours=None,
                                       no_neighbours=9):
        """
        Find the horizontally nearest neighbour, then relax the conditions
        to find the neighbouring point in the "no_neighbours" nearest nodes to
        the input coordinate that minimises the height difference. This is
        typically used for temperature, where vertical displacement can be much
        more important that horizontal displacement in determining the
        conditions.

        A vertical displacement bias may be applied with the vertical_bias
        keyword; whether to prefer grid points above or below the site, or
        neither.

        A land constraint may be applied that requires a land grid point be
        selected for a site that is over land. Currently this is established
        by checking that the nearest grid point barring any other conditions
        is a land point. If a site is a sea point it will use the nearest
        neighbour as there should be no vertical displacement difference with
        other sea points.

        Args:
            cube/sites : See process() above.

            default_neighbours (numpy.array):
                An existing list of neighbours from which variations are made
                using specified options (e.g. land_constraint). If unset the
                fast_nearest_neighbour method will be used to build this list.

            orography (numpy.array):
                Array of orography data extracted from an iris.cube.Cube that
                corresponds to the grids on which all other input diagnostics
                will be provided.

            land_mask (numpy.array):
                Array of land_mask data extracted from an iris.cube.Cube that
                corresponds to the grids on which all other input diagnostics
                will be provided.

            no_neighbours (int):
                Number of grid points about the site to consider when relaxing
                the nearest neighbour condition. If unset this defaults to 9.
                e.g. consider a 5x5 grid of points -> no_neighbours = 25.

        Returns:
            neighbours (numpy.array):
                See process() above.

        """

        # Use the default nearest neighbour list as a starting point, and
        # if for some reason it is missing, recreate the list using the fast
        # method.
        if default_neighbours is None:
            neighbours = self.fast_nearest_neighbour(cube, sites,
                                                     orography=orography)
        else:
            neighbours = default_neighbours

        for i_site, site in enumerate(sites.itervalues()):

            altitude = site['altitude']

            # If site altitude is set with np.nan this method cannot be used.
            if altitude == np.nan:
                continue

            i, j, dz_nearest = (neighbours['i'][i_site],
                                neighbours['j'][i_site],
                                neighbours['dz'][i_site])
            edgepoint = neighbours['edgepoint'][i_site]

            node_list = nearest_n_neighbours(i, j, no_neighbours)
            if edgepoint:
                node_list = node_edge_check(node_list, cube)

            if self.land_constraint:
                # Check that we are considering a land point and that at least
                # one neighbouring point is also land. If not no modification
                # is made to the nearest neighbour coordinates.

                neighbour_nodes = nearest_n_neighbours(i, j, no_neighbours,
                                                       exclude_self=True)
                if edgepoint:
                    neighbour_nodes = node_edge_check(neighbour_nodes, cube)
                if not land_mask[i, j] or not any(land_mask[neighbour_nodes]):
                    continue

                # Filter the node_list to keep only land points
                # (land_mask == 1).
                node_list = ConditionalListExtract('not_equal_to').process(
                    land_mask, node_list, 0)

            dzs = altitude - orography[node_list]
            dz_subset = apply_bias(self.vertical_bias, dzs)

            ij_min = index_of_minimum_difference(dzs, subset_list=dz_subset)
            i_min, j_min = list_entry_from_index(node_list, ij_min)
            dz_min = abs(altitude - orography[i_min, j_min])

            # Test to ensure that if multiple vertical displacements are the
            # same we don't select a more distant point because of array
            # ordering.
            if not np.isclose(dz_min, abs(dz_nearest)):
                neighbours[i_site] = i_min, j_min, dzs[ij_min], edgepoint

        return neighbours
