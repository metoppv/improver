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

"""Gridded data extraction for the Improver site specific process chain."""

import numpy as np
from iris.coords import AuxCoord, DimCoord
from iris import Constraint
from iris.cube import Cube
from iris.exceptions import InvalidCubeError
from numpy.linalg import lstsq
from improver.spotdata.common_functions import (nearest_n_neighbours,
                                                node_edge_test)
from improver.spotdata.read_input import data_from_dictionary
from improver.constants import (R_DRY_AIR,
                                CP_DRY_AIR)


class ExtractData(object):
    """
    A series of methods for extracting data from grid points to derive
    diagnostic values at off grid positions.

    """

    def __init__(self, method='use_nearest'):
        """
        The class is called with the desired method to be used in extracting/
        interpolating data to the site of interest from gridded data.

        """
        self.method = method

    def process(self, cube, sites, neighbours, ancillary_data, additional_data,
                no_neighbours=9, lower_level=1, upper_level=3):
        """
        Call the correct function to enact the method of data extraction
        specified. This function also handles multiple timesteps, consolidating
        the resulting data cubes into an Iris.CubeList.

        Args:
        -----
        cube : iris.cube.Cube
            Cube of diagnostic data from which to extract spotdata.

        sites : dict
            A dictionary containing the properties of spotdata sites.

        neighbours : numpy.array
            Array of neighbouring grid points that are associated with sites
            in the SortedDictionary of sites.

        additional_data : dict
            A dictionary containing any supplmentary time varying diagnostics
            that are needed for the selected extraction method.

        ancillary_data : dict
            A dictionary containing additional model data that is needed.
            e.g. {'orography': <cube of orography>}

        no_neighbours : int
            Number of grid points about the site to consider when calculating a
            local temperature lapse rate using orography variation.
            e.g. consider a 5x5 grid of points -> no_neighbours = 25.

        Returns:
        --------
        cube : iris.cube.Cube
            An irregularly (i.e. non-gridded) cube of diagnostic data extracted
            at the spotdata sites.

        """
        if self.method == 'use_nearest':
            return self.use_nearest(cube, sites, neighbours)
        elif self.method == 'orography_derived_temperature_lapse_rate':
            orography = data_from_dictionary(ancillary_data, 'orography').data
            return self.orography_derived_temperature_lapse_rate(
                cube, sites, neighbours, orography,
                no_neighbours=no_neighbours)
        elif self.method == 'model_level_temperature_lapse_rate':
            
            pressure_on_height_levels = (
                data_from_dictionary(
                    additional_data, 'pressure_on_height_levels')
                )
            lower_height, = pressure_on_height_levels.coord(
                'height')[lower_level].points
            upper_height, = pressure_on_height_levels.coord(
                'height')[upper_level].points
            height_constraint = Constraint(height=[lower_height, upper_height])

            pressure_on_height_levels = pressure_on_height_levels.extract(
                height_constraint)

            temperature_on_height_levels = (
                data_from_dictionary(
                    additional_data, 'temperature_on_height_levels').extract(
                        height_constraint)
                )
            surface_pressure = data_from_dictionary(additional_data,
                                                    'surface_pressure')

            # Ensure that pressure units are in Pa.
            pressure_on_height_levels.convert_units('Pa')
            surface_pressure.convert_units('Pa')

            return self.model_level_temperature_lapse_rate(
                cube, sites, neighbours, pressure_on_height_levels,
                surface_pressure, temperature_on_height_levels)

        raise AttributeError('Unknown method "{}" passed to {}.'.format(
            self.method, self.__class__.__name__))

    @staticmethod
    def _build_coordinates(latitudes, longitudes, site_ids, utc_offsets):
        """
        Construct coordinates for the irregular iris.Cube containing site data.
        A single dimensional coordinate is created using the running order,
        whilst the non-monotonically increasing coordinates (e.g. bestdata_id)
        are stored in AuxilliaryCoordinates.

        Args:
        -----
        latitudes : list
            A list of latitudes ordered to match the sites OrderedDict.

        longitudes : list
            A list of longitudes ordered to match the sites OrderedDict.

        site_ids   : list
            A list of bestdata site_ids ordered to match the sites OrderedDict.

        utc_offsets : list
            A list of UTC off sets in hours ordered to match the sites
            OrderedDict.

        Returns:
        --------
        Creates iris.DimCoord and iris.AuxCoord objects from the provided data
        for use in constructing new cubes.

        """
        indices = DimCoord(np.arange(len(latitudes)), long_name='index',
                           units='1')
        bd_ids = AuxCoord(site_ids, long_name='bestdata_id', units='1')
        latitude = AuxCoord(latitudes, standard_name='latitude',
                            units='degrees')
        longitude = AuxCoord(longitudes, standard_name='longitude',
                             units='degrees')
        utc_offset = AuxCoord(utc_offsets, long_name='utc_offset',
                              units='hours')
        return indices, bd_ids, latitude, longitude, utc_offset

    def make_cube(self, cube, data, sites):
        """
        Construct and return a cube containing the data extracted from the
        grids by the desired method for the sites provided.

        Args:
        -----
        cube : iris.cube.Cube
            The original diagnostic cube from which data has been extracted.

        data : numpy.array
            Array of diagnostic values extracted for the defined sites.

        sites : dict
            A dictionary containing the properties of spotdata sites.

        Returns:
        --------
        cube : iris.cube.Cube
            An irregularly (i.e. non-gridded) cube of diagnostic data extracted
            at the spotdata sites.

        """
        latitudes = [site['latitude'] for site in sites.itervalues()]
        longitudes = [site['longitude'] for site in sites.itervalues()]
        utc_offsets = [site['utc_offset'] for site in sites.itervalues()]
        site_ids = sites.keys()

        indices, bd_ids, latitude, longitude, utc_offset = (
            self._build_coordinates(
                latitudes, longitudes, site_ids, utc_offsets))

        # Add leading dimension for time.
        data.resize(1, len(data))
        result_cube = Cube(data,
                           long_name=cube.name(),
                           dim_coords_and_dims=[(cube.coord('time'), 0),
                                                (indices, 1)],
                           aux_coords_and_dims=[(latitude, 1),
                                                (longitude, 1),
                                                (utc_offset, 1),
                                                (bd_ids, 1)],
                           units=cube.units)

        # Enables use of long_name above for any name, and then moves it
        # to a standard name if possible.
        result_cube.rename(cube.name())
        return result_cube

    def use_nearest(self, cube, sites, neighbours):
        """
        Simplest case, in which the diagnostic data value at the nearest grid
        point, as determined by the chosen PointSelection method, is used for
        the site.

        Args:
        -----
        See process() above.

        """
        if (not cube.coord_dims(cube.coord(axis='y').name())[0] == 0 or
                not cube.coord_dims(cube.coord(axis='x').name())[0] == 1):
            raise InvalidCubeError("Cube dimensions not as expected.")

        data = cube.data[neighbours['i'], neighbours['j']]
        return self.make_cube(cube, data, sites)

    def orography_derived_temperature_lapse_rate(self, cube, sites, neighbours,
                                                 orography, no_neighbours=9):
        """
        Crude lapse rate method that uses temperature variation and height
        variation across local nodes to derive lapse rate. This is highly
        prone to noise given the small number of points involved and the
        variable degree to which elevation changes across the small number
        of points.

        Args:
        -----
        cube : iris.cube.Cube
            A cube of screen level temperatures at a single time.

        sites/neighbours/no_neighbours : See process() above.

        orography : numpy.array
            Array of orography data on a grid that corresponds to the grid of
            the the diagnostic cube.

        Returns:
        --------
        Data extracted from the screen level temperature cube at spotdata sites
        which has then been adjusted using a temperature lapse rate calculated
        using local variations in temperature with orography.

        """
        def local_lapse_rate(cube, orography, node_list):
            """
            Least-squares fit to local temperature and altitude data for grid
            points defined by node_list to calculate a local lapse rate.

            """
            y_data = cube.data[node_list]
            x_data = orography[node_list]
            matrix = np.vstack([x_data, np.ones(len(x_data))]).T
            gradient, intercept = lstsq(matrix, y_data)[0]
            return [gradient, intercept]

        data = np.empty(shape=(len(sites)))

        for i_site, site in enumerate(sites.itervalues()):
            altitude = site['altitude']

            i, j = neighbours['i'][i_site], neighbours['j'][i_site]
            edgepoint = neighbours['edgepoint'][i_site]
            node_list = nearest_n_neighbours(i, j, no_neighbours)
            if edgepoint:
                node_list = node_edge_test(node_list, cube)

            llr = local_lapse_rate(cube, orography, node_list)
            data[i_site] = llr[0]*altitude + llr[1]

        return self.make_cube(cube, data, sites)

    def model_level_temperature_lapse_rate(self, cube, sites, neighbours,
                                           pressure_on_height_levels,
                                           surface_pressure,
                                           temperature_on_height_levels):
        """
        Lapse rate method based on potential temperature. Follows the work of
        S.B. Vosper 2005 - Near-surface temperature variations over complex
        terrain; Milestone Report RC10JR Local forecasting in complex terrain;
        V 1.0 August 2005.

        Calculate potential temperature gradient and use this to adjust
        temperatures to the spotdata site altitude.

                     pref   R/cp                 p_site  R/cp
        Theta = T ( ------ )         T = Theta ( ------ )
                    p_site                        pref

        ln (Theta) = ln (T)     + kappa [ ln(pref) - ln(p) ]
        ln (T)     = ln (Theta) + kappa [ ln(p) - ln(pref) ]

        kappa = (R_DRY_AIR / CP_DRY_AIR)
        pref = 1000hPa (1.0E5Pa)
        p_site = pressure at spotdata site.


        Args:
        -----
        cube : iris.cube.Cube
            A cube of screen level temperatures at a single time.

        sites/neighbours : See process() above.

        pressure_on_height_levels : iris.cube.Cube
            Pressure data at 50m above the surface at an equivalent time to the
            cube of screen level temperatures.

        surface_pressure : iris.cube.Cube
            Cube of surface pressures at an equivalent time to the cube of
            screen level temperatures.

        temperature_on_height_levels : iris.cube.Cube
            Temperature data at 50m above the surface at an equivalent time to
            the cube of screen level temperatures.

        Returns:
        --------
        Data extracted from the screen level temperature cube at spotdata sites
        which has then been adjusted using a temperature lapse rate calculated
        using multi-level data.

        """
        def _adjust_temperature(theta_base, dthetadz, dz, p_site, kappa):
            # Default UKPP values.
            dz_max_adjustment = 70.
            p_ref = 1.0E5
            
            dz = min(abs(dz), dz_max_adjustment)*np.sign(dz)
            if dz < 0 or dthetadz > 0:
                return (theta_base + dz*dthetadz)*(p_site/p_ref)**kappa
            else:
                return theta_base*(p_site/p_ref)**kappa

        if not cube.name() == 'air_temperature':
            raise ValueError('{} should only be used for adjusting '
                             'temperatures. Cube of type {} is not '
                             'suitable.'.format(self.method, cube.name()))

        # Default UKPP values.
        kappa = R_DRY_AIR/CP_DRY_AIR
        dz_tolerance = 2.
        dthetadz_threshold = 0.02

        heights = temperature_on_height_levels.coord('height').points
        z_lower = heights[0]
        z_upper = heights[1] 
        dz_model_levels = z_upper - z_lower

        data = np.empty(shape=(len(sites)))
        temp_data = np.empty(shape=(4, len(sites)))
        for i_site in range(len(sites)):
            i, j, dz = (neighbours['i'][i_site], neighbours['j'][i_site],
                        neighbours['dz'][i_site])

            # Use neighbour grid point value if dz < dz_tolerance.
            if abs(dz) < dz_tolerance:
                data[i_site] = cube.data[i, j]
                continue

            # Temperatures at surface, lower model level and upper model level.
            t_surface = cube.data[i, j]
            t_lower = temperature_on_height_levels.data[0, i, j]
            t_upper = temperature_on_height_levels.data[1, i, j]

            # Pressures at surface, lower model level and upper model level.
            p_surface = surface_pressure.data[i, j]
            p_lower = pressure_on_height_levels.data[0, i, j]
            p_upper = pressure_on_height_levels.data[1, i, j]

            # Potential temperature at surface, lower model level and upper
            # model level.
            theta_surface = t_surface*(1.0E5/p_surface)**kappa
            theta_lower = t_lower*(1.0E5/p_lower)**kappa
            theta_upper = t_upper*(1.0E5/p_upper)**kappa

            # Calculate potential temperature gradient using levels away from
            # surface.
            dthetadz = (theta_upper - theta_lower)/dz_model_levels

            # If the potential temperature gradient is below the defined
            # dthetadz_threshold, use the non-surface levels.
            if dthetadz <= dthetadz_threshold:
                dz_from_model_level = dz - z_lower
                p_grad = (p_upper - p_lower)/dz_model_levels
                p_site = p_lower + p_grad*dz_from_model_level
                theta_base = theta_lower
            else:
                dthetadz = (theta_lower-theta_surface)/z_lower
                dz_from_model_level = dz
                p_grad = (p_lower - p_surface)/z_lower
                p_site = p_surface + p_grad*dz_from_model_level
                theta_base = theta_surface

            data[i_site] = _adjust_temperature(
                theta_base, dthetadz, dz_from_model_level, p_site, kappa)

        return self.make_cube(cube, data, sites)
