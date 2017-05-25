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
import warnings
from iris import FUTURE
from iris.coords import AuxCoord, DimCoord
from iris import Constraint
from iris.cube import Cube, CubeList
from numpy.linalg import lstsq
from improver.spotdata.common_functions import (nearest_n_neighbours,
                                                datetime_constraint,
                                                node_edge_test)
from improver.spotdata.ancillaries import data_from_ancillary
from improver.spotdata.read_input import get_additional_diagnostics
from improver.constants import (R_DRY_AIR,
                                CP_DRY_AIR)

FUTURE.cell_datetime_objects = True


class ExtractData(object):
    '''
    A series of methods for extracting data from grid points to derive
    diagnostic values at off grid positions.

    '''

    def __init__(self, method='use_nearest'):
        """
        The class is called with the desired method to be used in extracting/
        interpolating data to the site of interest from gridded data.

        """

        self.method = method

    def process(self, cubes, sites, neighbours, forecast_times,
                additional_data, **kwargs):
        """
        Call the correct function to enact the method of data extraction
        specified. This function also handles multiple timesteps, consolidating
        the resulting data cubes into an Iris.CubeList.

        Args:
        -----
        cubes:      iris.cube.CubeList of diagnostic data spanning available
                    times.
        sites:      Dictionary of site data, including lat/lon and altitude
                    information.
        forecast_times:
                    A list of datetime objects representing forecast times for
                    which data is required.
        neighbours: A list of neigbouring grid points that corresponds to sites
                    in the SortedDictionary of sites.
        additional_data:
                   A dictionary containing any supplmentary time varying
                   diagnostics that are needed for the selected extraction
                   method.
        ancillary_data:
                    A dictionary containing additional model data that
                    is needed. e.g. {'orography': <cube of orography>}

        Returns:
        --------
        resulting_cubes:
                    An iris.CubeList of irregular (i.e. non-gridded) cubes of
                    data that correspond to the sites of interest at the times
                    of interest.
        """

        if forecast_times is None:
            raise Exception("No forecast times provided.")

        resulting_cubes = CubeList()
        function = getattr(self, self.method)
        for a_time in forecast_times:
            time_extract = datetime_constraint(a_time)
            try:
                cube_in, = cubes.extract(time_extract)
            except:
                msg = ('Forecast time {} not found within data cubes.'.format(
                        a_time.strftime("%Y-%m-%d:%H:%M")))
                warnings.warn(msg)
                continue

            if additional_data is not None:
                for key in additional_data.keys():
                    ad_time, = additional_data[key].extract(time_extract)
                    kwargs.update({key: ad_time})

            resulting_cubes.append(
                function(cube_in, sites, neighbours, **kwargs)
                )

        return resulting_cubes

    @staticmethod
    def _build_coordinates(latitudes, longitudes, site_ids, gmtoffsets):
        '''
        Construct coordinates for the irregular iris.Cube containing site data.
        A single dimensional coordinate is created using the running order,
        whilst the non-monotonically increasing coordinates (e.g. bestdata_id)
        are stored in AuxilliaryCoordinates.

        Args:
        -----
        latitudes  : A list of latitudes ordered to correspond with the sites
                     OrderedDict.
        longitudes : A list of longitudes ordered to correspond with the sites
                     OrderedDict.
        site_ids   : A list of bestdata site_ids ordered to correspond with the
                     sites OrderedDict.
        gmtoffsets : A list of gmt off sets in hours ordered to correspond with
                     the sites OrderedDict.

        Returns:
        --------
        Creates iris.DimCoord and iris.AuxCoord objects from the provided data
        for use in constructing new cubes.
        '''
        indices = DimCoord(np.arange(len(latitudes)), long_name='index',
                           units='1')
        bd_ids = AuxCoord(site_ids, long_name='bestdata_id', units='1')
        latitude = AuxCoord(latitudes, standard_name='latitude',
                            units='degrees')
        longitude = AuxCoord(longitudes, standard_name='longitude',
                             units='degrees')
        gmtoffset = AuxCoord(gmtoffsets, long_name='gmtoffset',
                             units='hours')
        return indices, bd_ids, latitude, longitude, gmtoffset

    def make_cube(self, cube, data, sites):
        '''
        Construct and return a cube containing the data extracted from the
        grids by the desired method for the sites provided.

        '''
        latitudes = [site['latitude'] for site in sites.itervalues()]
        longitudes = [site['longitude'] for site in sites.itervalues()]
        gmtoffsets = [site['gmtoffset'] for site in sites.itervalues()]
        site_ids = sites.keys()

        indices, bd_ids, latitude, longitude, gmtoffset = (
            self._build_coordinates(
                latitudes, longitudes, site_ids, gmtoffsets))

        # Add leading dimension for time.
        data.resize(1, len(data))
        result_cube = Cube(data,
                           long_name=cube.name(),
                           dim_coords_and_dims=[(cube.coord('time'), 0),
                                                (indices, 1)],
                           aux_coords_and_dims=[(latitude, 1),
                                                (longitude, 1),
                                                (gmtoffset, 1),
                                                (bd_ids, 1)],
                           units=cube.units)

        # Enables use of long_name above for any name, and then moves it
        # to a standard name if possible.
        result_cube.rename(cube.name())
        return result_cube

    def use_nearest(self, cube, sites, neighbours, ancillary_data=None):
        '''
        Simplest case, in which the diagnostic data value at the nearest grid
        point, as determined by the chosen PointSelection method, is used for
        the site.

        '''
        if (not cube.coord_dims(cube.coord(axis='y').name())[0] == 0 or
                not cube.coord_dims(cube.coord(axis='x').name())[0] == 1):
            raise Exception("Cube dimensions not as expected.")

        data = cube.data[neighbours['i'], neighbours['j']]
        return self.make_cube(cube, data, sites)

    def orography_derived_temperature_lapse_rate(self, cube, sites, neighbours,
                                                 ancillary_data=None):
        '''
        Crude lapse rate method that uses temperature variation and height
        variation across local nodes to derive lapse rate. This is highly
        prone to noise given the small number of points involved and the
        variable degree to which elevation changes across the small number
        of points.

        '''
        def local_lapse_rate(cube, orography, node_list):
            '''
            Least-squares fit to local temperature and altitude data for grid
            points defined by node_list to calculate a local lapse rate.

            '''
            y_data = cube.data[node_list]
            x_data = orography[node_list]
            matrix = np.vstack([x_data, np.ones(len(x_data))]).T
            gradient, intercept = lstsq(matrix, y_data)[0]
            return [gradient, intercept]

        orography = data_from_ancillary(ancillary_data, 'orography')
        data = np.empty(shape=(len(sites)))

        for i_site, site in enumerate(sites.itervalues()):
            altitude = site['altitude']

            i, j = neighbours['i'][i_site], neighbours['j'][i_site]
            edgecase = neighbours['edge']
            node_list = nearest_n_neighbours(i, j, 9)
            if edgecase:
                node_list = node_edge_test(node_list, cube)

            llr = local_lapse_rate(cube, orography, node_list)
            data[i_site] = llr[0]*altitude + llr[1]

        return self.make_cube(cube, data, sites)

    def model_level_temperature_lapse_rate(self, cube, sites, neighbours,
                                           ancillary_data=None,
                                           pressure_on_height_levels=None,
                                           surface_pressure=None,
                                           temperature_on_height_levels=None):
        '''
        Lapse rate method based on potential temperature. Follows the work of
        S.B. Vosper 2005 (Near-surface temperature variations over complex
        terrain).

        '''
        if (pressure_on_height_levels is None or
                surface_pressure is None or
                temperature_on_height_levels is None):
            raise Exception(
                "Required additional data is unset: \n"
                "pressure_on_height_levels = {}\n"
                "temperature_on_height_levels = {}\n"
                "surface_pressure = {}\n".format(
                    pressure_on_height_levels,
                    temperature_on_height_levels,
                    surface_pressure)
                )

        pressure_on_height_levels.convert_units('hPa')
        surface_pressure.convert_units('hPa')

        h50con = Constraint(height=50)
        t50 = temperature_on_height_levels.extract(h50con)
        p50 = pressure_on_height_levels.extract(h50con)
        Kappa = R_DRY_AIR/CP_DRY_AIR

        data = np.empty(shape=(len(sites)))
        for i_site in range(len(sites)):

            i, j, dz = (neighbours['i'][i_site], neighbours['j'][i_site],
                        neighbours['dz'][i_site])

            # Use neighbour grid point value if vertical displacement=0.
            if dz == 0.:
                data[i_site] = cube.data[i, j]
                continue

            t_upper = t50.data[i, j]
            p_upper = p50.data[i, j]
            t_surface = cube.data[i, j]
            p_surface = surface_pressure.data[i, j]

            p_grad = (p_upper - p_surface)/50.
            p_site = p_surface + p_grad*dz

            theta_upper = t_upper*(1000./p_upper)**Kappa
            theta_surface = t_surface*(1000./p_surface)**Kappa
            dthetadz = (theta_upper - theta_surface)/50.

            if abs(dz) < 1.:
                t1p5 = t_surface
            else:
                dz = min(abs(dz), 70.)*np.sign(dz)
                if dthetadz > 0:
                    t1p5 = theta_surface*(p_site/1000.)**Kappa
                else:
                    t1p5 = (theta_surface + dz*dthetadz)*(p_site/1000.)**Kappa

            data[i_site] = t1p5

        return self.make_cube(cube, data, sites)


def get_method_prerequisites(method, diagnostic_data_path):
    '''
    Determine which additional diagnostics are required for a given
    method of data extraction.

    Args:
    -----
    method   : The method of data extraction that is being used.

    Returns:
    --------
    ad       : A dictionary keyed with the diagnostic names and containing the
               additional cubes that are required.

    '''
    if method == 'model_level_temperature_lapse_rate':
        additional_diagnostics = [
            'temperature_on_height_levels',
            'pressure_on_height_levels',
            'surface_pressure']
    else:
        return None

    ad = {}
    for item in additional_diagnostics:
        ad.update({item:
                   get_additional_diagnostics(
                       item, diagnostic_data_path)
                   })
    return ad
