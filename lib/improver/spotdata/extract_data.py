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
import iris
import warnings
import copy
from numpy.linalg import lstsq
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from improver.spotdata.common_functions import (nearest_n_neighbours,
                                                node_edge_check)
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
        interpolating data to the site of interest from gridded data. If no
        method is defined in the call to the class it defaults to the
        'use_nearest' method.

        """
        self.method = method

    def process(self, cube, sites, neighbours, ancillary_data, additional_data,
                no_neighbours=9, lower_level=1, upper_level=3,
                dz_tolerance=2., dthetadz_threshold=0.02,
                dz_max_adjustment=70.):
        """
        Call the correct function to enact the method of data extraction
        specified. This function also handles multiple timesteps, consolidating
        the resulting data cubes into an Iris.CubeList.

        Args:
            cube (iris.cube.Cube):
                Cube of diagnostic data from which to extract spotdata.

            sites (OrderedDict):
                A dictionary containing the properties of spotdata sites.

            neighbours (numpy.array):
                Array of neighbouring grid points that are associated with
                sites in the OrderedDict of sites.

            additional_data (dict):
                A dictionary containing any supplmentary time varying
                diagnostics that are needed for the selected extraction method.

            ancillary_data (dict):
                A dictionary containing additional model data that is needed.
                e.g. {'orography': <cube of orography>}

            no_neighbours (int):
                Number of grid points about the site to consider when
                calculating a local temperature lapse rate using orography
                variation.
                e.g. consider a 5x5 grid of points -> no_neighbours = 25.

            upper/lower_level (ints):
                Define the hybrid height model levels to use when calculating
                potential temperature gradients for use in lapse rate
                temperature adjustment.

            dz_tolerance (float (units: m)):
                Vertical displacement between spotdata site and neighbouring
                grid point below which there is no need to perform a lapse rate
                adjustment. Defaults to value of 2m.

            dthetadz_threshold (float (units: K/m)):
                Potential temperature gradient threshold, with gradients above
                this value deemed to have been calculated across an inversion.
                Defaults to UKPP value of 0.02 K/m.

            dz_max_adjustment (float (units: m)):
                Maximum vertical distance over which a temperature will be
                adjusted using the lapse rate. If the spotdata site is more
                than this distance above or below its neighbouring grid point,
                the adjustment will be made using dz = dz_max_adjustment.
                Defaults to UKPP value of 70m.

        Returns:
            cube (iris.cube.Cube):
                An irregular (i.e. non-gridded) cube of diagnostic data
                extracted at the spotdata sites.

        """
        if self.method == 'use_nearest':
            return self.use_nearest(cube, sites, neighbours)
        elif self.method == 'orography_derived_temperature_lapse_rate':
            orography = ancillary_data['orography'].data
            return self.orography_derived_temperature_lapse_rate(
                cube, sites, neighbours, orography,
                no_neighbours=no_neighbours)
        elif self.method == 'model_level_temperature_lapse_rate':
            pressure_on_height_levels = (
                additional_data['pressure_on_height_levels'])

            # Ensure that pressure units are in Pa.
            pressure_on_height_levels.convert_units('Pa')

            lower_pressure = pressure_on_height_levels[lower_level, ...]
            upper_pressure = pressure_on_height_levels[upper_level, ...]

            surface_pressure = additional_data['surface_pressure']
            surface_pressure.convert_units('Pa')

            temperature_on_height_levels = (
                additional_data['temperature_on_height_levels'])
            lower_temperature = temperature_on_height_levels[lower_level, ...]
            upper_temperature = temperature_on_height_levels[upper_level, ...]

            return self.model_level_temperature_lapse_rate(
                cube, sites, neighbours, surface_pressure, lower_pressure,
                upper_pressure, lower_temperature, upper_temperature,
                dz_tolerance, dthetadz_threshold, dz_max_adjustment)

        raise AttributeError('Unknown method "{}" passed to {}.'.format(
            self.method, self.__class__.__name__))

    @staticmethod
    def _build_coordinate(data, coordinate, coord_type=DimCoord,
                          data_type=float, units='1', bounds=None,
                          coord_system=None, custom_function=None):
        """
        Construct an iris.coord.Dim/Auxcoord using the provided options.

        Args:
            data (number/list/np.array):
                List or array of values to populate the coordinate points.
            coordinate (str):
                Name of the coordinate to be built.
            coord_type (iris.coord.AuxCoord or iris.coord.DimCoord (optional)):
                Selection between Dim and Aux coord.
            data_type (<type> (optional)):
                The data type of the coordinate points, e.g. int
            units (str (optional)):
                String defining the coordinate units.
            bounds (np.array (optional)):
                A (len(data), 2) array that defines coordinate bounds.
            coord_system(iris.coord_systems.<coord_system> (optional)):
                A coordinate system in which the dimension coordinates are
                defined.
            custom_function (function (optional)):
                A function to apply to the data values before constructing the
                coordinate, e.g. np.nan_to_num.

        Returns:
            iris coordinate:
                Dim or Auxcoord as chosen.

        """
        data = np.array(data, data_type)
        if custom_function is not None:
            data = custom_function(data)

        crd_out = coord_type(data, long_name=coordinate, units=units,
                             coord_system=coord_system, bounds=bounds)
        crd_out.rename(coordinate)
        return crd_out

    @staticmethod
    def _aux_coords_to_make():
        """
        Define coordinates that need to be made for the cube to be produced.

        Returns:
            dict:
                Dictionary of coordinates that are required for a spotdata
                cube.

        """
        return {'latitude': {'units': 'degrees', 'data_type': float,
                             'coord_type': AuxCoord},
                'longitude': {'units': 'degrees', 'data_type': float,
                              'coord_type': AuxCoord},
                'altitude': {'units': 'm', 'data_type': float,
                             'coord_type': AuxCoord,
                             'custom_function': np.nan_to_num},
                'wmo_site': {'data_type': int, 'coord_type': AuxCoord},
                'utc_offset': {'units': 'hours', 'data_type': float,
                               'coord_type': AuxCoord}
                }

    @staticmethod
    def make_stat_coordinate_first(cube):
        """
        Reorder cube dimension coordinates to ensure the statistical
        coordinate is first.

        Args:
            cube (iris.cube.Cube):
                The cube to be reordered.

        Returns:
            cube (iris.cube.Cube):
                Cube with the statistical coordinate moved to be first.

        Raises:
            Warning if more than one statistical dimension is found. Then
            promotes the first found to become the leading dimension.

        """
        stat_coordinates = ['realization', 'percentile_over']
        cube_dimension_order = {
            coord.name(): cube.coord_dims(coord.name())[0]
            for coord in cube.dim_coords}

        stat_coord = []
        for crd in stat_coordinates:
            stat_coord += [coord for coord in cube_dimension_order.keys()
                           if crd in coord]
        if len(stat_coord) >= 1:
            if len(stat_coord) > 1:
                msg = ('More than one statistical coordinate found. Promoting '
                       'the first found, {}, to the leading dimension.'.format(
                        stat_coord))
                warnings.warn(msg)

            stat_index = cube_dimension_order[stat_coord[0]]
            new_order = range(len(cube_dimension_order))
            new_order.pop(stat_index)
            new_order.insert(0, stat_index)
            cube.transpose(new_order)

        return cube

    def make_cube(self, cube, data, sites):
        """
        Construct and return a cube containing the data extracted from the
        grids by the desired method for the sites provided.

        Args:
            cube (iris.cube.Cube):
                The original diagnostic cube from which data has been
                extracted.

            data (numpy.array):
                Array of diagnostic values extracted for the defined sites.

            sites (OrderedDict):
                A dictionary containing the properties of spotdata sites.

        Returns:
            cube (iris.cube.Cube):
                An irregularly (i.e. non-gridded) cube of diagnostic data
                extracted at the spotdata sites.

        """

        # Ensure time is a dimension coordinate and convert to seconds.
        cube_coords = [coord.name() for coord in cube.dim_coords]
        if 'time' not in cube_coords:
            cube = iris.util.new_axis(cube, 'time')
        cube.coord('time').convert_units('seconds since 1970-01-01 00:00:00')

        cube_coords = [coord.name() for coord in cube.coords()]
        if 'forecast_reference_time' not in cube_coords:
            raise CoordinateNotFoundError(
                'No forecast reference time found on source cube.')
        cube.coord('forecast_reference_time').convert_units(
            'seconds since 1970-01-01 00:00:00')

        # Replicate all non spatial dimension coodinates.
        n_non_spatial_dimcoords = len(cube.dim_coords) - 2
        non_spatial_dimcoords = cube.dim_coords[0:n_non_spatial_dimcoords]
        dim_coords = [coord for coord in non_spatial_dimcoords]

        # Add an index coordinate as a dimension coordinate.
        indices = self._build_coordinate(np.arange(len(sites)), 'index',
                                         data_type=int)
        dim_coords.append(indices)

        # Record existing scalar coordinates on source cube. Aux coords
        # associated with dimensions cannot be preserved as the dimensions will
        # be reshaped and the auxiliarys no longer compatible.
        # Forecast period is ignored for the case where the input data has
        # an existing forecast_period scalar coordinate.
        scalar_coordinates = [coord.name() for coord in
                              cube.coords(dimensions=[])
                              if coord.name() != 'forecast_period']

        # Build a forecast_period dimension.
        forecast_periods = (cube.coord('time').points -
                            cube.coord('forecast_reference_time').points)
        forecast_period = self._build_coordinate(
            forecast_periods, 'forecast_period', units='seconds')

        # Build the new auxiliary coordinates.
        crds = self._aux_coords_to_make()
        aux_crds = []
        for key, kwargs in zip(crds.keys(), crds.itervalues()):
            aux_data = np.array([entry[key] for entry in sites.itervalues()])
            crd = self._build_coordinate(aux_data, key, **kwargs)
            aux_crds.append(crd)

        # Construct zipped lists of coordinates and indices. New aux coords are
        # associated with the index dimension.
        n_dim_coords = len(dim_coords)
        dim_coords = zip(dim_coords, range(n_dim_coords))
        aux_coords = zip(aux_crds, [n_dim_coords-1]*len(aux_crds))

        # Copy other cube metadata.
        metadata_dict = copy.deepcopy(cube.metadata._asdict())

        # Add leading dimension for time to the data array.
        data = np.expand_dims(data, axis=0)
        result_cube = Cube(data,
                           dim_coords_and_dims=dim_coords,
                           aux_coords_and_dims=aux_coords,
                           **metadata_dict)

        # Add back scalar coordinates from the original cube.
        for coord in scalar_coordinates:
            result_cube.add_aux_coord(cube.coord(coord))

        result_cube.add_aux_coord(forecast_period, cube.coord_dims('time'))

        # Enables use of long_name above for any name, and then moves it
        # to a standard name if possible.
        result_cube.rename(cube.name())

        # Promote any statistical coordinates to be first.
        result_cube = self.make_stat_coordinate_first(result_cube)
        return result_cube

    def use_nearest(self, cube, sites, neighbours):
        """
        Simplest case, in which the diagnostic data value at the nearest grid
        point, as determined by the chosen PointSelection method, is used for
        the site.

        Args:
            See process() above.

        Returns:
            iris.cube.Cube:
                Cube containing data extracted from the diagnostic cube
                grid points associated with spotdata sites.

        """
        data = cube.data[..., neighbours['i'], neighbours['j']]
        return self.make_cube(cube, data, sites)

    def orography_derived_temperature_lapse_rate(self, cube, sites, neighbours,
                                                 orography, no_neighbours=9):
        """
        Crude lapse rate method that uses temperature variation and height
        variation across local nodes to derive lapse rate. Temperature vs.
        height data points are fitted with a least-squares method to determine
        the gradient.

        This method is highly prone to noise given the small number of points
        involved and the variable degree to which elevation changes across
        these points.

        Args:
            cube (iris.cube.Cube):
                A cube of screen level temperatures at a single time.

            sites/neighbours/no_neighbours : See process() above.

            orography (numpy.array):
                Array of orography data on a grid that corresponds to the grid
                of the diagnostic cube.

        Returns:
            iris.cube.Cube:
                Cube containing data extracted from the screen level
                temperature cube at spotdata sites which has then been adjusted
                using a temperature lapse rate calculated using local
                variations in temperature with orography.

        """
        def _local_lapse_rate(cube, orography, node_list):
            """
            Least-squares fit to local temperature and altitude data for grid
            points defined by node_list to calculate a local lapse rate.

            """
            y_data = cube.data[node_list]
            x_data = orography[node_list]
            matrix = np.stack([x_data, np.ones(len(x_data))], axis=0).T
            gradient, intercept = lstsq(matrix, y_data)[0]
            return [gradient, intercept]

        data = np.empty(shape=(len(sites)), dtype=float)

        for i_site, site in enumerate(sites.itervalues()):
            i, j = neighbours['i'][i_site], neighbours['j'][i_site]

            altitude = site['altitude']
            if np.isnan(altitude):
                msg = ('orography_derived_temperature_lapse_rate method '
                       'requires site to have an altitude. Leaving value '
                       'unchanged.')
                warnings.warn(msg)
                data[i_site] = cube.data[i, j]
                continue

            edgepoint = neighbours['edgepoint'][i_site]
            node_list = nearest_n_neighbours(i, j, no_neighbours)
            if edgepoint:
                node_list = node_edge_check(node_list, cube)

            llr = _local_lapse_rate(cube, orography, node_list)
            data[i_site] = llr[0]*altitude + llr[1]

        return self.make_cube(cube, data, sites)

    def model_level_temperature_lapse_rate(
            self, cube, sites, neighbours, surface_pressure, lower_pressure,
            upper_pressure, lower_temperature, upper_temperature,
            dz_tolerance, dthetadz_threshold, dz_max_adjustment):

        """
        Args:
            cube (iris.cube.Cube):
                A cube of screen level temperatures at a single time.

            sites/neighbours : See process() above.

            surface_pressure (iris.cube.Cube):
                Cube of surface pressures at an equivalent time to the cube of
                screen level temperatures.

            lower/upper_pressure (iris.cube.Cube):
                Cubes of pressure data at the defined lower and upper model
                levels, each at an equivalent time to the cube of screen level
                temperatures.

            lower/upper_temperature (iris.cube.Cube):
                Cubes of temperature data at the defined lower and upper model
                levels, each at an equivalent time to the cube of screen level
                temperatures.

            dz_tolerance/dthetadz_threshold/dz_max_adjustment :
                See process docstring.

        Returns:
            iris.cube.Cube:
                Cube containing data extracted from the screen level
                temperature cube at spotdata sites which has then been adjusted
                using a temperature lapse rate calculated using multi-level
                temperature data.

        Examples:
            Lapse rate method based on potential temperature. Follows the work
            of S.B. Vosper 2005 - Near-surface temperature variations over
            complex terrain; Milestone Report RC10JR Local forecasting in
            complex terrain; V 1.0 August 2005.

            Calculate potential temperature gradient and use this to adjust
            temperatures to the spotdata site altitude. The method varies
            depending on whether the adjustment is an extrapolation to below
            the lowest available model level, or an interpolation between
            available levels.

            The essential equations are those converting between temperature
            and potential temperature (theta)::

                           pref   R/cp                 p_site  R/cp
              Theta = T ( ------ )         T = Theta ( ------ )
                          p_site                        pref

              ln (Theta) = ln (T)     + kappa [ ln(pref) - ln(p) ]
              ln (T)     = ln (Theta) + kappa [ ln(p) - ln(pref) ]

              kappa = (R_DRY_AIR / CP_DRY_AIR)
              pref = 1000 hPa (1.0E5 Pa)
              p_site = pressure interpolated/extrapolated to spotdata site
                       assuming a linear change in pressure with altitude; this
                       is assumption is reasonable if dz_max_adjustment is not
                       large (> several hundred metres).


            **Methodology**

            Use multi-level temperature data to calculate potential temperature
            gradients and use these to adjust extracted grid point temperatures
            to the altitudes of SpotData sites::


              ---upper_level--- Model level (k_upper)

              ---lower_level--- Model level (k_lower)

              --model_surface-- Model level (k=0)


            1. Calculate potential temperature gradient between lower and
               upper model levels.
            2. Compare the gradient with a defined threshold value that is
               used to indicate whether the gradient has been calculated
               across an inversion::

                 dtheta/dz <= threshold --> Keep value
                 dtheta/dz >  threshold --> Recalculate gradient between
                                            surface and lower_level to capture
                                            inversion.

            3. Determine if the SpotData site is below the lowest model level
               (the surface level, dz < 0). (This check is only at the
               neighbouring grid point, so doesn't actually guarantee we are
               below the model orography; combining this with neighbour finding
               with a below bias can ensure we are finding unresolved valleys/
               dips).


               **IF SpotData site height < model_surface --> Extrapolate
               downwards.**

               True surface below model surface (dz -ve) `Unresolved valley`::

                 ---upper_level--- Model level (k_upper)

                 ---lower_level--- Model level (k_lower)

                 --model_surface-- Model level (k=0)

                 ===site height=== SpotData site height

               4. Calculate pressure gradient between lower and upper model
                  levels.
               5. Use calculated gradients to extrapolate potential temperature
                  and pressure to the SpotData site height.
               6. Convert back to temperature using the equations given above.
               7. RETURN RESULT.

               **ELSE: SpotData site height > model_surface --> Interpolate to
               site height.**

               True surface above model surface (dz +ve) 'Unresolved hill'::

                 ---upper_level--- Model level (k_upper)

                 ===site height=== SpotData site height

                 ---lower_level--- Model level (k_lower)

                 --model_surface-- Model level (k=0)


               4. Use potential temperature gradient as an indicator of
                  atmospheric stability.

                  **IF: dtheta/dz > 0 --> Stable atmosphere**

                  5. Calculate pressure gradient between lower and upper model
                     levels.
                  6. Use calculated gradients to extrapolate potential
                     temperature and pressure to the SpotData site height.
                  7. Convert back to temperature using the equations given
                     above.
                  8. RETURN RESULT.

                  **ELSE: dtheta/dz <= 0 --> Neutral/well-mixed atmosphere**

                  5. Use potential temperature from surface level; for a well
                     mixed atmosphere dtheta/dz should be nearly constant.
                  6. Convert back to temperature using the pressure
                     interpolated to SpotData site height.
                  7. RETURN RESULT.
        """
        # Reference pressure of 1000hPa (1.0E5 Pa).
        p_ref = 1.0E5

        def _adjust_temperature(theta_base, dthetadz, dz, p_site, kappa):
            """
            Perform calculation of temperature at SpotData site.

            """
            if dz < 0 or dthetadz > 0:
                # Case for extrapolating downwards or a stable atmosphere.
                theta_site = theta_base + dz*dthetadz
                return theta_site*(p_site/p_ref)**kappa
            else:
                # Case for a well mixed atmosphere when calculating temperature
                # at a site above the neighbouring grid point.
                return theta_base*(p_site/p_ref)**kappa

        if not cube.name() == 'air_temperature':
            raise ValueError('{} should only be used for adjusting '
                             'temperatures. Cube of type {} is not '
                             'suitable.'.format(self.method, cube.name()))

        kappa = R_DRY_AIR/CP_DRY_AIR

        z_lower, = lower_pressure.coord('height').points
        z_upper, = upper_pressure.coord('height').points
        dz_model_levels = z_upper - z_lower

        data = np.empty(shape=(len(sites)), dtype=float)
        for i_site in range(len(sites)):
            i, j, dz = (neighbours['i'][i_site], neighbours['j'][i_site],
                        neighbours['dz'][i_site])

            # Use neighbour grid point value if dz < dz_tolerance.
            if abs(dz) < dz_tolerance:
                data[i_site] = cube.data[i, j]
                continue

            # Temperatures at surface, lower model level and upper model level.
            t_surface = cube.data[i, j]
            t_lower = lower_temperature.data[i, j]
            t_upper = upper_temperature.data[i, j]

            # Pressures at surface, lower model level and upper model level.
            p_surface = surface_pressure.data[i, j]
            p_lower = lower_pressure.data[i, j]
            p_upper = upper_pressure.data[i, j]

            # Potential temperature at surface, lower model level and upper
            # model level.
            theta_surface = t_surface*(p_ref/p_surface)**kappa
            theta_lower = t_lower*(p_ref/p_lower)**kappa
            theta_upper = t_upper*(p_ref/p_upper)**kappa

            # Enforce a maximum vertical displacement to which temperatures
            # will be adjusted using a lapse rate. This is to prevent excessive
            # changes based on what is unlikely to be a constant gradient
            # in potential temperature.
            dz = min(abs(dz), dz_max_adjustment)*np.sign(dz)

            # Calculate potential temperature gradient using levels away from
            # surface.
            dthetadz = (theta_upper - theta_lower)/dz_model_levels
            if dthetadz <= dthetadz_threshold:
                # If the potential temperature gradient is below the defined
                # dthetadz_threshold, use the non-surface levels.
                dz_from_model_level = dz - z_lower
                p_grad = (p_upper - p_lower)/dz_model_levels
                p_site = p_lower + p_grad*dz_from_model_level
                theta_base = theta_lower
            else:
                # A potential temperature gradient in excess of the threshold
                # value indicative of a lapse rate calculated across an
                # inversion. Recalulate using the surface and lower level to
                # better capture the inversion.
                dthetadz = (theta_lower-theta_surface)/z_lower
                dz_from_model_level = dz
                p_grad = (p_lower - p_surface)/z_lower
                p_site = p_surface + p_grad*dz_from_model_level
                theta_base = theta_surface

            data[i_site] = _adjust_temperature(
                theta_base, dthetadz, dz_from_model_level, p_site, kappa)

        return self.make_cube(cube, data, sites)
