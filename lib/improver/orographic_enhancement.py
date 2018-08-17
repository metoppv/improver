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
"""
This module contains a plugin to calculate the enhancement of precipitation
over orography.
"""

import numpy as np
from cf_units import Unit

import iris
from iris.analysis.cartography import rotate_winds

from improver.constants import R_WATER_VAPOUR
from improver.nbhood.square_kernel import SquareNeighbourhood
from improver.psychrometric_calculations.psychrometric_calculations \
    import WetBulbTemperature
from improver.utilities.cube_checker import check_for_x_and_y_axes
from improver.utilities.cube_manipulation import (
    compare_coords, set_increasing_spatial_coords)
from improver.utilities.spatial import (
    convert_number_of_grid_cells_into_distance,
    DifferenceBetweenAdjacentGridSquares)


class OrographicEnhancement(object):
    """
    Class to calculate orographic enhancement from horizontal wind components,
    temperature and relative humidity.
    """

    def __init__(self):
        """Initialise the plugin with thresholds from STEPS code"""
        self.orog_thresh_m = 20.
        self.rh_thresh_ratio = 0.8
        self.vgradz_thresh_ms = 0.0005
        self.upstream_range_of_influence_km = 15.
        self.efficiency_factor = 0.23265
        self.cloud_lifetime_s = 102.

    def __repr__(self):
        """Represent the plugin instance as a string"""
        msg = ('OrographicEnhancement() instance with orography threshold '
               '{} m, relative humidity threshold {}, v.gradz threshold {} m/s'
               ', maximum upstream influence {} km, upstream efficiency factor'
               ' {}, cloud lifetime {} s')
        return msg.format(self.orog_thresh_m, self.rh_thresh_ratio,
                          self.vgradz_thresh_ms,
                          self.upstream_range_of_influence_km,
                          self.efficiency_factor, self.cloud_lifetime_s)

    @staticmethod
    def _orography_gradients(topography):
        """
        Checks topography height is in same units as spatial dimensions, then
        calculates dimensionless gradient in both directions.

        Args:
            topography (iris.cube.Cube):
                Height of topography above sea level

        Returns:
            (tuple): tuple containing:
                **gradx**: iris.cube.Cube of dimensionless topography gradient
                    in the positive x direction
                **grady**: iris.cube.Cube of dimensionless topography gradient
                    in the positive y direction
        """
        topography.coord(axis='x').convert_units(topography.units)
        topography.coord(axis='y').convert_units(topography.units)

        gradx, grady = DifferenceBetweenAdjacentGridSquares(
            gradient=True).process(topography)

        gradx.units = '1'
        grady.units = '1'

        return gradx, grady

    def _regrid_and_populate(self, temperature, humidity, pressure,
                             uwind, vwind, topography):
        """
        Regrids input variables onto the high resolution orography field and
        calculates v.gradZ.  Populates class instance with the regridded
        variables.

        Args:
            temperature (iris.cube.Cube):
                Temperature at top of boundary layer
            humidity (iris.cube.Cube):
                Relative humidity at top of boundary layer
            pressure (iris.cube.Cube):
                Pressure at top of boundary layer
            uwind (iris.cube.Cube):
                Positive eastward wind vector component at top of boundary
                layer
            vwind (iris.cube.Cube):
                Positive northward wind vector component at top of boundary
                layer
            topography (iris.cube.Cube):
                Height of topography above sea level on 1 km UKPP domain grid
        """
        # set coordinates to be monotonically increasing
        # TODO check whether this has any impact once outputs are OK -
        #      if not, remove as it takes time
        for cube in [temperature, humidity, pressure,
                     uwind, vwind, topography]:
            set_increasing_spatial_coords(cube)

        # regrid variables to match the high resolution orography
        regridder = iris.analysis.Linear()
        self.temperature = temperature.regrid(topography, regridder)
        self.pressure = pressure.regrid(topography, regridder)
        self.humidity = humidity.regrid(topography, regridder)

        uwind, vwind = rotate_winds(uwind, vwind, topography.coord_system())
        self.uwind = uwind.regrid(topography, regridder)
        self.vwind = vwind.regrid(topography, regridder)

        # calculate orography gradients
        gradx, grady = self._orography_gradients(topography)

        # calculate v.gradZ
        self.vgradz = (np.multiply(gradx.data, self.uwind.data) +
                       np.multiply(grady.data, self.vwind.data))

    def _calculate_svp(self, method):
        """
        Calculates saturation vapour pressure using either the STEPS
        temperature polynomial approximation or the IMPROVER wet bulb
        temperature plugin functionality.

        Args:
            method (str):
                IMPROVER or STEPS.
        """
        if method == 'STEPS':
            print('Code the STEPS method')
        else:
            wbt = WetBulbTemperature()
            self.svp = wbt._pressure_correct_svp(
                wbt._lookup_svp(self.temperature),
                self.temperature, self.pressure)

    def _generate_mask(self, topography):
        """
        Generates a boolean mask of areas to calculate orographic enhancement.
        Criteria for calculation are:
            - 3x3 mean topography height >= threshold (20 m)
            - Relative humidity (fraction) >= threshold (0.8)
            - v dot grad z (wind x topography gradient) >= threshold (0.0005)

        Returns:
            mask (np.ndarray):
                Boolean mask - where True, set orographic enhancement to a
                default zero value
        """
        # calculate mean 3x3 (square nbhood) orography heights
        radius = convert_number_of_grid_cells_into_distance(topography, 1)
        topo_nbhood = SquareNeighbourhood().run(topography, radius)

        # create mask
        mask = np.full(topo_nbhood.shape, False, dtype=bool)
        mask = np.where(topo_nbhood.data < self.orog_thresh_m, True, mask)
        mask = np.where(self.humidity.data < self.rh_thresh_ratio, True, mask)
        mask = np.where(self.vgradz < self.vgradz_thresh_ms, True, mask)
        return mask

    def _site_orogenh(self):
        """
        Calculate precipitation enhancement over orography at each site using:

            orogenh = ((humidity * svp * vgradz) / 
                       (R_WATER_VAPOUR * temperature)) * 60 * 60

        Returns:
            site_orogenh (np.ndarray):
                Orographic enhancement values in mm/h
        """
        site_orogenh = np.zeros(self.temperature.data.shape, dtype=np.float32)

        prefactor = 3600./R_WATER_VAPOUR
        numerator = np.multiply(self.humidity.data, self.svp.data)
        numerator = np.multiply(numerator, self.vgradz)
        site_orogenh[~self.mask] = prefactor * np.divide(
            numerator[~self.mask], self.temperature.data[~self.mask])
        return np.where(site_orogenh > 0, site_orogenh, 0)

    def _add_upstream_component(self, site_orogenh, grid_spacing=1.):
        """
        Add upstream component to site orographic enhancement

        Args:
            site_orogenh (np.ndarray):
                Site orographic enhancement in mm h-1

        Kwargs:
            grid_spacing (int):
                Grid spacing of site_orogenh points in km

        Returns:
            orogenh (np.ndarray):
                Total orographic enhancement in mm h-1
        """
        # get wind speed and sin / cos direction wrt grid North
        wind_speed = np.sqrt(np.square(self.uwind.data) +
                             np.square(self.vwind.data))
        mask = np.where(np.isclose(wind_speed, 0), True, False)

        sin_wind_dir = np.zeros(wind_speed.shape, dtype=np.float32)
        sin_wind_dir[~mask] = np.divide(self.uwind.data[~mask],
                                        wind_speed[~mask])

        cos_wind_dir = np.zeros(wind_speed.shape, dtype=np.float32)
        cos_wind_dir[~mask] = np.divide(self.vwind.data[~mask],
                                        wind_speed[~mask])

        max_sin_cos = np.where(abs(sin_wind_dir) > abs(cos_wind_dir),
                               abs(sin_wind_dir), abs(cos_wind_dir))

        # calculate maximum upstream radius of influence at each grid cell
        upstream_roi = self.upstream_range_of_influence_km / grid_spacing
        max_roi = np.array((upstream_roi * max_sin_cos), dtype=int)

        # set standard deviation for Gaussian weighting function in grid
        # squares
        stddev = 0.001 * wind_speed * self.cloud_lifetime_s / grid_spacing
        variance = np.square(stddev)

        # generate 3d arrays to hold indices and weights of upstream components
        length = np.amax(max_roi)
        shape = (wind_speed.shape[0], wind_speed.shape[1], length)
        weights = [i / a if (i < b and not np.isclose(a, 0)) else np.nan
                   for (a, b) in zip(max_sin_cos.flatten(), max_roi.flatten())
                   for i in range(length)]
        weights = np.array(weights, dtype=np.float32).reshape(shape)

        # move the third axis to the front so that other arrays can be broadcast
        distance_weight = np.moveaxis(weights, -1, 0)

        # calculate positions of source points
        xpos, ypos = np.meshgrid(np.arange(wind_speed.shape[0]),
                                 np.arange(wind_speed.shape[1]))
        x_source = xpos - np.multiply(distance_weight,
                                      sin_wind_dir).astype(int)
        y_source = ypos - np.multiply(distance_weight,
                                      cos_wind_dir).astype(int)

        # force coordinates into bounds to avoid truncation at domain edges
        x_source = np.where(x_source < 0, 0, x_source)
        x_source = np.where(x_source > wind_speed.shape[1]-1,
                            wind_speed.shape[1]-1, x_source)

        y_source = np.where(y_source < 0, 0, y_source)
        y_source = np.where(y_source > wind_speed.shape[0]-1,
                            wind_speed.shape[0]-1, y_source)

        # extract values at source points
        source_values = [site_orogenh[y, x]
                         for (x, y) in zip(x_source.flatten(),
                                           y_source.flatten())]
        source_values = np.array(source_values).reshape(x_source.shape)

        # calculate weights of source points
        value_weight = np.where(
            np.isfinite(distance_weight),
            np.exp(np.divide(-0.5 * np.square(distance_weight), variance)), 0)
        sum_of_weights = np.sum(value_weight, axis=0)

        # multiply weights by values at source points
        weighted_values = np.multiply(source_values, value_weight)

        # sum along first axis and normalise
        total_orogenh = np.sum(weighted_values, axis=0)
        orogenh = self.efficiency_factor * np.divide(total_orogenh, sum_of_weights)

        return orogenh

    @staticmethod
    def _create_output_cubes(orogenh_data, high_res_cube, reference_cube):
        """
        Create two output cubes of orographic enhancement on different grids

        Args:
            orogenh_data (np.ndarray):
                Orographic enhancement value in mm h-1
            high_res_cube (iris.cube.Cube):
                Cube on the 1 km UKPP grid
            reference_cube (iris.cube.Cube):
                Cube with the correct time and forecast period coordinates on
                the UK 2 km standard grid

        Returns:
            (tuple): tuple containing:
                **orogenh** (iris.cube.Cube):
                    Orographic enhancement cube on 1 km UKPP grid
                **orogenh_standard_grid** (iris.cube.Cube):
                    Orographic enhancement cube on 2 km standard grid, padded
                    with masked np.nans
        """
        # create cube containing high resolution data in mm/h
        x_coord = high_res_cube.coord(axis='x')
        y_coord = high_res_cube.coord(axis='y')
        attributes = {'institution': 'Met Office', 'source': 'IMPROVER'}
        orogenh = iris.cube.Cube(
            orogenh_data, long_name="orographic_enhancement",
            units="mm h-1", attributes=attributes,
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        orogenh.add_aux_coord(reference_cube.coord('time'))
        orogenh.add_aux_coord(reference_cube.coord('forecast_reference_time'))
        orogenh.add_aux_coord(reference_cube.coord('forecast_period'))

        # regrid the orographic enhancement cube onto the standard grid and
        # mask extrapolated points
        orogenh_standard_grid = orogenh.regrid(
            reference_cube, iris.analysis.Linear(extrapolation_mode='mask'))

        return orogenh, orogenh_standard_grid

    def process(self, temperature, humidity, pressure, uwind, vwind,
                topography, svp_method='IMPROVER'):
        """
        Calculate precipitation enhancement over orography on standard (2 km)
        and high resolution (1 km UKPP domain) grids

        Args:
            temperature (iris.cube.Cube):
                Temperature at top of boundary layer
            humidity (iris.cube.Cube):
                Relative humidity at top of boundary layer
            pressure (iris.cube.Cube):
                Pressure at top of boundary layer
            uwind (iris.cube.Cube):
                Positive eastward wind vector component at top of boundary
                layer
            vwind (iris.cube.Cube):
                Positive northward wind vector component at top of boundary
                layer
            topography (iris.cube.Cube):
                Height of topography above sea level on 1 km UKPP domain grid

        Kwargs:
            svp_method (str):
                Which method to use to calculate the saturation vapour
                pressure (IMPROVER or STEPS).  Defaults to using the IMPROVER
                WetBulbTemperature plugin.

        Returns:
            (tuple): tuple containing:
                **orogenh** (iris.cube.Cube):
                    Precipitation enhancement due to orography in mm/h on the
                    1 km Transverse Mercator UKPP grid domain
                **orogenh_standard_grid** (iris.cube.Cube):
                    Precipitation enhancement due to orography in mm/h on the
                    2 km standard grid

        Reference:
            Alpert, P. and Shafir, H., 1989: Meso-Gamma-Scale Distribution of
            Orographic Precipitation: Numerical Study and Comparison with
            Precipitation Derived from Radar Measurements.  Journal of Applied
            Meteorology, 28, 1105-1117.
        """
        # check input variable cube coordinates match
        unmatched_coords = compare_coords(
            [temperature, pressure, humidity, uwind, vwind])

        if any(item.keys() for item in unmatched_coords):
            msg = 'Input cube coordinates {} are unmatched'
            raise ValueError(msg.format(unmatched_coords))

        # check all cubes are 2D spatial fields
        msg = 'Require 2D fields as input; found {} dimensions'
        if temperature.ndim > 2:
            raise ValueError(msg.format(temperature.ndim))
        check_for_x_and_y_axes(temperature)

        if topography.ndim > 2:
            raise ValueError(msg.format(topography.ndim))
        check_for_x_and_y_axes(topography)

        # convert input cube units
        # iris doesn't recognise 'mb' as a valid unit
        if pressure.units == Unit('mb'):
            pressure.units = Unit('hPa')
        pressure.convert_units('Pa')
        temperature.convert_units('kelvin')
        humidity.convert_units('1')
        uwind.convert_units('m s-1')
        vwind.convert_units('m s-1')
        topography.convert_units('m')

        # regrid variables to match topography and populate class instance
        self._regrid_and_populate(temperature, humidity, pressure,
                                  uwind, vwind, topography)

        # calculate saturation vapour pressure
        self._calculate_svp(method=svp_method)

        # generate mask defining where to calculate orographic enhancement
        self.mask = self._generate_mask(topography)

        # calculate site-specific orographic enhancement
        site_orogenh_data = self._site_orogenh()

        # integrate upstream component
        orogenh_data = self._add_upstream_component(site_orogenh_data)

        # create data cubes on the two required output grids
        orogenh, orogenh_standard_grid = self._create_output_cubes(
            orogenh_data, topography, temperature)

        return orogenh, orogenh_standard_grid
