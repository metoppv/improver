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
"""
This module contains a plugin to calculate the enhancement of precipitation
over orography.
"""

import iris
import numpy as np
from iris.analysis.cartography import rotate_winds
from scipy.ndimage import uniform_filter1d

from improver import BasePlugin
from improver.constants import R_WATER_VAPOUR
from improver.metadata.constants.mo_attributes import MOSG_GRID_ATTRIBUTES
from improver.metadata.utilities import generate_mandatory_attributes
from improver.nbhood.nbhood import NeighbourhoodProcessing
from improver.psychrometric_calculations.psychrometric_calculations \
    import calculate_svp_in_air
from improver.utilities.cube_checker import check_for_x_and_y_axes
from improver.utilities.cube_manipulation import (
    compare_coords, enforce_coordinate_ordering, sort_coord_in_cube)
from improver.utilities.spatial import (
    DifferenceBetweenAdjacentGridSquares,
    number_of_grid_cells_to_distance)


class OrographicEnhancement(BasePlugin):
    """
    Class to calculate orographic enhancement from horizontal wind components,
    temperature and relative humidity.

    References:
        Alpert, P. and Shafir, H., 1989: Meso-Gamma-Scale Distribution of
            Orographic Precipitation: Numerical Study and Comparison with
            Precipitation Derived from Radar Measurements.  Journal of
            Applied Meteorology, 28, 1105-1117.
        Roe, G., 2005: Orographic Precipitation.  Annual Review of Earth
            and Planetary Sciences, 33, 645-671.
    """

    def __init__(self):
        """
        Initialise the plugin with thresholds from STEPS code.  Usage as
        follows:

            Criteria for site orographic enhancement calculation:
                - 3x3 mean topography height >= self.orog_thresh_m (20 m)
                - Relative humidity (fraction) >= self.rh_thresh_ratio (0.8)
                - v dot grad z (wind x topography gradient) >=
                  self.vgradz_thresh_ms (0.0005 m/s)

            Parameters for calculating upstream contribution:
                - Maximum range of an upstream cell to contribute to the total
                  enhancement (self.upstream_range_of_influence_km).  This is
                  15 km in STEPS.
                - Cloud lifetime (self.cloud_lifetime_s) defines the standard
                  deviation of the distance weighting function for upstream
                  enhancement contributions.  This is 102 seconds in STEPS.
                - Scaling factor by which to multiply the weighted sum of
                  upstream contributions (self.efficiency_factor).  This is
                  0.23265 in STEPS.

        Create placeholder class members for regridded variable cubes
        (orography, temperature, humidity, pressure and wind components),
        saturation vapour pressure, V.gradZ (uplift) array and grid spacing.
        """
        self.orog_thresh_m = 20.
        self.rh_thresh_ratio = 0.8
        self.vgradz_thresh_ms = 0.0005

        self.upstream_range_of_influence_km = 15.
        self.cloud_lifetime_s = 102.
        self.efficiency_factor = 0.23265

        # initialise class members to store regridded variables for
        # orographic enhancement calculation
        self.topography = None
        self.temperature = None
        self.humidity = None
        self.pressure = None
        self.uwind = None
        self.vwind = None

        # initialise class members for derived variables and metadata
        self.vgradz = None
        self.svp = None
        self.grid_spacing_km = None

    def __repr__(self):
        """Represent the plugin instance as a string"""
        return '<OrographicEnhancement()>'

    def _orography_gradients(self):
        """
        Calculates the dimensionless gradient of self.topography along both
        spatial axes, smoothed along the perpendicular axis.  If spatial
        coordinates are not in the same units as topography height (m),
        converts coordinate units in place.

        Returns:
            (tuple): tuple containing:
                **gradx** (iris.cube.Cube):
                    2D cube of dimensionless topography gradients in the
                    positive x direction
                **grady** (iris.cube.Cube):
                    2D cube of dimensionless topography gradients in the
                    positive y direction
        """
        self.topography.coord(axis='x').convert_units(self.topography.units)
        xdim = self.topography.coord_dims(self.topography.coord(axis='x'))[0]
        self.topography.coord(axis='y').convert_units(self.topography.units)
        ydim = self.topography.coord_dims(self.topography.coord(axis='y'))[0]

        # smooth topography by +/- one grid cell along the perpendicular axis
        # before calculating each gradient (as done in STEPS)
        topo_smx = uniform_filter1d(self.topography.data, 3, axis=ydim)
        topo_smx_cube = self.topography.copy(data=topo_smx)
        gradx, _ = DifferenceBetweenAdjacentGridSquares(
            gradient=True).process(topo_smx_cube)
        gradx.units = '1'

        topo_smy = uniform_filter1d(self.topography.data, 3, axis=xdim)
        topo_smy_cube = self.topography.copy(data=topo_smy)
        _, grady = DifferenceBetweenAdjacentGridSquares(
            gradient=True).process(topo_smy_cube)
        grady.units = '1'

        return gradx, grady

    def _regrid_variable(self, var_cube, unit):
        """
        Sorts spatial coordinates in ascending order, regrids the input
        variable onto the topography grid and converts to the required
        units.  This function does not modify the input variable cube.

        Args:
            var_cube (iris.cube.Cube):
                Cube containing input variable data
            unit (str):
                Required unit for this variable

        Returns:
            iris.cube.Cube:
                Cube containing regridded variable data
        """
        for axis in ['x', 'y']:
            var_cube = sort_coord_in_cube(
                var_cube, var_cube.coord(axis=axis))

        enforce_coordinate_ordering(
            var_cube, [var_cube.coord(axis='y').name(),
                       var_cube.coord(axis='x').name()])

        regridder = iris.analysis.Linear()
        out_cube = var_cube.regrid(self.topography, regridder)
        out_cube.data = out_cube.data.astype(np.float32)
        out_cube.convert_units(unit)
        return out_cube

    def _regrid_and_populate(self, temperature, humidity, pressure,
                             uwind, vwind, topography):
        """
        Regrids input variables onto the high resolution orography field, then
        populates the class instance with regridded variables before converting
        to SI units.  Also calculates V.gradZ as a class member.

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
        # convert topography grid, datatype and units
        for axis in ['x', 'y']:
            topography = sort_coord_in_cube(
                topography, topography.coord(axis=axis))
        enforce_coordinate_ordering(
            topography, [topography.coord(axis='y').name(),
                         topography.coord(axis='x').name()])
        self.topography = topography.copy(
            data=topography.data.astype(np.float32))
        self.topography.convert_units('m')

        # rotate winds
        try:
            uwind, vwind = rotate_winds(
                uwind, vwind, topography.coord_system())
        except ValueError as err:
            if 'Duplicate coordinates are not permitted' in str(err):
                # ignore error raised if uwind and vwind do not need rotating
                pass
            else:
                raise ValueError(str(err))
        else:
            # remove auxiliary spatial coordinates from rotated winds
            for cube in [uwind, vwind]:
                for axis in ['x', 'y']:
                    cube.remove_coord(cube.coord(axis=axis, dim_coords=False))

        # regrid and convert input variables
        self.temperature = self._regrid_variable(temperature, 'kelvin')
        self.humidity = self._regrid_variable(humidity, '1')
        self.pressure = self._regrid_variable(pressure, 'Pa')
        self.uwind = self._regrid_variable(uwind, 'm s-1')
        self.vwind = self._regrid_variable(vwind, 'm s-1')

        # calculate orography gradients
        gradx, grady = self._orography_gradients()

        # calculate v.gradZ
        self.vgradz = (np.multiply(gradx.data, self.uwind.data) +
                       np.multiply(grady.data, self.vwind.data))

    def _generate_mask(self):
        """
        Generates a boolean mask of areas NOT to calculate orographic
        enhancement.  Criteria for calculating orographic enhancement are that
        all of the following are true:

            - 3x3 mean topography height >= threshold (20 m)
            - Relative humidity (fraction) >= threshold (0.8)
            - v dot grad z (wind x topography gradient) >= threshold (0.0005)

        The mask is therefore "True" if any of these conditions are false.

        Returns:
            numpy.ndarray:
                Boolean mask - where True, set orographic enhancement to a
                default zero value
        """
        # calculate mean 3x3 (square nbhood) orography heights
        radius = number_of_grid_cells_to_distance(self.topography, 1)
        topo_nbhood = NeighbourhoodProcessing(
            'square', radius).process(self.topography)
        topo_nbhood.convert_units('m')

        # create mask
        mask = np.full(topo_nbhood.shape, False, dtype=bool)
        mask = np.where(topo_nbhood.data < self.orog_thresh_m, True, mask)
        mask = np.where(self.humidity.data < self.rh_thresh_ratio, True, mask)
        mask = np.where(abs(self.vgradz) < self.vgradz_thresh_ms, True, mask)
        return mask

    def _point_orogenh(self):
        """
        Calculate the grid-point precipitation enhancement contribution due to
        orographic uplift using:

            orogenh = ((humidity * svp * vgradz) /
                       (R_WATER_VAPOUR * temperature)) * 60 * 60

        Returns:
            numpy.ndarray:
                Orographic enhancement values in mm/h
        """
        mask = np.logical_not(self._generate_mask())
        point_orogenh = np.zeros(self.temperature.data.shape, dtype=np.float32)

        prefactor = 3600./R_WATER_VAPOUR
        numerator = np.multiply(self.humidity.data, self.svp)
        numerator = np.multiply(numerator, self.vgradz)
        point_orogenh[mask] = prefactor * np.divide(
            numerator[mask], self.temperature.data[mask])
        return np.where(point_orogenh > 0, point_orogenh, 0)

    def _get_point_distances(self, wind_speed, max_sin_cos):
        """
        Generate 3d array of distances to upstream components

        Args:
            wind_speed (numpy.ndarray):
                2D array of wind speeds
            max_roi (numpy.ndarray):
                2D array of maximum ranges of influence in grid squares
            max_sin_cos (numpy.ndarray):
                2D array containing the larger of sin(wind_direction) or
                cos(wind_direction) with respect to grid north

        Returns:
            numpy.ndarray:
                3D array of source-to-destination distances in grid points,
                with np.nan filled in for out of range values
        """
        # calculate maximum upstream radius of influence at each grid cell
        upstream_roi = (
            self.upstream_range_of_influence_km / self.grid_spacing_km)
        max_roi = (upstream_roi * max_sin_cos).astype(int)

        length = np.amax(max_roi)
        shape = (length, wind_speed.shape[0], wind_speed.shape[1])
        distance = np.full(shape, np.nan, dtype=np.float32)
        for y in range(distance.shape[1]):
            for x in range(distance.shape[2]):
                distance[:max_roi[y, x], y, x] = (
                    np.arange(max_roi[y, x]) / max_sin_cos[y, x])

        return distance

    @staticmethod
    def _locate_source_points(
            wind_speed, distance, sin_wind_dir, cos_wind_dir):
        """
        Generate 3D arrays of source points from which to add upstream
        orographic enhancement contribution.  Assumes spatial coordinate
        ordering [y, x].

        Args:
            wind_speed (numpy.ndarray):
                2D array of wind speed magnitudes
            distance (numpy.ndarray):
                3D array of grid point source-to-destination distances
            sin_wind_dir (numpy.ndarray):
                2D array of sin wind direction wrt grid north
            cos_wind_dir (numpy.ndarray):
                2D array of cos wind direction wrt grid north

        Returns:
            (tuple): tuple containing:
                **x_source** (numpy.ndarray):
                    3D array of source point x-coordinates
                **y_source** (numpy.ndarray):
                    3D array of source point y-coordinates
        """
        xpos, ypos = np.meshgrid(np.arange(wind_speed.shape[1]),
                                 np.arange(wind_speed.shape[0]))
        x_source = np.around(xpos - np.multiply(distance,
                                                sin_wind_dir)).astype(int)
        y_source = np.around(ypos - np.multiply(distance,
                                                cos_wind_dir)).astype(int)

        # force coordinates into bounds to avoid truncation at domain edges
        x_source = np.where(x_source < 0, 0, x_source)
        x_source = np.where(x_source > wind_speed.shape[1]-1,
                            wind_speed.shape[1]-1, x_source)

        y_source = np.where(y_source < 0, 0, y_source)
        y_source = np.where(y_source > wind_speed.shape[0]-1,
                            wind_speed.shape[0]-1, y_source)

        return x_source, y_source

    def _compute_weighted_values(self, point_orogenh, x_source, y_source,
                                 distance, wind_speed):
        """
        Extract orographic enhancement values from source points and weight
        according to source-destination distance.

        Args:
            point_orogenh (numpy.ndarray):
                2D array of point orographic enhancement values
            x_source (numpy.ndarray):
                3D array of x-coordinates of source points from which to read
                upstream contribution
            y_source (numpy.ndarray):
                3D array of y-coordinates of source points from which to read
                upstream contribution
            distance:
                3D array of grid point source-to-destination distances
            wind_speed:
                2D array of wind speeds

        Returns:
            (tuple): tuple containing:
                **orogenh** (numpy.ndarray):
                    2D array containing a weighted sum of orographic
                    enhancement components from upstream source points
                **sum_of_weights** (numpy.ndarray):
                    2D array containing weights for normalisation
        """
        source_values = np.fromiter(
            (point_orogenh[y, x] for (x, y) in zip(x_source.flatten(),
                                                   y_source.flatten())),
            np.float32, count=x_source.size).reshape(x_source.shape)

        # set standard deviation for Gaussian weighting function in grid
        # squares
        grid_spacing_m = 1000.*self.grid_spacing_km
        stddev = wind_speed * self.cloud_lifetime_s / grid_spacing_m
        variance = np.square(stddev)

        # calculate weighted values at source points
        value_weight = np.where(
            (np.isfinite(distance)) & (variance > 0),
            np.exp(np.divide(-0.5 * np.square(distance), variance)), 0)
        sum_of_weights = np.sum(value_weight, axis=0)
        weighted_values = np.multiply(source_values, value_weight)

        return np.sum(weighted_values, axis=0), sum_of_weights

    def _add_upstream_component(self, point_orogenh):
        """
        Add upstream component to site orographic enhancement

        Args:
            point_orogenh (numpy.ndarray):
                Site orographic enhancement in mm h-1

        Returns:
            numpy.ndarray:
                Total orographic enhancement in mm h-1
        """
        # get wind speed and sin / cos direction wrt grid North
        wind_speed = np.sqrt(np.square(self.uwind.data) +
                             np.square(self.vwind.data))
        mask = np.where(np.isclose(wind_speed, 0), True, False)
        mask = np.logical_not(mask)

        sin_wind_dir = np.zeros(wind_speed.shape, dtype=np.float32)
        sin_wind_dir[mask] = np.divide(self.uwind.data[mask],
                                       wind_speed[mask])

        cos_wind_dir = np.zeros(wind_speed.shape, dtype=np.float32)
        cos_wind_dir[mask] = np.divide(self.vwind.data[mask],
                                       wind_speed[mask])

        max_sin_cos = np.where(abs(sin_wind_dir) > abs(cos_wind_dir),
                               abs(sin_wind_dir), abs(cos_wind_dir))

        # generate 3D array of distances to source points
        distance = self._get_point_distances(wind_speed, max_sin_cos)

        # calculate positions of source points
        x_source, y_source = self._locate_source_points(
            wind_speed, distance, sin_wind_dir, cos_wind_dir)

        # compute weighted enhancements summed over all source points
        orogenh, sum_of_weights = self._compute_weighted_values(
            point_orogenh, x_source, y_source, distance, wind_speed)

        # normalise by weights and scale by efficiency factor
        orogenh[mask] = self.efficiency_factor * np.divide(
            orogenh[mask], sum_of_weights[mask])

        return orogenh

    def _create_output_cube(self, orogenh_data, reference_cube):
        """Creates a cube containing orographic enhancement values in SI units.

        Args:
            orogenh_data (numpy.ndarray):
                Orographic enhancement value in mm h-1
            reference_cube (iris.cube.Cube):
                Cube with the correct time and forecast period coordinates on
                the UK standard grid

        Returns:
            iris.cube.Cube:
                Orographic enhancement cube (m s-1)
        """
        # create cube containing high resolution data in mm/h
        x_coord = self.topography.coord(axis='x')
        y_coord = self.topography.coord(axis='y')
        for coord in [x_coord, y_coord]:
            coord.points = coord.points.astype(np.float32)
            if coord.bounds is not None:
                coord.bounds = coord.bounds.astype(np.float32)

        aux_coords = []
        for coord in ['time', 'forecast_reference_time', 'forecast_period']:
            aux_coords.append((reference_cube.coord(coord), None))

        attributes = generate_mandatory_attributes([reference_cube])
        attributes["title"] = "unknown"  # remove possible wrong grid info.
        for key in MOSG_GRID_ATTRIBUTES:
            try:
                attributes[key] = self.topography.attributes[key]
            except KeyError:
                pass

        orog_enhance_cube = iris.cube.Cube(
            orogenh_data, long_name="orographic_enhancement",
            units="mm h-1", attributes=attributes,
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
            aux_coords_and_dims=aux_coords)
        orog_enhance_cube.convert_units("m s-1")

        return orog_enhance_cube

    def process(self, temperature, humidity, pressure, uwind, vwind,
                topography):
        """
        Calculate precipitation enhancement over orography on high resolution
        grid. Input diagnostics are all expected to be on the same grid, and
        are regridded to match the orography.

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
                Height of topography above sea level on high resolution (1 km)
                UKPP domain grid

        Returns:
            iris.cube.Cube:
                Precipitation enhancement due to orography in m/s.
        """
        # check input variable cube coordinates match
        unmatched_coords = compare_coords(
            [temperature, pressure, humidity, uwind, vwind])

        if any(item.keys() for item in unmatched_coords):
            msg = 'Input cube coordinates {} are unmatched'
            raise ValueError(msg.format(unmatched_coords))

        # check one of the input variable cubes is a 2D spatial field (this is
        # equivalent to checking all cubes whose coords are matched above)
        msg = 'Require 2D fields as input; found {} dimensions'
        if temperature.ndim > 2:
            raise ValueError(msg.format(temperature.ndim))
        check_for_x_and_y_axes(temperature)

        # check the topography cube is a 2D spatial field
        if topography.ndim > 2:
            raise ValueError(msg.format(topography.ndim))
        check_for_x_and_y_axes(topography)

        # regrid variables to match topography and populate class instance
        self._regrid_and_populate(temperature, humidity, pressure,
                                  uwind, vwind, topography)

        # calculate saturation vapour pressure
        self.svp = calculate_svp_in_air(
            self.temperature.data, self.pressure.data)

        # calculate site-specific orographic enhancement
        point_orogenh_data = self._point_orogenh()

        # integrate upstream component
        grid_coord_km = self.topography.coord(axis='x').copy()
        grid_coord_km.convert_units('km')
        self.grid_spacing_km = (
            grid_coord_km.points[1] - grid_coord_km.points[0])

        orogenh_data = self._add_upstream_component(point_orogenh_data)

        # create data cubes on the two required output grids
        orogenh = self._create_output_cube(orogenh_data, temperature)

        return orogenh
