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
"""Unit tests for the spotdata.ExtractData plugin."""


import unittest
import cf_units

from datetime import datetime as dt
from iris.coords import (DimCoord,
                         AuxCoord)
from iris import coord_systems
from iris.coord_systems import GeogCS
from iris.cube import (Cube,
                       CubeList)
from iris import Constraint
from iris.tests import IrisTest
from iris.time import PartialDateTime
import cartopy.crs as ccrs
from collections import OrderedDict
from iris import FUTURE
import numpy as np

from improver.spotdata.extract_data import ExtractData

FUTURE.cell_datetime_objects = True

class TestExtractData(IrisTest):

    """Test the extract data plugin."""

    def setUp(self):
        """
        Create a cube containing a regular lat-lon grid.

        Data is formatted to increase linearly in x/y dimensions,
        e.g.
              0 1 2 3
              1 2 3 4
              2 3 4 5
              3 4 5 6

        """
        data = np.arange(0, 20, 1)
        for i in range(1, 20):
            data = np.append(data, np.arange(i, 20+i))

        data.resize(1, 20, 20)
        latitudes = np.linspace(-90, 90, 20)
        longitudes = np.linspace(-180, 180, 20)
        latitude = DimCoord(latitudes, standard_name='latitude',
                            units='degrees', coord_system=GeogCS(6371229.0))
        longitude = DimCoord(longitudes, standard_name='longitude',
                             units='degrees', coord_system=GeogCS(6371229.0))

        # Use time of 2017-02-17 06:00:00
        time = DimCoord(
            [1487311200], standard_name='time',
            units=cf_units.Unit('seconds since 1970-01-01 00:00:00',
                                calendar='gregorian'))

        time_dt = [dt(2017, 02, 17, 06, 00)]
        # time_extract = Constraint(time=PartialDateTime(2017, 02, 17, 06, 00))

        cube = Cube(data,
                    long_name="test_data",
                    dim_coords_and_dims=[(time, 0),
                                         (latitude, 1),
                                         (longitude, 2)],
                    units="1")

        cubes = CubeList()
        cubes.append(cube)

        orography = Cube(np.ones((20, 20)),
                         long_name="surface_altitude",
                         dim_coords_and_dims=[(latitude, 0),
                                              (longitude, 1)],
                         units="m")

        # Western half of grid at altitude 0, eastern half at 10.
        # Note that the pressure_on_height_levels data is left unchanged,
        # so it is as if there is a sharp front running up the grid with
        # differing pressures on either side at equivalent heights above
        # the surface (e.g. east 1000hPa at 0m AMSL, west 1000hPa at 10m AMSL).
        # So there is higher pressure in the west.
        orography.data[0:10] = 0
        orography.data[10:] = 10
        ancillary_data = {}
        ancillary_data.update({'orography': orography})

        # Create additional vertical data used to calculate temperature lapse
        # rates from model levels.

        t_level0 = np.ones((1, 20, 20))*20.
        t_level1 = np.ones((1, 20, 20))*-20.
        t_level2 = np.ones((1, 20, 20))*-60.
        t_data = np.vstack((t_level0, t_level1, t_level2))
        t_data.resize((1, 3, 20, 20))

        p_level0 = np.ones((1, 20, 20))*1000.
        p_level1 = np.ones((1, 20, 20))*900.
        p_level2 = np.ones((1, 20, 20))*800.
        p_data = np.vstack((p_level0, p_level1, p_level2))
        p_data.resize((1, 3, 20, 20))

        height = DimCoord([0., 50., 100.], standard_name='height', units='m')

        temperature_on_height_levels = CubeList()
        temperature_on_height_levels.append(
            Cube(
                t_data,
                long_name="temperature_on_height_levels",
                dim_coords_and_dims=[(time, 0), (height, 1),
                                     (latitude, 2), (longitude, 3)],
                units="degree_Celsius"))

        pressure_on_height_levels = CubeList()
        pressure_on_height_levels.append(
            Cube(
                p_data,
                long_name="pressure_on_height_levels",
                dim_coords_and_dims=[(time, 0), (height, 1),
                                     (latitude, 2), (longitude, 3)],
                units="hPa"))

        surface_pressure = CubeList()
        surface_pressure.append(
            Cube(
                p_data[0, 0].reshape(1, 20, 20),
                long_name="surface_pressure",
                dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)],
                units="hPa"))

        ad = {'temperature_on_height_levels': temperature_on_height_levels,
              'pressure_on_height_levels': pressure_on_height_levels,
              'surface_pressure': surface_pressure}

        sites = OrderedDict()
        sites.update({'100':
                          {'latitude': 4.74,
                           'longitude': 9.47,
                           'altitude': 10,
                           'gmtoffset': 0
                           }
                      })

        neighbour_list = np.empty(1, dtype=[('i', 'i8'),
                                            ('j', 'i8'),
                                            ('dz', 'f8'),
                                            ('edge', 'bool_')])

        neighbour_list[0] = 10, 10, 0, False

        self.cubes = cubes
        self.ancillary_data = ancillary_data
        self.ad = ad
        self.sites = sites
        self.neighbour_list = neighbour_list
        self.time_dt = time_dt


    def return_type(self, method, additional_data, **kwargs):
        """Test that the plugin returns an iris.cube.CubeList."""
        plugin = ExtractData(method)
        result = plugin.process(self.cubes, self.sites, self.neighbour_list,
                                self.time_dt, additional_data, **kwargs)

        self.assertIsInstance(result, CubeList)

    def extracted_value(self, method, additional_data, expected, **kwargs):
        """Test that the plugin returns the correct value."""
        plugin = ExtractData(method)
        result = plugin.process(self.cubes, self.sites, self.neighbour_list,
                                self.time_dt, additional_data, **kwargs)
        self.assertAlmostEqual(result[0].data, expected)

    def different_projection(self, method, additional_data, expected, **kwargs):
        """Test that the plugin copes with non-lat/lon grids."""

        trg_crs = None
        src_crs = ccrs.PlateCarree()
        trg_crs = ccrs.LambertConformal(central_longitude=50,
                                        central_latitude=10)
        trg_crs_iris = coord_systems.LambertConformal(
            central_lon=50, central_lat=10)
        lons = self.cubes[0].coord('longitude').points
        lats = self.cubes[0].coord('latitude').points
        x, y = [], []
        for lon, lat in zip(lons, lats):
            x_trg, y_trg = trg_crs.transform_point(lon, lat, src_crs)
            x.append(x_trg)
            y.append(y_trg)

        new_x = AuxCoord(x, standard_name='projection_x_coordinate',
                         units='m', coord_system=trg_crs_iris)
        new_y = AuxCoord(y, standard_name='projection_y_coordinate',
                         units='m', coord_system=trg_crs_iris)

        cube = Cube(self.cubes[0].data,
                    long_name="test_data",
                    dim_coords_and_dims=[(self.cubes[0].coord('time'), 0)],
                    aux_coords_and_dims=[(new_y, 1), (new_x, 2)],
                    units="1")

        cubes = CubeList()
        cubes.append(cube)

        plugin = ExtractData(method)
        result = plugin.process(cubes, self.sites, self.neighbour_list,
                                self.time_dt, additional_data, **kwargs)

        self.assertEqual(cubes[0].coord_system(), trg_crs_iris)
        self.assertAlmostEqual(result[0].data, expected)
        self.assertEqual(result[0].coord(axis='y').name(), 'latitude')
        self.assertEqual(result[0].coord(axis='x').name(), 'longitude')
        self.assertAlmostEqual(result[0].coord(axis='y').points, 4.74)
        self.assertAlmostEqual(result[0].coord(axis='x').points, 9.47)

    def missing_ancillary_data(self, method, additional_data, **kwargs):
        """Test that the plugin copes with missing ancillary data."""
        plugin = ExtractData(method)
        msg = "Ancillary data"
        with self.assertRaisesRegexp(Exception, msg):
            result = plugin.process(
                self.cubes, self.sites, self.neighbour_list,
                self.time_dt, additional_data, **kwargs)

    def missing_additional_data(self, method, additional_data, **kwargs):
        """Test that the plugin copes with missing additional data."""
        plugin = ExtractData(method)
        msg = "Required additional data is unset"
        with self.assertRaisesRegexp(Exception, msg):
            result = plugin.process(
                self.cubes, self.sites, self.neighbour_list,
                self.time_dt, additional_data, **kwargs)


class use_nearest(TestExtractData):

    method = 'use_nearest'

    def test_return_type(self):
        self.return_type(self.method, None, ancillary_data=None)

    def test_extracted_value(self):
        """Test that the plugin returns the correct value."""
        expected = 20
        self.extracted_value(self.method, None, expected, ancillary_data=None)

    def test_different_projection(self):
        """Test that the plugin copes with non-lat/lon grids."""
        expected = 20.
        self.different_projection(self.method, None, expected,
                                  ancillary_data=None)


class orography_derived_temperature_lapse_rate(TestExtractData):

    method = 'orography_derived_temperature_lapse_rate'

    def test_return_type(self):
        self.return_type(self.method, None, ancillary_data=self.ancillary_data)

    def test_extracted_value(self):
        """
        Test that the plugin returns the correct value.

        Fit line given data above is: T = 0.15*altitude + 19
        Site defined with has altitude=10, so T+expected = 20.5.

        """
        expected = 20.5
        self.extracted_value(self.method, None, expected,
                             ancillary_data=self.ancillary_data)

    def test_different_projection(self):
        """
        Test that the plugin copes with non-lat/lon grids.

        Cube is transformed into a LambertConformal projection. The usual
        latitude/longitude coordinates are used to query the grid, with iris
        functionality used to convert the query coordinates to the correct
        projection.

        The returned cube has latitude/longitude dimensions.

        The expected value should be the same as the PlateCarree() projection
        case above.

        """
        expected = 20.5
        self.different_projection(self.method, None, expected,
                                  ancillary_data=self.ancillary_data)

    def test_missing_ancillary_data(self):
        self.missing_ancillary_data(self.method, None, ancillary_data=None)


class model_level_temperature_lapse_rate(TestExtractData):

    method = 'model_level_temperature_lapse_rate'

    def test_return_type(self):
        self.return_type(self.method, self.ad,
                         ancillary_data=self.ancillary_data)

    def test_extracted_value(self):
        """
        Test that the plugin returns the correct value.

        Site set to be 60m in altitude, which is a dz of +50m from the nearest
        grid point (its neighbour). As such it should fall on the 900hPa level
        and get a temperature of -20C.

        """
        self.sites['100']['altitude'] = 60.
        self.neighbour_list['dz'] = 50.
        expected = -20.
        self.extracted_value(self.method, self.ad, expected,
                             ancillary_data=self.ancillary_data)

    def test_different_projection(self):
        """
        Test that the plugin copes with non-lat/lon grids.

        Cube is transformed into a LambertConformal projection. The usual
        latitude/longitude coordinates are used to query the grid, with iris
        functionality used to convert the query coordinates to the correct
        projection.

        The returned cube has latitude/longitude dimensions.

        The expected value should be the same as the PlateCarree() projection
        case above.

        """
        self.sites['100']['altitude'] = 60.
        self.neighbour_list['dz'] = 50.
        expected = -20.
        plugin = ExtractData(self.method)
        self.different_projection(self.method, self.ad, expected,
                                  ancillary_data=self.ancillary_data)

    def test_missing_additional_data(self):
        plugin = ExtractData(self.method)
        self.missing_additional_data(self.method, None,
                                     ancillary_data=self.ancillary_data)


if __name__ == '__main__':
    unittest.main()
