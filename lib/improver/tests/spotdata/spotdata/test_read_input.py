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
"""Unit tests for the spotdata.read_data"""


import unittest
import numpy as np
import cf_units
import json
from datetime import datetime as dt
from subprocess import call as Call
from tempfile import mkdtemp
import iris
from iris import Constraint
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
from iris.time import PartialDateTime
from iris.exceptions import ConstraintMismatchError

from improver.spotdata.read_input import Load as Plugin
from improver.spotdata.read_input import get_method_prerequisites
from improver.spotdata.read_input import get_additional_diagnostics
from improver.spotdata.read_input import data_from_dictionary
from improver.spotdata.read_input import read_config


class Test_read_input(IrisTest):
    """Test the reading of ancillary data files and creation of an ancillaries
    dictionary."""

    def setUp(self):
        """Create a cube containing a regular lat-lon grid and other necessary
        ingredients for unit tests."""

        data = np.arange(0, 800, 1)
        data.resize(2, 20, 20)
        latitudes = np.linspace(-90, 90, 20)
        longitudes = np.linspace(-180, 180, 20)
        latitude = DimCoord(latitudes, standard_name='latitude',
                            units='degrees', var_name='latitude')
        longitude = DimCoord(longitudes, standard_name='longitude',
                             units='degrees', var_name='longitude')

        # Use time of 2017-02-17 06:00:00, 07:00:00
        time = DimCoord(
            [1487311200, 1487314800], standard_name='time',
            units=cf_units.Unit('seconds since 1970-01-01 00:00:00',
                                calendar='gregorian'), var_name='time')

        time_dt = dt(2017, 2, 17, 6, 0)
        time_extract = Constraint(time=PartialDateTime(
            time_dt.year, time_dt.month, time_dt.day, time_dt.hour))

        cube = Cube(data,
                    long_name="air_temperature",
                    dim_coords_and_dims=[(time, 0),
                                         (latitude, 1),
                                         (longitude, 2)],
                    units="K")
        cube2 = cube.copy()

        orography = Cube(np.ones((20, 20)),
                         long_name="surface_altitude",
                         dim_coords_and_dims=[(latitude, 0),
                                              (longitude, 1)],
                         units="m")

        land = orography.copy()
        land.rename('land_binary_mask')
        land.data = land.data + 1

        ancillary_data = {}
        ancillary_data.update({'orography': orography})
        ancillary_data.update({'land_mask': land})

        # Copies of cube simply renamed to be read as additional data.
        temperature_on_height_levels = cube.copy()
        temperature_on_height_levels.rename('temperature_on_height_levels')
        pressure_on_height_levels = cube.copy()
        pressure_on_height_levels.rename('pressure_on_height_levels')
        surface_pressure = cube.copy()
        surface_pressure.rename('surface_pressure')

        # Build reference copy of additional_data dictionary.
        with iris.FUTURE.context(cell_datetime_objects=True):
            additional_data = {
                'temperature_on_height_levels': CubeList(
                    [temperature_on_height_levels]),
                'pressure_on_height_levels': CubeList([
                    pressure_on_height_levels]),
                'surface_pressure': CubeList([surface_pressure])
                }

        self.data_directory = mkdtemp()

        self.cube_path = (self.data_directory +
                          '/01-temperature_at_screen_level.nc')
        self.cube_path2 = (self.data_directory +
                           '/02-temperature_at_screen_level.nc')
        orography_path = self.data_directory + '/orography.nc'
        land_path = self.data_directory + '/land_mask.nc'
        ad_path_temperature = (self.data_directory +
                               '/temperature_on_height_levels.nc')
        ad_path_pressure = (self.data_directory +
                            '/pressure_on_height_levels.nc')
        ad_path_s_pressure = self.data_directory + '/surface_pressure.nc'

        iris.save(cube, self.cube_path)
        iris.save(cube2, self.cube_path2)
        iris.save(orography, orography_path)
        iris.save(land, land_path)
        iris.save(temperature_on_height_levels, ad_path_temperature)
        iris.save(pressure_on_height_levels, ad_path_pressure)
        iris.save(surface_pressure, ad_path_s_pressure)

        diagnostic_recipe = {
            "temperature": {
                "diagnostic_name": "air_temperature",
                "extrema": True,
                "filepath": "temperature_at_screen_level",
                "neighbour_finding": {
                    "land_constraint": False,
                    "method": "fast_nearest_neighbour",
                    "vertical_bias": None
                    }
                }
            }

        self.config_path = self.data_directory + '/spotdata_diagnostics.json'
        ff = open(self.config_path, 'w')
        json.dump(diagnostic_recipe, ff, sort_keys=True, indent=4,
                  separators=(',', ': ',))
        ff.close()

        self.made_files = [self.cube_path, self.cube_path2, orography_path,
                           land_path, ad_path_temperature, ad_path_pressure,
                           ad_path_s_pressure, self.config_path]

        self.cube = cube
        self.cube2 = cube2
        self.temperature_on_height_levels = temperature_on_height_levels
        self.ancillary_data = ancillary_data
        self.additional_data = additional_data
        self.time_extract = time_extract

    def tearDown(self):
        """Remove temporary directories created for testing."""
        for a_file in self.made_files:
            Call(['rm', a_file])
        Call(['rmdir', self.data_directory])


class Test_Load(Test_read_input):
    """Test function used for loading data cubes."""

    def test_single_file(self):
        """Test loading of a single file as an iris.cube.Cube."""

        expected = self.cube
        result = Plugin('single_file').process(self.cube_path,
                                               'air_temperature')
        self.assertArrayEqual(expected.data, result.data)
        for ex_coord, re_coord in zip(expected.dim_coords, result.dim_coords):
            self.assertEqual(ex_coord, re_coord)

    def test_multi_file(self):
        """Test loading of multiple files as an iris.cube.CubeList."""

        expected = CubeList([self.cube, self.cube2])
        result = Plugin('multi_file').process(
            [self.cube_path, self.cube_path2], 'air_temperature')

        for ex, re in zip(expected, result):
            self.assertEqual(ex.name(), re.name())
            self.assertArrayEqual(ex.data, re.data)

    def test_invalid_method(self):
        """Test attempting to load data with an invalid method."""

        method = 'not_a_valid_method'
        msg = 'Unknown method ".*" passed to .*'
        with self.assertRaisesRegexp(AttributeError, msg):
            Plugin(method).process(self.cube_path, 'air_temperature')

    def test_single_file_invalid_diagnostic(self):
        """Test attempt to load a diagnostic cube from a file within which it
        cannot be found."""

        diagnostic = 'not_a_valid_diagnostic'
        msg = 'no cubes found'
        with self.assertRaisesRegexp(ConstraintMismatchError, msg):
            Plugin('single_file').process(self.cube_path, diagnostic)

    def test_multi_file_invalid_diagnostic(self):
        """Test attempt to load diagnostic cubes from files within which they
        cannot be found."""

        diagnostic = 'not_a_valid_diagnostic'
        msg = 'no cubes found'
        with self.assertRaisesRegexp(ConstraintMismatchError, msg):
            Plugin('single_file').process([self.cube_path, self.cube_path2],
                                          diagnostic)


class Test_get_method_prerequisites(Test_read_input):
    """Test the retrieval of a predefined dictionary of additional diagnostics
    required for given method of data extraction."""

    def test_known_method(self):
        """Test functionality with an expected method."""

        expected = self.additional_data
        method = 'model_level_temperature_lapse_rate'
        result = get_method_prerequisites(method, self.data_directory)

        self.assertArrayEqual(expected.keys(), result.keys())
        for diagnostic in expected.keys():
            self.assertArrayEqual(expected[diagnostic][0].data,
                                  result[diagnostic][0].data)

    def test_unknown_method(self):
        """Test functionality with an unexpected method."""

        expected = None
        method = 'not_a_valid_method'
        result = get_method_prerequisites(method, self.data_directory)
        self.assertArrayEqual(expected, result)


class Test_get_additional_diagnostics(Test_read_input):
    """Test method for loading additional diagnostic cubes into CubeLists that
    are returned with suitable keys in a dictionary."""

    def test_available_data_files(self):
        """Test with files available."""

        diagnostic_name = 'temperature_on_height_levels'
        result = get_additional_diagnostics(diagnostic_name,
                                            self.data_directory)
        self.assertIsInstance(result, CubeList)
        self.assertArrayEqual(result[0].data,
                              self.temperature_on_height_levels.data)

    def test_missing_data_files(self):
        """Test with missing files."""

        diagnostic_name = 'temperature_on_height_levels'
        msg = 'No relevant data files found in .*'
        with self.assertRaisesRegexp(IOError, msg):
            get_additional_diagnostics(diagnostic_name, 'not_a_valid_path')

    def test_available_data_files_with_time_extraction(self):
        """Test with files available and an extraction of data at a given
        time."""

        diagnostic_name = 'temperature_on_height_levels'
        result = get_additional_diagnostics(
            diagnostic_name, self.data_directory,
            time_extract=self.time_extract)

        self.assertIsInstance(result, CubeList)
        self.assertArrayEqual(result[0].data,
                              self.temperature_on_height_levels[0].data)

    def test_available_data_files_with_invalid_time_extraction(self):
        """Test with files available and an attempted extraction of data at a
        time that is not valid."""

        diagnostic_name = 'temperature_on_height_levels'
        msg = 'No diagnostics match .*'
        time_extract = Constraint(time=PartialDateTime(2018, 01, 01, 0))
        with self.assertRaisesRegexp(ValueError, msg):
            get_additional_diagnostics(diagnostic_name, self.data_directory,
                                       time_extract=time_extract)


class Test_data_from_dictionary(Test_read_input):
    """Test return of data from dictionary with error handling."""

    def test_valid_key(self):
        """Test valid key and return of data."""

        key = 'temperature_on_height_levels'
        result = data_from_dictionary(self.additional_data, key)
        self.assertIsInstance(result, CubeList)

    def test_invalid_key(self):
        """Test invalid key and raising of exception."""

        key = 'not_a_valid_key'
        msg = 'Data .* not found in dictionary.'
        with self.assertRaisesRegexp(KeyError, msg):
            data_from_dictionary(self.additional_data, key)

    def test_invalid_type(self):
        """Test case of something other than a dictionary being passed in."""

        key = 'temperature_on_height_levels'
        msg = 'Invalid type sent to data_from_dictionary'
        with self.assertRaisesRegexp(TypeError, msg):
            data_from_dictionary(self.cube, key)


class Test_read_config(Test_read_input):
    """Test reading of json config files that setup diagnostics and
    constants."""

    def test_available_config_file(self):
        """Test return from a valid json config file."""

        result = read_config(self.config_path)
        self.assertIsInstance(result, dict)

    def test_missing_config_file(self):
        """Test raising of exception for missing config file."""

        msg = 'No such file or directory'
        with self.assertRaisesRegexp(IOError, msg):
            read_config('not_a_valid_path')

    def test_invalid_config_file(self):
        """Test raising of exception for invalid config file.
        e.g. not a valid json file."""

        msg = 'Invalid json format. Unable to read configuration'
        with self.assertRaisesRegexp(ValueError, msg):
            read_config(self.cube_path)


if __name__ == '__main__':
    unittest.main()
