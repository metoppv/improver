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
"""Unit tests for the database.SpotDatabase plugin."""

import unittest

from datetime import datetime as dt
from tempfile import mkdtemp
from subprocess import call as Call

import pandas as pd
from pandas.util.testing import assert_frame_equal

from iris.coords import AuxCoord, DimCoord
from iris.coord_systems import GeogCS
from iris.cube import Cube, CubeList
from iris.tests import IrisTest

import cf_units
import numpy as np

from improver.database import SpotDatabase
from improver.utilities.warnings_handler import ManageWarnings


def set_up_spot_cube(point_data, validity_time=1487311200, forecast_period=0,
                     number_of_sites=3):
    """Set up a spot data cube at a given validity time and forecast period for
       a given number of sites.

       Produces a cube with dimension coordinates of time, percentile
       and index. There will be one point in the percentile and time
       coordinates, and as many points in index coordinate as number_of_sites.
       The output cube will also have auxillary coordinates for altitude,
       wmo_site, forecast_period, and forecast_reference_time.

       Args:
           point_data (float):
               The value for the data in the cube, which will be used for
               every site.
       Keyword Args:
           validity_time (float):
               The value for the validity time for your data, defaults to
               1487311200 i.e. 2017-02-17 06:00:00
           forecast_period (float):
               The forecast period for your cube in hours.
           number_of_sites (int):
               The number of sites you want in your output cube.
       Returns:
           cube (iris.cube.Cube):
               Example spot data cube.
    """
    # Set up a data array with all the values the same as point_data.
    data = np.ones((1, 1, number_of_sites)) * point_data
    # Set up dimension coordinates.
    time = DimCoord(np.array([validity_time]), standard_name='time',
                    units=cf_units.Unit('seconds since 1970-01-01 00:00:00',
                                        calendar='gregorian'))
    percentile = DimCoord(np.array([50.]), long_name="percentile", units='%')
    indices = np.arange(number_of_sites)
    index = DimCoord(indices, units=cf_units.Unit('1'),
                     long_name='index')
    # Set up auxillary coordinates.
    latitudes = np.ones(number_of_sites)*54
    latitude = AuxCoord(latitudes, standard_name='latitude',
                        units='degrees', coord_system=GeogCS(6371229.0))
    longitudes = np.arange(number_of_sites)
    longitude = AuxCoord(longitudes, standard_name='longitude',
                         units='degrees', coord_system=GeogCS(6371229.0))
    altitudes = np.arange(number_of_sites)+100
    altitude = DimCoord(altitudes, standard_name='altitude', units='m')
    wmo_sites = np.arange(number_of_sites)+1000
    wmo_site = AuxCoord(wmo_sites, units=cf_units.Unit('1'),
                        long_name='wmo_site')
    forecast_period_coord = AuxCoord(np.array(forecast_period*3600),
                                     standard_name='forecast_period',
                                     units='seconds')
    # Create cube
    cube = Cube(data,
                standard_name="air_temperature",
                dim_coords_and_dims=[(time, 0),
                                     (percentile, 1),
                                     (index, 2), ],
                aux_coords_and_dims=[(latitude, 2), (longitude, 2),
                                     (altitude, 2), (wmo_site, 2),
                                     (forecast_period_coord, 0)],
                units="K")
    # Add scalar forecast_reference_time.
    cycle_time = validity_time - forecast_period * 3600
    forecast_reference_time = AuxCoord(
        np.array([cycle_time]), standard_name='forecast_reference_time',
        units=cf_units.Unit('seconds since 1970-01-01 00:00:00',
                            calendar='gregorian'))
    cube.add_aux_coord(forecast_reference_time)
    return cube


class Test___repr__(IrisTest):
    """A basic test of the repr method"""
    def test_basic_repr(self):
        """Basic test of string representation"""
        expected_result = ('<SpotDatabase: csv, output, improver, time, index,'
                           ' primary_map=None, '
                           'primary_func=None, '
                           'pivot_dim=None, '
                           'pivot_map=None, '
                           'column_dims=None, '
                           'column_maps=None>')
        result = str(SpotDatabase("csv", "output", "improver", "time",
                                  "index"))
        self.assertEqual(expected_result, result)

    def test_repr_after_dataframe_created(self):
        """Basic test of string representation after a database has been
           created and added to self."""
        expected_result = ('<SpotDatabase: csv, output, improver, time, index,'
                           ' primary_map=None, '
                           'primary_func=None, '
                           'pivot_dim=None, '
                           'pivot_map=None, '
                           'column_dims=None, '
                           'column_maps=None>')
        plugin = SpotDatabase("csv", "output", "improver", "time", "index")
        plugin.df = pd.DataFrame()
        result = str(plugin)
        self.assertEqual(expected_result, result)


class Test_check_input_dimensions(IrisTest):
    """Test the check_input_dimensions function"""
    def setUp(self):
        """Set up the plugin and dataframe needed for these tests"""
        self.cube = set_up_spot_cube(280, number_of_sites=1)
        second_cube = self.cube.copy()
        second_cube.coord("percentile").points = np.array([60.0])
        cubelist = CubeList([self.cube, second_cube])
        self.cubes = cubelist.concatenate()
        self.plugin = SpotDatabase("csv", "output", "improver",
                                   "time", "IMPRO", 0)

    def test_basic_1D_cube(self):
        """Test it doesn't raise an exception if a 1D cube is input."""
        self.assertFalse(self.plugin.check_input_dimensions(self.cube))

    def test_raises_if_extra_dim(self):
        """Test it raises an exception if a 2D cube is input and
           no pivot_dim."""
        message = ("Dimensions that are not described by the pivot_dim or "
                   "coord_to_slice_over must only have one point in. "
                   "Dimension '1' has length '2' and is associated with the "
                   "'percentile' coordinate.")
        with self.assertRaisesRegex(ValueError, message):
            self.plugin.check_input_dimensions(self.cubes[0])

    def test_no_exception_if_pivot_dim_set(self):
        """Test it raises an exception if a 2D cube is input and
           no pivot_dim."""
        plugin = SpotDatabase("csv", "output", "improver", "time", "IMPRO",
                              0, pivot_dim="percentile")
        self.assertFalse(plugin.check_input_dimensions(self.cubes[0]))

    def test_exception_if_pivot_dim_set(self):
        """Test it raises an exception if a 2D cube is input and
           no pivot_dim."""
        plugin = SpotDatabase("csv", "output", "improver", "time", "IMPRO",
                              0, pivot_dim="percentile")
        cube = set_up_spot_cube(280, number_of_sites=3)
        second_cube = cube.copy()
        second_cube.coord("percentile").points = np.array([60.0])
        cubelist = CubeList([cube, second_cube])
        cubes = cubelist.concatenate()
        message = "Dimensions that are not described by the pivot_dim or "\
                  "coord_to_slice_over must only have one point in. "\
                  "Dimension '2' has length '3' and is associated with the "\
                  "'index' coordinate."
        with self.assertRaisesRegex(ValueError, message):
            plugin.check_input_dimensions(cubes[0])


class Test_pivot_table(IrisTest):
    """Test the pivot_table method"""
    def setUp(self):
        """Set up the plugin and dataframe needed for these tests"""
        self.cube = set_up_spot_cube(280, number_of_sites=1)
        self.plugin = SpotDatabase(
            "csv", "output", "improver", "time", "index",
            pivot_map=lambda x: 'T+{:03d}'.format(int(x/3600)),
            pivot_dim="forecast_period")
        data = [[1487311200, 280.]]
        columns = ["time", "values"]
        self.input_df = pd.DataFrame(data, columns=columns)

    def test_single_cube(self):
        """Basic test using one input cube."""
        # Set up expected dataframe.
        expected_data = [[280.]]
        expected_df = pd.DataFrame(expected_data, columns=["T+000"])
        expected_df.columns.name = "forecast_period"
        # Call the method.
        result = self.plugin.pivot_table(self.cube, self.input_df)
        assert_frame_equal(expected_df, result)

    def test_multiple_times_cube(self):
        """Test using one input cube, with one site and multiple times."""
        # Set up expected dataframe.
        expected_data = [[280., np.nan],
                         [np.nan, 281.]]
        expected_df = pd.DataFrame(expected_data, columns=["T+000", "T+001"])
        expected_df.columns.name = "forecast_period"

        # Set up cube with multiple lead times in.
        second_cube = set_up_spot_cube(281, number_of_sites=1,
                                       validity_time=1487311200+3600,
                                       forecast_period=1,)

        merged_cube = CubeList([self.cube, second_cube])
        merged_cube = merged_cube.concatenate()
        # Set up input dataframe
        data = [[1487311200, 280.],
                [1487311200+3600, 281.]]
        columns = ["time", "values"]
        input_df = pd.DataFrame(data, columns=columns)
        result = self.plugin.pivot_table(merged_cube[0], input_df)
        assert_frame_equal(expected_df, result)

    def test_extra_columns(self):
        """Test with an extra column in the input dataframe.
           In this case the extra column gets removed when the table
           is pivoted.
        """
        # Set up expected dataframe.
        expected_data = [[280., np.nan],
                         [np.nan, 281.]]
        expected_df = pd.DataFrame(expected_data, columns=["T+000", "T+001"])
        expected_df.columns.name = "forecast_period"

        # Set up cube with multiple lead times in.
        second_cube = set_up_spot_cube(281, number_of_sites=1,
                                       validity_time=1487311200+3600,
                                       forecast_period=1,)

        merged_cube = CubeList([self.cube, second_cube])
        merged_cube = merged_cube.concatenate()
        # Set up input dataframe
        data = [[1487311200, 3001, 280.],
                [1487311200+3600, 3002, 281.]]
        columns = ["time", "wmo_site", "values"]
        input_df = pd.DataFrame(data, columns=columns)
        result = self.plugin.pivot_table(merged_cube[0], input_df)
        assert_frame_equal(expected_df, result)


class Test_map_primary_index(IrisTest):
    """Test the map_primary_index method"""
    def setUp(self):
        """Set up the plugin and dataframe needed for these tests"""
        self.cube = set_up_spot_cube(280, number_of_sites=1,)
        self.plugin = SpotDatabase(
            "csv", "output", "improver", "time", "index",
            primary_map=['validity_date', 'validity_time'],
            primary_func=[lambda x: dt.utcfromtimestamp(x).date(),
                          lambda x: dt.utcfromtimestamp(x).hour*100])
        data = [[1487311200, 280.]]
        columns = ["time", "values"]
        self.input_df = pd.DataFrame(data, columns=columns)
        self.input_df = self.input_df.set_index(["time"])

    def test_single_cube(self):
        """Basic test using one input cube."""
        # Set up expected dataframe.

        validity_date = dt.utcfromtimestamp(1487311200).date()
        expected_data = [[validity_date, 600, 280.]]
        columns = ['validity_date', 'validity_time', "values"]
        expected_df = pd.DataFrame(expected_data, columns=columns)
        expected_df = expected_df.set_index(["validity_date", "validity_time"])
        # Call the method.
        self.plugin.map_primary_index(self.input_df)
        assert_frame_equal(expected_df, self.input_df)

    def test_multiple_times_cube(self):
        """Test using one input cube, with one site and multiple times."""
        # Set up expected dataframe.
        validity_date = dt.utcfromtimestamp(1487311200).date()
        expected_data = [[validity_date, 600, 280.],
                         [validity_date, 700, 281.]]
        columns = ['validity_date', 'validity_time', "values"]
        expected_df = pd.DataFrame(expected_data, columns=columns)
        expected_df = expected_df.set_index(["validity_date", "validity_time"])

        # Set up input dataframe
        data = [[1487311200, 280.],
                [1487311200+3600, 281.]]
        columns = ["time", "values"]
        input_df = pd.DataFrame(data, columns=columns)
        input_df = input_df.set_index(["time"])
        self.plugin.map_primary_index(input_df)
        assert_frame_equal(expected_df, input_df)


class Test_insert_extra_mapped_column(IrisTest):
    """Test the insert_extra_mapped_column method"""
    def setUp(self):
        """Set up the plugin and dataframe needed for these tests"""
        self.cube = set_up_spot_cube(280, number_of_sites=1)
        self.plugin = SpotDatabase("csv", "output", "improver", "time",
                                   "index")
        data = [[1487311200, 280.]]
        columns = ["time", "values"]
        self.input_df = pd.DataFrame(data, columns=columns)

    def test_extra_column_in_df(self):
        """Basic test using a column and dim for something that already exist.
           In this case the function does nothing to the dataframe."""
        # Set up expected dataframe.
        data = [[1487311200, 280.]]
        columns = ["time", "values"]
        expected_df = pd.DataFrame(data, columns=columns)
        # Call the method.
        self.plugin.insert_extra_mapped_column(
            self.input_df, self.cube, "values", [380.])
        assert_frame_equal(self.input_df, expected_df)

    def test_extra_column_from_coord(self):
        """Test for when we are taking data from a coordinate in the cube"""
        data = [[1487311200, 100, 280.]]
        columns = ["time", "altitude_of_site", "values"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df.set_index(["altitude_of_site"], append=True, inplace=True)
        # Call the method.
        self.plugin.insert_extra_mapped_column(
            self.input_df, self.cube, "altitude", "altitude_of_site")
        assert_frame_equal(self.input_df, expected_df)

    def test_extra_column_from_coord_multiple_points(self):
        """Test for when we are taking data from a coordinate in the cube.
           This test has multiple sites in the cube."""
        data = [[1487311200, 100, 280.],
                [1487311200, 101, 280.],
                [1487311200, 102, 280.]]
        columns = ["time", "altitude_of_site", "values"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df.set_index(["altitude_of_site"], append=True, inplace=True)
        # Call the method.
        cube = set_up_spot_cube(280, number_of_sites=3)
        data = [[1487311200, 280.],
                [1487311200, 280.],
                [1487311200, 280.]]
        columns = ["time", "values"]
        self.input_df = pd.DataFrame(data, columns=columns)
        self.plugin.insert_extra_mapped_column(
            self.input_df, cube, "altitude", "altitude_of_site")
        assert_frame_equal(self.input_df, expected_df)

    def test_extra_column_from_cube_name(self):
        """Test for when we are taking data from the cube name."""
        data = [[1487311200, "air_temperature", 280.]]
        columns = ["time", "name_of_cube", "values"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df.set_index(["name_of_cube"], append=True, inplace=True)
        # Call the method.
        self.plugin.insert_extra_mapped_column(
            self.input_df, self.cube, "name", "name_of_cube")
        assert_frame_equal(self.input_df, expected_df)

    def test_extra_column_from_cube_attribute(self):
        """Test for when we are taking data from the cube name."""
        data = [[1487311200, "K", 280.]]
        columns = ["time", "cube_units", "values"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df.set_index(["cube_units"], append=True, inplace=True)
        # Call the method.
        self.plugin.insert_extra_mapped_column(
            self.input_df, self.cube, "units", "cube_units")
        assert_frame_equal(self.input_df, expected_df)

    def test_static_extra_column(self):
        """Test for when we add a new column that does not come from the cube.
           """
        data = [[1487311200, "IMPRO_nbhood", 280.]]
        columns = ["time", "experiment_id", "values"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df.set_index(["experiment_id"], append=True, inplace=True)
        # Call the method.
        self.cube.attributes["source_grid"] = "ukvx"
        self.plugin.insert_extra_mapped_column(
            self.input_df, self.cube, "IMPRO_nbhood", "experiment_id")
        assert_frame_equal(self.input_df, expected_df)


class Test_to_dataframe(IrisTest):
    """Test the to_dataframe method"""
    def setUp(self):
        """Set up the plugin and dataframe needed for these tests"""
        self.cube = set_up_spot_cube(280, number_of_sites=1)
        self.cube2 = set_up_spot_cube(
            281, number_of_sites=1, validity_time=1487311200+3600,
            forecast_period=1)
        self.cube3 = set_up_spot_cube(
            282, number_of_sites=1, validity_time=1487311200+7200,
            forecast_period=2)
        self.cubelist = CubeList([self.cube])
        self.cubelist_multiple = CubeList([self.cube, self.cube2, self.cube3])
        self.plugin = SpotDatabase("csv", "output", "improver", "time",
                                   "index")
        data = [[1487311200, 280.]]
        columns = ["time", "values"]
        self.input_df = pd.DataFrame(data, columns=columns)

    def test_no_optional_args(self):
        """Test we create a datafram even when we have no optional
           arguements set"""
        # Set up expected dataframe.
        data = [[280.]]
        columns = ["values"]
        expected_df = pd.DataFrame(data, index=[1487311200], columns=columns)

        # Call the method.
        self.plugin.to_dataframe(self.cubelist)

        assert_frame_equal(self.plugin.df, expected_df)

    def test_all_optional_args(self):
        """Test we create a datafram even when we have all optional
           arguements set"""
        # Set up expected dataframe.
        data = [[600, "air_temperature", 280.]]
        columns = ["validity_time", "cf_name", "T+000"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df.set_index(["validity_time", "cf_name"], inplace=True)
        expected_df.columns.name = "forecast_period"
        plugin = SpotDatabase(
            "csv", "output", "improver", "time",
            primary_map=['validity_time'],
            primary_func=[lambda x: dt.utcfromtimestamp(x).hour*100],
            pivot_dim='forecast_period',
            pivot_map=lambda x: 'T+{:03d}'.format(int(x/3600)),
            column_dims=['name'], column_maps=['cf_name'],
            coord_to_slice_over="index")
        # Call the method.
        plugin.to_dataframe(self.cubelist)
        assert_frame_equal(plugin.df, expected_df)

    def test_all_optional_args_multiple_input_cubes(self):
        """Test we create a dataframe even when we have no optional
           arguements set and multiple cubes"""
        # Set up expected dataframe.
        data = [[600, "air_temperature", 280., np.nan, np.nan],
                [700, "air_temperature", np.nan, 281., np.nan],
                [800, "air_temperature", np.nan, np.nan, 282.]]
        columns = ["validity_time", "cf_name", "T+000", "T+001", "T+002"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df.set_index(["validity_time", "cf_name"], inplace=True)
        expected_df.columns.name = "forecast_period"
        plugin = SpotDatabase(
            "csv", "output", "improver", "time",
            primary_map=['validity_time'],
            primary_func=[lambda x: dt.utcfromtimestamp(x).hour*100],
            pivot_dim='forecast_period',
            pivot_map=lambda x: 'T+{:03d}'.format(int(x/3600)),
            column_dims=['name'], column_maps=['cf_name'],
            coord_to_slice_over="index")
        # Call the method.
        plugin.to_dataframe(self.cubelist_multiple)
        assert_frame_equal(plugin.df, expected_df)

    @staticmethod
    def test_all_optional_args_multiple_sites():
        """Test we create a datafram even when we have all optional
           arguements set and multiple spots"""
        # Set up expected dataframe.
        data = [[600, "air_temperature", 0, 280.],
                [600, "air_temperature", 1, 280.],
                [600, "air_temperature", 2, 280.]]
        columns = ["validity_time", "cf_name", "site", "T+000"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df.set_index(["validity_time", "cf_name", "site"],
                              inplace=True)
        expected_df.columns.name = "forecast_period"
        plugin = SpotDatabase(
            "csv", "output", "improver", "time",
            primary_map=['validity_time'],
            primary_func=[lambda x: dt.utcfromtimestamp(x).hour*100],
            pivot_dim='forecast_period',
            pivot_map=lambda x: 'T+{:03d}'.format(int(x/3600)),
            column_dims=['name', "index"], column_maps=['cf_name', "site"],
            coord_to_slice_over="index")
        # Call the method.
        cube = set_up_spot_cube(280, number_of_sites=3,)
        plugin.to_dataframe(CubeList([cube]))
        assert_frame_equal(plugin.df, expected_df)


class Test_determine_schema(IrisTest):
    """A set of tests for the determine_schema method"""
    def setUp(self):
        """Set up the plugin and dataframe needed for this test"""
        cubes = CubeList([set_up_spot_cube(280)])
        self.plugin = SpotDatabase("csv", "output", "improver", "time",
                                   "index")
        self.dataframe = self.plugin.to_dataframe(cubes)

    def test_full_schema(self):
        """Basic test using a basic dataframe as input"""
        schema = self.plugin.determine_schema("improver")
        expected_schema = 'CREATE TABLE "improver" (\n"index" INTEGER,\n  '\
                          '"values" REAL,\n  CONSTRAINT improver_pk '\
                          'PRIMARY KEY ("index")\n)'
        self.assertEqual(schema, expected_schema)


class Test_process(IrisTest):
    """A set of tests for the determine_schema method"""
    def setUp(self):
        """Set up the plugin and dataframe needed for this test"""
        self.cubes = CubeList([set_up_spot_cube(280)])
        self.data_directory = mkdtemp()
        self.plugin = SpotDatabase("csv", self.data_directory + "/test.csv",
                                   "improver", "time", "index")

    def tearDown(self):
        """Remove temporary directories created for testing."""
        Call(['rm', '-f', self.data_directory + '/test.csv'])
        Call(['rmdir', self.data_directory])

    def test_save_as_csv(self):
        """Basic test using a basic dataframe as input"""
        self.plugin.process(self.cubes)
        with open(self.data_directory + '/test.csv') as f:
            resulting_string = f.read()
        expected_string = ',values\n1487311200,280.0\n'
        self.assertEqual(resulting_string, expected_string)

    def test_unknown_output_type(self):
        """Test what happens if you give an unknown output type."""
        plugin = SpotDatabase("kitten", self.data_directory + "/test.csv",
                              "improver", "time", "index")
        message = ("Unrecognised output type. Current options are 'sqlite'"
                   " or 'csv', 'kitten' given.")
        with self.assertRaisesRegex(ValueError, message):
            plugin.process(self.cubes)


if __name__ == '__main__':
    unittest.main()
    set_up_spot_cube(280, number_of_sites=5)
