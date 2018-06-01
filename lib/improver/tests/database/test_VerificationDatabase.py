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

import iris
from iris.tests import IrisTest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from improver.database import VerificationTable


from improver.tests.database.test_SpotDatabase import set_up_spot_cube


class Test___repr__(IrisTest):
    """A basic test of the repr method"""
    def test_basic_repr(self):
        """Basic test of string representation"""
        expected_result = ("<VerificationTable: csv, output, improver, "
                           "nbhood, 54>")
        result = str(VerificationTable("csv", "output", "improver",
                                       "nbhood", 54))
        self.assertEqual(expected_result, result)


class Test_to_dataframe(IrisTest):
    """A set of tests for the to_dataframe method"""

    @staticmethod
    def test_single_cube():
        """Basic test using one input cube."""
        # Set up expected dataframe.
        validity_date = dt.utcfromtimestamp(1487311200).date()
        data = [[validity_date, 600, 1000, "air_temperature", "IMPRO", 280.],
                [validity_date, 600, 1001, "air_temperature", "IMPRO", 280.],
                [validity_date, 600, 1002, "air_temperature", "IMPRO", 280.]]
        columns = ["validity_date", "validity_time", "station_id", "cf_name",
                   "exp_id", "fcr_tplus000"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df = expected_df.set_index(["validity_date", "validity_time",
                                             "station_id", "cf_name",
                                             "exp_id"])
        expected_df.columns.name = "forecast_period"
        # Call the plugin.
        cubes = iris.cube.CubeList([set_up_spot_cube(280)])
        plugin = VerificationTable("csv", "output", "improver", "IMPRO", 0)
        plugin.to_dataframe(cubes)
        result = plugin.df
        assert_frame_equal(expected_df, result)

    def test_single_cube_extra_data(self):
        """Basic test using one input cube with an extra point in the
           percentile dimension."""
        # Set up cubes
        cube = set_up_spot_cube(280)
        second_cube = cube.copy()
        second_cube.coord("percentile").points = np.array([60.0])
        cubelist = iris.cube.CubeList([cube, second_cube])
        cubes = cubelist.concatenate()
        plugin = VerificationTable("csv", "output", "improver", "IMPRO", 0)
        message = "Dimensions that are not described by the pivot_dim or "\
                  "coord_to_slice_over must only have one point in. "\
                  "Dimension '1' has length '2' and is associated with the "\
                  "'percentile' coordinate."
        with self.assertRaisesRegex(ValueError, message):
            plugin.to_dataframe(cubes)

    @staticmethod
    def test_single_cube_single_site():
        """Basic test using one input cube with a single site in it."""
        # Set up expected dataframe.
        validity_date = dt.utcfromtimestamp(1487311200).date()
        data = [[validity_date, 600, 1000, "air_temperature", "IMPRO", 280.]]
        columns = ["validity_date", "validity_time", "station_id", "cf_name",
                   "exp_id", "fcr_tplus000"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df = expected_df.set_index(["validity_date", "validity_time",
                                             "station_id", "cf_name",
                                             "exp_id"])
        expected_df.columns.name = "forecast_period"
        # Call the plugin.
        cubes = iris.cube.CubeList([set_up_spot_cube(280, number_of_sites=1)])
        plugin = VerificationTable("output", "csv", "improver", "IMPRO", 0)
        plugin.to_dataframe(cubes)
        result = plugin.df
        assert_frame_equal(expected_df, result)

    @staticmethod
    def test_multiple_cubes_same_validity_time():
        """Basic test using several input cubes with same validity times
        but with different forecast_period."""
        # Set up expected dataframe.
        validity_date = dt.utcfromtimestamp(1487311200).date()
        data = [[validity_date, 600, 1000, "air_temperature", "IMPRO",
                 280., 281., 282.]]

        columns = ["validity_date", "validity_time", "station_id", "cf_name",
                   "exp_id", "fcr_tplus000", "fcr_tplus001", "fcr_tplus002"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df = expected_df.set_index(["validity_date", "validity_time",
                                             "station_id", "cf_name",
                                             "exp_id"])
        expected_df.columns.name = "forecast_period"
        # Call the plugin.
        cubes = [set_up_spot_cube(280+i, forecast_period=i, number_of_sites=1)
                 for i in range(3)]
        cubes = iris.cube.CubeList(cubes)
        plugin = VerificationTable("output", "csv", "improver", "IMPRO", 0)
        plugin.to_dataframe(cubes)
        result = plugin.df
        assert_frame_equal(expected_df, result)

    @staticmethod
    def test_multiple_forecast_periods():
        """Basic test using several input cubes with different validity times
           and different forecast_periods.This is what the output will look
           like if you load in multiple cubes from the same cycle."""
        # Set up expected dataframe.
        validity_date = dt.utcfromtimestamp(1487311200).date()
        data = [[validity_date, 600, 1000, "air_temperature", "IMPRO",
                 280., np.nan, np.nan],
                [validity_date, 700, 1000, "air_temperature", "IMPRO",
                 np.nan, 281., np.nan],
                [validity_date, 800, 1000, "air_temperature", "IMPRO",
                 np.nan, np.nan, 282.]]
        columns = ["validity_date", "validity_time", "station_id", "cf_name",
                   "exp_id", "fcr_tplus000", "fcr_tplus001", "fcr_tplus002"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df = expected_df.set_index(["validity_date", "validity_time",
                                             "station_id", "cf_name",
                                             "exp_id"])
        expected_df.columns.name = "forecast_period"
        # Call the plugin.
        cubes = [set_up_spot_cube(
            280+i, validity_time=1487311200+3600*i,
            forecast_period=i, number_of_sites=1) for i in range(3)]
        cubes = iris.cube.CubeList(cubes)
        plugin = VerificationTable("output", "csv", "improver", "IMPRO", 0)
        plugin.to_dataframe(cubes)
        result = plugin.df
        assert_frame_equal(expected_df, result)

    @staticmethod
    def test_single_cube_max_lead_time():
        """Basic test using one input cube with larger max lead time
           in output"""
        # Set up expected dataframe.
        validity_date = dt.utcfromtimestamp(1487311200).date()
        data = [[validity_date, 600, 1000, "air_temperature", "IMPRO",
                 280., np.nan]]
        columns = ["validity_date", "validity_time", "station_id", "cf_name",
                   "exp_id", "fcr_tplus000", "fcr_tplus001"]
        expected_df = pd.DataFrame(data, columns=columns)
        expected_df = expected_df.set_index(["validity_date", "validity_time",
                                             "station_id", "cf_name",
                                             "exp_id"])
        expected_df.columns.name = "forecast_period"
        # Call the plugin.
        cubes = iris.cube.CubeList([set_up_spot_cube(280, number_of_sites=1)])
        plugin = VerificationTable("output", "csv", "improver", "IMPRO", 3600)
        plugin.to_dataframe(cubes)
        result = plugin.df
        assert_frame_equal(expected_df, result)


class Test_ensure_ensure_all_forecast_columns(IrisTest):
    """A set of tests for the ensure_all_forecast_collumns method"""
    @staticmethod
    def test_single_cube():
        """Basic test using one input cube."""

        plugin = VerificationTable("csv", "output", "improver",
                                   "nbhood", 3600)
        test_dataframe = pd.DataFrame(data=np.array([280.0, 280.0, 280.0]),
                                      columns=["fcr_tplus000"])
        plugin.ensure_all_forecast_columns(test_dataframe)
        result = test_dataframe
        expected_dataframe = pd.DataFrame(data=np.array([[280.0, np.nan],
                                                         [280.0, np.nan],
                                                         [280.0, np.nan]]),
                                          columns=["fcr_tplus000",
                                                   "fcr_tplus001"])
        assert_frame_equal(expected_dataframe, result)


if __name__ == '__main__':
    unittest.main()
