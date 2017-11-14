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
"""Unit tests for the database.SpotDatabase plugin."""

import unittest

import iris
from iris.coords import AuxCoord, DimCoord
from iris.coord_systems import GeogCS
from iris.cube import Cube
from iris.tests import IrisTest
import cf_units
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from improver.database import SpotDatabase


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
    forecast_period_coord = AuxCoord(np.array(forecast_period),
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
        expected_result = "some_string"
        result = str(SpotDatabase())
        self.assertEqual(expected_result, result)


class Test_to_dataframe(IrisTest):
    """A set of tests for the to_dataframe method"""
    def test_single_cube(self):
        """Basic test using one input cube."""
        cubes = iris.cube.CubeList([set_up_spot_cube(280)])
        plugin = SpotDatabase(cubes, "output", "csv", "improver", pivot_max=1)
        plugin.to_dataframe()
        result = plugin.df
        print result
        expected_dataframe = pd.DataFrame(data=np.array([280.0, 280.0, 280.0]),
                                          columns=["fcr_tplus000"])
        print expected_dataframe
        assert_frame_equal(expected_dataframe, result)


if __name__ == '__main__':
    unittest.main()
    set_up_spot_cube(280, number_of_sites=5)
