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
"""Unit tests for weather code utilities."""

import unittest
from subprocess import call as Call
from tempfile import mkdtemp

import numpy as np


import iris
from iris.cube import Cube
from iris.tests import IrisTest
from cf_units import Unit

from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf

from improver.wxcode.wxcode_utilities import (WX_DICT,
                                              add_wxcode_metadata,
                                              expand_nested_lists)
from improver.tests.ensemble_calibration.ensemble_calibration. \
    helper_functions import set_up_cube


class Test_wx_dict(IrisTest):
    """ Test WX_DICT set correctly """

    def test_wxcode_values(self):
        """Check wxcode values are set correctly."""
        self.assertEqual(WX_DICT[0], 'Clear_Night')
        self.assertEqual(WX_DICT[1], 'Sunny_Day')
        self.assertEqual(WX_DICT[2], 'Partly_Cloudy_Night')
        self.assertEqual(WX_DICT[3], 'Partly_Cloudy_Day')
        self.assertEqual(WX_DICT[4], 'Dust')
        self.assertEqual(WX_DICT[5], 'Mist')
        self.assertEqual(WX_DICT[6], 'Fog')
        self.assertEqual(WX_DICT[7], 'Cloudy')
        self.assertEqual(WX_DICT[8], 'Overcast')
        self.assertEqual(WX_DICT[9], 'Light_Shower_Night')
        self.assertEqual(WX_DICT[10], 'Light_Shower_Day')
        self.assertEqual(WX_DICT[11], 'Drizzle')
        self.assertEqual(WX_DICT[12], 'Light_Rain')
        self.assertEqual(WX_DICT[13], 'Heavy_Shower_Night')
        self.assertEqual(WX_DICT[14], 'Heavy_Shower_Day')
        self.assertEqual(WX_DICT[15], 'Heavy_Rain')
        self.assertEqual(WX_DICT[16], 'Sleet_Shower_Night')
        self.assertEqual(WX_DICT[17], 'Sleet_Shower_Day')
        self.assertEqual(WX_DICT[18], 'Sleet')
        self.assertEqual(WX_DICT[19], 'Hail_Shower_Night')
        self.assertEqual(WX_DICT[20], 'Hail_Shower_Day')
        self.assertEqual(WX_DICT[21], 'Hail')
        self.assertEqual(WX_DICT[22], 'Light_Snow_Shower_Night')
        self.assertEqual(WX_DICT[23], 'Light_Snow_Shower_Day')
        self.assertEqual(WX_DICT[24], 'Light_Snow')
        self.assertEqual(WX_DICT[25], 'Heavy_Snow_Shower_Night')
        self.assertEqual(WX_DICT[26], 'Heavy_Snow_Shower_Day')
        self.assertEqual(WX_DICT[27], 'Heavy_Snow')
        self.assertEqual(WX_DICT[28], 'Thunder_Shower_Night')
        self.assertEqual(WX_DICT[29], 'Thunder_Shower_Day')
        self.assertEqual(WX_DICT[30], 'Thunder')


class Test_add_wxcode_metadata(IrisTest):
    """ Test add_wxcode_metadata is working correctly """

    def setUp(self):
        """Set up cube """
        data = np.array([0, 1, 5, 11, 20, 5, 9, 10, 4,
                         2, 0, 1, 29, 30, 1, 5, 6, 6]).reshape(2, 1, 3, 3)
        self.cube = set_up_cube(data, 'air_temperature', 'K',
                                realizations=np.array([0, 1]))
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())
        self.data_directory = mkdtemp()
        self.nc_file = self.data_directory + '/wxcode.nc'
        Call(['touch', self.nc_file])

    def tearDown(self):
        """Remove temporary directories created for testing."""
        Call(['rm', self.nc_file])
        Call(['rmdir', self.data_directory])

    def test_basic(self):
        """Test that the function returns a cube."""
        result = add_wxcode_metadata(self.cube)
        self.assertIsInstance(result, Cube)

    def test_metadata_set(self):
        """Test that the metadata is set correctly."""
        result = add_wxcode_metadata(self.cube)
        self.assertEqual(result.name(), 'weather_code')
        self.assertEqual(result.units, Unit("1"))
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)

    def test_metadata_saves(self):
        """Test that the metadata saves as NetCDF correctly."""
        cube = add_wxcode_metadata(self.cube)
        save_netcdf(cube, self.nc_file)
        result = load_cube(self.nc_file)
        self.assertEqual(result.name(), 'weather_code')
        self.assertEqual(result.units, Unit("1"))
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)


class Test_expand_nested_lists(IrisTest):
    """ Test expand_nested_lists is working correctly """

    def setUp(self):
        """ Set up dictionary for testing """
        self.dictionary = {'list': ['a', 'a'],
                           'list_of_lists': [['a', 'a'], ['a', 'a']]}

    def test_basic(self):
        """Test that the expand_nested_lists returns a list."""
        result = expand_nested_lists(self.dictionary, 'list')
        self.assertIsInstance(result, list)

    def test_simple_list(self):
        """Testexpand_nested_lists returns a expanded list if given a list."""
        result = expand_nested_lists(self.dictionary, 'list')
        for val in result:
            self.assertEqual(val, 'a')

    def test_list_of_lists(self):
        """Returns a expanded list if given a list of lists."""
        result = expand_nested_lists(self.dictionary, 'list_of_lists')
        for val in result:
            self.assertEqual(val, 'a')


if __name__ == '__main__':
    unittest.main()
