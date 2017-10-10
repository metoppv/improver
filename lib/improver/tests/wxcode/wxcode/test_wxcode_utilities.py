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
"""Unit tests for weather code utilities."""

import unittest

from iris.cube import Cube
from iris.tests import IrisTest
from cf_units import Unit
import numpy as np

from improver.wxcode.wxcode_utilities import WXCODE, WXMEANING, \
    add_wxcode_metadata
from improver.tests.ensemble_calibration.ensemble_calibration. \
    helper_functions import (set_up_cube)


class Test_wxcode(IrisTest):
    """ Test WXCODE set correctly """

    def test_basic(self):
        """Check number of WXCODE is equal to number of WXMEANINGS."""
        self.assertEqual(len(WXCODE), len(WXMEANING))

    def test_wxcode_values(self):
        """Check wxcode values are set correctly."""
        wx_dict = dict(zip(WXCODE, WXMEANING))
        self.assertEqual(wx_dict[0], 'Clear Night')
        self.assertEqual(wx_dict[1], 'Sunny Day')
        self.assertEqual(wx_dict[2], 'Partly Cloudy Night')
        self.assertEqual(wx_dict[3], 'Partly Cloudy Day')
        self.assertEqual(wx_dict[4], 'Dust')
        self.assertEqual(wx_dict[5], 'Mist')
        self.assertEqual(wx_dict[6], 'Fog')
        self.assertEqual(wx_dict[7], 'Cloudy')
        self.assertEqual(wx_dict[8], 'Overcast')
        self.assertEqual(wx_dict[9], 'Light Shower Night')
        self.assertEqual(wx_dict[10], 'Light Shower Day')
        self.assertEqual(wx_dict[11], 'Drizzle')
        self.assertEqual(wx_dict[12], 'Light Rain')
        self.assertEqual(wx_dict[13], 'Heavy Shower Night')
        self.assertEqual(wx_dict[14], 'Heavy Shower Day')
        self.assertEqual(wx_dict[15], 'Heavy Rain')
        self.assertEqual(wx_dict[16], 'Sleet Shower Night')
        self.assertEqual(wx_dict[17], 'Sleet Shower Day')
        self.assertEqual(wx_dict[18], 'Sleet')
        self.assertEqual(wx_dict[19], 'Hail Shower Night')
        self.assertEqual(wx_dict[20], 'Hail Shower Day')
        self.assertEqual(wx_dict[21], 'Hail')
        self.assertEqual(wx_dict[22], 'Light Snow Shower Night')
        self.assertEqual(wx_dict[23], 'Light Snow Shower Day')
        self.assertEqual(wx_dict[24], 'Light Snow')
        self.assertEqual(wx_dict[25], 'Heavy Snow Shower Night')
        self.assertEqual(wx_dict[26], 'Heavy Snow Shower Day')
        self.assertEqual(wx_dict[27], 'Heavy Snow')
        self.assertEqual(wx_dict[28], 'Thunder Shower Night')
        self.assertEqual(wx_dict[29], 'Thunder Shower Day')
        self.assertEqual(wx_dict[30], 'Thunder')


class Test_add_wxcode_metadata(IrisTest):
    """ Test add_wxcode_metadata is working correctly """

    def setUp(self):
        """Set up cube """
        data = np.array([0, 1, 5, 11, 20, 5, 9, 10, 4,
                         2, 0, 1, 29, 30, 1, 5, 6, 6]).reshape(2, 1, 3, 3)
        cube = set_up_cube(data, 'air_temperature', 'K',
                           realizations=np.array([0, 1]))
        self.cube = cube

    def test_basic(self):
        """Test that the function returns a cube."""
        result = add_wxcode_metadata(self.cube)
        self.assertIsInstance(result, Cube)

    def test_metadata_set(self):
        """Test that the metadata is set correctly."""
        result = add_wxcode_metadata(self.cube)
        self.assertEqual(result.name(), 'weather_code')
        self.assertEqual(result.units, Unit("1"))
        self.assertEqual(result.attributes['weather_code'], WXCODE)
        self.assertEqual(result.attributes['weather_code_meaning'], WXMEANING)
