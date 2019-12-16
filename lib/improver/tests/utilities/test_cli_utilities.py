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
"""Unit tests for utilities.cli_utilities."""

import os
import numpy as np
import unittest
from unittest.mock import patch, mock_open
from tempfile import mkdtemp

from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.tests.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cli_utilities import (radius_or_radii_and_lead,
                                              load_json_or_none,
                                              make_cubes_non_lazy)


class Test_load_json_or_none(unittest.TestCase):
    """Tests load_json_or_none to call loading json or return None."""

    @patch('builtins.open', new_callable=mock_open, read_data='{"k": "v"}')
    def test_loading_file(self, m):
        """Tests if called with a filepath, loads a dict."""
        file_path = 'filename'
        dict_read = load_json_or_none(file_path)
        self.assertEqual(dict_read, {"k": "v"})
        m.assert_called_with('filename', 'r')

    @patch('builtins.open', new_callable=mock_open, read_data='{"k": "v"}')
    def test_none(self, m):
        """Tests if called with None returns None."""
        file_path = None
        dict_read = load_json_or_none(file_path)
        self.assertIsNone(dict_read)
        m.assert_not_called()


class Test_radius_or_radii_and_lead(unittest.TestCase):
    """Tests radius_or_radii_and_lead to split the data correctly."""

    def test_radius(self):
        """Tests a correct radius and a radii/lead of None.
        radius_or_radii is the first input, lead is None."""
        radius, lead = radius_or_radii_and_lead(3.3, None)

        self.assertEqual(radius, 3.3)
        self.assertIsNone(lead)

    def test_both_None(self):
        """ Tests if both are None and raises a TypeError"""
        msg = ("Neither radius or radii_by_lead_time have been set. "
               "One option should be specified.")
        with self.assertRaisesRegex(TypeError, msg):
            radius_or_radii_and_lead(None, None)

    def test_radii_and_lead(self):
        """Tests if radius is None and radii/lead is a list of csv.
        radius_or_radii is the [0] of the second argument separated by
        commas lead is the [1] of the second argument separated by commas."""
        radii, lead = radius_or_radii_and_lead(
            None, ["0,36,72,144", "18000,54000,90000,162000"])

        self.assertEqual(radii, ['0', '36', '72', '144'])
        self.assertEqual(lead, ['18000', '54000', '90000', '162000'])

    def test_both_used(self):
        """Tests if both arguments are used and raises a TypeError"""
        radius = 3.3
        radii_lead = ["0,36,72,144", "18000,54000,90000,162000"]

        msg = ("Both radius and radii_by_lead_time have been set. "
               "Only one option should be specified.")
        with self.assertRaisesRegex(TypeError, msg):
            radius_or_radii_and_lead(radius, radii_lead)


class Test_make_cubes_non_lazy(unittest.TestCase):
    """Tests make_cubes_non_lazy loads cubes data."""

    def setUp(self):
        self.directory = mkdtemp()
        self.filepath = os.path.join(self.directory, "temp.nc")
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        save_netcdf(cube, self.filepath)
        self.cube = load_cube(self.filepath)

    def test_single_cube_made_non_lazy(self):
        """Test that a single cube which initially contains lazy data is loaded
        fully by use of the make_cubes_non_lazy function."""

        self.assertTrue(self.cube.has_lazy_data())
        make_cubes_non_lazy(self.cube)
        self.assertFalse(self.cube.has_lazy_data())

    def test_multiple_cubes_made_non_lazy(self):
        """Test that a list of cubes which initially contain lazy data are
        loaded fully by use of the make_cubes_non_lazy function."""

        cubes = [self.cube.copy(), self.cube.copy()]
        self.assertTrue(cubes[0].has_lazy_data())
        self.assertTrue(cubes[1].has_lazy_data())
        make_cubes_non_lazy(cubes)
        self.assertFalse(cubes[0].has_lazy_data())
        self.assertFalse(cubes[1].has_lazy_data())


if __name__ == '__main__':
    unittest.main()
