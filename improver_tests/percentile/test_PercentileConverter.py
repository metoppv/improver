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
"""Unit tests for the percentile.PercentileConverter plugin."""

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.percentile import PercentileConverter
from improver.utilities.warnings_handler import ManageWarnings

from ..set_up_test_cubes import set_up_variable_cube


class Test_process(IrisTest):

    """Test the creation of percentiles by the plugin."""

    def setUp(self):
        """Create a cube with collapsable coordinates.

        Data is formatted to increase linearly in x/y dimensions,
        e.g.
              0 0 0 0
              1 1 1 1
              2 2 2 2
              3 3 3 3

        """
        data = [[list(range(0, 11, 1))]*11]*3
        data = np.array(data).astype(np.float32)
        data.resize((3, 11, 11))
        self.cube = set_up_variable_cube(data, realizations=[0, 1, 2])
        self.default_percentiles = np.array([0, 5, 10, 20, 25, 30, 40, 50,
                                             60, 70, 75, 80, 90, 95, 100])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_valid_single_coord_string(self):
        """Test that the plugin handles a valid collapse_coord passed in
        as a string."""

        collapse_coord = 'longitude'

        plugin = PercentileConverter(collapse_coord)
        result = plugin.process(self.cube)

        # Check percentile values.
        self.assertArrayAlmostEqual(result.data[:, 0, 0],
                                    self.default_percentiles*0.1)
        # Check coordinate name.
        self.assertEqual(result.coords()[0].name(),
                         'percentile')
        # Check coordinate units.
        self.assertEqual(result.coords()[0].units, '%')
        # Check coordinate points.
        self.assertArrayEqual(
            result.coord('percentile').points,
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100])
        # Check resulting data shape.
        self.assertEqual(result.data.shape, (15, 3, 11))

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_valid_single_coord_string_for_time(self):
        """Test that the plugin handles time being the collapse_coord that is
        passed in as a string."""
        data = [[list(range(1, 12, 1))]*11]*3
        data = np.array(data).astype(np.float32)
        data.resize((3, 11, 11))
        new_cube = set_up_variable_cube(
            data, time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0), realizations=[0, 1, 2])
        cube = iris.cube.CubeList([self.cube, new_cube]).merge_cube()

        collapse_coord = 'time'

        plugin = PercentileConverter(collapse_coord)
        result = plugin.process(cube)

        # Check percentile values.
        self.assertArrayAlmostEqual(result.data[:, 0, 0, 0],
                                    self.default_percentiles*0.01)
        # Check coordinate name.
        self.assertEqual(result.coords()[0].name(),
                         'percentile')
        # Check coordinate units.
        self.assertEqual(result.coords()[0].units, '%')
        # Check coordinate points.
        self.assertArrayEqual(
            result.coord('percentile').points,
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100])
        # Check resulting data shape.
        self.assertEqual(result.data.shape, (15, 3, 11, 11))

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_valid_multi_coord_string_list(self):
        """Test that the plugin handles a valid list of collapse_coords passed
        in as a list of strings."""

        collapse_coord = ['longitude', 'latitude']

        plugin = PercentileConverter(collapse_coord)
        result = plugin.process(self.cube)

        # Check percentile values.
        self.assertArrayAlmostEqual(
            result.data[:, 0], [0., 0., 1., 2., 2., 3., 4., 5., 6., 7., 8.,
                                8., 9., 10., 10.])
        # Check coordinate name.
        self.assertEqual(result.coords()[0].name(),
                         'percentile')
        # Check coordinate units.
        self.assertEqual(result.coords()[0].units, '%')
        # Check coordinate points.
        self.assertArrayEqual(
            result.coord('percentile').points,
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100])
        # Check resulting data shape.
        self.assertEqual(result.data.shape, (15, 3))

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_use_with_masked_data(self):
        """Test that the plugin handles masked data, this requiring the option
        fast_percentile_method=False."""

        mask = np.zeros((3, 11, 11))
        mask[:, :, 1:-1:2] = 1
        masked_data = np.ma.array(self.cube.data, mask=mask)
        cube = self.cube.copy(data=masked_data)
        collapse_coord = 'longitude'

        plugin = PercentileConverter(collapse_coord,
                                     fast_percentile_method=False)
        result = plugin.process(cube)

        # Check percentile values.
        self.assertArrayAlmostEqual(result.data[:, 0, 0],
                                    self.default_percentiles*0.1)
        # Check coordinate name.
        self.assertEqual(result.coords()[0].name(),
                         'percentile')
        # Check coordinate units.
        self.assertEqual(result.coords()[0].units, '%')
        # Check coordinate points.
        self.assertArrayEqual(
            result.coord('percentile').points,
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100])
        # Check resulting data shape.
        self.assertEqual(result.data.shape, (15, 3, 11))

    def test_unavailable_collapse_coord(self):
        """Test that the plugin handles a collapse_coord that is not
        available in the cube."""

        collapse_coord = 'not_a_coordinate'
        plugin = PercentileConverter(collapse_coord)
        msg = "Coordinate "
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            plugin.process(self.cube)

    def test_invalid_collapse_coord_type(self):
        """Test that the plugin handles invalid collapse_coord type."""

        collapse_coord = self.cube
        msg = "collapse_coord is "
        with self.assertRaisesRegex(TypeError, msg):
            PercentileConverter(collapse_coord)


if __name__ == '__main__':
    unittest.main()
