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
"""Unit tests for the percentile.PercentileConverter plugin."""


import unittest

from cf_units import Unit
from iris.cube import Cube
from iris.coords import DimCoord
from iris.tests import IrisTest
from iris.exceptions import CoordinateNotFoundError
import numpy as np

from improver.percentile import PercentileConverter
from improver.utilities.warnings_handler import ManageWarnings


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
        data = [[range(0, 11, 1)]*11]*3
        data = np.array(data).astype('float32')
        data.resize(3, 1, 11, 11)

        realization = DimCoord([0, 1, 2], 'realization', units=1)
        time = DimCoord([402192.5], standard_name='time',
                        units=Unit('hours since 1970-01-01 00:00:00',
                                   calendar='gregorian'))
        latitude = DimCoord(np.linspace(-90, 90, 11),
                            standard_name='latitude', units='degrees')
        longitude = DimCoord(np.linspace(-180, 180, 11),
                             standard_name='longitude', units='degrees')

        cube = Cube(data, standard_name="air_temperature",
                    dim_coords_and_dims=[(realization, 0),
                                         (time, 1),
                                         (latitude, 2),
                                         (longitude, 3)],
                    units="K")

        self.cube = cube
        self.longitude = longitude
        self.latitude = latitude
        self.default_percentiles = np.array([0, 5, 10, 20, 25, 30, 40, 50,
                                             60, 70, 75, 80, 90, 95, 100])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_valid_single_coord_string(self):
        """
        Test that the plugin handles a valid collapse_coord passed in
        as a string.

        """
        collapse_coord = 'longitude'

        plugin = PercentileConverter(collapse_coord)
        result = plugin.process(self.cube)

        # Check percentile values.
        self.assertArrayAlmostEqual(result.data[:, 0, 0, 0],
                                    self.default_percentiles*0.1)
        # Check coordinate name.
        self.assertEqual(result.coords()[0].name(),
                         'percentile_over_longitude')
        # Check coordinate points.
        self.assertArrayEqual(
            result.coord('percentile_over_longitude').points,
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100])
        # Check resulting data shape.
        self.assertEqual(result.data.shape, (15, 3, 1, 11))
        # Check demoted longitude coordinate exists as scalar with bounds.
        self.assertArrayEqual(result.coord('longitude').bounds,
                              [[-180., 180.]])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_valid_multi_coord_string_list(self):
        """
        Test that the plugin handles a valid list of collapse_coords passed in
        as a list of strings.

        """
        collapse_coord = ['longitude', 'latitude']

        plugin = PercentileConverter(collapse_coord)
        result = plugin.process(self.cube)

        # Check percentile values.
        self.assertArrayAlmostEqual(
            result.data[:, 0, 0], [0., 0., 1., 2., 2., 3., 4., 5., 6., 7., 8.,
                                   8., 9., 10., 10.])
        # Check coordinate name.
        self.assertEqual(result.coords()[0].name(),
                         'percentile_over_latitude_longitude')
        # Check coordinate points.
        self.assertArrayEqual(
            result.coord('percentile_over_latitude_longitude').points,
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100])
        # Check resulting data shape.
        self.assertEqual(result.data.shape, (15, 3, 1))
        # Check demoted dimension coordinates exists as scalars with bounds.
        self.assertArrayEqual(result.coord('longitude').bounds,
                              [[-180., 180.]])
        self.assertArrayEqual(result.coord('latitude').bounds, [[-90., 90.]])

    def test_unavailable_collapse_coord(self):
        """
        Test that the plugin handles a collapse_coord that is not
        available in the cube.

        """
        collapse_coord = 'not_a_coordinate'
        plugin = PercentileConverter(collapse_coord)
        msg = "Coordinate "
        with self.assertRaisesRegexp(CoordinateNotFoundError, msg):
            plugin.process(self.cube)

    def test_invalid_collapse_coord_type(self):
        """
        Test that the plugin handles invalid collapse_coord type.

        """
        collapse_coord = self.cube
        msg = "collapse_coord is "
        with self.assertRaisesRegexp(TypeError, msg):
            PercentileConverter(collapse_coord)


if __name__ == '__main__':
    unittest.main()
