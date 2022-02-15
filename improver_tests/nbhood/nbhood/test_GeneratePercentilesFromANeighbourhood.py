# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Unit tests for the
nbhood.circular_kernel.GeneratePercentilesFromANeighbourhood
plugin."""


import unittest
from datetime import datetime

import iris
import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.constants import DEFAULT_PERCENTILES
from improver.nbhood.circular_kernel import GeneratePercentilesFromANeighbourhood
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)


class Test_make_percentile_cube(IrisTest):

    """Test the make_percentile_cube method from
       GeneratePercentilesFromANeighbourhood."""

    def setUp(self):
        """Set up a 2D cube."""

        data = np.ones((5, 5), dtype=np.float32)
        data[2, 2] = 0
        self.cube = set_up_variable_cube(data, spatial_grid="equalarea",)

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""

        result = GeneratePercentilesFromANeighbourhood(2000).make_percentile_cube(
            self.cube
        )
        self.assertIsInstance(result, Cube)

    def test_coord_present(self):
        """Test that the percentile coord is added."""

        result = GeneratePercentilesFromANeighbourhood(2000).make_percentile_cube(
            self.cube
        )
        expected_data = self.cube.data.copy()

        self.assertIsInstance(result.coord("percentile"), iris.coords.Coord)
        self.assertArrayEqual(result.coord("percentile").points, DEFAULT_PERCENTILES)
        self.assertArrayEqual(result[0].data, expected_data)
        self.assertDictEqual(self.cube.metadata._asdict(), result.metadata._asdict())

    def test_coord_is_dim_vector(self):
        """Test that the percentile coord is added as the zeroth dimension when
        multiple percentiles are used."""

        result = GeneratePercentilesFromANeighbourhood(2000).make_percentile_cube(
            self.cube
        )
        self.assertEqual(result.coord_dims("percentile")[0], 0)

    def test_coord_is_dim_scalar(self):
        """Test that the percentile coord is added as the zeroth dimension when
        a single percentile is used."""

        result = GeneratePercentilesFromANeighbourhood(
            2000, percentiles=[50.0]
        ).make_percentile_cube(self.cube)
        self.assertEqual(result.coord_dims("percentile")[0], 0)


class Test_pad_and_unpad_cube(IrisTest):

    """Test the padding and unpadding of the data within a cube."""

    def setUp(self):
        """Set up a 2D cube."""

        data = np.ones((5, 5), dtype=np.float32)
        self.cube = set_up_variable_cube(data, spatial_grid="equalarea",)

    def test_2d_slice(self):
        """Test a 2D slice."""

        expected = np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.4, 1.0, 1.0],
                    [1.0, 0.4, 0.4, 0.4, 1.0],
                    [1.0, 1.0, 0.4, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ],
        )
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        self.cube.data[2, 2] = 0
        plugin = GeneratePercentilesFromANeighbourhood(2000)
        plugin.percentiles = np.array([10, 50, 90])
        result = plugin.pad_and_unpad_cube(self.cube, kernel)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_irregular_kernel(self):
        """Test a 2d slice."""
        expected = np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.3, 1.0, 1.0, 1.0],
                    [1.0, 0.3, 1.0, 0.3, 1.0],
                    [1.0, 1.0, 0.3, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ],
        )
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        self.cube.data[2, 2] = 0
        plugin = GeneratePercentilesFromANeighbourhood(2000)
        plugin.percentiles = np.array([10, 50, 90])
        result = plugin.pad_and_unpad_cube(self.cube, kernel)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_almost_edge(self):
        """Test behaviour for a non-zero grid cell quite near the edge."""

        data = np.ones((3, 3), dtype=np.float32)
        data[1, 1] = 0
        cube = set_up_variable_cube(data, spatial_grid="equalarea",)

        # Just within range of the edge.

        expected = np.array(
            [
                [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        )
        percentiles = np.array([10, 50, 90])
        kernel = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        result = GeneratePercentilesFromANeighbourhood(
            2000, percentiles=percentiles
        ).pad_and_unpad_cube(cube, kernel)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_adjacent_edge(self):
        """Test behaviour for a single non-zero grid cell near the edge."""

        self.cube.data[2, 1] = 0

        # Range 3 goes over the edge

        expected = np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.4, 1.0, 1.0, 1.0],
                    [0.4, 0.4, 0.4, 1.0, 1.0],
                    [1.0, 0.4, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )
        percentiles = np.array([10, 50, 90])
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        result = GeneratePercentilesFromANeighbourhood(
            2000, percentiles=percentiles
        ).pad_and_unpad_cube(self.cube, kernel)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_on_edge(self):
        """Test behaviour for a non-zero grid cell on the edge."""

        expected = np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.4, 1, 1.0, 1.0, 1.0],
                    [0.0, 0.4, 1.0, 1.0, 1.0],
                    [0.4, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        self.cube.data[2, 0] = 0

        percentiles = np.array([10, 50, 90])
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        result = GeneratePercentilesFromANeighbourhood(
            2000, percentiles=percentiles
        ).pad_and_unpad_cube(self.cube, kernel)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_on_corner(self):
        """Test behaviour for a single non-zero grid cell on the corner."""

        expected = np.array(
            [
                [
                    [0.0, 0.4, 1.0, 1.0, 1.0],
                    [0.4, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        self.cube.data[0, 0] = 0

        # Point is right on the corner.
        percentiles = np.array([10, 50, 90])
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        result = GeneratePercentilesFromANeighbourhood(
            2000, percentiles=percentiles
        ).pad_and_unpad_cube(self.cube, kernel)
        self.assertArrayAlmostEqual(result.data, expected)


class Test_process(IrisTest):

    """Test the process method within the plugin to calculate percentile values
    from a neighbourhood."""

    def setUp(self):
        """Set up a 2D cube."""

        data = np.ones((5, 5), dtype=np.float32)
        self.cube = set_up_variable_cube(data, spatial_grid="equalarea",)

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""

        self.cube.data[2, 2] = 0
        radius = 4000.0
        result = GeneratePercentilesFromANeighbourhood(radius).process(self.cube)
        self.assertIsInstance(result, Cube)

    def test_single_point(self):
        """Test behaviour for a single non-zero grid cell."""

        expected = np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.4, 1.0, 1.0],
                    [1.0, 0.4, 0.4, 0.4, 1.0],
                    [1.0, 1.0, 0.4, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        data = np.ones((5, 5), dtype=np.float32)
        data[2, 2] = 0
        cube = set_up_variable_cube(data, spatial_grid="equalarea",)
        percentiles = np.array([10, 50, 90])
        radius = 2000.0
        result = GeneratePercentilesFromANeighbourhood(
            radius, percentiles=percentiles
        ).process(cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_multi_point_multitimes(self):
        """Test behaviour for points over multiple times."""

        data = np.ones((5, 5), dtype=np.float32)
        cube = set_up_variable_cube(data, spatial_grid="equalarea",)
        time_points = [
            datetime(2017, 11, 10, 2),
            datetime(2017, 11, 10, 3),
        ]
        cube = add_coordinate(
            cube, coord_points=time_points, coord_name="time", is_datetime="true",
        )
        cube.data[0, 2, 2] = 0
        cube.data[1, 2, 1] = 0

        expected = np.array(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 0.4, 1.0, 1.0],
                        [1.0, 0.4, 0.4, 0.4, 1.0],
                        [1.0, 1.0, 0.4, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.4, 1.0, 1.0, 1.0],
                        [0.4, 0.4, 0.4, 1.0, 1.0],
                        [1.0, 0.4, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                ],
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                ],
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                ],
            ]
        )
        percentiles = np.array([10, 50, 90])
        radius = 2000.0
        result = GeneratePercentilesFromANeighbourhood(
            radius, percentiles=percentiles
        ).process(cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_lat_long(self):
        """Test behaviour for a single grid cell on lat long grid."""

        data = np.ones((16, 16), dtype=np.float32)
        data[7, 7] = 0
        cube = set_up_variable_cube(data, spatial_grid="latlon",)

        msg = "Unable to convert from"
        radius = 6000.0
        with self.assertRaisesRegex(ValueError, msg):
            GeneratePercentilesFromANeighbourhood(radius).process(cube)

    def test_single_point_masked_to_null(self):
        """Test behaviour with a masked non-zero point.
        The underlying numpy.percentile base behaviour does not support
        masked arrays so raise an error.
        """
        mask = np.zeros_like(self.cube.data)
        mask[2, 2] = 1
        self.cube.data = np.ma.masked_array(self.cube.data, mask=mask)
        percentiles = np.array([10, 50, 90])
        radius = 2000.0
        message = (
            "The use of masked input cubes is not yet implemented in "
            "the GeneratePercentilesFromANeighbourhood plugin."
        )
        with self.assertRaisesRegex(NotImplementedError, message):
            GeneratePercentilesFromANeighbourhood(
                radius, percentiles=percentiles
            ).process(self.cube)

    def test_single_point_low_percentiles(self):
        """Test behaviour with low percentiles."""

        expected = np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.2, 1.0, 1.0],
                    [1.0, 0.2, 0.2, 0.2, 1.0],
                    [1.0, 1.0, 0.2, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.4, 1.0, 1.0],
                    [1.0, 0.4, 0.4, 0.4, 1.0],
                    [1.0, 1.0, 0.4, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.8, 1.0, 1.0],
                    [1.0, 0.8, 0.8, 0.8, 1.0],
                    [1.0, 1.0, 0.8, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )
        self.cube.data[2, 2] = 0

        percentiles = np.array([5, 10, 20])
        radius = 2000.0
        result = GeneratePercentilesFromANeighbourhood(
            radius, percentiles=percentiles
        ).process(self.cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_point_pair(self):
        """Test behaviour for two nearby non-zero grid cells."""

        expected = np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )
        self.cube.data[2, 2] = 0
        self.cube.data[2, 1] = 0

        percentiles = np.array([25, 50, 75])
        radius = 2000.0
        result = GeneratePercentilesFromANeighbourhood(
            radius, percentiles=percentiles
        ).process(self.cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_number_of_percentiles_equals_number_of_points(self):
        """Test when the number of percentiles is equal to the number of points
        used to construct the percentiles."""

        expected = np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.4, 0.4, 0.4, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.6, 0.6, 0.6, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.8, 0.8, 0.8, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        data = np.ones((7, 7), dtype=np.float32)
        data[3, 3] = 0
        cube = set_up_variable_cube(data, spatial_grid="equalarea",)
        percentiles = np.array([5, 10, 15, 20, 25])
        radius = 2000.0
        result = GeneratePercentilesFromANeighbourhood(
            radius, percentiles=percentiles
        ).process(cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_number_of_points_half_of_number_of_percentiles(self):
        """Test when the number of points is half the number of percentiles."""

        expected = np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.3, 0.3, 0.3, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.4, 0.4, 0.4, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.6, 0.6, 0.6, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.7, 0.7, 0.7, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.8, 0.8, 0.8, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        data = np.ones((7, 7), dtype=np.float32)
        data[3, 3] = 0
        cube = set_up_variable_cube(data, spatial_grid="equalarea",)

        percentiles = np.array([2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25])
        radius = 2000.0
        result = GeneratePercentilesFromANeighbourhood(
            radius, percentiles=percentiles
        ).process(cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_circle_bigger_than_domain(self):
        """Test that an exception is raised if the circle requested is bigger
        than the size of the domain."""

        self.cube.data[2, 2] = 0
        radius = 50000.0
        msg = "Distance of {}m exceeds max domain distance".format(radius)
        with self.assertRaisesRegex(ValueError, msg):
            GeneratePercentilesFromANeighbourhood(radius).process(self.cube)


if __name__ == "__main__":
    unittest.main()
