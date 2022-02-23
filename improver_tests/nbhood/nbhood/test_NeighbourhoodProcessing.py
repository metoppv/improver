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
"""Unit tests for the nbhood.NeighbourhoodProcessing plugin."""


import unittest

import numpy as np
from iris.coords import CellMethod
from iris.cube import Cube
from iris.tests import IrisTest

from improver.nbhood.square_kernel import NeighbourhoodProcessing
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube


class Test__init__(IrisTest):

    """Test the __init__ method of NeighbourhoodProcessing."""

    def test_neighbourhood_method_does_not_exist(self):
        """Test that desired error message is raised, if the neighbourhood
        method does not exist."""
        neighbourhood_method = "nonsense"
        radii = 10000
        msg = "nonsense is not a valid neighbourhood_method"
        with self.assertRaisesRegex(ValueError, msg):
            NeighbourhoodProcessing(neighbourhood_method, radii)

    def test_square_nbhood_with_weighted_mode(self):
        """Test that desired error message is raised, if the neighbourhood
        method is square and the weighted_mode option is used."""
        radii = 10000
        msg = "weighted_mode can only be used if neighbourhood_method is circular"
        with self.assertRaisesRegex(ValueError, msg):
            NeighbourhoodProcessing("square", radii, weighted_mode=True)


class Test__calculate_neighbourhood(IrisTest):

    """Test the _calculate_neighbourhood method."""

    RADIUS = 2500

    def setUp(self):
        """Set up data arrays and kernels"""

        self.data = np.ones((5, 5), dtype=np.float32)
        self.data[2, 2] = 0
        self.nbhood_size = 3
        self.circular_kernel = np.array(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
        )
        self.weighted_circular_kernel = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.75, 0.5, 0.0],
                [0.0, 0.75, 1.0, 0.75, 0.0],
                [0.0, 0.5, 0.75, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        # set up data for tests using masks
        self.data_for_masked_tests = np.array(
            [
                [1.0, 1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 1.0],
            ]
        )
        self.mask = np.array(
            [
                [0, 0, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
            ]
        )
        self.expected_array = np.array(
            [
                [1.0000, 0.666667, 0.600000, 0.500000, 0.50],
                [1.0000, 0.750000, 0.571429, 0.428571, 0.25],
                [1.0000, 1.000000, 0.714286, 0.571429, 0.25],
                [np.nan, 1.000000, 0.666667, 0.571429, 0.25],
                [np.nan, 1.000000, 0.750000, 0.750000, 0.50],
            ]
        )
        self.expected_mask = ~self.mask.astype(bool)

    def test_basic_square(self):
        """Test the _calculate_neighbourhood method with a square neighbourhood."""
        expected_array = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 8 / 9, 8 / 9, 8 / 9, 1.0],
                [1.0, 8 / 9, 8 / 9, 8 / 9, 1.0],
                [1.0, 8 / 9, 8 / 9, 8 / 9, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        plugin = NeighbourhoodProcessing("square", self.RADIUS)
        plugin.nb_size = self.nbhood_size
        result = plugin._calculate_neighbourhood(self.data)
        self.assertArrayAlmostEqual(result, expected_array)

    def test_basic_circular(self):
        """Test the _calculate_neighbourhood method with a circular neighbourhood."""
        expected_array = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.8, 1.0, 1.0],
                [1.0, 0.8, 0.8, 0.8, 1.0],
                [1.0, 1.0, 0.8, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        plugin = NeighbourhoodProcessing("circular", self.RADIUS)
        plugin.kernel = self.circular_kernel
        result = plugin._calculate_neighbourhood(self.data)
        self.assertArrayAlmostEqual(result.data, expected_array)

    def test_basic_weighted_circular(self):
        """Test the _calculate_neighbourhood method with a
        weighted circular neighbourhood."""
        expected_array = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.916667, 0.875, 0.916667, 1.0],
                [1.0, 0.875, 0.833333, 0.875, 1.0],
                [1.0, 0.916667, 0.875, 0.916667, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        plugin = NeighbourhoodProcessing("circular", self.RADIUS)
        plugin.kernel = self.weighted_circular_kernel
        result = plugin._calculate_neighbourhood(self.data)
        self.assertArrayAlmostEqual(result.data, expected_array)

    def test_basic_square_sum(self):
        """Test the _calculate_neighbourhood method calculating a sum in
        a square neighbourhood."""
        expected_array = np.array(
            [
                [4.0, 6.0, 6.0, 6.0, 4.0],
                [6.0, 8.0, 8.0, 8.0, 6.0],
                [6.0, 8.0, 8.0, 8.0, 6.0],
                [6.0, 8.0, 8.0, 8.0, 6.0],
                [4.0, 6.0, 6.0, 6.0, 4.0],
            ]
        )
        plugin = NeighbourhoodProcessing("square", self.RADIUS, sum_only=True)
        plugin.nb_size = self.nbhood_size
        result = plugin._calculate_neighbourhood(self.data)
        self.assertArrayAlmostEqual(result, expected_array)

    def test_basic_circular_sum(self):
        """Test the _calculate_neighbourhood method calculating a sum in
        a circular neighbourhood."""
        expected_array = np.array(
            [
                [5.0, 5.0, 5.0, 5.0, 5.0],
                [5.0, 5.0, 4.0, 5.0, 5.0],
                [5.0, 4.0, 4.0, 4.0, 5.0],
                [5.0, 5.0, 4.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0, 5.0],
            ]
        )
        plugin = NeighbourhoodProcessing("circular", self.RADIUS, sum_only=True)
        plugin.kernel = self.circular_kernel
        result = plugin._calculate_neighbourhood(self.data)
        self.assertArrayAlmostEqual(result.data, expected_array)

    def test_masked_array_re_mask_true_square(self):
        """Test the _calculate_neighbourhood method when masked data is
        passed in and re-masking is applied."""

        input_data = np.ma.masked_where(self.mask == 0, self.data_for_masked_tests)
        plugin = NeighbourhoodProcessing("square", self.RADIUS)
        plugin.nb_size = self.nbhood_size
        result = plugin._calculate_neighbourhood(input_data)
        self.assertArrayAlmostEqual(result.data, self.expected_array)
        self.assertArrayAlmostEqual(result.mask, self.expected_mask)

    def test_masked_array_re_mask_true_circular(self):
        """Test the _calculate_neighbourhood method when masked data is
        passed in and re-masking is applied with a circular neighbourhood."""

        expected_array = np.array(
            [
                [np.nan, 0.5, 0.5, 0.5, 1.0],
                [1.0, 1.0, 0.6, 0.5, 0.0],
                [np.nan, 1.0, 0.75, 0.4, 0.0],
                [np.nan, 1.0, 1.0, 0.5, 0.5],
                [np.nan, 1.0, 0.75, 0.5, 0.0],
            ]
        )
        input_data = np.ma.masked_where(self.mask == 0, self.data_for_masked_tests)
        plugin = NeighbourhoodProcessing("circular", self.RADIUS)
        plugin.kernel = self.circular_kernel
        result = plugin._calculate_neighbourhood(input_data)

        self.assertArrayAlmostEqual(result.data, expected_array)
        self.assertArrayAlmostEqual(result.mask, self.expected_mask)

    def test_masked_array_re_mask_false(self):
        """Test the _calculate_neighbourhood method when masked data is
        passed in and re-masking is not applied."""

        input_data = np.ma.masked_where(self.mask == 0, self.data_for_masked_tests)
        plugin = NeighbourhoodProcessing("square", self.RADIUS, re_mask=False)
        plugin.nb_size = self.nbhood_size
        result = plugin._calculate_neighbourhood(input_data)
        self.assertArrayAlmostEqual(result, self.expected_array)
        with self.assertRaises(AttributeError):
            result.mask

    def test_masked_array_with_nans_re_mask_true(self):
        """Test the _calculate_neighbourhood method when masked data
        (with masked Nans in) is passed in and re-masking is applied."""
        self.data_for_masked_tests[0, 0] = np.nan
        self.expected_array[0, 0] = np.nan
        input_data = np.ma.masked_where(self.mask == 0, self.data_for_masked_tests)
        plugin = NeighbourhoodProcessing("square", self.RADIUS)
        plugin.nb_size = self.nbhood_size
        result = plugin._calculate_neighbourhood(input_data)
        self.assertArrayAlmostEqual(result, self.expected_array)
        self.assertArrayAlmostEqual(result.mask, self.expected_mask)

    def test_complex(self):
        """Test that data containing complex numbers is sensibly processed"""
        self.data = self.data.astype(complex)
        self.data[1, 3] = 0.5 + 0.5j
        self.data[4, 3] = 0.4 + 0.6j
        expected_array = np.array(
            [
                [
                    1.0 + 0.0j,
                    1.0 + 0.0j,
                    0.91666667 + 0.083333333j,
                    0.91666667 + 0.083333333j,
                    0.875 + 0.125j,
                ],
                [
                    1.0 + 0.0j,
                    0.88888889 + 0.0j,
                    0.83333333 + 0.055555556j,
                    0.83333333 + 0.055555556j,
                    0.91666667 + 0.083333333j,
                ],
                [
                    1.0 + 0.0j,
                    0.88888889 + 0.0j,
                    0.83333333 + 0.055555556j,
                    0.83333333 + 0.055555556j,
                    0.91666667 + 0.083333333j,
                ],
                [
                    1.0 + 0.0j,
                    0.88888889 + 0.0j,
                    0.82222222 + 0.066666667j,
                    0.82222222 + 0.066666667j,
                    0.9 + 0.1j,
                ],
                [1.0 + 0.0j, 1.0 + 0.0j, 0.9 + 0.1j, 0.9 + 0.1j, 0.85 + 0.15j],
            ]
        )
        plugin = NeighbourhoodProcessing("square", self.RADIUS)
        plugin.nb_size = self.nbhood_size
        result = plugin._calculate_neighbourhood(self.data)
        self.assertArrayAlmostEqual(result, expected_array)

    def test_external_mask_square(self):
        """Test the _calculate_neighbourhood method when an external mask is
        passed in and re-masking is applied."""
        plugin = NeighbourhoodProcessing("square", self.RADIUS)
        plugin.nb_size = self.nbhood_size
        result = plugin._calculate_neighbourhood(
            self.data_for_masked_tests, mask=self.mask
        )
        self.assertArrayAlmostEqual(result.data, self.expected_array)
        self.assertArrayAlmostEqual(result.mask, self.expected_mask)

    def test_external_mask_with_masked_data_square(self):
        """Test the _calculate_neighbourhood method when masked data is
        passed in and an external mask is passed in and re-masking is applied."""
        mask = np.array(
            [
                [1, 0, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 0],
                [1, 0, 1, 1, 0],
            ]
        )
        external_mask = np.array(
            [
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
            ]
        )

        self.data = np.ma.masked_where(mask == 0, self.data_for_masked_tests)
        plugin = NeighbourhoodProcessing("square", self.RADIUS)
        plugin.nb_size = self.nbhood_size
        result = plugin._calculate_neighbourhood(self.data, external_mask)
        self.assertArrayAlmostEqual(result.data, self.expected_array)
        self.assertArrayAlmostEqual(result.mask, self.expected_mask)


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        """Set up a cube."""

        data = np.ones((3, 5, 5), dtype=np.float32)
        data[:, 2, 2] = 0
        self.cube = set_up_probability_cube(
            data,
            thresholds=np.array([278, 281, 284], dtype=np.float32),
            spatial_grid="equalarea",
        )

    def test_square_neighbourhood(self):
        """Test that the square neighbourhood processing is successful."""
        nbhood_result = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.88888889, 0.88888889, 0.88888889, 1.0],
                [1.0, 0.88888889, 0.88888889, 0.88888889, 1.0],
                [1.0, 0.88888889, 0.88888889, 0.88888889, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        expected = np.broadcast_to(nbhood_result, (3, 5, 5))
        neighbourhood_method = "square"
        radii = 2000
        result = NeighbourhoodProcessing(neighbourhood_method, radii)(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)
        self.assertTupleEqual(result.cell_methods, self.cube.cell_methods)
        self.assertDictEqual(result.attributes, self.cube.attributes)

    def test_cube_metadata(self):
        """Test the result has the correct attributes and cell methods"""
        neighbourhood_method = "square"
        radii = 2000
        self.cube.attributes = {"Conventions": "CF-1.5"}
        self.cube.add_cell_method(CellMethod("mean", coords="time"))
        result = NeighbourhoodProcessing(neighbourhood_method, radii)(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertTupleEqual(result.cell_methods, self.cube.cell_methods)
        self.assertDictEqual(result.attributes, self.cube.attributes)


if __name__ == "__main__":
    unittest.main()
