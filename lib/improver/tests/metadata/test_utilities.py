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
"""Tests for the improver.metadata.utilities module"""

import unittest
import numpy as np

from improver.metadata.utilities import generate_hash, create_coordinate_hash
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_generate_hash(unittest.TestCase):
    """Test utility to generate md5 hash codes from a multitude of inputs."""

    def test_string_input(self):
        """Test the expected hash is returned when input is a string type."""

        hash_input = 'this is a test string'
        result = generate_hash(hash_input)
        expected = "8e502f6a5b4a2e0f226649210895cebc"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_numeric_input(self):
        """Test the expected hash is returned when input is a numeric type."""

        hash_input = 1000
        result = generate_hash(hash_input)
        expected = "d017763f19ef64f920c43fc57413d171"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_dictionary_input(self):
        """Test the expected hash is returned when input is a dictionary."""

        hash_input = {'one': 1, 'two': 2}
        result = generate_hash(hash_input)
        expected = "4735f4a74dd17d27b383de504a87e324"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_dictionary_order_variant(self):
        """Test the expected hash is different if the dictionary order is
        different."""

        hash_input1 = {'one': 1, 'two': 2}
        hash_input2 = {'two': 2, 'one': 1}
        result1 = generate_hash(hash_input1)
        result2 = generate_hash(hash_input2)
        self.assertNotEqual(result1, result2)

    def test_cube_input(self):
        """Test the expected hash is returned when input is a cube."""

        hash_input = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        result = generate_hash(hash_input)
        expected = "ad664992debed0bdf8f20804e4164691"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_coordinate_input(self):
        """Test the expected hash is returned when input is a cube
        coordinate."""

        cube = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        hash_input = cube.coord('latitude')
        result = generate_hash(hash_input)
        expected = "8c8846a4be49f7aab487353d9ecf623c"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_numpy_array_type_variant(self):
        """Test the expected hash is different if the numpy array type is
        different."""

        hash_input32 = np.array([np.sqrt(2.)], dtype=np.float32)
        hash_input64 = np.array([np.sqrt(2.)], dtype=np.float64)
        result32 = generate_hash(hash_input32)
        result64 = generate_hash(hash_input64)
        self.assertNotEqual(result32, result64)

    def test_equivalent_input_gives_equivalent_hash(self):
        """Test that creating a hash twice using the same input results in the
        same hash being generated."""

        cube = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        hash_input = cube.coord('latitude')
        result1 = generate_hash(hash_input)
        result2 = generate_hash(hash_input)
        self.assertEqual(result1, result2)


class Test_create_coordinate_hash(unittest.TestCase):
    """Test wrapper to hash generation to return a hash based on the x and y
    coordinates of a given cube."""

    def test_basic(self):
        """Test the expected hash is returned for a given cube."""

        hash_input = set_up_variable_cube(np.zeros((3, 3)).astype(np.float32))
        result = create_coordinate_hash(hash_input)
        expected = "fd40f6d5a8e0a347f181d87bcfd445fa"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_variation(self):
        """Test that two cubes with slightly different coordinates return
        different hashes."""

        hash_input1 = set_up_variable_cube(np.zeros((3, 3)).astype(np.float32))
        hash_input2 = hash_input1.copy()
        latitude = hash_input2.coord('latitude')
        latitude_values = latitude.points * 1.001
        latitude = latitude.copy(points=latitude_values)
        hash_input2.remove_coord("latitude")
        hash_input2.add_dim_coord(latitude, 0)

        result1 = create_coordinate_hash(hash_input1)
        result2 = create_coordinate_hash(hash_input2)
        self.assertNotEqual(result1, result2)


if __name__ == '__main__':
    unittest.main()
