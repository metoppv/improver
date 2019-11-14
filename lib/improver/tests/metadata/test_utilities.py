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
import iris
import numpy as np

from improver.metadata.utilities import (
    create_new_diagnostic_cube, generate_hash, create_coordinate_hash)
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_create_new_diagnostic_cube(unittest.TestCase):
    """Test utility to create new diagnostic cubes"""

    def setUp(self):
        """Set up template with data, coordinates, attributes and cell
        methods"""
        self.template_cube = set_up_variable_cube(
            280*np.ones((3, 5, 5), dtype=np.float32),
            standard_grid_metadata='uk_det')
        self.template_cube.add_cell_method('time (max): 1 hour')
        self.name = "lwe_precipitation_rate"
        self.units = "mm h-1"

    def test_basic(self):
        """Test result is a cube that inherits coordinates only"""
        result = create_new_diagnostic_cube(
            self.name, self.units, self.template_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.standard_name, "lwe_precipitation_rate")
        self.assertEqual(result.units, "mm h-1")
        self.assertSequenceEqual(result.coords(dim_coords=True),
                                 self.template_cube.coords(dim_coords=True))
        self.assertSequenceEqual(result.coords(dim_coords=False),
                                 self.template_cube.coords(dim_coords=False))
        self.assertFalse(np.allclose(result.data, self.template_cube.data))
        self.assertFalse(result.attributes)
        self.assertFalse(result.cell_methods)
        self.assertEqual(result.data.dtype, np.float32)

    def test_attributes(self):
        """Test attributes can be set on the output cube"""
        attributes = {"source": "IMPROVER"}
        result = create_new_diagnostic_cube(
            self.name, self.units, self.template_cube, attributes=attributes)
        self.assertDictEqual(result.attributes, attributes)

    def test_data(self):
        """Test data can be set on the output cube"""
        data = np.arange(3*5*5).reshape((3, 5, 5)).astype(np.float32)
        result = create_new_diagnostic_cube(
            self.name, self.units, self.template_cube, data=data)
        self.assertTrue(np.allclose(result.data, data))

    def test_dtype(self):
        """Test dummy data of a different type can be set"""
        result = create_new_diagnostic_cube(
            self.name, self.units, self.template_cube, dtype=np.int32)
        self.assertEqual(result.data.dtype, np.int32)

    def test_non_standard_name(self):
        """Test cube can be created with a non-CF-standard name"""
        result = create_new_diagnostic_cube(
            "RainRate Composite", self.units, self.template_cube)
        self.assertEqual(result.long_name, "RainRate Composite")
        self.assertIsNone(result.standard_name)


class Test_generate_hash(unittest.TestCase):
    """Test utility to generate md5 hash codes from a multitude of inputs."""

    def test_string_input(self):
        """Test the expected hash is returned when input is a string type."""

        hash_input = 'this is a test string'
        result = generate_hash(hash_input)
        expected = (
            "7a5a4f1716b08d290d5782da904cc076315376889e9bf641ae527889704fd314"
        )
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_numeric_input(self):
        """Test the expected hash is returned when input is a numeric type."""

        hash_input = 1000
        result = generate_hash(hash_input)
        expected = (
            "40510175845988f13f6162ed8526f0b09f73384467fa855e1e79b44a56562a58"
        )
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_dictionary_input(self):
        """Test the expected hash is returned when input is a dictionary."""

        hash_input = {'one': 1, 'two': 2}
        result = generate_hash(hash_input)
        expected = (
            "c261139b6339f880f4f75a3bf5a08f7c2d6f208e2720760f362e4464735e3845"
        )
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_dictionary_order_invariant(self):
        """Test the expected hash is the same for different dict ordering."""

        hash_input1 = {'one': 1, 'two': 2}
        hash_input2 = {'two': 2, 'one': 1}
        result1 = generate_hash(hash_input1)
        result2 = generate_hash(hash_input2)
        self.assertEqual(result1, result2)

    def test_cube_input(self):
        """Test the expected hash is returned when input is a cube."""

        hash_input = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        result = generate_hash(hash_input)
        expected = (
            "4d82994200559c90234b0186bccc59b9b9d6436284f29bab9a15dc97172d1fb8"
        )
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_coordinate_input(self):
        """Test the expected hash is returned when input is a cube
        coordinate."""

        cube = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        hash_input = cube.coord('latitude')
        result = generate_hash(hash_input)
        expected = (
            "62267c5827656790244ef2f26b708a1be5dcb491e4ae36b9db9b47c2aaaadf7e"
        )
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
        expected = (
            "b26ca16d28f6e06ea4573fd745f55750c6dd93a06891f1b4ff0c6cd50585ac08"
        )
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
