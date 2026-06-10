# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Extends unittest.TestCase class with additional useful tests."""

import unittest

import numpy as np
from iris.cube import Cube, CubeList


class ImproverTest(unittest.TestCase):
    """Extends unittest.TestCase with a method for comparing cubes and cubelists"""

    def assertCubeEqual(self, cube_a: Cube, cube_b: Cube):
        """Uses Cube.xml method to create an easily-comparable string containing all
        meta-data and data"""
        self.assertEqual(
            cube_a.xml(checksum=True, order=False, byteorder=False),
            cube_b.xml(checksum=True, order=False, byteorder=False),
        )

    def assertCubeListEqual(self, cubelist_a: CubeList, cubelist_b: CubeList):
        """Uses CubeList.xml method to create an easily-comparable string containing all
        meta-data and data"""
        self.assertEqual(
            cubelist_a.xml(checksum=True, order=False, byteorder=False),
            cubelist_b.xml(checksum=True, order=False, byteorder=False),
        )

    def assertDictEqual(self, dict_a, dict_b):
        """Asserts that two dictionaries are equal. Improves on the default unittest
        assertDictEqual method to allow handling of numpy arrays / lists as values."""
        assert set(dict_a.keys()) == set(dict_b.keys())
        for key in dict_a.keys():
            try:
                assert dict_a[key] == dict_b[key]
            except ValueError:
                np.testing.assert_array_equal(dict_a[key], dict_b[key])
