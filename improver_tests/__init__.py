# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Extends IrisTest class with additional useful tests."""

from iris.cube import Cube, CubeList
from iris.tests import IrisTest


class ImproverTest(IrisTest):
    """Extends IrisTest with a method for comparing cubes and cubelists"""

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
