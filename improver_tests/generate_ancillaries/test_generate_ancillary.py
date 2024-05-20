# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the the stand alone functions in generate_ancillary.py"""

import unittest

import numpy as np
from cf_units import Unit
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from iris.fileformats.pp import EARTH_RADIUS
from iris.tests import IrisTest

from improver.generate_ancillaries.generate_ancillary import _make_mask_cube


def _make_test_cube(long_name):
    """Make a cube to run tests upon"""
    cs = GeogCS(EARTH_RADIUS)
    data = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    cube = Cube(data, long_name=long_name)
    x_coord = DimCoord(
        np.linspace(-45.0, 45.0, 3), "latitude", units="degrees", coord_system=cs
    )
    y_coord = DimCoord(
        np.linspace(120, 180, 3), "longitude", units="degrees", coord_system=cs
    )
    cube.add_dim_coord(x_coord, 0)
    cube.add_dim_coord(y_coord, 1)
    return cube


class Test__make_mask_cube(IrisTest):
    """Test private function to make cube from generated mask"""

    def setUp(self):
        """setting up test data"""
        premask = np.array([[0.0, 3.0, 2.0], [0.5, 0.0, 1.5], [0.2, 0.0, 0]])
        self.mask = np.ma.masked_where(premask > 1.0, premask)

        self.x_coord = DimCoord([1, 2, 3], long_name="longitude")
        self.y_coord = DimCoord([1, 2, 3], long_name="latitude")
        self.coords = [self.x_coord, self.y_coord]
        self.upper = 100.0
        self.lower = 0.0
        self.units = "m"

    def test_wrong_number_of_bounds(self):
        """test checking that an exception is raised when the _make_mask_cube
        method is called with an incorrect number of bounds."""
        emsg = "should have only an upper and lower limit"
        with self.assertRaisesRegex(TypeError, emsg):
            _make_mask_cube(self.mask, self.coords, [0], self.units)
        with self.assertRaisesRegex(TypeError, emsg):
            _make_mask_cube(self.mask, self.coords, [0, 2, 4], self.units)

    def test_upperbound_fails(self):
        """test checking that an exception is raised when the _make_mask_cube
        method is called with only an upper bound."""
        emsg = "should have both an upper and lower limit"
        with self.assertRaisesRegex(TypeError, emsg):
            _make_mask_cube(self.mask, self.coords, [None, self.upper], self.units)

    def test_bothbounds(self):
        """test creating cube with both thresholds set"""
        result = _make_mask_cube(
            self.mask, self.coords, [self.lower, self.upper], self.units
        )
        self.assertEqual(result.coord("topographic_zone").bounds[0][1], self.upper)
        self.assertEqual(result.coord("topographic_zone").bounds[0][0], self.lower)
        self.assertEqual(
            result.coord("topographic_zone").points, np.mean([self.lower, self.upper])
        )
        self.assertEqual(result.coord("topographic_zone").units, Unit("m"))

    def test_cube_attribute_no_seapoints(self):
        """Test the new attribute is added to the cube."""
        result = _make_mask_cube(
            self.mask, self.coords, [self.lower, self.upper], self.units
        )
        self.assertEqual(
            result.attributes["topographic_zones_include_seapoints"], "False"
        )

    def test_cube_attribute_include_seapoints(self):
        """Test the new attribute is added to the cube when seapoints
           included."""
        result = _make_mask_cube(
            self.mask,
            self.coords,
            [self.lower, self.upper],
            self.units,
            sea_points_included="True",
        )
        self.assertEqual(
            result.attributes["topographic_zones_include_seapoints"], "True"
        )


if __name__ == "__main__":
    unittest.main()
