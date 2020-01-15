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
"""
Unit tests for the function "cube_manipulation.build_coordinate".
"""

import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.coord_systems import TransverseMercator
from iris.coords import AuxCoord, DimCoord
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import build_coordinate


class Test_build_coordinate(IrisTest):
    """Test the build_coordinate utility."""

    def test_basic(self):
        """Test that the utility returns a coord."""
        result = build_coordinate([1.0], long_name='testing')
        self.assertIsInstance(result, DimCoord)

    def test_use_many_keyword_arguments(self):
        """Test that a coordinate is built when most of the keyword arguments
        are specified."""
        standard_name = "height"
        long_name = "height"
        var_name = "height"
        coord_type = AuxCoord
        data_type = np.int64
        units = "m"
        bounds = np.array([0.5, 1.5])
        coord_system = TransverseMercator
        result = build_coordinate(
            [1.0], standard_name=standard_name, long_name=long_name,
            var_name=var_name, coord_type=coord_type, data_type=data_type,
            units=units, bounds=bounds, coord_system=coord_system)
        self.assertIsInstance(result, AuxCoord)
        self.assertEqual(result.standard_name, "height")
        self.assertEqual(result.long_name, "height")
        self.assertEqual(result.var_name, "height")
        self.assertIsInstance(result.points[0], np.int64)
        self.assertEqual(result.units, Unit("m"))
        self.assertArrayAlmostEqual(result.bounds, np.array([[0.5, 1.5]]))
        self.assertArrayAlmostEqual(result.points, np.array([1.0]))
        self.assertEqual(
            result.coord_system, TransverseMercator)

    def test_template_coord(self):
        """Test that a coordinate can be built from a template coordinate."""
        template_coord = DimCoord([2.0], standard_name="height", units="m")
        result = build_coordinate([5.0, 10.0], template_coord=template_coord)
        self.assertIsInstance(result, DimCoord)
        self.assertEqual(result.standard_name, "height")
        self.assertEqual(result.units, Unit("m"))
        self.assertArrayAlmostEqual(result.points, np.array([5.0, 10.0]))

    def test_custom_function(self):
        """Test that a coordinate can be built when using a custom function."""
        def divide_data(data):
            """Basic custom function for testing in build_coordinate"""
            return data/2
        result = build_coordinate(
            [1.0], long_name="realization", custom_function=divide_data)
        self.assertArrayAlmostEqual(result.points, np.array([0.5]))

    def test_build_latitude_coordinate(self):
        """Test building a latitude coordinate."""
        latitudes = np.linspace(-90, 90, 20)
        coord_system = iris.coord_systems.GeogCS(6371229.0)
        result = build_coordinate(latitudes, long_name='latitude',
                                  units='degrees',
                                  coord_system=coord_system)
        self.assertArrayEqual(result.points, latitudes)
        self.assertEqual(result.name(), 'latitude')
        self.assertIsInstance(result, DimCoord)
        self.assertEqual(result.units, 'degrees')


if __name__ == '__main__':
    unittest.main()
