# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Unit tests for the windgust_diagnostic.WindGustDiagnostic plugin."""
import unittest
import warnings
import numpy as np

import iris
from iris.tests import IrisTest
from iris.cube import Cube
from iris.coords import DimCoord
from iris.exceptions import CoordinateNotFoundError
from cf_units import Unit

from improver.cube_combiner import CubeCombiner


def create_cube_with_threshold_coord(data=None,
                                     long_name=None,
                                     threshold_values=None,
                                     units=None):
    """Create a cube with threshold coord."""
    if threshold_values is None:
        threshold_values = [1.0]
    if data is None:
        data = np.zeros((len(threshold_values), 2, 2, 2))
        data[:, 0, :, :] = 0.5
        data[:, 1, :, :] = 0.1
    if long_name is None:
        long_name = "probability_of_rainfall_rate"
    if units is None:
        units = "m s^-1"

    cube = Cube(data, long_name=long_name, units='1')
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                                units='degrees'), 2)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                units='degrees'), 3)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(threshold_values,
                                long_name='threshold',
                                units=units), 0)
    return cube


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = CubeCombiner('+')
        self.assertEqual(plugin.operation, '+')


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(CubeCombiner('+'))
        msg = '<CubeCombiner: operation=+>'
        self.assertEqual(result, msg)


class Test_add_metadata(IrisTest):

    """Test the add_metadata method."""
    def test_basic(self):
        """Test that the function returns a Cube. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold_coord()
        result = plugin.add_metadata(cube)
        self.assertIsInstance(result, Cube)


class Test_process(IrisTest):

    """Test the plugin combines the cubelist into a cube."""

    def test_basic(self):
        """Test that the plugin returns a Cube. """
        plugin = CubeCombiner('+')
        cube = create_cube_with_threshold_coord()
        cubelist = iris.cube.CubeList([cube,cube])
        result = plugin.process(cubelist)
        self.assertIsInstance(result, Cube)


if __name__ == '__main__':
    unittest.main()
