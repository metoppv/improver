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
"""Unit tests for psychrometric_calculations PrecipPhaseProbability plugin."""

import unittest

import numpy as np
from cf_units import Unit
import iris
from iris.tests import IrisTest

from improver.psychrometric_calculations.precip_phase_probability import (
    PrecipPhaseProbability)
from improver.nbhood.nbhood import GeneratePercentilesFromANeighbourhood

from improver_tests.set_up_test_cubes import set_up_variable_cube


class Test__init__(IrisTest):

    """Test the init method."""

    def test_basic(self):
        """Test that the __init__ method configures the plugin as expected."""

        plugin = PrecipPhaseProbability()
        self.assertTrue(plugin.percentile_plugin is
                        GeneratePercentilesFromANeighbourhood)
        self.assertEqual(plugin._nbhood_shape, 'circular')
        self.assertAlmostEqual(plugin.radius, 10000.)


class Test_process(IrisTest):

    """Test the PhaseChangeLevel processing works"""

    def setUp(self):
        """Set up orography cube (as zeros) and falling_phase_level cube with
        multiple realizations designed to return snow, sleet and rain. The
        middle realization gives both not-snow and not-rain because both the
        20th percentile is <= zero and the 80th percentile is >= zero."""

        # cubes for testing have a grid-length of 333333m.
        self.plugin = PrecipPhaseProbability(radius=350000.)
        self.mandatory_attributes = {
            'title': 'mandatory title',
            'source': 'mandatory_source',
            'institution': 'mandatory_institution'
        }

        data = np.zeros((3, 3), dtype=np.float32)

        orog_cube = set_up_variable_cube(
            data, name='surface_altitude', units='m',
            spatial_grid='equalarea', attributes=self.mandatory_attributes)

        falling_level_data = np.array(
            [[[-1, -1, -1],
              [-1, -1, -1],
              [-1, -1, -1]],
             [[0, -1, 0],
              [0, 1, 0],
              [0, -1, 0]],
             [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]], dtype=np.float32)

        falling_level_cube = set_up_variable_cube(
            falling_level_data, units='m', spatial_grid='equalarea',
            name='altitude_of_snow_falling_level', realizations=[0, 1, 2],
            attributes=self.mandatory_attributes)

        self.cubes = iris.cube.CubeList([falling_level_cube, orog_cube])

    def test_prob_snow(self):
        """Test that process returns a cube with the right name, units and
        values. In this instance the phase change is from snow to sleet."""
        result = self.plugin.process(self.cubes)
        expected = np.zeros((3, 3, 3), dtype=np.float32)
        expected[0] = 1.
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "probability_of_snow_at_surface")
        self.assertEqual(result.units, Unit('1'))
        self.assertDictEqual(result.attributes, self.mandatory_attributes)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_prob_rain(self):
        """Test that process returns a cube with the right name, units and
        values. In this instance the phase change is from sleet to rain."""
        self.cubes[0].rename('altitude_of_rain_falling_level')
        result = self.plugin.process(self.cubes)
        expected = np.zeros((3, 3, 3), dtype=np.float32)
        expected[2] = 1.
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "probability_of_rain_at_surface")
        self.assertEqual(result.units, Unit('1'))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_unit_conversion(self):
        """Test that process returns the same as test_prob_rain when the
        orography cube units are in feet."""
        self.cubes[1].units = Unit('feet')
        self.cubes[0].rename('altitude_of_rain_falling_level')
        result = self.plugin.process(self.cubes)
        expected = np.zeros((3, 3, 3), dtype=np.float32)
        expected[2] = 1.
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "probability_of_rain_at_surface")
        self.assertEqual(result.units, Unit('1'))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_unit_synonyms(self):
        """Test that process returns the same as test_prob_rain when the
        orography cube units are "metres" (a synonym of "m")."""
        self.cubes[1].units = Unit('metres')
        self.cubes[0].rename('altitude_of_rain_falling_level')
        result = self.plugin.process(self.cubes)
        expected = np.zeros((3, 3, 3), dtype=np.float32)
        expected[2] = 1.
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "probability_of_rain_at_surface")
        self.assertEqual(result.units, Unit('1'))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_bad_phase_cube(self):
        """Test that process raises an exception when the input phase cube is
        incorrectly named."""
        self.cubes[0].rename('altitude_of_kittens')
        msg = 'Could not extract a rain or snow falling-level cube from'
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(self.cubes)

    def test_bad_orography_cube(self):
        """Test that process raises an exception when the input orography
        cube is incorrectly named."""
        self.cubes[1].rename('altitude_of_kittens')
        msg = 'Could not extract surface_altitude cube from'
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(self.cubes)

    def test_bad_units(self):
        """Test that process raises an exception when the input cubes cannot
        be coerced into the same units."""
        self.cubes[1].units = Unit('seconds')
        msg = 'Unable to convert from '
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(self.cubes)

    def test_spatial_mismatch(self):
        """Test that process raises an exception when the input cubes have
        different spatial coordinates."""
        self.cubes[1] = set_up_variable_cube(
            self.cubes[1].data, name='surface_altitude', units='m',
            spatial_grid='latlon', attributes=self.mandatory_attributes)
        msg = 'Spatial coords mismatch between'
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(self.cubes)


if __name__ == '__main__':
    unittest.main()
