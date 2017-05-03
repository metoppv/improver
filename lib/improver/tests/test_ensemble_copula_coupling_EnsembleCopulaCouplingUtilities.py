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
"""
Unit tests for the
`ensemble_copula_coupling.EnsemeblCopulaCouplingUtilities` class.
"""
import unittest

import iris
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_copula_coupling import (
    EnsembleCopulaCouplingUtilities as Plugin)
from improver.tests.helper_functions_ensemble_calibration import(
    set_up_temperature_cube, _add_forecast_reference_time_and_forecast_period)


class Test_create_cube_with_percentiles(IrisTest):

    """Test the _create_cube_with_percentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube."""
        cube = self.current_temperature_forecast_cube
        cube_data = cube.data + 2
        percentiles = [0.1, 0.5, 0.9]
        result = Plugin.create_cube_with_percentiles(
            percentiles, cube, cube_data)
        self.assertIsInstance(result, Cube)

    def test_many_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with many
        percentiles.
        """
        cube = self.current_temperature_forecast_cube
        percentiles = np.linspace(0, 1, 100)
        cube_data = np.zeros(
            [len(percentiles), len(cube.coord("time").points),
             len(cube.coord("latitude").points),
             len(cube.coord("longitude").points)])
        result = Plugin.create_cube_with_percentiles(
            percentiles, cube, cube_data)
        self.assertEqual(cube_data.shape, result.data.shape)

    def test_incompatible_percentiles(self):
        """
        Test that the plugin fails if the percentile values requested
        are not numbers.
        """
        cube = self.current_temperature_forecast_cube
        percentiles = ["cat", "dog", "elephant"]
        cube_data = np.zeros(
            [len(percentiles), len(cube.coord("time").points),
             len(cube.coord("latitude").points),
             len(cube.coord("longitude").points)])
        msg = "could not convert string to float"
        with self.assertRaisesRegexp(ValueError, msg):
            Plugin.create_cube_with_percentiles(
                percentiles, cube, cube_data)

    def test_percentile_points(self):
        """
        Test that the plugin returns an Iris.cube.Cube
        with a percentile coordinate with the desired points.
        """
        cube = self.current_temperature_forecast_cube
        cube_data = cube.data + 2
        percentiles = [0.1, 0.5, 0.9]
        result = Plugin.create_cube_with_percentiles(
            percentiles, cube, cube_data)
        self.assertIsInstance(result.coord("percentile"), DimCoord)
        self.assertArrayAlmostEqual(
            result.coord("percentile").points, percentiles)


class Test_create_percentiles(IrisTest):

    """Test the create_percentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

    def test_basic(self):
        """
        Test that the plugin returns a list with the expected number of
        percentiles.
        """
        cube = self.current_temperature_forecast_cube
        no_of_percentiles = 3
        result = Plugin.create_percentiles(no_of_percentiles)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), no_of_percentiles)

    def test_data(self):
        """
        Test that the plugin returns a list with the expected data values
        for the percentiles.
        """
        data = np.array([0.25, 0.5, 0.75])

        cube = self.current_temperature_forecast_cube
        no_of_percentiles = 3
        result = Plugin.create_percentiles(no_of_percentiles)
        self.assertArrayAlmostEqual(result, data)

    def test_random(self):
        """
        Test that the plugin returns a list with the expected number of
        percentiles, if the random sampling option is selected.
        """
        cube = self.current_temperature_forecast_cube
        no_of_percentiles = 3
        result = Plugin.create_percentiles(
            no_of_percentiles, sampling="random")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), no_of_percentiles)

    def test_unknown_sampling_option(self):
        """
        Test that the plugin returns the expected error message,
        if an unknown sampling option is selected.
        """
        cube = self.current_temperature_forecast_cube
        no_of_percentiles = 3
        msg = "The unknown sampling option is not yet implemented"
        with self.assertRaisesRegexp(ValueError, msg):
            Plugin.create_percentiles(
                no_of_percentiles, sampling="unknown")
