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
"""Unit tests for the vicinity_nbhood.ProbabilityOfOccurrence plugin."""

import unittest

from iris.coords import AuxCoord
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.nbhood.vicinity import ProbabilityOfOccurrence
from improver.tests.utilities.test_OccurrenceWithinVicinity import (
    set_up_cube)


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_exception(self):
        """Test the expected exception is raised."""
        distance = 2000
        radius = 2000
        msg = "Only a square neighbourhood is accepted"
        with self.assertRaisesRegexp(ValueError, msg):
            ProbabilityOfOccurrence(distance, "circular", radius)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(ProbabilityOfOccurrence(2000, "square", 2000))
        msg = ('<ProbabilityOfOccurrence: distance: 2000; '
               'neighbourhood_method: square; radii: 2000; '
               'lead_times: None; weighted_mode: True; '
               'ens_factor: 1.0>')
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        """Set up a cube."""
        data = np.zeros((1, 1, 5, 5))
        data[0, 0, 0, 1] = 1.0
        data[0, 0, 2, 3] = 1.0
        y_dimension_values = np.arange(0.0, 10000.0, 2000.0)
        self.cube = set_up_cube(data, "lwe_precipitation_rate", "m s-1",
                                y_dimension_values=y_dimension_values,
                                x_dimension_values=y_dimension_values)

    def test_with_realization(self):
        """Test when a realization coordinate is present."""
        expected = np.array(
            [[[1, 1., 0.77777778, 0.55555556, 0.33333333],
              [0.66666667, 0.77777778, 0.77777778, 0.77777778, 0.66666667],
              [0.33333333, 0.55555556, 0.77777778, 1., 1.],
              [0., 0.22222222, 0.44444444, 0.66666667, 0.66666667],
              [0., 0.11111111, 0.22222222, 0.33333333, 0.33333333]]])
        distance = 2000
        neighbourhood_method = "square"
        radii = 2000
        result = (
            ProbabilityOfOccurrence(
                distance, neighbourhood_method, radii).process(self.cube))
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_without_realization(self):
        """Test when a realization coordinate is not present."""
        expected = np.array(
            [[[1, 1., 0.77777778, 0.55555556, 0.33333333],
              [0.66666667, 0.77777778, 0.77777778, 0.77777778, 0.66666667],
              [0.33333333, 0.55555556, 0.77777778, 1., 1.],
              [0., 0.22222222, 0.44444444, 0.66666667, 0.66666667],
              [0., 0.11111111, 0.22222222, 0.33333333, 0.33333333]]])
        cube = self.cube[0, :, :, :]
        cube.remove_coord("realization")
        distance = 2000
        neighbourhood_method = "square"
        radii = 2000
        result = (
            ProbabilityOfOccurrence(
                distance, neighbourhood_method, radii).process(self.cube))
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_additional_arguments(self):
        """Test when all keyword arguments are passed in."""
        expected = np.array(
            [[[1., 1., 0.77777778, 0.55555556, 0.333333],
              [0.66666667, 0.77777778, 0.77777778, 0.77777778, 0.66666667],
              [0.33333333, 0.55555556, 0.77777778, 1., 1.],
              [0., 0.22222222, 0.44444444, 0.66666667, 0.66666667],
              [0., 0.11111111, 0.22222222, 0.33333333, 0.33333333]],
             [[0.84, 0.8, 0.76, 0.72, 0.68],
              [0.68, 0.7, 0.72, 0.74, 0.76],
              [0.48, 0.52, 0.56, 0.6, 0.64],
              [0.3, 0.4, 0.5, 0.6, 0.7],
              [0.12, 0.24, 0.36, 0.48, 0.6]]])
        data = np.zeros((1, 2, 5, 5))
        data[0, :, 0, 1] = 1.0
        data[0, :, 2, 3] = 1.0
        y_dimension_values = np.arange(0.0, 10000.0, 2000.0)
        cube = set_up_cube(data, "lwe_precipitation_rate", "m s-1",
                           y_dimension_values=y_dimension_values,
                           x_dimension_values=y_dimension_values,
                           timesteps=np.array([402192.5, 402195.5]))
        distance = 2000
        neighbourhood_method = "square"
        radii = [2000, 4000]
        lead_times = [3, 6]
        cube.add_aux_coord(AuxCoord(
            lead_times, "forecast_period", units="hours"), 1)
        weighted_mode = False
        ens_factor = 0.9
        result = (
            ProbabilityOfOccurrence(
                distance, neighbourhood_method, radii, lead_times=lead_times,
                weighted_mode=weighted_mode, ens_factor=ens_factor
                ).process(cube))
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
