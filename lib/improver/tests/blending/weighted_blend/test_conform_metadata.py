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
"""Module to test the conform_metadata function."""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.blending.weighted_blend import conform_metadata
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period, set_up_cube


class Test_conform_metadata(IrisTest):

    """Test the conform_metadata function."""

    def setUp(self):
        """Set up cubes for testing."""
        data = np.full((3, 3), 275.15, dtype=np.float)
        cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube(data, "air_temperature", "Kelvin"))
        cube = cube[0].remove_coord("realization")
        #cube.add_aux_coord(AuxCoord([402190.0], "forecast_reference_time",
            #units=cube.coord("time").units), data_dims=1)
        #cube.add_aux_coord(
            #AuxCoord([3.0], "forecast_period", units=tunit), data_dims=0)

        # Cube with multiple times.
        cube_orig = add_forecast_reference_time_and_forecast_period(
            set_up_cube(data, "air_temperature", "Kelvin", timesteps=2))
        cube_orig = cube_orig[0].remove_coord("realization")
        #cube_orig.add_aux_coord(AuxCoord([402190.0, 402191.0], "forecast_reference_time",
            #units=cube_orig.coord("time").units), data_dims=1)
        #cube_orig.add_aux_coord(
            #AuxCoord([3.0, 4.0], "forecast_period", units=tunit), data_dims=0)

        # Cube without forecast_period
        cube_orig_without_fp = cube_orig.copy()
        cube_orig_without_fp.remove_coord("forecast_period")

    def test_basic(self):
       result = conform_metadata(self.cube, self.cube_orig)
       self.assertIsInstance(result, Cube)

# Cube returned

    def test_with_forecast_reference_time_and_forecast_period(self):
       result = conform_metadata(self.cube, self.cube_orig)
       self.assertEqual(
           result.coord("forecast_reference_time").points,
           np.max(self.cube_orig.coord("forecast_reference_time").points))
       self.assertFalse(result.coord("forecast_reference_time").bounds)
       self.assertEqual(
           result.coord("forecast_period").points,
           np.min(self.cube_orig.coord("forecast_period").points))
       self.assertFalse(result.coord("forecast_period").bounds)

# Cube with frt and fp

    def test_with_forecast_reference_time_and_without_forecast_period(self):
       result = conform_metadata(self.cube, self.cube_orig)
       self.assertEqual(
           result.coord("forecast_reference_time").points,
           np.max(self.cube_orig.coord("forecast_reference_time").points))
       self.assertFalse(result.coord("forecast_reference_time").bounds)
       self.assertEqual(
           result.coord("forecast_period").points,
           np.min(self.cube_orig.coord("forecast_period").points))
       self.assertFalse(result.coord("forecast_period").bounds)

# Cube with frt and without fp

# Cube with model_id, model_realization

    def test_with_model_model_id_and_model_realization(self):
       result = conform_metadata(self.cube, self.cube_orig)
       self.assertEqual(
           result.coord("forecast_reference_time").points,
           np.max(self.cube_orig.coord("forecast_reference_time").points))
       self.assertFalse(result.coord("forecast_reference_time").bounds)
       self.assertEqual(
           result.coord("forecast_period").points,
           np.min(self.cube_orig.coord("forecast_period").points))
       self.assertFalse(result.coord("forecast_period").bounds)


if __name__ == '__main__':
    unittest.main()