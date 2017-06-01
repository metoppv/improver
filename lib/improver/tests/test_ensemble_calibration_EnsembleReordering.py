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
Unit tests for the `ensemble_calibration.EnsembleReordering`
class.

"""
import unittest

from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_calibration.ensemble_calibration import (
    EnsembleReordering as Plugin)
from improver.tests.helper_functions_ensemble_calibration import(
    set_up_temperature_cube,
    add_forecast_reference_time_and_forecast_period)


class Test_rank_ecc(IrisTest):

    """Test the rank_ecc method in the EnsembleReordering plugin."""

    def setUp(self):
        """
        Create a cube with forecast_reference_time and
        forecast_period coordinates.
        """
        self.cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        raw_data = np.array([[[[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]]],
                             [[[2, 2, 2],
                               [2, 2, 2],
                               [2, 2, 2]]],
                             [[[3, 3, 3],
                               [3, 3, 3],
                               [3, 3, 3]]]])
        calibrated_data = np.array(
            [[[[0.71844843, 0.71844843, 0.71844843],
               [0.71844843, 0.71844843, 0.71844843],
               [0.71844843, 0.71844843, 0.71844843]]],
             [[[2., 2., 2.],
               [2., 2., 2.],
               [2., 2., 2.]]],
             [[[3.28155157, 3.28155157, 3.28155157],
               [3.28155157, 3.28155157, 3.28155157],
               [3.28155157, 3.28155157, 3.28155157]]]])

        raw_cube = self.cube.copy()
        raw_cube.data = raw_data
        calibrated_cube = self.cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)
        self.assertIsInstance(result, Cube)

    def test_ordered_data(self):
        """
        Test that the plugin returns an Iris Cube where the cube data is an
        ordered numpy array for the calibrated data with the same ordering
        as the raw data.
        """
        raw_data = np.array([[[[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]]],
                             [[[2, 2, 2],
                               [2, 2, 2],
                               [2, 2, 2]]],
                             [[[3, 3, 3],
                               [3, 3, 3],
                               [3, 3, 3]]]])

        calibrated_data = raw_data

        raw_cube = self.cube.copy()
        raw_cube.data = raw_data
        calibrated_cube = self.cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)
        result.transpose([1, 0, 2, 3])
        self.assertArrayAlmostEqual(result.data, calibrated_data)

    def test_unordered_data(self):
        """
        Test that the plugin returns an iris.cube.Cube with the correct data.
        ECC orders the calibrated data based on the ordering of the raw data.
        This could mean that the calibrated data appears out of order.
        ECC does not reorder the calibrated data in a monotonically-increasing
        order.
        """
        raw_data = np.array([[[[5, 5, 5],
                               [7, 5, 5],
                               [5, 5, 5]]],
                             [[[4, 4, 4],
                               [4, 4, 4],
                               [4, 4, 4]]],
                             [[[6, 6, 6],
                               [6, 6, 6],
                               [6, 6, 6]]]])

        calibrated_data = np.array([[[[4, 5, 4],
                                      [4, 5, 4],
                                      [4, 5, 4]]],
                                    [[[5, 6, 5],
                                      [5, 6, 5],
                                      [5, 6, 5]]],
                                    [[[6, 7, 6],
                                      [6, 7, 6],
                                      [6, 7, 6]]]])

        # This reordering does not pay attention to the values within the
        # calibrated data, the rankings created to perform the sorting are
        # taken exclusively from the raw_data.
        result_data = np.array([[[[5, 6, 5],
                                  [6, 6, 5],
                                  [5, 6, 5]]],
                                [[[4, 5, 4],
                                  [4, 5, 4],
                                  [4, 5, 4]]],
                                [[[6, 7, 6],
                                  [5, 7, 6],
                                  [6, 7, 6]]]])

        raw_cube = self.cube.copy()
        raw_cube.data = raw_data
        calibrated_cube = self.cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)
        result.transpose([1, 0, 2, 3])
        self.assertArrayAlmostEqual(result.data, result_data)

    def test_3d_cube(self):
        """Test that the plugin returns the correct cube data for a
        3d input cube."""
        raw_data = np.array(
            [[[1, 1]],
             [[3, 2]],
             [[2, 3]]])

        calibrated_data = np.array(
            [[[1, 1]],
             [[2, 2]],
             [[3, 3]]])

        # Reordering of the calibrated_data array to match
        # the raw_data ordering
        result_data = np.array(
            [[[1, 1]],
             [[3, 2]],
             [[2, 3]]])

        cube = self.cube.copy()
        cube = cube[:, :, :2, 0]

        raw_cube = cube.copy()
        raw_cube.data = raw_data

        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)

        result.transpose([1, 0, 2])

        self.assertArrayAlmostEqual(result.data, result_data)

    def test_3d_cube_tied_values(self):
        """
        Test that the plugin returns the correct cube data for a
        3d input cube, when there are tied values witin the
        raw ensemble members. As there are two possible options for the
        result data, as the tie is decided randomly, both possible result
        data options are checked.
        """
        raw_data = np.array(
            [[[1, 1]],
             [[3, 2]],
             [[2, 2]]])

        calibrated_data = np.array(
            [[[1, 1]],
             [[2, 2]],
             [[3, 3]]])

        # Reordering of the calibrated_data array to match
        # the raw_data ordering
        result_data_first = np.array(
            [[[1, 1]],
             [[3, 2]],
             [[2, 3]]])

        result_data_second = np.array(
            [[[1, 1]],
             [[3, 3]],
             [[2, 2]]])

        cube = self.cube.copy()
        cube = cube[:, :, :2, 0]

        raw_cube = cube.copy()
        raw_cube.data = raw_data

        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data
        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)
        result.transpose([1, 0, 2])

        err_count = 0
        try:
            self.assertArrayAlmostEqual(result.data, result_data_first)
        except Exception as err1:
            err_count += 1

        try:
            self.assertArrayAlmostEqual(result.data, result_data_second)
        except Exception as err2:
            err_count += 1

        if err_count == 2:
            raise ValueError("Exceptions raised by both accepted forms of the "
                             "calibrated data. {} {}".format(err1, err2))

    def test_2d_cube(self):
        """
        Test that the plugin returns the correct cube data for a
        2d input cube.
        """
        raw_data = np.array([[3],
                             [2],
                             [1]])

        calibrated_data = np.array([[1],
                                    [2],
                                    [3]])

        result_data = np.array([[3],
                                [2],
                                [1]])

        cube = self.cube.copy()
        cube = cube[:, :, 0, 0]
        raw_cube = cube.copy()
        raw_cube.data = raw_data
        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)
        result.transpose([1, 0])
        self.assertArrayAlmostEqual(result.data, result_data)


class Test_process(IrisTest):

    """Test the EnsembleReordering plugin."""

    def setUp(self):
        """
        Create a raw and calibrated cube with with forecast_reference_time and
        forecast_period coordinates.
        """
        self.raw_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))
        self.calibrated_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        plugin = Plugin()
        result = plugin.process(self.raw_cube, self.calibrated_cube)
        self.assertIsInstance(result, Cube)
        self.assertTrue(result.coords("realization"))


if __name__ == '__main__':
    unittest.main()
