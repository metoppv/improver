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
`ensemble_copula_coupling.EnsembleReordering` plugin.

"""
import unittest

from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_copula_coupling import EnsembleReordering as Plugin
from improver.tests.helper_functions_ensemble_calibration import(
    set_up_cube, set_up_temperature_cube,
    _add_forecast_reference_time_and_forecast_period)


class Test_mismatch_between_length_of_raw_members_and_percentiles(IrisTest):

    """
    Test the mismatch_between_length_of_raw_members_and_percentiles
    method in the EnsembleReordering plugin.
    """

    def setUp(self):
        """
        Create a cube with forecast_reference_time and
        forecast_period coordinates.
        """
        data = np.tile(np.linspace(5, 10, 9), 3).reshape(3, 1, 3, 3)
        data[0] -= 1
        data[1] += 1
        data[2] += 3
        cube = set_up_cube(data, "air_temperature", "degreesC")
        self.realization_cube = (
            _add_forecast_reference_time_and_forecast_period(cube.copy()))
        cube.coord("realization").rename("percentile")
        self.percentile_cube = (
            _add_forecast_reference_time_and_forecast_period(cube))

    def test_types_length_of_percentiles_equals_length_of_members(self):
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_members = self.realization_cube
        plugin = Plugin()
        result = plugin.mismatch_between_length_of_raw_members_and_percentiles(
            post_processed_forecast_percentiles, raw_forecast_members)
        self.assertIsInstance(result, tuple)
        for aresult in result:
            self.assertIsInstance(aresult, Cube)

    def test_types_length_of_percentiles_greater_than_length_of_members(self):
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_members = self.realization_cube
        raw_forecast_members = raw_forecast_members[:2, :, :, :]
        plugin = Plugin()
        result = plugin.mismatch_between_length_of_raw_members_and_percentiles(
            post_processed_forecast_percentiles, raw_forecast_members)
        for aresult in result:
            self.assertIsInstance(aresult, Cube)

    def test_types_length_of_percentiles_less_than_length_of_members(self):
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_members = self.realization_cube
        post_processed_forecast_percentiles = (
            post_processed_forecast_percentiles[:2, :, :, :])
        plugin = Plugin()
        result = plugin.mismatch_between_length_of_raw_members_and_percentiles(
            post_processed_forecast_percentiles, raw_forecast_members)
        for aresult in result:
            self.assertIsInstance(aresult, Cube)

    def test_realization_for_equal(self):
        data = [0, 1, 2]
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_members = self.realization_cube
        plugin = Plugin()
        result = plugin.mismatch_between_length_of_raw_members_and_percentiles(
            post_processed_forecast_percentiles, raw_forecast_members)
        self.assertArrayAlmostEqual(
            data, result[1].coord("realization").points)

    def test_realization_for_greater_than(self):
        data = [0, 1, 2]
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_members = self.realization_cube
        raw_forecast_members = raw_forecast_members[:2, :, :, :]
        plugin = Plugin()
        result = plugin.mismatch_between_length_of_raw_members_and_percentiles(
            post_processed_forecast_percentiles, raw_forecast_members)
        self.assertArrayAlmostEqual(
            data, result[1].coord("realization").points)

    def test_realization_for_less_than(self):
        data = [0, 1]
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_members = self.realization_cube
        post_processed_forecast_percentiles = (
            post_processed_forecast_percentiles[:2, :, :, :])
        plugin = Plugin()
        result = plugin.mismatch_between_length_of_raw_members_and_percentiles(
            post_processed_forecast_percentiles, raw_forecast_members)
        self.assertArrayAlmostEqual(
            data, result[1].coord("realization").points)


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

    def test_2d_cube_random_ordering(self):
        """
        Test that the plugin returns the correct cube data for a
        2d input cube, if random ordering is selected.
        """
        raw_data = np.array([[3],
                             [2],
                             [1]])

        calibrated_data = np.array([[1],
                                    [2],
                                    [3]])

        result_data_first = np.array([[1],
                                      [2],
                                      [3]])

        result_data_second = np.array([[1],
                                      [3],
                                      [2]])

        result_data_third = np.array([[2],
                                      [1],
                                      [3]])

        result_data_fourth = np.array([[2],
                                      [3],
                                      [1]])

        result_data_fifth = np.array([[3],
                                      [1],
                                      [2]])

        result_data_sixth = np.array([[3],
                                      [2],
                                      [1]])

        cube = self.cube.copy()
        cube = cube[:, :, 0, 0]
        raw_cube = cube.copy()
        raw_cube.data = raw_data
        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube,
                                 random_ordering=True)
        result.transpose([1, 0])

        err_count = 0
        try:
            self.assertArrayAlmostEqual(result.data, result_data_first)
        except Exception as err1:
            err_count += 1

        try:
            self.assertArrayAlmostEqual(result.data, result_data_second)
        except Exception as err2:
            err_count += 1

        try:
            self.assertArrayAlmostEqual(result.data, result_data_third)
        except Exception as err3:
            err_count += 1

        try:
            self.assertArrayAlmostEqual(result.data, result_data_fourth)
        except Exception as err4:
            err_count += 1

        try:
            self.assertArrayAlmostEqual(result.data, result_data_fifth)
        except Exception as err5:
            err_count += 1

        try:
            self.assertArrayAlmostEqual(result.data, result_data_sixth)
        except Exception as err6:
            err_count += 1

        if err_count == 6:
            raise ValueError("Exceptions raised as all accepted forms of the "
                             "calibrated data were not matched."
                             "1. {}"
                             "2. {}"
                             "3. {}"
                             "4. {}"
                             "5. {}"
                             "6. {}".format(err1, err2, err3,
                                            err4, err5, err6))


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
        self.calibrated_cube.coord("realization").rename("percentile")

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        plugin = Plugin()
        result = plugin.process(self.calibrated_cube, self.raw_cube)
        self.assertIsInstance(result, Cube)
        self.assertTrue(result.coords("realization"))


if __name__ == '__main__':
    unittest.main()
