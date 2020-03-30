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
Unit tests for the
`ensemble_copula_coupling.EnsembleReordering` plugin.

"""
import itertools
import unittest

import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import \
    EnsembleReordering as Plugin
from improver.utilities.warnings_handler import ManageWarnings

from ...calibration.ensemble_calibration.helper_functions import (
    add_forecast_reference_time_and_forecast_period, set_up_cube,
    set_up_temperature_cube)


class Test__recycle_raw_ensemble_realizations(IrisTest):

    """
    Test the _recycle_raw_ensemble_realizations
    method in the EnsembleReordering plugin.
    """

    def setUp(self):
        """
        Create a cube with a realization coordinate and a cube with a
        percentile coordinate with forecast_reference_time and
        forecast_period coordinates.
        """
        data = np.tile(np.linspace(5, 10, 9), 3).reshape(3, 1, 3, 3)
        data[0] -= 1
        data[1] += 1
        data[2] += 3
        cube = set_up_cube(data, "air_temperature", "degreesC")
        self.realization_cube = (
            add_forecast_reference_time_and_forecast_period(cube.copy()))
        self.perc_coord = "percentile"
        cube.coord("realization").rename(self.perc_coord)
        self.percentile_cube = (
            add_forecast_reference_time_and_forecast_period(cube))

    def test_realization_for_equal(self):
        """
        Test to check the behaviour whether the number of percentiles equals
        the number of realizations. For when the length of the percentiles
        equals the length of the realizations, check that the points of the
        realization coordinate is as expected.
        """
        data = [0, 1, 2]
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_realizations = self.realization_cube
        plu = Plugin()
        result = plu._recycle_raw_ensemble_realizations(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            self.perc_coord)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            data, result.coord("realization").points)

    def test_realization_for_greater_than(self):
        """
        Test to check the behaviour whether the number of percentiles is
        greater than the number of realizations. For when the length of the
        percentiles is greater than the length of the realizations,
        check that the points of the realization coordinate is as expected.
        """
        data = [12, 13, 14]
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_realizations = self.realization_cube
        raw_forecast_realizations = raw_forecast_realizations[:2, :, :, :]
        raw_forecast_realizations.coord("realization").points = [12, 13]
        plu = Plugin()
        result = plu._recycle_raw_ensemble_realizations(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            self.perc_coord)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            data, result.coord("realization").points)

    def test_realization_for_less_than(self):
        """
        Test to check the behaviour whether the number of percentiles is
        less than the number of realizations. For when the length of the
        percentiles is less than the length of the realizations, check that
        the points of the realization coordinate is as expected.
        """
        data = [0, 1]
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_realizations = self.realization_cube
        post_processed_forecast_percentiles = (
            post_processed_forecast_percentiles[:2, :, :, :])
        plu = Plugin()
        result = plu._recycle_raw_ensemble_realizations(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            self.perc_coord)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            data, result.coord("realization").points)

    def test_realization_for_equal_check_data(self):
        """
        Test to check the behaviour whether the number of percentiles equals
        the number of realizations. For when the length of the percentiles
        equals the length of the realizations, check that the points of the
        realization coordinate is as expected.
        """
        data = np.array([[[[4., 4.625, 5.25],
                           [5.875, 6.5, 7.125],
                           [7.75, 8.375, 9.]]],
                         [[[6., 6.625, 7.25],
                           [7.875, 8.5, 9.125],
                           [9.75, 10.375, 11.]]],
                         [[[8., 8.625, 9.25],
                           [9.875, 10.5, 11.125],
                           [11.75, 12.375, 13.]]]])

        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_realizations = self.realization_cube
        plu = Plugin()
        result = plu._recycle_raw_ensemble_realizations(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            self.perc_coord)
        self.assertArrayAlmostEqual(data, result.data)

    def test_realization_for_greater_than_check_data(self):
        """
        Test to check the behaviour whether the number of percentiles is
        greater than the number of realizations. For when the length of the
        percentiles is greater than the length of the realizations, check
        that the points of the realization coordinate is as expected.
        """
        data = np.array([[[[4., 4.625, 5.25],
                           [5.875, 6.5, 7.125],
                           [7.75, 8.375, 9.]],
                          [[6., 6.625, 7.25],
                           [7.875, 8.5, 9.125],
                           [9.75, 10.375, 11.]],
                          [[4., 4.625, 5.25],
                           [5.875, 6.5, 7.125],
                           [7.75, 8.375, 9.]]]])
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_realizations = self.realization_cube
        # Slice number of raw forecast realizations, so that there are fewer
        # realizations than percentiles.
        raw_forecast_realizations = raw_forecast_realizations[:2, :, :, :]
        plu = Plugin()
        result = plu._recycle_raw_ensemble_realizations(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            self.perc_coord)
        self.assertArrayAlmostEqual(data, result.data)

    def test_realization_for_less_than_check_data(self):
        """
        Test to check the behaviour whether the number of percentiles is
        less than the number of realizations. For when the length of the
        percentiles is less than the length of the realizations, check that
        the points of the realization coordinate is as expected.
        """
        data = np.array([[[[4., 4.625, 5.25],
                           [5.875, 6.5, 7.125],
                           [7.75, 8.375, 9.]],
                          [[6., 6.625, 7.25],
                           [7.875, 8.5, 9.125],
                           [9.75, 10.375, 11.]]]])
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_realizations = self.realization_cube
        post_processed_forecast_percentiles = (
            post_processed_forecast_percentiles[:2, :, :, :])
        plu = Plugin()
        result = plu._recycle_raw_ensemble_realizations(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            self.perc_coord)
        self.assertArrayAlmostEqual(data, result.data)

    def test_realization_for_greater_than_check_data_many_realizations(self):
        """
        Test to check the behaviour whether the number of percentiles is
        greater than the number of realizations. For when the length of the
        percentiles is greater than the length of the realizations, check
        that the points of the realization coordinate is as expected.
        """
        data = np.tile(np.linspace(5, 10, 9), 9).reshape(9, 1, 3, 3)
        data[0] -= 1
        data[1] += 1
        data[2] += 3
        cube = set_up_cube(
            data, "air_temperature", "degreesC",
            realizations=np.arange(0, 9))

        self.realization_cube = (
            add_forecast_reference_time_and_forecast_period(cube.copy()))
        cube.coord("realization").rename(self.perc_coord)
        self.percentile_cube = (
            add_forecast_reference_time_and_forecast_period(cube))

        expected = np.array([[[[4., 4.625, 5.25],
                               [5.875, 6.5, 7.125],
                               [7.75, 8.375, 9.]],
                              [[6., 6.625, 7.25],
                               [7.875, 8.5, 9.125],
                               [9.75, 10.375, 11.]],
                              [[4., 4.625, 5.25],
                               [5.875, 6.5, 7.125],
                               [7.75, 8.375, 9.]],
                              [[6., 6.625, 7.25],
                               [7.875, 8.5, 9.125],
                               [9.75, 10.375, 11.]],
                              [[4., 4.625, 5.25],
                               [5.875, 6.5, 7.125],
                               [7.75, 8.375, 9.]],
                              [[6., 6.625, 7.25],
                               [7.875, 8.5, 9.125],
                               [9.75, 10.375, 11.]],
                              [[4., 4.625, 5.25],
                               [5.875, 6.5, 7.125],
                               [7.75, 8.375, 9.]],
                              [[6., 6.625, 7.25],
                               [7.875, 8.5, 9.125],
                               [9.75, 10.375, 11.]],
                              [[4., 4.625, 5.25],
                               [5.875, 6.5, 7.125],
                               [7.75, 8.375, 9.]]]])
        post_processed_forecast_percentiles = self.percentile_cube
        raw_forecast_realizations = self.realization_cube
        raw_forecast_realizations = raw_forecast_realizations[:2, :, :, :]
        plu = Plugin()
        result = plu._recycle_raw_ensemble_realizations(
            post_processed_forecast_percentiles, raw_forecast_realizations,
            self.perc_coord)
        self.assertArrayAlmostEqual(expected, result.data)


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

        cube = self.cube[:, :, :2, 0].copy()

        raw_cube = cube.copy()
        raw_cube.data = raw_data

        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)

        self.assertArrayAlmostEqual(result.data, result_data)

    def test_3d_cube_masked(self):
        """Test that the plugin returns the correct cube data for a
        3d input cube with a mask applied to each realization."""
        mask = np.array([[[True, False]],
                         [[True, False]],
                         [[True, False]]])
        raw_data = np.array(
            [[[1, 9]],
             [[3, 5]],
             [[2, 7]]])

        calibrated_data = np.ma.MaskedArray(
            [[[1, 6]],
             [[2, 8]],
             [[3, 10]]], mask=mask, dtype=np.float32)

        # Reordering of the calibrated_data array to match
        # the raw_data ordering
        result_data = np.array(
            [[[np.nan, 10]],
             [[np.nan, 6]],
             [[np.nan, 8]]], dtype=np.float32)

        cube = self.cube[:, :, :2, 0].copy()

        raw_cube = cube.copy()
        raw_cube.data = raw_data

        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)
        self.assertArrayAlmostEqual(result.data.data, result_data)
        self.assertArrayEqual(result.data.mask, mask)
        self.assertEqual(result.data.dtype, np.float32)

    def test_3d_cube_masked_nans(self):
        """Test that the plugin returns the correct cube data for a
        3d input cube with a mask applied to each realization, and there are
        nans under the mask."""
        mask = np.array([[[True, False]],
                         [[True, False]],
                         [[True, False]]])
        raw_data = np.array(
            [[[1, 9]],
             [[3, 5]],
             [[2, 7]]])

        calibrated_data = np.ma.MaskedArray(
            [[[np.nan, 6]],
             [[np.nan, 8]],
             [[np.nan, 10]]], mask=mask, dtype=np.float32)

        # Reordering of the calibrated_data array to match
        # the raw_data ordering
        result_data = np.array(
            [[[np.nan, 10]],
             [[np.nan, 6]],
             [[np.nan, 8]]], dtype=np.float32)

        cube = self.cube[:, :, :2, 0].copy()

        raw_cube = cube.copy()
        raw_cube.data = raw_data

        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)
        self.assertArrayAlmostEqual(result.data.data, result_data)
        self.assertArrayEqual(result.data.mask, mask)
        self.assertEqual(result.data.dtype, np.float32)

    def test_3d_cube_tied_values(self):
        """
        Test that the plugin returns the correct cube data for a
        3d input cube, when there are tied values witin the
        raw ensemble realizations. As there are two possible options for the
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

        cube = self.cube[:, :, :2, 0].copy()

        raw_cube = cube.copy()
        raw_cube.data = raw_data

        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data
        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)
        permutations = [result_data_first, result_data_second]
        matches = [
            np.array_equal(aresult, result.data) for aresult in permutations]
        self.assertIn(True, matches)

    def test_3d_cube_tied_values_random_seed(self):
        """
        Test that the plugin returns the correct cube data for a
        3d input cube, when there are tied values witin the
        raw ensemble realizations. The random seed is specified to ensure that
        only one option, out of the two possible options will be returned.
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
        result_data = np.array(
            [[[1, 1]],
             [[3, 2]],
             [[2, 3]]])

        cube = self.cube[:, :, :2, 0].copy()

        raw_cube = cube.copy()
        raw_cube.data = raw_data

        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data
        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube, random_seed=0)
        self.assertArrayAlmostEqual(result.data, result_data)

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

        cube = self.cube[:, :, 0, 0].copy()
        raw_cube = cube.copy()
        raw_cube.data = raw_data
        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube)
        self.assertArrayAlmostEqual(result.data, result_data)

    def test_2d_cube_random_ordering(self):
        """
        Test that the plugin returns the correct cube data for a
        2d input cube, if random ordering is selected.

        Random ordering does not use the ordering from the raw ensemble,
        and instead just orders the input values randomly.
        """
        raw_data = np.array([[3],
                             [2],
                             [1]])

        calibrated_data = np.array([[1],
                                    [2],
                                    [3]])

        cube = self.cube[:, :, 0, 0].copy()
        raw_cube = cube.copy()
        raw_cube.data = raw_data
        calibrated_cube = cube.copy()
        calibrated_cube.data = calibrated_data

        plugin = Plugin()
        result = plugin.rank_ecc(calibrated_cube, raw_cube,
                                 random_ordering=True)

        permutations = list(itertools.permutations(raw_data))
        permutations = [np.array(permutation) for permutation in permutations]

        matches = [
            np.array_equal(aresult, result.data) for aresult in permutations]
        self.assertIn(True, matches)


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
        self.post_processed_percentiles = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))
        self.perc_coord = "percentile"
        self.post_processed_percentiles.coord("realization").rename(
            self.perc_coord)
        self.post_processed_percentiles.coord(
            self.perc_coord).points = [10, 50, 90]

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences"])
    def test_basic(self):
        """
        Test that the plugin returns an iris.cube.Cube and the cube has a
        realization coordinate.
        """
        plugin = Plugin()
        result = plugin.process(self.post_processed_percentiles, self.raw_cube)
        self.assertIsInstance(result, Cube)
        self.assertTrue(result.coords("realization"))
        self.assertArrayAlmostEqual(
            result.coord("realization").points, [0, 1, 2])

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences"])
    def test_2d_cube_random_ordering(self):
        """
        Test that the plugin returns the correct cube data for a
        2d input cube, if random ordering is selected.
        """
        raw_data = np.array([[3],
                             [2],
                             [1]])

        post_processed_percentiles_data = np.array([[1],
                                                    [2],
                                                    [3]])

        raw_cube = self.raw_cube[:, :, 0, 0]
        raw_cube.data = raw_data
        post_processed_percentiles = (
            self.post_processed_percentiles[:, :, 0, 0])
        post_processed_percentiles.data = post_processed_percentiles_data

        plugin = Plugin()
        result = plugin.process(post_processed_percentiles, raw_cube,
                                random_ordering=True)

        permutations = list(itertools.permutations(raw_data))
        permutations = [np.array(permutation) for permutation in permutations]

        matches = [
            np.array_equal(aresult, result.data) for aresult in permutations]
        self.assertIn(True, matches)

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences"])
    def test_2d_cube_recycling_raw_ensemble_realizations(self):
        """
        Test that the plugin returns the correct cube data for a
        2d input cube, if the number of raw ensemble realizations is fewer
        than the number of percentiles required, and therefore, raw
        ensemble realization recycling is required.

        Case where two raw ensemble realizations are exactly the same,
        after the raw ensemble realizations have been recycled.
        The number of raw ensemble realizations are recycled in order to match
        the number of percentiles.

        After recycling the raw _data will be
        raw_data = np.array([[1],
                             [2],
                             [1]])

        If there's a tie, the re-ordering randomly allocates the ordering
        for the data from the raw ensemble realizations, which is why there are
        two possible options for the resulting post-processed ensemble
        realizations.

        Raw ensemble realizations
        1,  2
        Post-processed percentiles
        1,  2,  3
        After recycling raw ensemble realizations
        1,  2,  1
        As the second ensemble realization(with a data value of 2), is the
        highest value, the highest value from the post-processed percentiles
        will be the second ensemble realization data value within the
        post-processed realizations. The data values of 1 and 2 from the
        post-processed percentiles will then be split between the first
        and third post-processed ensemble realizations.

        """
        raw_data = np.array([[1],
                             [2]])

        post_processed_percentiles_data = np.array([[1],
                                                    [2],
                                                    [3]])

        expected_first = np.array([[1],
                                   [3],
                                   [2]])

        expected_second = np.array([[2],
                                    [3],
                                    [1]])

        raw_cube = self.raw_cube[:2, :, 0, 0]
        raw_cube.data = raw_data
        post_processed_percentiles = (
            self.post_processed_percentiles[:, :, 0, 0])
        post_processed_percentiles.data = post_processed_percentiles_data

        plugin = Plugin()
        result = plugin.process(post_processed_percentiles, raw_cube)
        permutations = [expected_first, expected_second]
        matches = [
            np.array_equal(aresult, result.data) for aresult in permutations]
        self.assertIn(True, matches)


if __name__ == '__main__':
    unittest.main()
