# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the
`ensemble_copula_coupling.EnsembleReordering` plugin.

"""

import itertools
import unittest

import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    EnsembleReordering as Plugin,
)
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_variable_cube,
)

from .ecc_test_data import ECC_TEMPERATURE_REALIZATIONS


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
        data = np.tile(np.linspace(5, 10, 9), 3).reshape(3, 3, 3)
        data[0] -= 1
        data[1] += 1
        data[2] += 3
        self.realization_cube = set_up_variable_cube(
            data.astype(np.float32), name="air_temperature", units="degC"
        )
        self.percentile_cube = set_up_percentile_cube(
            np.sort(data.astype(np.float32), axis=0),
            np.array([10, 50, 90], dtype=np.float32),
            name="air_temperature",
            units="degC",
        )
        self.perc_coord = "percentile"

    def test_realization_for_equal(self):
        """
        Test to check the behaviour when the number of percentiles equals
        the number of realizations.
        """
        expected_data = np.array(
            [
                [[4.0, 4.625, 5.25], [5.875, 6.5, 7.125], [7.75, 8.375, 9.0]],
                [[6.0, 6.625, 7.25], [7.875, 8.5, 9.125], [9.75, 10.375, 11.0]],
                [[8.0, 8.625, 9.25], [9.875, 10.5, 11.125], [11.75, 12.375, 13.0]],
            ]
        )

        result = Plugin()._recycle_raw_ensemble_realizations(
            self.percentile_cube, self.realization_cube, self.perc_coord
        )
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.coord("realization").points, [0, 1, 2])
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_realization_for_greater_than(self):
        """
        Test to check the behaviour when the number of percentiles is
        greater than the number of realizations.
        """
        expected_data = np.array(
            [
                [[4.0, 4.625, 5.25], [5.875, 6.5, 7.125], [7.75, 8.375, 9.0]],
                [[6.0, 6.625, 7.25], [7.875, 8.5, 9.125], [9.75, 10.375, 11.0]],
                [[4.0, 4.625, 5.25], [5.875, 6.5, 7.125], [7.75, 8.375, 9.0]],
            ]
        )
        raw_forecast_realizations = self.realization_cube[:2, :, :]
        raw_forecast_realizations.coord("realization").points = [12, 13]
        result = Plugin()._recycle_raw_ensemble_realizations(
            self.percentile_cube, raw_forecast_realizations, self.perc_coord
        )
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.coord("realization").points, [12, 13, 14])
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_realization_for_less_than(self):
        """
        Test to check the behaviour when the number of percentiles is
        less than the number of realizations.
        """
        expected_data = np.array(
            [
                [[4.0, 4.625, 5.25], [5.875, 6.5, 7.125], [7.75, 8.375, 9.0]],
                [[6.0, 6.625, 7.25], [7.875, 8.5, 9.125], [9.75, 10.375, 11.0]],
            ]
        )

        post_processed_forecast_percentiles = self.percentile_cube[:2, :, :]
        result = Plugin()._recycle_raw_ensemble_realizations(
            post_processed_forecast_percentiles, self.realization_cube, self.perc_coord
        )
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.coord("realization").points, [0, 1])
        self.assertArrayAlmostEqual(result.data, expected_data)


class Test_rank_ecc(IrisTest):
    """Test the rank_ecc method in the EnsembleReordering plugin."""

    def setUp(self):
        """
        Create a cube with forecast_reference_time and
        forecast_period coordinates.
        """
        self.cube = set_up_variable_cube(ECC_TEMPERATURE_REALIZATIONS.copy())
        self.cube_2d = self.cube[:, :2, 0].copy()

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        raw_data = np.array(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            ]
        )

        calibrated_data = np.array(
            [
                [
                    [0.71844843, 0.71844843, 0.71844843],
                    [0.71844843, 0.71844843, 0.71844843],
                    [0.71844843, 0.71844843, 0.71844843],
                ],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                [
                    [3.28155157, 3.28155157, 3.28155157],
                    [3.28155157, 3.28155157, 3.28155157],
                    [3.28155157, 3.28155157, 3.28155157],
                ],
            ]
        )

        raw_cube = self.cube.copy(data=raw_data)
        calibrated_cube = self.cube.copy(data=calibrated_data)

        result = Plugin().rank_ecc(calibrated_cube, raw_cube)
        self.assertIsInstance(result, Cube)

    def test_ordered_data(self):
        """
        Test that the plugin returns an Iris Cube where the cube data is an
        ordered numpy array for the calibrated data with the same ordering
        as the raw data.
        """
        raw_data = np.array(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            ]
        )
        raw_cube = self.cube.copy(data=raw_data)
        calibrated_cube = raw_cube.copy()

        result = Plugin().rank_ecc(calibrated_cube, raw_cube)
        self.assertArrayAlmostEqual(result.data, calibrated_cube.data)

    def test_unordered_data(self):
        """
        Test that the plugin returns an iris.cube.Cube with the correct data.
        ECC orders the calibrated data based on the ordering of the raw data.
        This could mean that the calibrated data appears out of order.
        ECC does not reorder the calibrated data in a monotonically-increasing
        order.
        """
        raw_data = np.array(
            [
                [[5, 5, 5], [7, 5, 5], [5, 5, 5]],
                [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
                [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
            ]
        )

        calibrated_data = np.array(
            [
                [[4, 5, 4], [4, 5, 4], [4, 5, 4]],
                [[5, 6, 5], [5, 6, 5], [5, 6, 5]],
                [[6, 7, 6], [6, 7, 6], [6, 7, 6]],
            ]
        )

        # This reordering does not pay attention to the values within the
        # calibrated data, the rankings created to perform the sorting are
        # taken exclusively from the raw_data.
        result_data = np.array(
            [
                [[5, 6, 5], [6, 6, 5], [5, 6, 5]],
                [[4, 5, 4], [4, 5, 4], [4, 5, 4]],
                [[6, 7, 6], [5, 7, 6], [6, 7, 6]],
            ]
        )

        raw_cube = self.cube.copy(data=raw_data)
        calibrated_cube = self.cube.copy(data=calibrated_data)

        result = Plugin().rank_ecc(calibrated_cube, raw_cube)
        self.assertArrayAlmostEqual(result.data, result_data)

    def test_2d_cube(self):
        """Test that the plugin returns the correct cube data for a
        2d input cube."""
        raw_data = np.array([[1, 1], [3, 2], [2, 3]])
        calibrated_data = np.array([[1, 1], [2, 2], [3, 3]])
        result_data = raw_data.copy()

        raw_cube = self.cube_2d.copy(data=raw_data)
        calibrated_cube = self.cube_2d.copy(data=calibrated_data)

        result = Plugin().rank_ecc(calibrated_cube, raw_cube)
        self.assertArrayAlmostEqual(result.data, result_data)

    def test_2d_cube_masked(self):
        """Test that the plugin returns the correct cube data for a
        2d input cube with a mask applied to each realization."""
        mask = np.array([[True, False], [True, False], [True, False]])
        raw_data = np.array([[1, 9], [3, 5], [2, 7]])
        calibrated_data = np.ma.MaskedArray(
            [[1, 6], [2, 8], [3, 10]], mask=mask, dtype=np.float32
        )
        result_data = np.array(
            [[np.nan, 10], [np.nan, 6], [np.nan, 8]], dtype=np.float32
        )

        raw_cube = self.cube_2d.copy(data=raw_data)
        calibrated_cube = self.cube_2d.copy(data=calibrated_data)

        result = Plugin().rank_ecc(calibrated_cube, raw_cube)
        self.assertArrayAlmostEqual(result.data.data, result_data)
        self.assertArrayEqual(result.data.mask, mask)
        self.assertEqual(result.data.dtype, np.float32)

    def test_2d_cube_masked_nans(self):
        """Test that the plugin returns the correct cube data for a
        2d input cube with a mask applied to each realization, and there are
        nans under the mask."""
        mask = np.array([[True, False], [True, False], [True, False]])
        raw_data = np.array([[1, 9], [3, 5], [2, 7]])
        calibrated_data = np.ma.MaskedArray(
            [[np.nan, 6], [np.nan, 8], [np.nan, 10]], mask=mask, dtype=np.float32
        )
        result_data = np.array(
            [[np.nan, 10], [np.nan, 6], [np.nan, 8]], dtype=np.float32
        )

        raw_cube = self.cube_2d.copy(data=raw_data)
        calibrated_cube = self.cube_2d.copy(data=calibrated_data)

        result = Plugin().rank_ecc(calibrated_cube, raw_cube)
        self.assertArrayAlmostEqual(result.data.data, result_data)
        self.assertArrayEqual(result.data.mask, mask)
        self.assertEqual(result.data.dtype, np.float32)

    def test_2d_cube_tied_values_random(self):
        """
        Test that the plugin returns the correct cube data for a
        2d input cube, when there are tied values within the
        raw ensemble realizations. As there are two possible options for the
        result data, as the tie is decided randomly, both possible result
        data options are checked.
        """
        raw_data = np.array([[1, 1], [3, 2], [2, 2]])
        calibrated_data = np.array([[1, 1], [2, 2], [3, 3]])

        # Reordering of the calibrated_data array to match
        # the raw_data ordering
        result_data_first = np.array([[1, 1], [3, 2], [2, 3]])
        result_data_second = np.array([[1, 1], [3, 3], [2, 2]])

        raw_cube = self.cube_2d.copy(data=raw_data)
        calibrated_cube = self.cube_2d.copy(data=calibrated_data)

        result = Plugin().rank_ecc(calibrated_cube, raw_cube)
        permutations = [result_data_first, result_data_second]
        matches = [np.array_equal(aresult, result.data) for aresult in permutations]
        self.assertIn(True, matches)

    def test_2d_cube_tied_values_random_with_seed(self):
        """
        Test that the plugin returns the correct cube data for a
        2d input cube, when there are tied values within the
        raw ensemble realizations. The random seed is specified to ensure that
        only one option, out of the two possible options will be returned.
        """
        raw_data = np.array([[1, 1], [3, 2], [2, 2]])
        calibrated_data = np.array([[1, 1], [2, 2], [3, 3]])
        result_data = np.array([[1, 1], [3, 3], [2, 2]])

        raw_cube = self.cube_2d.copy(data=raw_data)
        calibrated_cube = self.cube_2d.copy(data=calibrated_data)

        result = Plugin().rank_ecc(calibrated_cube, raw_cube, random_seed=1)
        self.assertArrayAlmostEqual(result.data, result_data)

    def test_2d_cube_tied_values_realization(self):
        """
        Test that the plugin returns the correct cube data for a
        2d input cube, when there are tied values within the
        raw ensemble realizations. In this test, ties should be broken by assigning
        values to the highest realization number first.
        """
        raw_data = np.array([[1, 1], [3, 2], [2, 2]])
        calibrated_data = np.array([[1, 1], [2, 2], [3, 3]])
        result_data = np.array([[1, 1], [3, 2], [2, 3]])

        raw_cube = self.cube_2d.copy(data=raw_data)
        calibrated_cube = self.cube_2d.copy(data=calibrated_data)

        result = Plugin().rank_ecc(calibrated_cube, raw_cube, tie_break="realization")
        self.assertArrayAlmostEqual(result.data, result_data)

    def test_1d_cube(self):
        """
        Test that the plugin returns the correct cube data for a
        1d input cube.
        """
        raw_data = np.array([3, 2, 1])
        calibrated_data = np.array([1, 2, 3])
        result_data = raw_data.copy()

        cube = self.cube[:, 0, 0].copy()
        raw_cube = cube.copy(data=raw_data)
        calibrated_cube = cube.copy(data=calibrated_data)

        result = Plugin().rank_ecc(calibrated_cube, raw_cube)
        self.assertArrayAlmostEqual(result.data, result_data)

    def test_1d_cube_random_ordering(self):
        """
        Test that the plugin returns the correct cube data for a
        1d input cube, if random ordering is selected.

        Random ordering does not use the ordering from the raw ensemble,
        and instead just orders the input values randomly.
        """
        raw_data = np.array([3, 2, 1])
        calibrated_data = np.array([1, 2, 3])

        cube = self.cube[:, 0, 0].copy()
        raw_cube = cube.copy(data=raw_data)
        calibrated_cube = cube.copy(data=calibrated_data)

        result = Plugin().rank_ecc(calibrated_cube, raw_cube, random_ordering=True)

        permutations = list(itertools.permutations(raw_data))
        permutations = [np.array(permutation) for permutation in permutations]

        matches = [np.array_equal(aresult, result.data) for aresult in permutations]
        self.assertIn(True, matches)

    def test_bad_tie_break_exception(self):
        """
        Test that the correct exception is raised when an unknown method is input for
        tie_break.
        """

        raw_data = np.array([[1, 1], [3, 2], [2, 2]])
        calibrated_data = np.array([[1, 1], [2, 2], [3, 3]])

        raw_cube = self.cube_2d.copy(data=raw_data)
        calibrated_cube = self.cube_2d.copy(data=calibrated_data)

        message = (
            'Input tie_break must be either "random", or "realization", not "kittens".'
        )
        with self.assertRaisesRegex(ValueError, message):
            Plugin().rank_ecc(calibrated_cube, raw_cube, tie_break="kittens")


class Test__check_input_cube_masks(IrisTest):
    """Test the _check_input_cube_masks method in the EnsembleReordering plugin."""

    def setUp(self):
        """
        Create a raw realization forecast with a fixed set of realization
        numbers and a percentile cube following ensemble reordering.
        """
        self.raw_cube = set_up_variable_cube(
            ECC_TEMPERATURE_REALIZATIONS.copy(), realizations=[10, 11, 12]
        )
        self.post_processed_percentiles = set_up_percentile_cube(
            np.sort(ECC_TEMPERATURE_REALIZATIONS.copy(), axis=0),
            np.array([25, 50, 75], dtype=np.float32),
        )

    def test_unmasked_data(self):
        """Test unmasked data does not raise any errors."""
        Plugin._check_input_cube_masks(self.post_processed_percentiles, self.raw_cube)

    def test_only_one_post_processed_forecast_masked(self):
        """
        Test no error is raised if only the post_processed_forecast is masked.
        """
        self.post_processed_percentiles.data[:, 0, 0] = np.nan
        self.post_processed_percentiles.data = np.ma.masked_invalid(
            self.post_processed_percentiles.data
        )
        Plugin._check_input_cube_masks(self.post_processed_percentiles, self.raw_cube)

    def test_only_raw_cube_masked(self):
        """
        Test an error is raised if only the raw_cube is masked.
        """
        self.raw_cube.data[:, 0, 0] = np.nan
        self.raw_cube.data = np.ma.masked_invalid(self.raw_cube.data)
        message = (
            "The raw_forecast provided has a mask, but the post_processed_forecast "
            "isn't masked. The post_processed_forecast and the raw_forecast "
            "should have the same mask applied to them."
        )
        with self.assertRaisesRegex(ValueError, message):
            Plugin._check_input_cube_masks(
                self.post_processed_percentiles, self.raw_cube
            )

    def test_post_processed_forecast_inconsistent_mask(self):
        """Test an error is raised if the post_processed_forecast has an
        inconsistent mask.
        """
        self.post_processed_percentiles.data[2, 0, 0] = np.nan
        self.post_processed_percentiles.data = np.ma.masked_invalid(
            self.post_processed_percentiles.data
        )
        self.raw_cube.data[:, 0, 0] = np.nan
        self.raw_cube.data = np.ma.masked_invalid(self.raw_cube.data)

        message = (
            "The post_processed_forecast does not have same mask on all x-y slices"
        )
        with self.assertRaisesRegex(ValueError, message):
            Plugin._check_input_cube_masks(
                self.post_processed_percentiles, self.raw_cube
            )

    def test_raw_forecast_inconsistent_mask(self):
        """Test an error is raised if the raw_forecast has an
        inconsistent mask.
        """
        self.post_processed_percentiles.data[:, 0, 0] = np.nan
        self.post_processed_percentiles.data = np.ma.masked_invalid(
            self.post_processed_percentiles.data
        )
        self.raw_cube.data[2, 0, 0] = np.nan
        self.raw_cube.data = np.ma.masked_invalid(self.raw_cube.data)

        message = (
            "The raw_forecast x-y slices do not all have the"
            " same mask as the post_processed_forecast."
        )
        with self.assertRaisesRegex(ValueError, message):
            Plugin._check_input_cube_masks(
                self.post_processed_percentiles, self.raw_cube
            )

    def test_consistent_masks(self):
        """Test no error is raised if the raw_forecast and
        post_processed_forecast have consistent masks.
        """
        self.post_processed_percentiles.data[:, 0, 0] = np.nan
        self.post_processed_percentiles.data = np.ma.masked_invalid(
            self.post_processed_percentiles.data
        )
        self.raw_cube.data[:, 0, 0] = np.nan
        self.raw_cube.data = np.ma.masked_invalid(self.raw_cube.data)

        Plugin._check_input_cube_masks(self.post_processed_percentiles, self.raw_cube)


class Test_process(IrisTest):
    """Test the EnsembleReordering plugin."""

    def setUp(self):
        """
        Create a raw realization forecast with a fixed set of realization
        numbers and a percentile cube following ensemble reordering.
        """
        self.raw_cube = set_up_variable_cube(
            ECC_TEMPERATURE_REALIZATIONS.copy(), realizations=[10, 11, 12]
        )
        self.post_processed_percentiles = set_up_percentile_cube(
            np.sort(ECC_TEMPERATURE_REALIZATIONS.copy(), axis=0),
            np.array([25, 50, 75], dtype=np.float32),
        )

    def test_basic(self):
        """
        Test that the plugin returns an iris.cube.Cube, the cube has a
        realization coordinate with specific realization numbers and is
        correctly re-ordered to match the source realizations.
        """
        expected_data = self.raw_cube.data.copy()
        result = Plugin().process(self.post_processed_percentiles, self.raw_cube)
        self.assertIsInstance(result, Cube)
        self.assertTrue(result.coords("realization"))
        self.assertEqual(
            result.coord("realization"), self.raw_cube.coord("realization")
        )
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_basic_masked_input_data(self):
        """
        Test that the plugin returns an iris.cube.Cube, the cube has a
        realization coordinate with specific realization numbers and is
        correctly re-ordered to match the source realizations, when the
        input data is masked.
        """
        # Assuming input data and raw ensemble are masked in the same way.
        self.raw_cube.data[:, 0, 0] = np.nan
        self.raw_cube.data = np.ma.masked_invalid(self.raw_cube.data)
        self.post_processed_percentiles.data[:, 0, 0] = np.nan
        self.post_processed_percentiles.data = np.ma.masked_invalid(
            self.post_processed_percentiles.data
        )
        expected_data = self.raw_cube.data.copy()
        result = Plugin().process(self.post_processed_percentiles, self.raw_cube)
        self.assertIsInstance(result, Cube)
        self.assertTrue(result.coords("realization"))
        self.assertEqual(
            result.coord("realization"), self.raw_cube.coord("realization")
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertArrayEqual(result.data.mask, expected_data.mask)

    def test_basic_masked_input_data_not_nans(self):
        """
        Test that the plugin returns an iris.cube.Cube, the cube has a
        realization coordinate with specific realization numbers and is
        correctly re-ordered to match the source realizations, when the
        input data is masked and the masked data is not a nan.
        """
        # Assuming input data and raw ensemble are masked in the same way.
        self.raw_cube.data[:, 0, 0] = 1000
        self.raw_cube.data = np.ma.masked_equal(self.raw_cube.data, 1000)
        self.post_processed_percentiles.data[:, 0, 0] = 1000
        self.post_processed_percentiles.data = np.ma.masked_equal(
            self.post_processed_percentiles.data, 1000
        )
        expected_data = self.raw_cube.data.copy()
        result = Plugin().process(self.post_processed_percentiles, self.raw_cube)
        self.assertIsInstance(result, Cube)
        self.assertTrue(result.coords("realization"))
        self.assertEqual(
            result.coord("realization"), self.raw_cube.coord("realization")
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertArrayEqual(result.data.mask, expected_data.mask)

    def test_1d_cube_random_ordering(self):
        """
        Test that the plugin returns the correct cube data for a
        1d input cube, if random ordering is selected.
        """
        raw_data = np.array([3, 2, 1])

        post_processed_percentiles_data = np.array([1, 2, 3])

        raw_cube = self.raw_cube[:, 0, 0]
        raw_cube.data = raw_data
        post_processed_percentiles = self.post_processed_percentiles[:, 0, 0]
        post_processed_percentiles.data = post_processed_percentiles_data

        result = Plugin().process(
            post_processed_percentiles, raw_cube, random_ordering=True
        )

        permutations = list(itertools.permutations(raw_data))
        permutations = [np.array(permutation) for permutation in permutations]

        matches = [np.array_equal(aresult, result.data) for aresult in permutations]
        self.assertIn(True, matches)

    def test_1d_cube_recycling_raw_ensemble_realizations(self):
        """
        Test that the plugin returns the correct cube data for a
        1d input cube, if the number of raw ensemble realizations is fewer
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
        raw_data = np.array([1, 2])
        post_processed_percentiles_data = np.array([1, 2, 3])
        expected_first = np.array([1, 3, 2])
        expected_second = np.array([2, 3, 1])

        raw_cube = self.raw_cube[:2, 0, 0]
        raw_cube.data = raw_data
        post_processed_percentiles = self.post_processed_percentiles[:, 0, 0]
        post_processed_percentiles.data = post_processed_percentiles_data

        result = Plugin().process(post_processed_percentiles, raw_cube)
        permutations = [expected_first, expected_second]
        matches = [np.array_equal(aresult, result.data) for aresult in permutations]
        self.assertIn(True, matches)


if __name__ == "__main__":
    unittest.main()
