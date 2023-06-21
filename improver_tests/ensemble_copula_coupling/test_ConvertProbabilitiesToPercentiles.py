# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
`ensemble_copula_coupling.ConvertProbabilitiesToPercentiles` class.
"""
import unittest
from datetime import datetime

import cf_units as unit
import numpy as np
import pytest
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertProbabilitiesToPercentiles as Plugin,
)
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
)

from .ecc_test_data import (
    ECC_TEMPERATURE_PROBABILITIES,
    ECC_TEMPERATURE_THRESHOLDS,
    set_up_spot_test_cube,
)


class Test__add_bounds_to_thresholds_and_probabilities(IrisTest):

    """
    Test the _add_bounds_to_thresholds_and_probabilities method of the
    ConvertProbabilitiesToPercentiles.
    """

    def setUp(self):
        """Set up data for testing."""
        self.probabilities_for_cdf = ECC_TEMPERATURE_PROBABILITIES.reshape(3, 9)
        self.threshold_points = ECC_TEMPERATURE_THRESHOLDS
        self.bounds_pairing = (-40, 50)

    def test_basic(self):
        """Test that the plugin returns two numpy arrays."""
        result = Plugin()._add_bounds_to_thresholds_and_probabilities(
            self.threshold_points, self.probabilities_for_cdf, self.bounds_pairing
        )
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_bounds_of_threshold_points(self):
        """
        Test that the plugin returns the expected results for the
        threshold_points, where they've been padded with the values from
        the bounds_pairing.
        """
        result = Plugin()._add_bounds_to_thresholds_and_probabilities(
            self.threshold_points, self.probabilities_for_cdf, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result[0][0], self.bounds_pairing[0])
        self.assertArrayAlmostEqual(result[0][-1], self.bounds_pairing[1])

    def test_probability_data(self):
        """
        Test that the plugin returns the expected results for the
        probabilities, where they've been padded with zeros and ones to
        represent the extreme ends of the Cumulative Distribution Function.
        """
        zero_array = np.zeros(self.probabilities_for_cdf[:, 0].shape)
        one_array = np.ones(self.probabilities_for_cdf[:, 0].shape)
        result = Plugin()._add_bounds_to_thresholds_and_probabilities(
            self.threshold_points, self.probabilities_for_cdf, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result[1][:, 0], zero_array)
        self.assertArrayAlmostEqual(result[1][:, -1], one_array)

    def test_endpoints_of_distribution_exceeded(self):
        """
        Test that the plugin raises a ValueError when the constant
        end points of the distribution are exceeded by a threshold value
        used in the forecast.
        """
        probabilities_for_cdf = np.array([[0.05, 0.7, 0.95]])
        threshold_points = np.array([8, 10, 60])
        msg = (
            "The calculated threshold values \\[-40   8  10  60  50\\] are "
            "not in ascending order as required for the cumulative distribution "
            "function \\(CDF\\). This is due to the threshold values exceeding "
            "the range given by the ECC bounds \\(-40, 50\\)."
        )
        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._add_bounds_to_thresholds_and_probabilities(
                threshold_points, probabilities_for_cdf, self.bounds_pairing
            )

    def test_endpoints_of_distribution_exceeded_warning(self):
        """
        Test that the plugin raises a warning message when the constant
        end points of the distribution are exceeded by a threshold value
        used in the forecast and the ecc_bounds_warning keyword argument
        has been specified.
        """
        probabilities_for_cdf = np.array([[0.05, 0.7, 0.95]])
        threshold_points = np.array([8, 10, 60])
        plugin = Plugin(ecc_bounds_warning=True)
        warning_msg = (
            "The calculated threshold values \\[-40   8  10  60  50\\] are "
            "not in ascending order as required for the cumulative distribution "
            "function \\(CDF\\). This is due to the threshold values exceeding "
            "the range given by the ECC bounds \\(-40, 50\\). The threshold "
            "points that have exceeded the existing bounds will be used as "
            "new bounds."
        )
        with pytest.warns(UserWarning, match=warning_msg):
            plugin._add_bounds_to_thresholds_and_probabilities(
                threshold_points, probabilities_for_cdf, self.bounds_pairing
            )

    def test_new_endpoints_generation(self):
        """Test that the plugin re-applies the threshold bounds using the
        maximum and minimum threshold points values when the original bounds
        have been exceeded and ecc_bounds_warning has been set."""
        probabilities_for_cdf = np.array([[0.05, 0.7, 0.95]])
        threshold_points = np.array([-50, 10, 60])
        plugin = Plugin(ecc_bounds_warning=True)
        result = plugin._add_bounds_to_thresholds_and_probabilities(
            threshold_points, probabilities_for_cdf, self.bounds_pairing
        )
        self.assertEqual(max(result[0]), max(threshold_points))
        self.assertEqual(min(result[0]), min(threshold_points))


class Test__probabilities_to_percentiles(IrisTest):

    """Test the _probabilities_to_percentiles method of the
    ConvertProbabilitiesToPercentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.cube = set_up_probability_cube(
            ECC_TEMPERATURE_PROBABILITIES,
            ECC_TEMPERATURE_THRESHOLDS,
            threshold_units="degC",
        )
        self.percentiles = [10, 50, 90]
        self.bounds_pairing = (-40, 50)

    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube with the expected name"""
        result = Plugin()._probabilities_to_percentiles(
            self.cube, self.percentiles, self.bounds_pairing
        )
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "air_temperature")

    def test_unknown_thresholding(self):
        """Test an error is raised for "between thresholds" probability cubes"""
        self.cube.coord(var_name="threshold").attributes[
            "spp__relative_to_threshold"
        ] = "between"
        msg = "Probabilities to percentiles only implemented for"
        with self.assertRaisesRegex(NotImplementedError, msg):
            Plugin()._probabilities_to_percentiles(
                self.cube, self.percentiles, self.bounds_pairing
            )

    def test_percentile_coord(self):
        """Test that the plugin returns an Iris.cube.Cube with an appropriate
        percentile coordinate with suitable units.
        """
        result = Plugin()._probabilities_to_percentiles(
            self.cube, self.percentiles, self.bounds_pairing
        )
        self.assertIsInstance(result.coord("percentile"), DimCoord)
        self.assertArrayEqual(result.coord("percentile").points, self.percentiles)
        self.assertEqual(result.coord("percentile").units, unit.Unit("%"))

    def test_transpose_cube_dimensions(self):
        """
        Test that the plugin returns an the expected data, when comparing
        input cubes which have dimensions in a different order.
        """
        # Calculate result for nontransposed cube.
        nontransposed_result = Plugin()._probabilities_to_percentiles(
            self.cube, self.percentiles, self.bounds_pairing
        )

        # Calculate result for transposed cube.
        # Original cube dimensions are [P, Y, X].
        # Transposed cube dimensions are [X, Y, P].
        self.cube.transpose([2, 1, 0])
        transposed_result = Plugin()._probabilities_to_percentiles(
            self.cube, self.percentiles, self.bounds_pairing
        )

        # Result cube will be [P, X, Y]
        # Transpose cube to be [P, Y, X]
        transposed_result.transpose([0, 2, 1])
        self.assertArrayAlmostEqual(nontransposed_result.data, transposed_result.data)

    def test_simple_check_data_above(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles when input probabilities are given
        for being above a threshold.
        The input cube contains probabilities that values are above a given
        threshold.
        """
        expected = np.array([8.15384615, 9.38461538, 11.6])
        expected = expected[:, np.newaxis, np.newaxis]

        data = np.array([0.95, 0.3, 0.05])
        data = data[:, np.newaxis, np.newaxis]

        cube = set_up_probability_cube(
            data.astype(np.float32), ECC_TEMPERATURE_THRESHOLDS, threshold_units="degC"
        )

        result = Plugin()._probabilities_to_percentiles(
            cube, self.percentiles, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result.data, expected)

    def test_simple_check_data_below(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles when input probabilities are given
        for being below a threshold.
        The input cube contains probabilities that values are below a given
        threshold.
        """
        expected = np.array([8.4, 10.61538462, 11.84615385])
        expected = expected[:, np.newaxis, np.newaxis]

        data = np.array([0.95, 0.3, 0.05])[::-1]
        data = data[:, np.newaxis, np.newaxis]

        cube = set_up_probability_cube(
            data.astype(np.float32),
            ECC_TEMPERATURE_THRESHOLDS,
            threshold_units="degC",
            spp__relative_to_threshold="below",
        )

        result = Plugin()._probabilities_to_percentiles(
            cube, self.percentiles, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result.data, expected)

    def test_check_data_multiple_timesteps(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles.
        """
        expected = np.array(
            [
                [[[8.0, 8.0], [-8.0, 8.66666667]], [[8.0, -16.0], [8.0, -16.0]]],
                [[[12.0, 12.0], [12.0, 12.0]], [[10.5, 10.0], [10.5, 10.0]]],
                [[[31.0, 31.0], [31.0, 31.0]], [[11.5, 11.33333333], [11.5, 12.0]]],
            ],
            dtype=np.float32,
        )

        cube = set_up_probability_cube(
            np.zeros((3, 2, 2), dtype=np.float32),
            ECC_TEMPERATURE_THRESHOLDS,
            threshold_units="degC",
            time=datetime(2015, 11, 23, 7),
            frt=datetime(2015, 11, 23, 6),
        )
        cube = add_coordinate(
            cube,
            [datetime(2015, 11, 23, 7), datetime(2015, 11, 23, 8)],
            "time",
            is_datetime=True,
            order=[1, 0, 2, 3],
        )

        cube.data = np.array(
            [
                [[[0.8, 0.8], [0.7, 0.9]], [[0.8, 0.6], [0.8, 0.6]]],
                [[[0.6, 0.6], [0.6, 0.6]], [[0.5, 0.4], [0.5, 0.4]]],
                [[[0.4, 0.4], [0.4, 0.4]], [[0.1, 0.1], [0.1, 0.2]]],
            ],
            dtype=np.float32,
        )

        percentiles = [20, 60, 80]
        result = Plugin()._probabilities_to_percentiles(
            cube, percentiles, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result.data, expected, decimal=5)

    def test_probabilities_not_monotonically_increasing(self):
        """
        Test that the plugin raises a Warning when the probabilities
        of the Cumulative Distribution Function are not monotonically
        increasing.
        """
        data = np.array([0.05, 0.7, 0.95])
        data = data[:, np.newaxis, np.newaxis]
        cube = set_up_probability_cube(
            data.astype(np.float32), ECC_TEMPERATURE_THRESHOLDS, threshold_units="degC"
        )

        warning_msg = "The probability values used to construct the"
        with pytest.warns(UserWarning, match=warning_msg):
            Plugin()._probabilities_to_percentiles(
                cube, self.percentiles, self.bounds_pairing
            )

    def test_result_cube_has_no_air_temperature_threshold_coordinate(self):
        """
        Test that the plugin returns a cube with coordinates that
        do not include a threshold-type coordinate.
        """
        result = Plugin()._probabilities_to_percentiles(
            self.cube, self.percentiles, self.bounds_pairing
        )
        try:
            threshold_coord = find_threshold_coordinate(result)
        except CoordinateNotFoundError:
            threshold_coord = None
        self.assertIsNone(threshold_coord)

    def test_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles.
        """
        data = np.array(
            [
                [[15.8, 8.0, 10.4], [-16.0, 8.0, -30.4], [-30.4, -34.0, -35.2]],
                [[31.0, 10.0, 12.0], [10.0, 10.0, 8.0], [8.0, -10.0, -16.0]],
                [[46.2, 31.0, 42.4], [31.0, 11.6, 12.0], [11.0, 9.0, 3.2]],
            ],
            dtype=np.float32,
        )

        result = Plugin()._probabilities_to_percentiles(
            self.cube, self.percentiles, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result.data, data, decimal=5)

    def test_check_single_threshold(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if a single threshold is used for
        constructing the percentiles.
        """
        data = np.array(
            [
                [[12.2, 8.0, 12.2], [-16.0, 8.0, -30.4], [-30.4, -34.0, -35.2]],
                [
                    [29.0, 26.66666667, 29.0],
                    [23.75, 26.66666667, 8.0],
                    [8.0, -10.0, -16.0],
                ],
                [
                    [45.8, 45.33333333, 45.8],
                    [44.75, 45.33333333, 41.6],
                    [41.6, 29.0, 3.2],
                ],
            ],
            dtype=np.float32,
        )

        threshold_coord = find_threshold_coordinate(self.cube)
        cube = next(self.cube.slices_over(threshold_coord))

        result = Plugin()._probabilities_to_percentiles(
            cube, self.percentiles, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result.data, data, decimal=5)

    def test_lots_of_probability_thresholds(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if there are lots of thresholds.
        """
        data = np.array(
            [
                [[2.9, 2.9, 2.9], [2.9, 2.9, 2.9], [2.9, 2.9, 2.9]],
                [[14.5, 14.5, 14.5], [14.5, 14.5, 14.5], [14.5, 14.5, 14.5]],
                [
                    [26.099998, 26.099998, 26.099998],
                    [26.099998, 26.099998, 26.099998],
                    [26.099998, 26.099998, 26.099998],
                ],
            ],
            dtype=np.float32,
        )

        input_probs = np.tile(np.linspace(1, 0, 30), (3, 3, 1)).T
        cube = set_up_probability_cube(
            input_probs.astype(np.float32),
            np.arange(30).astype(np.float32),
            threshold_units="degC",
        )

        result = Plugin()._probabilities_to_percentiles(
            cube, self.percentiles, self.bounds_pairing
        )

        self.assertArrayAlmostEqual(result.data, data)

    def test_lots_of_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if lots of percentile values are
        requested.
        """
        data = np.array(
            [
                [[13.9, -16.0, 10.2], [-28.0, -16.0, -35.2], [-35.2, -37.0, -37.6]],
                [[17.7, 8.25, 10.6], [-4.0, 8.25, -25.6], [-25.6, -31.0, -32.8]],
                [[21.5, 8.75, 11.0], [8.33333333, 8.75, -16.0], [-16.0, -25.0, -28.0]],
                [[25.3, 9.25, 11.4], [9.0, 9.25, -6.4], [-6.4, -19.0, -23.2]],
                [[29.1, 9.75, 11.8], [9.66666667, 9.75, 3.2], [3.2, -13.0, -18.4]],
                [
                    [32.9, 10.33333333, 15.8],
                    [10.33333333, 10.2, 8.5],
                    [8.33333333, -7.0, -13.6],
                ],
                [[36.7, 11.0, 23.4], [11.0, 10.6, 9.5], [9.0, -1.0, -8.8]],
                [
                    [40.5, 11.66666667, 31.0],
                    [11.66666667, 11.0, 10.5],
                    [9.66666667, 5.0, -4.0],
                ],
                [[44.3, 21.5, 38.6], [21.5, 11.4, 11.5], [10.5, 8.5, 0.8]],
                [[48.1, 40.5, 46.2], [40.5, 11.8, 31.0], [11.5, 9.5, 5.6]],
            ],
            dtype=np.float32,
        )

        percentiles = np.arange(5, 100, 10)
        result = Plugin()._probabilities_to_percentiles(
            self.cube, percentiles, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result.data, data, decimal=5)

    def test_check_data_spot_forecasts(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles for spot forecasts.
        """
        data = np.array(
            [
                [15.8, 8.0, 10.4, -16.0, 8.0, -30.4, -30.4, -34.0, -35.2],
                [31.0, 10.0, 12.0, 10.0, 10.0, 8.0, 8.0, -10.0, -16.0],
                [46.2, 31.0, 42.4, 31.0, 11.6, 12.0, 11.0, 9.0, 3.2],
            ],
            dtype=np.float32,
        )

        cube = set_up_spot_test_cube(cube_type="probability")
        result = Plugin()._probabilities_to_percentiles(
            cube, self.percentiles, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result.data, data, decimal=5)

    def test_masked_data_below(self):
        """Test that if mask_percentiles is true, data is masked as
        expected when input probability data is below a threshold"""

        expected_mask = np.full_like(self.cube.data, False, dtype=bool)
        expected_mask[:, 0, 0] = True
        expected_mask[1, 0, 2] = True
        expected_mask[2, 0] = True
        expected_mask[2, 1, 2] = True
        expected_mask[2, 1, 0] = True

        cube = set_up_probability_cube(
            1 - self.cube.data,
            [200, 1000, 15000],
            variable_name="cloud_base_altitude_assuming_only_consider_cloud_\
                area_fraction_greater_than_4p5_oktas",
            threshold_units="m",
            spp__relative_to_threshold="below",
        )

        result = Plugin(mask_percentiles=True)._probabilities_to_percentiles(
            cube, self.percentiles, [0, 22000]
        )
        self.assertArrayEqual(result.data.mask, expected_mask)

    def test_masked_data_above(self):
        """Test that if mask_percentiles is true, data is masked as expected
        when input probability data is above a threshold"""

        expected_mask = np.full_like(self.cube.data, False, dtype=bool)
        expected_mask[:, 0, 0] = True
        expected_mask[1, 0, 2] = True
        expected_mask[2, 0] = True
        expected_mask[2, 1, 2] = True
        expected_mask[2, 1, 0] = True

        cube = set_up_probability_cube(
            self.cube.data,
            [200, 1000, 15000],
            variable_name="cloud_base_altitude_assuming_only_consider_cloud_\
                area_fraction_greater_than_4p5_oktas",
            threshold_units="m",
            spp__relative_to_threshold="above",
        )

        result = Plugin(mask_percentiles=True)._probabilities_to_percentiles(
            cube, self.percentiles, [0, 22000]
        )

        self.assertArrayEqual(result.data.mask, expected_mask)


class Test_process(IrisTest):

    """
    Test the process method of the ConvertProbabilitiesToPercentiles plugin.
    """

    def setUp(self):
        """Set up temperature probability cube and expected output percentiles."""
        self.cube = set_up_probability_cube(
            ECC_TEMPERATURE_PROBABILITIES,
            ECC_TEMPERATURE_THRESHOLDS,
            threshold_units="degC",
        )

        self.percentile_25 = np.array(
            [[24.0, 8.75, 11.0], [8.33333333, 8.75, -46.0], [-46.0, -66.25, -73.0]],
            dtype=np.float32,
        )
        self.percentile_50 = np.array(
            [[36.0, 10.0, 12.0], [10.0, 10.0, 8.0], [8.0, -32.5, -46.0]],
            dtype=np.float32,
        )
        self.percentile_75 = np.array(
            [
                [48.0, 11.66666667, 36.0],
                [11.66666667, 11.0, 10.5],
                [9.66666667, 1.25, -19.0],
            ],
            dtype=np.float32,
        )

    def test_check_data_specifying_no_of_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific number of percentiles.
        """
        expected_data = np.array(
            [self.percentile_25, self.percentile_50, self.percentile_75]
        )
        result = Plugin().process(self.cube, no_of_percentiles=3)
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)

    def test_check_data_specifying_single_percentile(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific percentile passes in as a single realization
        list.
        """
        expected_data = np.array(self.percentile_25)
        result = Plugin().process(self.cube, percentiles=[25])
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)

    def test_check_data_specifying_single_percentile_not_as_list(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific percentile passed in as a value.
        """
        expected_data = np.array(self.percentile_25)
        result = Plugin().process(self.cube, percentiles=25)
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)

    def test_check_data_specifying_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific set of percentiles.
        """
        expected_data = np.array(
            [self.percentile_25, self.percentile_50, self.percentile_75]
        )
        result = Plugin().process(self.cube, percentiles=[25, 50, 75])
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)

    def test_check_data_not_specifying_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values without specifying the number of percentiles.
        """
        expected_data = np.array(
            [self.percentile_25, self.percentile_50, self.percentile_75]
        )
        result = Plugin().process(self.cube)
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)

    def test_check_data_masked_input_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values when the input data is masked.
        """
        cube = self.cube.copy()
        cube.data[:, 0, 0] = np.nan
        cube.data = np.ma.masked_invalid(cube.data)
        expected_data = np.array(
            [self.percentile_25, self.percentile_50, self.percentile_75]
        )
        expected_data[:, 0, 0] = np.nan
        expected_data = np.ma.masked_invalid(expected_data)
        result = Plugin().process(cube)
        self.assertArrayAlmostEqual(result.data.data, expected_data.data, decimal=5)
        self.assertArrayEqual(result.data.mask, expected_data.mask)

    def test_check_data_masked_input_data_non_nans(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values when the input data is masked without underlying nans.
        """
        cube = self.cube.copy()
        cube.data[:, 0, 0] = 1000
        cube.data = np.ma.masked_equal(cube.data, 1000)
        expected_data = np.array(
            [self.percentile_25, self.percentile_50, self.percentile_75]
        )
        expected_data[:, 0, 0] = np.nan
        expected_data = np.ma.masked_invalid(expected_data)
        result = Plugin().process(cube)
        self.assertArrayAlmostEqual(result.data.data, expected_data.data, decimal=5)
        self.assertArrayEqual(result.data.mask, expected_data.mask)

    def test_check_data_over_specifying_percentiles(self):
        """
        Test that the plugin raises a suitable error when both a number and set
        or percentiles are specified.
        """
        msg = "Cannot specify both no_of_percentiles and percentiles"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin().process(self.cube, no_of_percentiles=3, percentiles=[25, 50, 75])

    def test_metadata(self):
        """Test name and cell methods are updated as expected after conversion"""
        threshold_coord = find_threshold_coordinate(self.cube)
        expected_name = threshold_coord.name()
        expected_units = threshold_coord.units
        # add a cell method indicating "max in period" for the underlying data
        self.cube.add_cell_method(
            CellMethod("max", coords="time", comments=f"of {expected_name}")
        )
        expected_cell_method = CellMethod("max", coords="time")
        result = Plugin().process(self.cube)
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(result.units, expected_units)
        self.assertEqual(result.cell_methods[0], expected_cell_method)

    def test_vicinity_metadata(self):
        """Test vicinity cube name is correctly regenerated after processing"""
        self.cube.rename("probability_of_air_temperature_in_vicinity_above_threshold")
        result = Plugin().process(self.cube)
        self.assertEqual(result.name(), "air_temperature_in_vicinity")


if __name__ == "__main__":
    unittest.main()
