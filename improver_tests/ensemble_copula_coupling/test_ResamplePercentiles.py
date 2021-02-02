# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
Unit tests for the `ensemble_copula_coupling.ResamplePercentiles` class.
"""
import unittest
from datetime import datetime

import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ResamplePercentiles as Plugin,
)
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_percentile_cube,
)
from improver.utilities.warnings_handler import ManageWarnings

from .ecc_test_data import set_up_spot_test_cube


class Test__add_bounds_to_percentiles_and_forecast_values(IrisTest):

    """
    Test the _add_bounds_to_percentiles_and_forecast_values method of the
    ResamplePercentiles plugin.
    """

    def setUp(self):
        """Set up realization and percentile cubes for testing."""
        data = np.tile(np.linspace(5, 10, 9), 3).reshape(3, 3, 3)
        data[0] -= 1
        data[1] += 1
        data[2] += 3
        self.forecast_at_percentiles = data.reshape(3, 9)
        self.percentiles = np.array([10, 50, 90], dtype=np.float32)
        self.bounds_pairing = (-40, 50)

    def test_basic(self):
        """Test that the plugin returns two numpy arrays."""
        result = Plugin()._add_bounds_to_percentiles_and_forecast_at_percentiles(
            self.percentiles, self.forecast_at_percentiles, self.bounds_pairing
        )
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_bounds_of_percentiles(self):
        """
        Test that the plugin returns the expected results for the
        percentiles, where the percentile values have been padded with 0 and 1.
        """
        result = Plugin()._add_bounds_to_percentiles_and_forecast_at_percentiles(
            self.percentiles, self.forecast_at_percentiles, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result[0][0], 0)
        self.assertArrayAlmostEqual(result[0][-1], 100)

    def test_probability_data(self):
        """
        Test that the plugin returns the expected results for the
        forecast values, where they've been padded with the values from the
        bounds_pairing.
        """
        lower_array = np.full(
            self.forecast_at_percentiles[:, 0].shape,
            self.bounds_pairing[0],
            dtype=np.float32,
        )
        upper_array = np.full(
            self.forecast_at_percentiles[:, 0].shape,
            self.bounds_pairing[1],
            dtype=np.float32,
        )
        result = Plugin()._add_bounds_to_percentiles_and_forecast_at_percentiles(
            self.percentiles, self.forecast_at_percentiles, self.bounds_pairing
        )
        self.assertArrayAlmostEqual(result[1][:, 0], lower_array)
        self.assertArrayAlmostEqual(result[1][:, -1], upper_array)

    def test_endpoints_of_distribution_exceeded(self):
        """
        Test that the plugin raises a ValueError when the constant
        end points of the distribution are exceeded by a forecast value.
        The end points must be outside the minimum and maximum within the
        forecast values.
        """
        forecast_at_percentiles = np.array([[8, 10, 60]])
        percentiles = np.array([5, 70, 95])
        msg = "Forecast values exist that fall outside the expected extrema"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._add_bounds_to_percentiles_and_forecast_at_percentiles(
                percentiles, forecast_at_percentiles, self.bounds_pairing
            )

    @ManageWarnings(record=True)
    def test_endpoints_of_distribution_exceeded_warning(self, warning_list=None):
        """
        Test that the plugin raises a warning message when the constant
        end points of the distribution are exceeded by a percentile value
        used in the forecast and the ecc_bounds_warning keyword argument
        has been specified.
        """
        forecast_at_percentiles = np.array([[8, 10, 60]])
        percentiles = np.array([5, 70, 95])
        plugin = Plugin(ecc_bounds_warning=True)
        warning_msg = "Forecast values exist that fall outside the expected extrema"
        plugin._add_bounds_to_percentiles_and_forecast_at_percentiles(
            percentiles, forecast_at_percentiles, self.bounds_pairing
        )
        self.assertTrue(any(warning_msg in str(item) for item in warning_list))

    @ManageWarnings(ignored_messages=["The end points added to the forecast at"])
    def test_new_endpoints_generation(self):
        """Test that the plugin re-applies the percentile bounds using the
        maximum and minimum percentile values when the original bounds have
        been exceeded and ecc_bounds_warning has been set."""
        forecast_at_percentiles = np.array([[8, 10, 60]])
        percentiles = np.array([5, 70, 95])
        plugin = Plugin(ecc_bounds_warning=True)
        result = plugin._add_bounds_to_percentiles_and_forecast_at_percentiles(
            percentiles, forecast_at_percentiles, self.bounds_pairing
        )
        self.assertEqual(
            np.max(result[1]),
            max([forecast_at_percentiles.max(), max(self.bounds_pairing)]),
        )
        self.assertEqual(
            np.min(result[1]),
            min([forecast_at_percentiles.min(), min(self.bounds_pairing)]),
        )

    def test_percentiles_not_ascending(self):
        """
        Test that the plugin raises a ValueError, if the percentiles are
        not in ascending order.
        """
        forecast_at_percentiles = np.array([[8, 10, 12]])
        percentiles = np.array([100, 0, -100])
        msg = "The percentiles must be in ascending order"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._add_bounds_to_percentiles_and_forecast_at_percentiles(
                percentiles, forecast_at_percentiles, self.bounds_pairing
            )


class Test__interpolate_percentiles(IrisTest):

    """
    Test the _interpolate_percentiles method of the ResamplePercentiles plugin.
    """

    def setUp(self):
        """Set up percentile cube."""
        data = np.tile(np.linspace(5, 10, 9), 3).reshape(3, 3, 3)
        data[0] -= 1
        data[1] += 1
        data[2] += 3
        self.perc_coord = "percentile"
        self.percentiles = np.array([10, 50, 90], dtype=np.float32)
        self.cube = set_up_percentile_cube(
            np.sort(data.astype(np.float32), axis=0),
            self.percentiles,
            name="air_temperature",
            units="degC",
        )
        self.bounds_pairing = (-40, 50)

    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube."""
        result = Plugin()._interpolate_percentiles(
            self.cube, self.percentiles, self.bounds_pairing, self.perc_coord
        )
        self.assertIsInstance(result, Cube)

    def test_transpose_cube_dimensions(self):
        """
        Test that the plugin returns an the expected data, when comparing
        input cubes which have dimensions in a different order.
        """
        # Calculate result for nontransposed cube.
        nontransposed_result = Plugin()._interpolate_percentiles(
            self.cube, self.percentiles, self.bounds_pairing, self.perc_coord
        )

        # Calculate result for transposed cube.
        # Original cube dimensions are [P, Y, X].
        # Transposed cube dimensions are [X, Y, P].
        self.cube.transpose([2, 1, 0])
        transposed_result = Plugin()._interpolate_percentiles(
            self.cube, self.percentiles, self.bounds_pairing, self.perc_coord
        )

        # Result cube will be [P, X, Y]
        # Transpose cube to be [P, Y, X]
        transposed_result.transpose([0, 2, 1])
        self.assertArrayAlmostEqual(nontransposed_result.data, transposed_result.data)

    def test_simple_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        forecast values at each percentile.
        """
        data = np.array([8, 10, 12])
        data = data[:, np.newaxis, np.newaxis]
        expected = data.copy()

        cube = set_up_percentile_cube(
            data.astype(np.float32),
            self.percentiles,
            name="air_temperature",
            units="degC",
        )

        result = Plugin()._interpolate_percentiles(
            cube, self.percentiles, self.bounds_pairing, self.perc_coord
        )
        self.assertArrayAlmostEqual(result.data, expected)

    def test_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles.
        """
        data = np.array(
            [
                [[4.5, 5.125, 5.75], [6.375, 7.0, 7.625], [8.25, 8.875, 9.5]],
                [[6.5, 7.125, 7.75], [8.375, 9.0, 9.625], [10.25, 10.875, 11.5]],
                [[7.5, 8.125, 8.75], [9.375, 10.0, 10.625], [11.25, 11.875, 12.5]],
            ]
        )

        percentiles = [20, 60, 80]
        result = Plugin()._interpolate_percentiles(
            self.cube, percentiles, self.bounds_pairing, self.perc_coord
        )
        self.assertArrayAlmostEqual(result.data, data)

    def test_check_data_multiple_timesteps(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles.
        """
        expected = np.array(
            [
                [
                    [[4.5, 5.21428571], [5.92857143, 6.64285714]],
                    [[7.35714286, 8.07142857], [8.78571429, 9.5]],
                ],
                [
                    [[6.5, 7.21428571], [7.92857143, 8.64285714]],
                    [[9.35714286, 10.07142857], [10.78571429, 11.5]],
                ],
                [
                    [[7.5, 8.21428571], [8.92857143, 9.64285714]],
                    [[10.35714286, 11.07142857], [11.78571429, 12.5]],
                ],
            ]
        )

        cube = set_up_percentile_cube(
            np.zeros((3, 2, 2), dtype=np.float32),
            self.percentiles,
            name="air_temperature",
            units="degC",
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

        data = np.tile(np.linspace(5, 10, 8), 3).reshape(3, 2, 2, 2)
        data[0] -= 1
        data[1] += 1
        data[2] += 3
        cube.data = data.astype(np.float32)

        percentiles = [20, 60, 80]
        result = Plugin()._interpolate_percentiles(
            cube, percentiles, self.bounds_pairing, self.perc_coord
        )
        self.assertArrayAlmostEqual(result.data, expected)

    def test_check_single_threshold(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if a single percentile is used within
        the input set of percentiles.
        """
        expected = np.array(
            [
                [[4.0, 4.625, 5.25], [5.875, 6.5, 7.125], [7.75, 8.375, 9.0]],
                [
                    [24.44444444, 24.79166667, 25.13888889],
                    [25.48611111, 25.83333333, 26.18055556],
                    [26.52777778, 26.875, 27.22222222],
                ],
                [
                    [44.88888889, 44.95833333, 45.02777778],
                    [45.09722222, 45.16666667, 45.23611111],
                    [45.30555556, 45.375, 45.44444444],
                ],
            ],
            dtype=np.float32,
        )

        cube = next(self.cube.slices_over(self.perc_coord))

        result = Plugin()._interpolate_percentiles(
            cube, self.percentiles, self.bounds_pairing, self.perc_coord
        )
        self.assertArrayAlmostEqual(result.data, expected)

    def test_lots_of_input_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if there are lots of thresholds.
        """
        expected_data = np.array(
            [
                [[11.0, 11.0, 11.0], [11.0, 11.0, 11.0], [11.0, 11.0, 11.0]],
                [[15.0, 15.0, 15.0], [15.0, 15.0, 15.0], [15.0, 15.0, 15.0]],
                [[19.0, 19.0, 19.0], [19.0, 19.0, 19.0], [19.0, 19.0, 19.0]],
            ]
        )

        input_forecast_values = np.tile(np.linspace(10, 20, 30), (3, 3, 1)).T
        cube = set_up_percentile_cube(
            input_forecast_values.astype(np.float32),
            np.linspace(0, 100, 30).astype(np.float32),
            name="air_temperature",
            units="degC",
        )

        result = Plugin()._interpolate_percentiles(
            cube, self.percentiles, self.bounds_pairing, self.perc_coord
        )
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_lots_of_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if lots of percentile values are
        requested.
        """
        data = np.array(
            [
                [
                    [-18.0, -17.6875, -17.375],
                    [-17.0625, -16.75, -16.4375],
                    [-16.125, -15.8125, -15.5],
                ],
                [[4.25, 4.875, 5.5], [6.125, 6.75, 7.375], [8.0, 8.625, 9.25]],
                [[4.75, 5.375, 6.0], [6.625, 7.25, 7.875], [8.5, 9.125, 9.75]],
                [[5.25, 5.875, 6.5], [7.125, 7.75, 8.375], [9.0, 9.625, 10.25]],
                [[5.75, 6.375, 7.0], [7.625, 8.25, 8.875], [9.5, 10.125, 10.75]],
                [[6.25, 6.875, 7.5], [8.125, 8.75, 9.375], [10.0, 10.625, 11.25]],
                [[6.75, 7.375, 8.0], [8.625, 9.25, 9.875], [10.5, 11.125, 11.75]],
                [[7.25, 7.875, 8.5], [9.125, 9.75, 10.375], [11.0, 11.625, 12.25]],
                [[7.75, 8.375, 9.0], [9.625, 10.25, 10.875], [11.5, 12.125, 12.75]],
                [
                    [29.0, 29.3125, 29.625],
                    [29.9375, 30.25, 30.5625],
                    [30.875, 31.1875, 31.5],
                ],
            ]
        )

        percentiles = np.arange(5, 100, 10)
        result = Plugin()._interpolate_percentiles(
            self.cube, percentiles, self.bounds_pairing, self.perc_coord
        )
        self.assertArrayAlmostEqual(result.data, data)

    def test_check_data_spot_forecasts(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles for spot forecasts.
        """
        spot_percentile_cube = set_up_spot_test_cube(cube_type="percentile")
        spot_percentile_cube.data = np.tile(np.linspace(5, 10, 3), 9).reshape(3, 9)

        data = np.array(
            [
                [5.0, 7.5, 10.0, 5.0, 7.5, 10.0, 5.0, 7.5, 10.0],
                [5.0, 7.5, 10.0, 5.0, 7.5, 10.0, 5.0, 7.5, 10.0],
                [5.0, 7.5, 10.0, 5.0, 7.5, 10.0, 5.0, 7.5, 10.0],
            ]
        )
        percentiles = spot_percentile_cube.coord("percentile").points
        result = Plugin()._interpolate_percentiles(
            spot_percentile_cube, percentiles, self.bounds_pairing, self.perc_coord
        )
        self.assertArrayAlmostEqual(result.data, data)


class Test_process(IrisTest):

    """Test the process plugin of the Resample Percentiles plugin."""

    def setUp(self):
        """Set up percentile cube."""
        data = np.tile(np.linspace(5, 10, 9), 3).reshape(3, 3, 3)
        data[0] -= 1
        data[1] += 1
        data[2] += 3
        self.percentile_cube = set_up_percentile_cube(
            data.astype(np.float32),
            np.array([10, 50, 90], dtype=np.float32),
            name="air_temperature",
            units="degC",
        )

    @ManageWarnings(ignored_messages=["Only a single cube so no differences"])
    def test_check_data_specifying_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific number of percentiles.
        """
        data = np.array(
            [
                [[4.75, 5.375, 6.0], [6.625, 7.25, 7.875], [8.5, 9.125, 9.75]],
                [[6.0, 6.625, 7.25], [7.875, 8.5, 9.125], [9.75, 10.375, 11.0]],
                [[7.25, 7.875, 8.5], [9.125, 9.75, 10.375], [11.0, 11.625, 12.25]],
            ]
        )
        result = Plugin().process(self.percentile_cube, no_of_percentiles=3)
        self.assertArrayAlmostEqual(result.data, data)

    @ManageWarnings(ignored_messages=["Only a single cube so no differences"])
    def test_check_data_not_specifying_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values without specifying the number of percentiles.
        """
        data = np.array(
            [
                [[4.75, 5.375, 6.0], [6.625, 7.25, 7.875], [8.5, 9.125, 9.75]],
                [[6.0, 6.625, 7.25], [7.875, 8.5, 9.125], [9.75, 10.375, 11.0]],
                [[7.25, 7.875, 8.5], [9.125, 9.75, 10.375], [11.0, 11.625, 12.25]],
            ]
        )
        result = Plugin().process(self.percentile_cube)
        self.assertArrayAlmostEqual(result.data, data)


if __name__ == "__main__":
    unittest.main()
