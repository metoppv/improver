# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the `ensemble_copula_coupling.ResamplePercentiles` class.
"""
import unittest
from datetime import datetime

import numpy as np
import pytest
from iris.cube import Cube
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ResamplePercentiles as Plugin,
)
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_percentile_cube,
)

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

        msg = (
            "Forecast values exist that fall outside the expected extrema "
            "values that are defined as bounds in ensemble_copula_coupling"
            "\\/constants.py. Applying the extrema values as end points to "
            "the distribution would result in non-monotonically increasing "
            "values. The defined extremes are \\(-40, 50\\), whilst the "
            "following forecast values exist outside this range: \\[60\\]."
        )

        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._add_bounds_to_percentiles_and_forecast_at_percentiles(
                percentiles, forecast_at_percentiles, self.bounds_pairing
            )

    def test_endpoints_of_distribution_exceeded_warning(self):
        """
        Test that the plugin raises a warning message when the constant
        end points of the distribution are exceeded by a percentile value
        used in the forecast and the ecc_bounds_warning keyword argument
        has been specified.
        """
        forecast_at_percentiles = np.array([[8, 10, 60]])
        percentiles = np.array([5, 70, 95])
        plugin = Plugin(ecc_bounds_warning=True)
        warning_msg = (
            "Forecast values exist that fall outside the expected extrema "
            "values that are defined as bounds in ensemble_copula_coupling"
            "/constants.py. Applying the extrema values as end points to "
            "the distribution would result in non-monotonically increasing "
            "values. The defined extremes are \\(-40, 50\\), whilst the "
            "following forecast values exist outside this range: \\[60\\]. "
            "The percentile values that have exceeded the existing bounds "
            "will be used as new bounds."
        )
        with pytest.warns(UserWarning, match=warning_msg):
            plugin._add_bounds_to_percentiles_and_forecast_at_percentiles(
                percentiles, forecast_at_percentiles, self.bounds_pairing
            )

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

    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube."""
        result = Plugin()._interpolate_percentiles(
            self.cube, self.percentiles, self.perc_coord
        )
        self.assertIsInstance(result, Cube)

    def test_transpose_cube_dimensions(self):
        """
        Test that the plugin returns the expected data, when comparing
        input cubes which have dimensions in a different order.
        """
        # Calculate result for nontransposed cube.
        nontransposed_result = Plugin()._interpolate_percentiles(
            self.cube, self.percentiles, self.perc_coord
        )

        # Calculate result for transposed cube.
        # Original cube dimensions are [P, Y, X].
        # Transposed cube dimensions are [X, Y, P].
        self.cube.transpose([2, 1, 0])
        transposed_result = Plugin()._interpolate_percentiles(
            self.cube, self.percentiles, self.perc_coord
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
            cube, self.percentiles, self.perc_coord
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
            self.cube, percentiles, self.perc_coord
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
        result = Plugin()._interpolate_percentiles(cube, percentiles, self.perc_coord)
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
                    [28.88889, 29.23611, 29.583334],
                    [29.930555, 30.277779, 30.625],
                    [30.972221, 31.319445, 31.666666],
                ],
                [
                    [53.77778, 53.84722, 53.916668],
                    [53.98611, 54.055557, 54.125],
                    [54.194443, 54.26389, 54.333332],
                ],
            ],
            dtype=np.float32,
        )

        cube = next(self.cube.slices_over(self.perc_coord))

        result = Plugin()._interpolate_percentiles(
            cube, self.percentiles, self.perc_coord
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
            cube, self.percentiles, self.perc_coord
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
                    [-48.0, -47.6875, -47.375],
                    [-47.0625, -46.75, -46.4375],
                    [-46.125, -45.8125, -45.5],
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
                    [34.0, 34.3125, 34.625],
                    [34.9375, 35.25, 35.5625],
                    [35.875, 36.1875, 36.5],
                ],
            ]
        )

        percentiles = np.arange(5, 100, 10)
        result = Plugin()._interpolate_percentiles(
            self.cube, percentiles, self.perc_coord
        )
        self.assertArrayAlmostEqual(result.data, data)

    def test_check_data_spot_forecasts(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles for spot forecasts.
        """
        spot_percentile_cube = set_up_spot_test_cube(cube_type="percentile")
        spot_percentile_cube.data = (
            np.tile(np.linspace(5, 10, 3), 9).reshape(3, 9) + 273.15
        )

        data = (
            np.array(
                [
                    [5.0, 7.5, 10.0, 5.0, 7.5, 10.0, 5.0, 7.5, 10.0],
                    [5.0, 7.5, 10.0, 5.0, 7.5, 10.0, 5.0, 7.5, 10.0],
                    [5.0, 7.5, 10.0, 5.0, 7.5, 10.0, 5.0, 7.5, 10.0],
                ]
            )
            + 273.15
        )
        percentiles = spot_percentile_cube.coord("percentile").points

        result = Plugin()._interpolate_percentiles(
            spot_percentile_cube, percentiles, self.perc_coord
        )
        self.assertArrayAlmostEqual(result.data, data, decimal=5)


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
        self.expected = np.array(
            [
                [[4.75, 5.375, 6.0], [6.625, 7.25, 7.875], [8.5, 9.125, 9.75]],
                [[6.0, 6.625, 7.25], [7.875, 8.5, 9.125], [9.75, 10.375, 11.0]],
                [[7.25, 7.875, 8.5], [9.125, 9.75, 10.375], [11.0, 11.625, 12.25]],
            ]
        )

    def test_check_data_specifying_percentile_number(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific number of percentiles.
        """
        result = Plugin().process(self.percentile_cube, no_of_percentiles=3)
        self.assertArrayAlmostEqual(result.data, self.expected)

    def test_check_data_not_specifying_percentile_number(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values without specifying the number of percentiles.
        """
        result = Plugin().process(self.percentile_cube)
        self.assertArrayAlmostEqual(result.data, self.expected)

    def test_check_data_masked_input_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values when the input data is masked.
        """
        self.percentile_cube.data[:, 0, 0] = np.nan
        self.percentile_cube.data = np.ma.masked_invalid(self.percentile_cube.data)
        self.expected[:, 0, 0] = np.nan
        self.expected = np.ma.masked_invalid(self.expected)
        result = Plugin().process(self.percentile_cube)
        self.assertArrayAlmostEqual(result.data.data, self.expected.data)
        self.assertArrayEqual(result.data.mask, self.expected.mask)

    def test_check_data_masked_input_data_non_nans(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values when the input data is masked and underlying data is not NaNs.
        """
        self.percentile_cube.data[:, 0, 0] = 1000
        self.percentile_cube.data = np.ma.masked_equal(self.percentile_cube.data, 1000)
        self.expected[:, 0, 0] = np.nan
        self.expected = np.ma.masked_invalid(self.expected)
        result = Plugin().process(self.percentile_cube)
        self.assertArrayAlmostEqual(result.data.data, self.expected.data)
        self.assertArrayEqual(result.data.mask, self.expected.mask)

    def test_check_data_specifying_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values corresponding to the set of percentiles requested.
        """
        result = Plugin().process(self.percentile_cube, percentiles=[35, 60, 85])
        self.assertArrayAlmostEqual(result.data, self.expected + 0.5)

    def test_check_data_specifying_extreme_percentiles_with_ecc_bounds(self):
        """
        Test that the plugin returns data with the expected
        values corresponding to percentiles that require ECC bounds to resolve.
        """
        expected = np.array(
            [
                [
                    [-89.6, -89.5375, -89.475],
                    [-89.4125, -89.35, -89.2875],
                    [-89.225, -89.1625, -89.1],
                ],
                self.expected[1] + 0.5,
                [
                    [54.8, 54.8625, 54.925],
                    [54.9875, 55.05, 55.1125],
                    [55.175, 55.2375, 55.3],
                ],
            ]
        )
        result = Plugin().process(self.percentile_cube, percentiles=[1, 60, 99])
        self.assertArrayAlmostEqual(result.data, expected, decimal=5)

    def test_check_data_specifying_extreme_percentiles_without_ecc_bounds(self):
        """
        Test that the plugin returns data with the expected
        values where percentiles outside of the range given by the input percentiles
        have been filled using nearest neighbour interpolation.
        """
        expected = np.array(
            [
                self.percentile_cube[0].data,
                self.expected[1] + 0.5,
                self.percentile_cube[-1].data,
            ]
        )
        result = Plugin(skip_ecc_bounds=True).process(
            self.percentile_cube, percentiles=[1, 60, 99]
        )
        self.assertArrayAlmostEqual(result.data, expected)

    def test_percentiles_too_low(self):
        """
        Test that an exception is raised if a percentile value is below 0.
        """
        msg = "The percentiles supplied must be between 0 and 100"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin().process(self.percentile_cube, percentiles=[-5, 50, 75])

    def test_percentiles_too_high(self):
        """
        Test that an exception is raised if a percentile value is above 100.
        """
        msg = "The percentiles supplied must be between 0 and 100"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin().process(self.percentile_cube, percentiles=[25, 50, 105])


if __name__ == "__main__":
    unittest.main()
