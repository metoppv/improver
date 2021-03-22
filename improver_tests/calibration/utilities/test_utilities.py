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
Unit tests for the utilities within the `calibration.utilities`
module.

"""
import datetime
import importlib
import unittest

import iris
import numpy as np
from iris.tests import IrisTest
from iris.util import squeeze
from numpy.testing import assert_array_equal

from improver.calibration.utilities import (
    check_forecast_consistency,
    check_predictor,
    convert_cube_data_to_2d,
    create_unified_frt_coord,
    filter_non_matching_cubes,
    flatten_ignoring_masked_data,
    forecast_coords_match,
    get_frt_hours,
    merge_land_and_sea,
    statsmodels_available,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_percentile_cube,
    set_up_variable_cube,
)

from ..ensemble_calibration.helper_functions import SetupCubes
from ..reliability_calibration.test_AggregateReliabilityCalibrationTables import (
    Test_Aggregation,
)

try:
    importlib.import_module("statsmodels")
except (ModuleNotFoundError, ImportError):
    STATSMODELS_FOUND = False
else:
    STATSMODELS_FOUND = True


class Test_convert_cube_data_to_2d(IrisTest):

    """Test the convert_cube_data_to_2d utility."""

    def setUp(self):
        """Set up a 3d temperature cube"""
        data = np.arange(226.15, 230.1, 0.15, dtype=np.float32).reshape(3, 3, 3)
        self.cube = set_up_variable_cube(data)
        self.data = np.array(
            [data[0].flatten(), data[1].flatten(), data[2].flatten()]
        ).T

    def test_basic(self):
        """Test that the utility returns an iris.cube.Cube."""
        result = convert_cube_data_to_2d(self.cube)
        self.assertIsInstance(result, np.ndarray)

    def test_check_values(self):
        """Test that the utility returns the expected data values."""
        result = convert_cube_data_to_2d(self.cube)
        self.assertArrayAlmostEqual(result, self.data)

    def test_change_coordinate(self):
        """
        Test that the utility returns the expected data values
        when the cube is sliced along the longitude dimension.
        """
        data = self.data.flatten().reshape(9, 3).T.reshape(9, 3)
        result = convert_cube_data_to_2d(self.cube, coord="longitude")
        self.assertArrayAlmostEqual(result, data)

    def test_no_transpose(self):
        """
        Test that the utility returns the expected data values
        when the cube is not transposed after slicing.
        """
        data = self.data.T
        result = convert_cube_data_to_2d(self.cube, transpose=False)
        self.assertArrayAlmostEqual(result, data)

    def test_2d_cube(self):
        """
        Test that the utility returns the expected data values
        when a 2d cube is input.
        """
        cube = next(self.cube.slices_over("realization"))
        expected_data = np.array([cube.data.flatten()]).T
        result = convert_cube_data_to_2d(cube)
        self.assertArrayAlmostEqual(result, expected_data, decimal=5)

    def test_1d_cube(self):
        """
        Test that the utility returns the expected data values
        when a 1d cube is input.
        """
        cube = self.cube[0, 0]
        expected_data = np.array([cube.data.flatten()]).T
        result = convert_cube_data_to_2d(cube)
        self.assertArrayAlmostEqual(result, expected_data, decimal=5)

    def test_5d_cube(self):
        """
        Test that the utility returns the expected data values
        when a 5d cube is input.
        """
        cube = add_coordinate(self.cube, [5, 10], "height", coord_units="m")
        expected_data = np.array(
            [
                cube.data[:, 0, :, :].flatten(),
                cube.data[:, 1, :, :].flatten(),
                cube.data[:, 2, :, :].flatten(),
            ]
        ).T
        result = convert_cube_data_to_2d(cube)
        self.assertArrayAlmostEqual(result, expected_data, decimal=5)


class Test_flatten_ignoring_masked_data(IrisTest):

    """Test the flatten_ignoring_masked_data utility."""

    def setUp(self):
        """Set up a basic 3D data array to use in the tests."""
        self.data_array = np.array(
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
                [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ],
            dtype=np.float32,
        )
        self.mask = np.array(
            [
                [[True, False, True, True], [True, False, True, True]],
                [[True, False, True, True], [True, False, True, True]],
                [[True, False, True, True], [True, False, True, True]],
            ]
        )
        self.expected_result_preserve_leading_dim = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
            ],
            dtype=np.float32,
        )

    def test_basic_not_masked(self):
        """Test a basic unmasked array"""
        expected_result = np.arange(0, 24, 1, dtype=np.float32)
        result = flatten_ignoring_masked_data(self.data_array)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.dtype, np.float32)

    def test_basic_masked(self):
        """Test a basic masked array"""
        masked_data_array = np.ma.MaskedArray(self.data_array, self.mask)
        expected_result = np.array([1.0, 5.0, 9.0, 13.0, 17.0, 21.0], dtype=np.float32)
        result = flatten_ignoring_masked_data(masked_data_array)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.dtype, np.float32)

    def test_basic_not_masked_preserver_leading_dim(self):
        """Test a basic unmasked array, with preserve_leading_dimension"""
        result = flatten_ignoring_masked_data(
            self.data_array, preserve_leading_dimension=True
        )
        self.assertArrayAlmostEqual(result, self.expected_result_preserve_leading_dim)
        self.assertEqual(result.dtype, np.float32)

    def test_basic_masked_preserver_leading_dim(self):
        """Test a basic masked array, with preserve_leading_dimension"""

        masked_data_array = np.ma.MaskedArray(self.data_array, self.mask)
        expected_result = np.array(
            [[1.0, 5.0], [9.0, 13.0], [17.0, 21.0]], dtype=np.float32
        )
        result = flatten_ignoring_masked_data(
            masked_data_array, preserve_leading_dimension=True
        )
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.dtype, np.float32)

    def test_all_masked(self):
        """Test empty array is returned when all points are masked."""
        mask = np.ones((3, 2, 4)) * True
        masked_data_array = np.ma.MaskedArray(self.data_array, mask)
        expected_result = np.array([], dtype=np.float32)
        result = flatten_ignoring_masked_data(masked_data_array)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.dtype, np.float32)

    def test_1D_input(self):
        """Test input array is unchanged when input in 1D"""
        data_array = self.data_array.flatten()
        expected_result = data_array.copy()
        result = flatten_ignoring_masked_data(data_array)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.dtype, np.float32)

    def test_4D_input_not_masked_preserve_leading_dim(self):
        """Test input array is unchanged when input in 4D.
           This should give the same answer as the corresponding 3D array."""
        data_array = self.data_array.reshape((3, 2, 2, 2))
        result = flatten_ignoring_masked_data(
            data_array, preserve_leading_dimension=True
        )
        self.assertArrayAlmostEqual(result, self.expected_result_preserve_leading_dim)
        self.assertEqual(result.dtype, np.float32)

    def test_inconsistent_mask_along_leading_dim(self):
        """Test an inconsistently masked array raises an error."""
        mask = np.array(
            [
                [[True, False, False, True], [True, False, True, True]],
                [[True, False, True, True], [True, False, True, True]],
                [[True, False, True, True], [True, False, True, False]],
            ]
        )
        masked_data_array = np.ma.MaskedArray(self.data_array, mask)
        expected_message = "The mask on the input array is not the same"
        with self.assertRaisesRegex(ValueError, expected_message):
            flatten_ignoring_masked_data(
                masked_data_array, preserve_leading_dimension=True
            )


class Test_check_predictor(IrisTest):

    """
    Test to check the predictor.
    """

    @staticmethod
    def test_mean():
        """
        Test that the utility does not raise an exception when
        predictor = "mean".
        """
        check_predictor("mean")

    @staticmethod
    def test_realizations():
        """
        Test that the utility does not raise an exception when
        predictor = "realizations".
        """
        check_predictor("realizations")

    def test_invalid_predictor(self):
        """
        Test that the utility raises an exception when
        predictor = "foo", a name not present in the list of
        accepted values for the predictor.
        """
        msg = "The requested value for the predictor"
        with self.assertRaisesRegex(ValueError, msg):
            check_predictor("foo")


class Test__filter_non_matching_cubes(SetupCubes):
    """Test the _filter_non_matching_cubes method."""

    def setUp(self):
        super().setUp()
        # Create historical forecasts and truth cubes where some items
        # are missing.
        self.partial_historic_forecasts = (
            self.historic_forecasts[:2] + self.historic_forecasts[3:]
        ).merge_cube()
        self.partial_truth = (self.truth[:2] + self.truth[3:]).merge_cube()

    def test_all_matching(self):
        """Test for when the historic forecast and truth cubes all match."""
        hf_result, truth_result = filter_non_matching_cubes(
            self.historic_temperature_forecast_cube, self.temperature_truth_cube
        )
        self.assertEqual(hf_result, self.historic_temperature_forecast_cube)
        self.assertEqual(truth_result, self.temperature_truth_cube)

    def test_bounded_variables(self):
        """Test for when the historic forecast and truth cubes all match
        inclusive of both the points and bounds on the time coordinate."""
        # Define bounds so that the lower bound is one hour preceding the point
        # whilst the upper bound is equal to the point.
        points = self.historic_temperature_forecast_cube.coord("time").points
        bounds = []
        for point in points:
            bounds.append([point - 1 * 60 * 60, point])

        self.historic_temperature_forecast_cube.coord("time").bounds = bounds
        self.temperature_truth_cube.coord("time").bounds = bounds

        hf_result, truth_result = filter_non_matching_cubes(
            self.historic_temperature_forecast_cube, self.temperature_truth_cube
        )
        self.assertEqual(hf_result, self.historic_temperature_forecast_cube)
        self.assertEqual(truth_result, self.temperature_truth_cube)

    def test_fewer_historic_forecasts(self):
        """Test for when there are fewer historic forecasts than truths,
        for example, if there is a missing forecast cycle."""
        hf_result, truth_result = filter_non_matching_cubes(
            self.partial_historic_forecasts, self.temperature_truth_cube
        )
        self.assertEqual(hf_result, self.partial_historic_forecasts)
        self.assertEqual(truth_result, self.partial_truth)

    def test_fewer_truths(self):
        """Test for when there are fewer truths than historic forecasts,
        for example, if there is a missing analysis."""
        hf_result, truth_result = filter_non_matching_cubes(
            self.historic_temperature_forecast_cube, self.partial_truth
        )
        self.assertEqual(hf_result, self.partial_historic_forecasts)
        self.assertEqual(truth_result, self.partial_truth)

    def test_mismatching(self):
        """Test for when there is both a missing historic forecasts and a
        missing truth at different validity times. This results in the
        expected historic forecasts and the expected truths containing cubes
        at three matching validity times."""
        partial_truth = self.truth[1:].merge_cube()
        expected_historical_forecasts = iris.cube.CubeList(
            [self.historic_forecasts[index] for index in (1, 3, 4)]
        ).merge_cube()
        expected_truth = iris.cube.CubeList(
            [self.truth[index] for index in (1, 3, 4)]
        ).merge_cube()
        hf_result, truth_result = filter_non_matching_cubes(
            self.partial_historic_forecasts, partial_truth
        )
        self.assertEqual(hf_result, expected_historical_forecasts)
        self.assertEqual(truth_result, expected_truth)

    def test_no_matches_exception(self):
        """Test for when no matches in validity time are found between the
        historic forecasts and the truths. In this case, an exception is
        raised."""
        partial_truth = self.truth[2]
        msg = "The filtering has found no matches in validity time "
        with self.assertRaisesRegex(ValueError, msg):
            filter_non_matching_cubes(self.partial_historic_forecasts, partial_truth)


class Test_create_unified_frt_coord(Test_Aggregation):

    """Test the create_unified_frt_coord method."""

    def test_coordinate(self):
        """Test the forecast reference time coordinate has the expected point,
        bounds, and type for an input with multiple forecast reference time
        points."""

        frt = "forecast_reference_time"
        frt_coord = self.forecasts.coord(frt)

        expected_points = self.forecast_2.coord(frt).points[0]
        expected_bounds = [[self.forecast_1.coord(frt).points[0], expected_points]]
        result = create_unified_frt_coord(frt_coord)

        self.assertIsInstance(result, iris.coords.DimCoord)
        assert_array_equal(result.points, expected_points)
        assert_array_equal(result.bounds, expected_bounds)
        self.assertEqual(result.name(), frt_coord.name())
        self.assertEqual(result.units, frt_coord.units)

    def test_coordinate_single_frt_input(self):
        """Test the forecast reference time coordinate has the expected point,
        bounds, and type for an input with a single forecast reference time
        point."""

        frt = "forecast_reference_time"
        frt_coord = self.forecast_1.coord(frt)

        expected_points = self.forecast_1.coord(frt).points[0]
        expected_bounds = [[self.forecast_1.coord(frt).points[0], expected_points]]
        result = create_unified_frt_coord(frt_coord)

        self.assertIsInstance(result, iris.coords.DimCoord)
        assert_array_equal(result.points, expected_points)
        assert_array_equal(result.bounds, expected_bounds)
        self.assertEqual(result.name(), frt_coord.name())
        self.assertEqual(result.units, frt_coord.units)

    def test_coordinate_input_with_bounds(self):
        """Test the forecast reference time coordinate has the expected point,
        bounds, and type for an input multiple forecast reference times, each
        with bounds."""

        frt = "forecast_reference_time"
        cube = iris.cube.CubeList(
            [self.reliability_cube, self.different_frt]
        ).merge_cube()
        frt_coord = cube.coord(frt)

        expected_points = self.different_frt.coord(frt).points[0]
        expected_bounds = [
            [
                self.reliability_cube.coord(frt).bounds[0][0],
                self.different_frt.coord(frt).bounds[0][-1],
            ]
        ]
        result = create_unified_frt_coord(frt_coord)

        self.assertIsInstance(result, iris.coords.DimCoord)
        assert_array_equal(result.points, expected_points)
        assert_array_equal(result.bounds, expected_bounds)
        self.assertEqual(result.name(), frt_coord.name())
        self.assertEqual(result.units, frt_coord.units)


class Test_merge_land_and_sea(IrisTest):

    """Test merge_land_and_sea"""

    def setUp(self):
        """Set up a percentile cube"""
        # Create a percentile cube
        land_data = np.ones((2, 3, 4), dtype=np.float32)
        sea_data = np.ones((2, 3, 4), dtype=np.float32) * 3.0
        mask = np.array(
            [
                [
                    [True, False, False, False],
                    [True, False, False, False],
                    [False, False, False, True],
                ],
                [
                    [True, False, False, False],
                    [True, False, False, False],
                    [False, False, False, True],
                ],
            ]
        )
        land_data = np.ma.MaskedArray(land_data, mask)
        self.percentiles_land = set_up_percentile_cube(land_data, [30, 60])
        self.percentiles_sea = set_up_percentile_cube(sea_data, [30, 60])

    def test_missing_dim(self):
        """Check that an error is raised if missing dimensional coordinate"""
        single_percentile = squeeze(self.percentiles_land[0])
        message = "Input cubes do not have the same dimension coordinates"
        with self.assertRaisesRegex(ValueError, message):
            merge_land_and_sea(single_percentile, self.percentiles_sea)

    def test_mismatch_dim_length(self):
        """Check an error is raised if a dim coord has a different length"""
        land_slice = self.percentiles_land[:, 1:, :]
        message = "Input cubes do not have the same dimension coordinates"
        with self.assertRaisesRegex(ValueError, message):
            merge_land_and_sea(land_slice, self.percentiles_sea)

    def test_merge(self):
        """Test merged data."""
        expected_merged_data = np.array(
            [
                [[3.0, 1.0, 1.0, 1.0], [3.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 3.0]],
                [[3.0, 1.0, 1.0, 1.0], [3.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 3.0]],
            ],
            dtype=np.float32,
        )
        expected_cube = self.percentiles_land.copy()
        expected_cube.data = expected_merged_data
        merge_land_and_sea(self.percentiles_land, self.percentiles_sea)
        self.assertArrayEqual(self.percentiles_land.data, expected_merged_data)
        self.assertEqual(
            expected_cube.xml(checksum=True), self.percentiles_land.xml(checksum=True)
        )
        self.assertFalse(np.ma.is_masked(self.percentiles_land.data))
        self.assertEqual(self.percentiles_land.data.dtype, np.float32)

    def test_nothing_to_merge(self):
        """Test case where there is no missing data to fill in."""
        input_mask = np.ones((2, 3, 4)) * False
        self.percentiles_land.data.mask = input_mask
        expected_cube = self.percentiles_land.copy()
        merge_land_and_sea(self.percentiles_land, self.percentiles_sea)
        self.assertArrayEqual(self.percentiles_land.data, expected_cube.data)
        self.assertEqual(
            expected_cube.xml(checksum=True), self.percentiles_land.xml(checksum=True)
        )
        self.assertFalse(np.ma.is_masked(self.percentiles_land.data))
        self.assertEqual(self.percentiles_land.data.dtype, np.float32)

    def test_input_not_masked(self):
        """Test case where input cube is not masked."""
        self.percentiles_land.data = np.ones((2, 3, 4), dtype=np.float32)
        expected_cube = self.percentiles_land.copy()
        merge_land_and_sea(self.percentiles_land, self.percentiles_sea)
        self.assertArrayEqual(self.percentiles_land.data, expected_cube.data)
        self.assertEqual(
            expected_cube.xml(checksum=True), self.percentiles_land.xml(checksum=True)
        )
        self.assertEqual(self.percentiles_land.data.dtype, np.float32)


class Test_forecast_coords_match(IrisTest):

    """Test for function that tests if forecast period and the hour of the
     forecast_reference_time coordinate match between two cubes."""

    def setUp(self):
        """Set-up testing."""
        self.data = np.ones((3, 3), dtype=np.float32)
        self.ref_cube = set_up_variable_cube(
            self.data,
            frt=datetime.datetime(2017, 11, 10, 1, 0),
            time=datetime.datetime(2017, 11, 10, 4, 0),
        )
        self.message = "The following coordinates of the two cubes do not match"

    def test_match(self):
        """Test returns None when cubes time coordinates match."""
        self.assertIsNone(forecast_coords_match(self.ref_cube, self.ref_cube.copy()))

    def test_forecast_period_mismatch(self):
        """Test an error is raised when the forecast period mismatches."""
        self.adjusted_cube = set_up_variable_cube(
            self.data,
            frt=datetime.datetime(2017, 11, 10, 1, 0),
            time=datetime.datetime(2017, 11, 10, 5, 0),
        )

        with self.assertRaisesRegex(ValueError, self.message):
            forecast_coords_match(self.ref_cube, self.adjusted_cube)

    def test_frt_hour_mismatch(self):
        """Test an error is raised when the forecast_reference_time mismatches"""
        self.adjusted_cube = set_up_variable_cube(
            self.data,
            frt=datetime.datetime(2017, 11, 10, 2, 0),
            time=datetime.datetime(2017, 11, 10, 5, 0),
        )

        with self.assertRaisesRegex(ValueError, self.message):
            forecast_coords_match(self.ref_cube, self.adjusted_cube)


class Test_get_frt_hours(IrisTest):

    """Test the get_frt_hours function."""

    def test_single_value(self):
        """Test that the expected forecast reference time hour value is
        returned in a set."""

        frt = iris.coords.DimCoord(
            [0],
            standard_name="forecast_reference_time",
            units=TIME_COORDS["forecast_reference_time"].units,
        )
        result = get_frt_hours(frt)
        self.assertEqual(result, {0})

    def test_multiple_values(self):
        """Test that the expected forecast reference time hour values are
        returned in a set."""
        expected = np.array([0, 1, 4], dtype=np.float32)
        frt = iris.coords.DimCoord(
            expected * 3600,
            standard_name="forecast_reference_time",
            units=TIME_COORDS["forecast_reference_time"].units,
        )
        result = get_frt_hours(frt)
        self.assertEqual(result, set(expected))


class Test_check_forecast_consistency(IrisTest):

    """Test the check_forecast_consistency function."""

    def setUp(self):
        """Set-up cubes for testing."""
        self.forecast1 = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            frt=datetime.datetime(2017, 11, 10, 1, 0),
            time=datetime.datetime(2017, 11, 10, 4, 0),
        )
        forecast2 = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            frt=datetime.datetime(2017, 11, 11, 1, 0),
            time=datetime.datetime(2017, 11, 11, 4, 0),
        )
        self.forecasts = iris.cube.CubeList([self.forecast1, forecast2]).merge_cube()

    def test_matching_forecasts(self):
        """Test case in which forecasts share frt hour and forecast period
        values. No result is expected in this case, hence there is no value
        comparison; the test is the absence of an exception."""

        check_forecast_consistency(self.forecasts)

    def test_mismatching_frt_hours(self):
        """Test case in which forecast reference time hours differ."""
        forecast2 = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            frt=datetime.datetime(2017, 11, 11, 2, 0),
            time=datetime.datetime(2017, 11, 11, 4, 0),
        )
        forecasts = iris.cube.CubeList([self.forecast1, forecast2]).merge_cube()

        msg = (
            "Forecasts have been provided with differing hours for the "
            "forecast reference time {1, 2}"
        )

        with self.assertRaisesRegex(ValueError, msg):
            check_forecast_consistency(forecasts)

    def test_mismatching_forecast_periods(self):
        """Test case in which the forecast periods differ."""
        forecast2 = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            frt=datetime.datetime(2017, 11, 11, 1, 0),
            time=datetime.datetime(2017, 11, 11, 5, 0),
        )
        forecasts = iris.cube.CubeList([self.forecast1, forecast2]).merge_cube()

        msg = (
            "Forecasts have been provided with differing forecast periods "
            "\[10800 14400\]"
        )

        with self.assertRaisesRegex(ValueError, msg):
            check_forecast_consistency(forecasts)


class Test_statsmodels_available(IrisTest):

    """Test statsmodels_available function."""

    @unittest.skipIf(STATSMODELS_FOUND is False, "statsmodels module not available.")
    def test_statsmodels_available(self):
        """Test when the statsmodels module is available."""
        self.assertTrue(statsmodels_available())

    @unittest.skipIf(STATSMODELS_FOUND is True, "statsmodels module not available.")
    def test_statsmodels_not_available(self):
        """Test when the statsmodels module is available."""
        self.assertFalse(statsmodels_available())


if __name__ == "__main__":
    unittest.main()
