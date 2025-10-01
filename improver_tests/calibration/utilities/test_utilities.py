# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the utilities within the `calibration.utilities`
module.

"""

import datetime
import importlib
import re
import unittest
from pathlib import Path

import iris
import numpy as np
import pandas as pd
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import CubeList
from iris.tests import IrisTest
from iris.util import squeeze
from numpy.testing import assert_array_equal

from improver.calibration.utilities import (
    broadcast_data_to_time_coord,
    check_data_sufficiency,
    check_forecast_consistency,
    check_predictor,
    convert_cube_data_to_2d,
    convert_parquet_to_cube,
    create_unified_frt_coord,
    filter_non_matching_cubes,
    flatten_ignoring_masked_data,
    forecast_coords_match,
    get_frt_hours,
    merge_land_and_sea,
    prepare_cube_no_calibration,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.cube_manipulation import sort_coord_in_cube

from ..emos_calibration.helper_functions import SetupCubes

pyarrow_installed = True
if not importlib.util.find_spec("pyarrow"):
    pyarrow_installed = False


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


class Test_check_predictor(unittest.TestCase):
    """
    Test to check the predictor.
    """

    def test_mean(self):
        """
        Test that the result is lowercase and an exception
        is not raised when predictor = "mean".
        """
        expected = "mean"
        result = check_predictor("mean")
        self.assertEqual(result, expected)

    def test_realizations(self):
        """
        Test that the result is lowercase and an exception
        is not raised when predictor = "realizations".
        """
        expected = "realizations"
        result = check_predictor("realizations")
        self.assertEqual(result, expected)

    def test_invalid_predictor(self):
        """
        Test that the utility raises an exception when
        predictor = "foo", a name not present in the list of
        accepted values for the predictor.
        """
        msg = "The requested value for the predictor"
        with self.assertRaisesRegex(ValueError, msg):
            check_predictor("foo")

    def test_lowercasing(self):
        """
        Test that the result has been lowercased.
        """
        expected = "mean"
        result = check_predictor("MeaN")
        self.assertEqual(result, expected)


class Test__filter_non_matching_cubes(SetupCubes):
    """Test the _filter_non_matching_cubes method."""

    def setUp(self):
        super().setUp()
        # Create historical forecasts and truth cubes where some items
        # are missing.
        self.partial_historic_forecasts = (
            self.historic_forecasts[:2] + self.historic_forecasts[3:]
        ).merge_cube()
        # Ensure the forecast coordinates are in the order: realization, time, lat, lon.
        self.partial_historic_forecasts.transpose([1, 0, 2, 3])
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

    def test_all_nan_historic_forecasts(self):
        """Test for when a time slice of the historic forecasts contains only NaNs."""
        historic_temperature_forecast_cube = (
            self.historic_temperature_forecast_cube.copy()
        )
        historic_temperature_forecast_cube.data[:, 0] = np.full((3, 3, 3), np.nan)
        hf_result, truth_result = filter_non_matching_cubes(
            historic_temperature_forecast_cube, self.temperature_truth_cube
        )
        self.assertEqual(hf_result, self.historic_temperature_forecast_cube[:, 1:])
        self.assertEqual(truth_result, self.temperature_truth_cube[1:])

    def test_duplicate_truths(self):
        """Test for when two forecasts match a single truth in terms of validity time.
        In this case, the first forecast that matches the truth will be kept.
        This can occur when processing a cube with a multi-dimensional time
        coordinate."""
        expected_historical_forecasts = self.historic_forecasts[0].copy()
        expected_truth = self.truth[0].copy()
        partial_truth = self.truth[0].copy()

        # Create an historic forecast cube with a multi-dimensional time coordinate.
        historic_forecast1 = self.historic_forecasts[0].copy()
        historic_forecast2 = self.historic_forecasts[0].copy()
        historic_forecast3 = self.historic_forecasts[0].copy()
        historic_forecast4 = self.historic_forecasts[0].copy()

        # Increase forecast period and validity time.
        historic_forecast2.coord("forecast_period").points = (
            historic_forecast2.coord("forecast_period").points + 3600
        )
        historic_forecast2.coord("time").points = (
            historic_forecast2.coord("time").points + 3600
        )

        # Increase forecast reference time and validity time.
        historic_forecast3.coord("forecast_reference_time").points = (
            historic_forecast3.coord("forecast_reference_time").points + 3600
        )
        historic_forecast3.coord("time").points = (
            historic_forecast3.coord("time").points + 3600
        )

        # Increase forecast period and forecast reference time
        historic_forecast4.coord("forecast_period").points = (
            historic_forecast4.coord("forecast_period").points + 3600
        )
        historic_forecast4.coord("forecast_reference_time").points = (
            historic_forecast4.coord("forecast_reference_time").points + 3600
        )

        # Pre-process historic forecasts so that they can be combined.
        cube1 = CubeList([historic_forecast1, historic_forecast2]).merge_cube()
        cube2 = CubeList([historic_forecast3, historic_forecast4]).merge_cube()

        time_points = []
        cubelist = CubeList()
        for cube in [cube1, cube2]:
            cube = iris.util.new_axis(cube, "forecast_reference_time")
            iris.util.promote_aux_coord_to_dim_coord(cube, "forecast_period")
            cube = sort_coord_in_cube(cube, "forecast_period")
            time_points.append(cube.coord("time").points)
            cube.remove_coord("time")
            cubelist.append(cube)
        cube = cubelist.concatenate_cube()

        # Add multi-dimensional time coordinate.
        time_coord = AuxCoord(
            np.array(np.reshape(time_points, (2, 2)), dtype=TIME_COORDS["time"].dtype),
            "time",
            units=TIME_COORDS["time"].units,
        )
        cube.add_aux_coord(time_coord, data_dims=(0, 1))

        hf_result, truth_result = filter_non_matching_cubes(cube, partial_truth)
        self.assertEqual(hf_result, expected_historical_forecasts)
        self.assertEqual(truth_result, expected_truth)

    def test_mismatching(self):
        """Test for when there is both a missing historic forecasts and a
        missing truth at different validity times. This results in the
        expected historic forecasts and the expected truths containing cubes
        at three matching validity times."""
        partial_truth = self.truth[1:].merge_cube()
        expected_historical_forecasts = iris.cube.CubeList(
            [self.historic_forecasts[index] for index in (1, 3, 4)]
        ).merge_cube()
        expected_historical_forecasts.transpose([1, 0, 2, 3])
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


def test_create_unified_frt_coordinate(forecast_grid):
    """Test the forecast reference time coordinate has the expected point,
    bounds, and type for an input with multiple forecast reference time
    points."""

    frt = "forecast_reference_time"
    frt_coord = forecast_grid.coord(frt)
    forecast_1 = forecast_grid[0, ...]
    forecast_2 = forecast_grid[1, ...]
    expected_points = forecast_2.coord(frt).points[0]
    expected_bounds = [[forecast_1.coord(frt).points[0], expected_points]]
    result = create_unified_frt_coord(frt_coord)

    assert isinstance(result, DimCoord)
    assert_array_equal(result.points, expected_points)
    assert_array_equal(result.bounds, expected_bounds)
    assert result.name() == frt_coord.name()
    assert result.units == frt_coord.units


def test_create_unified_frt_single_frt_input(forecast_grid):
    """Test the forecast reference time coordinate has the expected point,
    bounds, and type for an input with a single forecast reference time
    point."""

    frt = "forecast_reference_time"
    forecast_1 = forecast_grid[0, ...]
    frt_coord = forecast_1.coord(frt)

    expected_points = forecast_1.coord(frt).points[0]
    expected_bounds = [[forecast_1.coord(frt).points[0], expected_points]]
    result = create_unified_frt_coord(frt_coord)

    assert isinstance(result, iris.coords.DimCoord)
    assert_array_equal(result.points, expected_points)
    assert_array_equal(result.bounds, expected_bounds)
    assert result.name() == frt_coord.name()
    assert result.units == frt_coord.units


def test_create_unified_frt_input_with_bounds(reliability_cube, different_frt):
    """Test the forecast reference time coordinate has the expected point,
    bounds, and type for an input multiple forecast reference times, each
    with bounds."""

    frt = "forecast_reference_time"
    cube = iris.cube.CubeList([reliability_cube, different_frt]).merge_cube()
    frt_coord = cube.coord(frt)

    expected_points = different_frt.coord(frt).points[0]
    expected_bounds = [
        [
            reliability_cube.coord(frt).bounds[0][0],
            different_frt.coord(frt).bounds[0][-1],
        ]
    ]
    result = create_unified_frt_coord(frt_coord)

    assert isinstance(result, iris.coords.DimCoord)
    assert_array_equal(result.points, expected_points)
    assert_array_equal(result.bounds, expected_bounds)
    assert result.name() == frt_coord.name()
    assert result.units == frt_coord.units


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

    def test_15_minute_frt_offset_match(self):
        """Test returns None when cubes time coordinates match with an
        allowed leniency for a 15 minute offset."""
        adjusted_cube = set_up_variable_cube(
            self.data,
            frt=datetime.datetime(2017, 11, 10, 1, 15),
            time=datetime.datetime(2017, 11, 10, 4, 0),
        )

        self.assertIsNone(forecast_coords_match(self.ref_cube, adjusted_cube))

    def test_45_minute_frt_offset_match(self):
        """Test returns None when cubes time coordinates match with an
        allowed leniency for a 45 minute offset."""
        adjusted_cube = set_up_variable_cube(
            self.data,
            frt=datetime.datetime(2017, 11, 10, 1, 45),
            time=datetime.datetime(2017, 11, 10, 4, 0),
        )

        self.assertIsNone(forecast_coords_match(self.ref_cube, adjusted_cube))

    def test_forecast_period_mismatch(self):
        """Test an error is raised when the forecast period mismatches."""
        adjusted_cube = set_up_variable_cube(
            self.data,
            frt=datetime.datetime(2017, 11, 10, 1, 0),
            time=datetime.datetime(2017, 11, 10, 5, 0),
        )

        with self.assertRaisesRegex(ValueError, self.message):
            forecast_coords_match(self.ref_cube, adjusted_cube)

    def test_frt_hour_mismatch(self):
        """Test an error is raised when the forecast_reference_time mismatches"""
        adjusted_cube = set_up_variable_cube(
            self.data,
            frt=datetime.datetime(2017, 11, 10, 2, 0),
            time=datetime.datetime(2017, 11, 10, 5, 0),
        )

        with self.assertRaisesRegex(ValueError, self.message):
            forecast_coords_match(self.ref_cube, adjusted_cube)


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
            "Forecasts have been provided with differing hours for the forecast "
            "reference time {1, 2}"
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
            r"Forecasts have been provided with differing forecast periods "
            r"\[10800 14400\]"
        )

        with self.assertRaisesRegex(ValueError, msg):
            check_forecast_consistency(forecasts)


class Test_broadcast_data_to_time_coord(IrisTest):
    """Test the broadcast_data_to_time_coord function."""

    def setUp(self):
        """Set-up cubes for testing."""
        frts = [
            datetime.datetime(2017, 11, 10, 1, 0),
            datetime.datetime(2017, 11, 11, 1, 0),
            datetime.datetime(2017, 11, 12, 1, 0),
        ]
        forecast_cubes = CubeList()
        for frt in frts:
            forecast_cubes.append(
                set_up_variable_cube(
                    np.ones((2, 3, 3), dtype=np.float32),
                    frt=frt,
                    time=frt + datetime.timedelta(hours=3),
                )
            )
        self.forecast = forecast_cubes.merge_cube()
        self.forecast.transpose([1, 0, 2, 3])

        self.altitude = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32), name="surface_altitude", units="m"
        )
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.altitude.remove_coord(coord)

        self.expected_forecast = self.forecast.data.shape
        self.expected_altitude = (
            len(self.forecast.coord("time").points),
        ) + self.altitude.shape

    def test_one_forecast_predictor(self):
        """Test handling one forecast predictor"""
        self.forecast_predictors = iris.cube.CubeList([self.forecast])
        results = broadcast_data_to_time_coord(self.forecast_predictors)
        self.assertEqual(len(results), 1)
        self.assertTupleEqual(results[0].shape, self.expected_forecast)
        self.assertArrayEqual(results[0], self.forecast.data)

    def test_two_forecast_predictors(self):
        """Test handling two forecast predictors, where one is a static predictor."""
        self.forecast_predictors = iris.cube.CubeList([self.forecast, self.altitude])
        results = broadcast_data_to_time_coord(self.forecast_predictors)
        self.assertEqual(len(results), 2)
        self.assertTupleEqual(results[0].shape, self.expected_forecast)
        self.assertTupleEqual(results[1].shape, self.expected_altitude)

    def test_scalar_time_coord(self):
        """Test handling of a scalar time coordinate on the historic forecasts,
        which may occur during the spinning-up of a training dataset."""
        self.forecast_predictors = iris.cube.CubeList(
            [self.forecast[:, 0], self.altitude]
        )
        results = broadcast_data_to_time_coord(self.forecast_predictors)
        self.assertEqual(len(results), 2)
        self.assertTupleEqual(results[0].shape, self.forecast[:, 0].shape)
        self.assertTupleEqual(results[1].shape, self.altitude.shape)


class Test_check_data_sufficiency(SetupCubes):
    """Test the _check_data_sufficiency function."""

    def setUp(self):
        """Set up for testing."""
        super().setUp()
        self.hf_with_nans = self.historic_forecast_spot_cube.copy()
        self.truth_with_nans = self.truth_spot_cube.copy()
        for site_index in range(len(self.hf_with_nans.coord("wmo_id").points)):
            self.hf_with_nans.data[:, : site_index + 2, site_index] = np.nan
        for site_index in range(len(self.truth_with_nans.coord("wmo_id").points)):
            self.truth_with_nans.data[: site_index + 2, site_index] = np.nan
        self.point_by_point = False
        self.proportion_of_nans = 0.5

    def test_gridded(self):
        """Test providing gridded historic forecasts and truths."""
        check_data_sufficiency(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube,
            self.point_by_point,
            self.proportion_of_nans,
        )

    def test_spot(self):
        """Test providing spot historic forecasts and truths."""
        check_data_sufficiency(
            self.historic_forecast_spot_cube,
            self.truth_spot_cube,
            self.point_by_point,
            self.proportion_of_nans,
        )

    def test_spot_nans(self):
        """Test providing spot historic forecasts and truths with
        a proportion of NaNs above the allowable proportion."""
        msg = (
            "The proportion of NaNs detected is 0.7. This is higher than "
            "the allowable proportion of NaNs within the historic forecasts "
            "and truth pairs: 0.5."
        )
        with self.assertRaisesRegex(ValueError, msg):
            check_data_sufficiency(
                self.hf_with_nans,
                self.truth_with_nans,
                self.point_by_point,
                self.proportion_of_nans,
            )

    def test_spot_nans_alternative_proportion(self):
        """Test providing spot historic forecasts and truths with
        a proportion of NaNs above the allowable proportion for an
        alternative choice of proportion."""
        proportion_of_nans = 0.3
        msg = (
            "The proportion of NaNs detected is 0.7. This is higher than "
            "the allowable proportion of NaNs within the historic forecasts "
            "and truth pairs: 0.3."
        )
        with self.assertRaisesRegex(ValueError, msg):
            check_data_sufficiency(
                self.hf_with_nans,
                self.truth_with_nans,
                self.point_by_point,
                proportion_of_nans,
            )

    def test_spot_point_by_point(self):
        """Test providing spot historic forecasts and truths when using
        point-by-point processing."""
        point_by_point = True
        check_data_sufficiency(
            self.historic_forecast_spot_cube,
            self.truth_spot_cube,
            point_by_point,
            self.proportion_of_nans,
        )

    def test_spot_point_by_point_nans(self):
        """Test providing spot historic forecasts and truths when using
        point-by-point processing with a proportion of NaNs above the
        allowable proportion."""
        point_by_point = True
        msg = (
            "3 sites have a proportion of NaNs that is higher than the "
            "allowable proportion of NaNs within the historic forecasts "
            "and truth pairs. The allowable proportion is 0.5. "
            "The maximum proportion of NaNs is 1.0."
        )
        with self.assertRaisesRegex(ValueError, msg):
            check_data_sufficiency(
                self.hf_with_nans,
                self.truth_with_nans,
                point_by_point,
                self.proportion_of_nans,
            )

    def test_spot_point_by_point_nans_alternative_proportion(self):
        """Test providing spot historic forecasts and truths when using
        point-by-point processing with a proportion of NaNs above the
        allowable proportion for an alternative choice of proportion."""
        proportion_of_nans = 0.7
        point_by_point = True
        msg = (
            "2 sites have a proportion of NaNs that is higher than the "
            "allowable proportion of NaNs within the historic forecasts "
            "and truth pairs. The allowable proportion is 0.7. "
            "The maximum proportion of NaNs is 1.0."
        )
        with self.assertRaisesRegex(ValueError, msg):
            check_data_sufficiency(
                self.hf_with_nans,
                self.truth_with_nans,
                point_by_point,
                proportion_of_nans,
            )


@pytest.fixture
def forecast_cube():
    return set_up_variable_cube(
        data=np.array(
            [[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [1.0, 3.0, 3.0]], dtype=np.float32
        ),
        name="wind_speed",
        units="m/s",
    )


@pytest.fixture
def forecast_percentile_cube():
    return set_up_percentile_cube(
        data=np.array(
            [
                [[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [1.0, 3.0, 3.0]],
                [[3.0, 4.0, 4.0], [4.0, 3.0, 5.0], [3.0, 5.0, 5.0]],
            ],
            dtype=np.float32,
        ),
        percentiles=np.array([50, 60], dtype=np.float32),
        name="wind_speed",
        units="m/s",
    )


@pytest.fixture
def prob_template():
    return set_up_probability_cube(
        data=np.ones((2, 3, 3), dtype=np.float32),
        thresholds=[1.5, 2.5],
        variable_name="wind_speed",
    )


@pytest.fixture
def emos_coefficient_cubes():
    # Set-up coefficient cubes
    fp_names = ["wind_speed"]
    predictor_index = DimCoord(
        np.array(range(len(fp_names)), dtype=np.int32),
        long_name="predictor_index",
        units="1",
    )
    dim_coords_and_dims = ((predictor_index, 0),)
    predictor_name = AuxCoord(fp_names, long_name="predictor_name", units="no_unit")
    aux_coords_and_dims = ((predictor_name, 0),)

    attributes = {
        "diagnostic_standard_name": "wind_speed",
        "distribution": "norm",
    }
    alpha = iris.cube.Cube(
        np.array(0, dtype=np.float32),
        long_name="emos_coefficients_alpha",
        units="K",
        attributes=attributes,
    )
    beta = iris.cube.Cube(
        np.array([0.5], dtype=np.float32),
        long_name="emos_coefficients_beta",
        units="1",
        attributes=attributes,
        dim_coords_and_dims=dim_coords_and_dims,
        aux_coords_and_dims=aux_coords_and_dims,
    )
    gamma = iris.cube.Cube(
        np.array(0, dtype=np.float32),
        long_name="emos_coefficients_gamma",
        units="K",
        attributes=attributes,
    )
    delta = iris.cube.Cube(
        np.array(1, dtype=np.float32),
        long_name="emos_coefficients_delta",
        units="1",
        attributes=attributes,
    )

    return CubeList([alpha, beta, gamma, delta])


def create_multi_forecast_period_forecast_parquet_file(tmp_path):
    """Create a parquet file with multi-forecast period forecast data."""

    data_dict = {
        "percentile": [50, 50, 50, 50],
        "forecast": [277, 270, 280, 269],
        "altitude": [10, 83, 10, 83],
        "blend_time": [pd.Timestamp("2017-01-02 00:00:00", tz="utc")] * 4,
        "forecast_period": np.repeat(
            [
                [pd.Timedelta(int(6 * 3.6e3), unit="seconds")],
                [pd.Timedelta(int(12 * 3.6e3), unit="seconds")],
            ],
            2,
        ),
        "forecast_reference_time": [pd.Timestamp("2017-01-02 00:00:00", tz="utc")] * 4,
        "latitude": [60.1, 59.9, 60.1, 59.9],
        "longitude": [1, 2, 1, 2],
        "time": np.repeat(
            [
                pd.Timestamp("2017-01-02 06:00:00", tz="utc"),
                pd.Timestamp("2017-01-02 12:00:00", tz="utc"),
            ],
            2,
        ),
        "wmo_id": ["03001", "03002", "03001", "03002"],
        "station_id": ["03001", "03002", "03001", "03002"],
        "cf_name": ["air_temperature"] * 4,
        "units": ["K"] * 4,
        "experiment": ["latestblend"] * 4,
        "period": [pd.NaT] * 4,
        "height": [1.5] * 4,
        "diagnostic": ["temperature_at_screen_level"] * 4,
    }
    # Add wind speed to demonstrate filtering.
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["forecast"] = [6, 16, 12, 15]
    wind_speed_dict["cf_name"] = "wind_speed"
    wind_speed_dict["units"] = "m s-1"
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    output_dir = tmp_path / "forecast_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "forecast.parquet")
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")

    return data_df, output_dir


def create_multi_forecast_period_truth_parquet_file(tmp_path):
    """Create a parquet file with multi-forecast period truth data."""
    data_dict = {
        "diagnostic": ["temperature_at_screen_level"] * 4,
        "latitude": [60.1, 59.9, 60.1, 59.9],
        "longitude": [1, 2, 1, 2],
        "altitude": [10, 83, 10, 83],
        "time": np.repeat(
            [
                pd.Timestamp("2017-01-02 06:00:00", tz="utc"),
                pd.Timestamp("2017-01-02 12:00:00", tz="utc"),
            ],
            2,
        ),
        "wmo_id": ["03001", "03002", "03001", "03002"],
        "ob_value": [280, 273, 284, 275],
    }
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["ob_value"] = [2, 11, 10, 14]
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    output_dir = tmp_path / "truth_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "truth.parquet")
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return data_df, output_dir


@pytest.mark.parametrize(
    "include_coeffs,validity_times",
    [
        [True, None],
        [True, ["0400"]],  # matches forecast cube validity time
        [True, ["0500"]],  # does not match forecast cube validity time
        [False, None],
    ],
)
def test_prepare_cube_no_calibration_basic(
    forecast_cube, emos_coefficient_cubes, include_coeffs, validity_times
):
    """Test that the function returns the expected result when coefficients are
    provided with and without using the validity_time argument.
    """
    result = prepare_cube_no_calibration(
        forecast_cube,
        emos_coefficients=emos_coefficient_cubes if include_coeffs else None,
        validity_times=validity_times,
    )

    if not include_coeffs or validity_times == ["0500"]:
        # No matching coefficients so should return the input cube with some
        # additional metadata.
        assert isinstance(result, iris.cube.Cube)
        assert result.coords() == forecast_cube.coords()
        assert np.all(result.data == forecast_cube.data)
        assert result.attributes["comment"] == (
            "Warning: Calibration of this forecast has been attempted, however, no "
            "calibration has been applied."
        )
    else:
        assert result is None


@pytest.mark.parametrize(
    "include_coeffs,validity_times",
    [
        [True, None],
        [True, ["0400"]],  # matches forecast cube validity time
        [True, ["0500"]],  # does not match forecast cube validity time
        [False, None],
    ],
)
def test_prepare_cube_no_calibration_prob_template(
    forecast_cube, emos_coefficient_cubes, prob_template, include_coeffs, validity_times
):
    """Test that the function returns the expected result when coefficients and a
    probability template are provided with and without using the validity_time argument.
    """
    result = prepare_cube_no_calibration(
        forecast_cube,
        emos_coefficients=emos_coefficient_cubes if include_coeffs else None,
        validity_times=validity_times,
        prob_template=prob_template,
    )

    if not include_coeffs or validity_times == ["0500"]:
        # No matching coefficients so should return the probability template cube with
        # some additional metadata.
        assert isinstance(result, iris.cube.Cube)
        assert result.coords() == prob_template.coords()
        assert np.all(result.data == prob_template.data)
        assert result.attributes["comment"] == (
            "Warning: Calibration of this forecast has been attempted, however, no "
            "calibration has been applied."
        )
    else:
        assert result is None


@pytest.mark.skipif(not pyarrow_installed, reason="pyarrow not installed")
@pytest.mark.parametrize(
    "include_coeffs,validity_times",
    [
        [True, None],
        [True, ["0400"]],  # matches forecast cube validity time
        [True, ["0500"]],  # does not match forecast cube validity time
        [False, None],
    ],
)
def test_prepare_cube_no_calibration_percentiles(
    forecast_percentile_cube, emos_coefficient_cubes, include_coeffs, validity_times
):
    """Test that the function returns the expected result when coefficients and a set
    of percentiles are provided."""
    percentiles = [55]

    expected = set_up_percentile_cube(
        data=np.array(
            [
                [[2.0, 3.0, 3.0], [3.0, 2.0, 4.0], [2.0, 4.0, 4.0]],
            ],
            dtype=np.float32,
        ),
        percentiles=np.array(percentiles, dtype=np.float32),
        name="wind_speed",
        units="m/s",
    )

    result = prepare_cube_no_calibration(
        forecast_percentile_cube,
        emos_coefficients=emos_coefficient_cubes if include_coeffs else None,
        validity_times=validity_times,
        percentiles=percentiles,
    )

    if not include_coeffs or validity_times == ["0500"]:
        # No matching coefficients so should return a cube with percentiles
        # resampled.
        assert isinstance(result, iris.cube.Cube)
        assert result.coords() == expected.coords()
        assert np.all(result.data == expected.data)
        assert result.attributes["comment"] == (
            "Warning: Calibration of this forecast has been attempted, however, no "
            "calibration has been applied."
        )
    else:
        assert result is None


@pytest.mark.skipif(not pyarrow_installed, reason="pyarrow not installed")
@pytest.mark.parametrize("cycletime", ["20170103T0000Z", "20170104T0000Z"])
def test_convert_parquet_to_cube_basic(tmp_path, cycletime):
    """Test that this function returns the expected cubes when provided with valid
    inputs."""
    fcs_df, fcs_path = create_multi_forecast_period_forecast_parquet_file(tmp_path)

    truth_df, truth_path = create_multi_forecast_period_truth_parquet_file(tmp_path)

    fcs_cube, truth_cube = convert_parquet_to_cube(
        Path(fcs_path),
        Path(truth_path),
        forecast_period=6 * 3.6e3,  # seconds
        cycletime=cycletime,
        training_length=1,
        diagnostic="temperature_at_screen_level",
        percentiles=[50],
        experiment="latestblend",
    )

    if cycletime == "20170103T0000Z":
        # There is valid training data in the dataframes for this cycletime.
        assert isinstance(fcs_cube, iris.cube.Cube)
        assert isinstance(truth_cube, iris.cube.Cube)

        for cube in [fcs_cube, truth_cube]:
            assert cube.name() == "air_temperature"
            assert (
                cube.coord("time").points
                == pd.Timestamp("2017-01-02 06:00:00", tz="utc").timestamp()
            )
            np.testing.assert_array_almost_equal(
                cube.coord("altitude").points, np.array([10, 83], dtype=np.float32)
            )
            np.testing.assert_array_almost_equal(
                cube.coord("latitude").points, np.array([60.1, 59.9], dtype=np.float32)
            )
            np.testing.assert_array_almost_equal(
                cube.coord("longitude").points, np.array([1, 2], dtype=np.float32)
            )
            np.testing.assert_array_equal(
                cube.coord("wmo_id").points, np.array(["03001", "03002"])
            )

        assert (
            fcs_cube.coord("forecast_reference_time").points
            == pd.Timestamp("2017-01-02 00:00:00", tz="utc").timestamp()
        )
        assert fcs_cube.coord("forecast_period").points == 6 * 3.6e3
    else:
        # There is no valid training data in the dataframes for this cycletime.
        assert fcs_cube is None
        assert truth_cube is None


@pytest.mark.skipif(not pyarrow_installed, reason="pyarrow not installed")
def test_convert_parquet_to_cube_exception(tmp_path):
    """Test that the correct exception is raised when filtering returns an empty truth
    dataframe."""
    fcs_df, fcs_path = create_multi_forecast_period_forecast_parquet_file(tmp_path)

    truth_df, truth_path = create_multi_forecast_period_truth_parquet_file(tmp_path)

    # This input diagnostic does not exist in the forecast or truth dataframes.
    diagnostic = "lwe_precipitation_rate"
    msg = re.escape(
        f"The requested filepath {truth_path} does not contain the requested contents: "
        "[[('diagnostic', '==', 'lwe_precipitation_rate')]]"
    )
    with pytest.raises(IOError, match=msg):
        convert_parquet_to_cube(
            Path(fcs_path),
            Path(truth_path),
            forecast_period=6 * 3.6e3,  # seconds
            cycletime="20170103T0000Z",
            training_length=1,
            diagnostic=diagnostic,
            percentiles=[50],
            experiment="latestblend",
        )


if __name__ == "__main__":
    unittest.main()
