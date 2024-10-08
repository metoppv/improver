# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for calibration.__init__"""

import os
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import mkdtemp

import iris
import numpy as np
import pytest
from iris.cube import CubeList
from iris.tests import IrisTest

from improver.calibration import (
    add_warning_comment,
    get_cube_from_directory,
    split_forecasts_and_bias_files,
    split_forecasts_and_coeffs,
    split_forecasts_and_truth,
    validity_time_check,
)
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.save import save_netcdf
from improver_tests import ImproverTest


class Test_split_forecasts_and_truth(unittest.TestCase):

    """Test the split_forecasts_and_truth method."""

    def setUp(self):
        """Create cubes for testing the split_forecasts_and_truth method.
        Forecast data is all set to 1, and truth data to 0, allowing for a
        simple check that the cubes have been separated as expected."""

        thresholds = [283, 288]
        probability_data = np.ones((2, 4, 4), dtype=np.float32)
        realization_data = np.ones((4, 4), dtype=np.float32)

        self.truth_attribute = "mosg__model_configuration=uk_det"
        truth_attributes = {"mosg__model_configuration": "uk_det"}

        probability_forecast_1 = set_up_probability_cube(probability_data, thresholds)
        probability_forecast_2 = set_up_probability_cube(
            probability_data,
            thresholds,
            time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0),
        )
        self.probability_forecasts = [probability_forecast_1, probability_forecast_2]

        probability_truth_1 = probability_forecast_1.copy(
            data=np.zeros((2, 4, 4), dtype=np.float32)
        )
        probability_truth_2 = probability_forecast_2.copy(
            data=np.zeros((2, 4, 4), dtype=np.float32)
        )
        probability_truth_1.attributes.update(truth_attributes)
        probability_truth_2.attributes.update(truth_attributes)
        self.probability_truths = [probability_truth_1, probability_truth_2]

        realization_forecast_1 = set_up_variable_cube(realization_data)
        realization_forecast_2 = set_up_variable_cube(
            realization_data,
            time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0),
        )
        self.realization_forecasts = [realization_forecast_1, realization_forecast_2]

        realization_truth_1 = realization_forecast_1.copy(
            data=np.zeros((4, 4), dtype=np.float32)
        )
        realization_truth_2 = realization_forecast_2.copy(
            data=np.zeros((4, 4), dtype=np.float32)
        )
        realization_truth_1.attributes.update(truth_attributes)
        realization_truth_2.attributes.update(truth_attributes)
        self.realization_truths = [realization_truth_1, realization_truth_2]

        self.landsea_mask = realization_truth_1.copy()
        self.landsea_mask.rename("land_binary_mask")

    def test_probability_data(self):
        """Test that when multiple probability forecast cubes and truth cubes
        are provided, the groups are created as expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.probability_forecasts + self.probability_truths, self.truth_attribute
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsNone(land_sea_mask, None)
        self.assertSequenceEqual((2, 2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))

    def test_probability_data_with_land_sea_mask(self):
        """Test that when multiple probability forecast cubes, truth cubes, and
        a single land-sea mask are provided, the groups are created as
        expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.probability_forecasts + self.probability_truths + [self.landsea_mask],
            self.truth_attribute,
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(land_sea_mask, iris.cube.Cube)
        self.assertEqual("land_binary_mask", land_sea_mask.name())
        self.assertSequenceEqual((2, 2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))
        self.assertSequenceEqual((4, 4), land_sea_mask.shape)

    def test_realization_data(self):
        """Test that when multiple forecast cubes and truth cubes are provided,
        the groups are created as expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.realization_forecasts + self.realization_truths, self.truth_attribute
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsNone(land_sea_mask, None)
        self.assertSequenceEqual((2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))

    def test_realization_data_with_land_sea_mask(self):
        """Test that when multiple forecast cubes, truth cubes, and
        a single land-sea mask are provided, the groups are created as
        expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.realization_forecasts + self.realization_truths + [self.landsea_mask],
            self.truth_attribute,
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(land_sea_mask, iris.cube.Cube)
        self.assertEqual("land_binary_mask", land_sea_mask.name())
        self.assertSequenceEqual((2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))
        self.assertSequenceEqual((4, 4), land_sea_mask.shape)

    def test_exception_for_multiple_land_sea_masks(self):
        """Test that when multiple land-sea masks are provided an exception is
        raised."""

        msg = "Expected one cube for land-sea mask."
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask, self.landsea_mask],
                self.truth_attribute,
            )

    def test_exception_for_unintended_cube_combination(self):
        """Test that when the forecast and truth cubes have different names,
        indicating different diagnostics, an exception is raised."""

        self.realization_truths[0].rename("kitten_density")

        msg = "Must have cubes with 1 or 2 distinct names."
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask, self.landsea_mask],
                self.truth_attribute,
            )

    def test_exception_for_missing_truth_inputs(self):
        """Test that when all truths are missing an exception is raised."""

        self.realization_truths = []

        msg = "Missing truth input."
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask],
                self.truth_attribute,
            )

    def test_exception_for_missing_forecast_inputs(self):
        """Test that when all forecasts are missing an exception is raised."""

        self.realization_forecasts = []

        msg = "Missing historical forecast input."
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask],
                self.truth_attribute,
            )


class Test_split_forecasts_and_coeffs(ImproverTest):

    """Test the split_forecasts_and_coeffs function."""

    def setUp(self):
        """Set-up cubes for testing."""
        thresholds = [283, 288]
        percentiles = [25, 75]
        probability_data = np.ones((2, 4, 4), dtype=np.float32)
        realization_data = np.ones((4, 4), dtype=np.float32)

        self.truth_attribute = "mosg__model_configuration=uk_det"

        # Set-up probability and realization forecast cubes
        self.probability_forecast = CubeList(
            [set_up_probability_cube(probability_data, thresholds)]
        )
        self.realization_forecast = CubeList([set_up_variable_cube(realization_data)])
        self.percentile_forecast = CubeList(
            [set_up_percentile_cube(probability_data, percentiles)]
        )

        # Set-up coefficient cubes
        fp_names = [self.realization_forecast[0].name()]
        predictor_index = iris.coords.DimCoord(
            np.array(range(len(fp_names)), dtype=np.int32),
            long_name="predictor_index",
            units="1",
        )
        dim_coords_and_dims = ((predictor_index, 0),)
        predictor_name = iris.coords.AuxCoord(
            fp_names, long_name="predictor_name", units="no_unit"
        )
        aux_coords_and_dims = ((predictor_name, 0),)

        attributes = {
            "diagnostic_standard_name": self.realization_forecast[0].name(),
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

        self.coefficient_cubelist = CubeList([alpha, beta, gamma, delta])

        # Set-up land-sea mask.
        self.land_sea_mask_name = "land_binary_mask"
        self.land_sea_mask = set_up_variable_cube(
            np.zeros((4, 4), dtype=np.float32), name=self.land_sea_mask_name
        )
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.land_sea_mask.remove_coord(coord)
        self.land_sea_mask = CubeList([self.land_sea_mask])

        altitude = set_up_variable_cube(
            np.ones((4, 4), dtype=np.float32), name="surface_altitude", units="m"
        )
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            altitude.remove_coord(coord)
        self.additional_predictors = CubeList([altitude])

    def test_realization_forecast_and_coefficients(self):
        """Test a realization forecast input."""
        (
            forecast,
            coeffs,
            additional_predictors,
            land_sea_mask,
            template,
        ) = split_forecasts_and_coeffs(
            CubeList([self.realization_forecast, self.coefficient_cubelist]),
            self.land_sea_mask_name,
        )

        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(additional_predictors, None)
        self.assertEqual(land_sea_mask, None)
        self.assertEqual(template, None)

    def test_percentile_forecast_and_coefficients(self):
        """Test a percentile forecast input."""
        (
            forecast,
            coeffs,
            additional_predictors,
            land_sea_mask,
            template,
        ) = split_forecasts_and_coeffs(
            CubeList([self.percentile_forecast, self.coefficient_cubelist]),
            self.land_sea_mask_name,
        )
        self.assertCubeEqual(forecast, self.percentile_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(additional_predictors, None)
        self.assertEqual(land_sea_mask, None)
        self.assertEqual(template, None)

    def test_probability_forecast_and_coefficients(self):
        """Test a probability forecast input."""
        (
            forecast,
            coeffs,
            additional_predictors,
            land_sea_mask,
            template,
        ) = split_forecasts_and_coeffs(
            CubeList([self.probability_forecast, self.coefficient_cubelist]),
            self.land_sea_mask_name,
        )
        self.assertCubeEqual(forecast, self.probability_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(additional_predictors, None)
        self.assertEqual(land_sea_mask, None)
        self.assertEqual(template, None)

    def test_forecast_coefficients_additional_predictor(self):
        """Test the addition of a static additional predictor."""
        (
            forecast,
            coeffs,
            additional_predictors,
            land_sea_mask,
            template,
        ) = split_forecasts_and_coeffs(
            CubeList(
                [
                    self.realization_forecast,
                    self.coefficient_cubelist,
                    self.additional_predictors,
                ]
            ),
            self.land_sea_mask_name,
        )
        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertCubeListEqual(additional_predictors, self.additional_predictors)
        self.assertEqual(land_sea_mask, None)
        self.assertEqual(template, None)

    def test_forecast_coefficients_and_land_sea_mask(self):
        """Test the addition of a land-sea mask."""
        (
            forecast,
            coeffs,
            additional_predictors,
            land_sea_mask,
            template,
        ) = split_forecasts_and_coeffs(
            CubeList(
                [
                    self.realization_forecast,
                    self.coefficient_cubelist,
                    self.land_sea_mask,
                ]
            ),
            self.land_sea_mask_name,
        )

        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(additional_predictors, None)
        self.assertCubeEqual(land_sea_mask, self.land_sea_mask[0])
        self.assertEqual(template, None)

    def test_no_land_sea_mask_name(self):
        """Test when not providing the land_sea_mask_name option."""
        (
            forecast,
            coeffs,
            additional_predictors,
            land_sea_mask,
            template,
        ) = split_forecasts_and_coeffs(
            CubeList([self.realization_forecast, self.coefficient_cubelist]),
        )

        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(additional_predictors, None)
        self.assertEqual(land_sea_mask, None)
        self.assertEqual(template, None)

    def test_forecast_coefficients_prob_template(self):
        """Test the addition of a probability template cube."""
        (
            forecast,
            coeffs,
            additional_predictors,
            land_sea_mask,
            template,
        ) = split_forecasts_and_coeffs(
            CubeList(
                [
                    self.realization_forecast,
                    self.coefficient_cubelist,
                    self.probability_forecast,
                ]
            ),
            self.land_sea_mask_name,
        )
        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(additional_predictors, None)
        self.assertEqual(land_sea_mask, None)
        self.assertCubeEqual(template, self.probability_forecast[0])

    def test_all_options(self):
        """Test providing a forecast, coefficients, additional predictor,
        land-sea mask and a probability template."""
        (
            forecast,
            coeffs,
            additional_predictors,
            land_sea_mask,
            template,
        ) = split_forecasts_and_coeffs(
            CubeList(
                [
                    self.realization_forecast,
                    self.coefficient_cubelist,
                    self.additional_predictors,
                    self.land_sea_mask,
                    self.probability_forecast,
                ]
            ),
            self.land_sea_mask_name,
        )
        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertCubeListEqual(additional_predictors, self.additional_predictors)
        self.assertCubeEqual(land_sea_mask, self.land_sea_mask[0])
        self.assertCubeEqual(template, self.probability_forecast[0])

    def test_probability_forecast_coefficients_prob_template(self):
        """Test providing a probability template with a probability forecast."""
        msg = "Providing multiple probability cubes"
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_coeffs(
                CubeList(
                    [
                        self.probability_forecast,
                        self.coefficient_cubelist,
                        self.probability_forecast,
                    ]
                ),
                self.land_sea_mask_name,
            )

    def test_no_coefficients(self):
        """Test if no EMOS coefficients are provided."""
        _, coeffs, _, _, _ = split_forecasts_and_coeffs(
            CubeList([self.percentile_forecast]), self.land_sea_mask_name
        )
        self.assertIsNone(coeffs)

    def test_no_forecast(self):
        """Test if no forecast is present."""
        msg = "No forecast is present"
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_coeffs(
                CubeList([self.coefficient_cubelist]), self.land_sea_mask_name
            )

    def test_duplicate_forecasts(self):
        """Test if a duplicate forecast is provided."""
        msg = "Multiple items have been provided"
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_coeffs(
                CubeList(
                    [
                        self.percentile_forecast,
                        self.coefficient_cubelist,
                        self.land_sea_mask,
                        self.probability_forecast,
                        self.percentile_forecast,
                    ]
                ),
                self.land_sea_mask_name,
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
def forecast_error_cubelist():
    bias_cubes = CubeList()
    for bias_index in range(-1, 2):
        bias_cube = set_up_variable_cube(
            data=np.array(
                [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-2.0, 0.0, 1.0]], dtype=np.float32
            )
            + (-1) * bias_index,
            name="forecast_error_of_wind_speed",
            units="m/s",
            frt=(datetime(2017, 11, 10, 0, 0) - timedelta(days=(2 - bias_index))),
            attributes={"title": "Forecast bias data"},
        )
        bias_cube.remove_coord("time")
        bias_cubes.append(bias_cube)
    return bias_cubes


@pytest.mark.parametrize("multiple_bias_cubes", [(True, False)])
def test_split_forecasts_and_bias_files(
    forecast_cube, forecast_error_cubelist, multiple_bias_cubes
):
    """Test that split_forecasts_and_bias_files correctly separates out
    the forecast cube from the forecast error cube(s)."""
    if not multiple_bias_cubes:
        forecast_error_cubelist = forecast_error_cubelist[0:]
    merged_input_cubelist = forecast_error_cubelist.copy()
    merged_input_cubelist.append(forecast_cube)

    result_forecast_cube, result_bias_cubes = split_forecasts_and_bias_files(
        merged_input_cubelist
    )

    assert result_forecast_cube == forecast_cube
    assert result_bias_cubes == forecast_error_cubelist
    if not multiple_bias_cubes:
        assert len(result_bias_cubes) == 1


@pytest.mark.parametrize("multiple_bias_cubes", [(True, False)])
def test_split_forecasts_and_bias_files_missing_fcst(
    forecast_error_cubelist, multiple_bias_cubes
):
    """Test that split_forecasts_and_bias_files raises a ValueError when
    no forecast cube is provided."""
    if not multiple_bias_cubes:
        forecast_error_cubelist = forecast_error_cubelist[0:]
    with pytest.raises(ValueError, match="No forecast is present"):
        split_forecasts_and_bias_files(forecast_error_cubelist)


def test_split_forecasts_and_bias_files_multiple_fcsts(
    forecast_cube, forecast_error_cubelist
):
    """Test that split_forecasts_and_bias_files raises a ValueError when
    multiple forecast cubes are provided."""
    forecast_error_cubelist.append(forecast_cube)
    forecast_error_cubelist.append(forecast_cube)

    with pytest.raises(ValueError, match="Multiple forecast inputs"):
        split_forecasts_and_bias_files(forecast_error_cubelist)


@pytest.mark.parametrize(
    "time,validity_times,expected",
    [
        (datetime(2017, 11, 10, 4, 0), ["0400", "0500", "0600"], True),
        (datetime(2017, 11, 10, 4, 15), ["0415", "0430", "0445"], True),
        (datetime(2017, 11, 10, 4, 0), ["0000", "0100", "0200"], False),
    ],
)
def test_matching_validity_times(time, validity_times, expected):
    """Test that True is returned if the forecast contains a validity time that
    matches with a validity time within the list provided.
    Otherwise, False is returned."""
    data = np.zeros((2, 2), dtype=np.float32)
    forecast = set_up_variable_cube(data, time=time)
    result = validity_time_check(forecast, validity_times)
    assert result is expected


@pytest.mark.parametrize(
    "comment", [(None), ("Example comment")],
)
def test_add_warning_to_comment(comment):
    """Test the addition of a warning comment if calibration has been attempted
    but not applied successfully."""
    expected = (
        "Warning: Calibration of this forecast has been attempted, "
        "however, no calibration has been applied."
    )
    data = np.zeros((2, 2), dtype=np.float32)
    cube = set_up_variable_cube(data)
    if comment:
        cube.attributes["comment"] = comment
        expected = "\n".join([comment, expected])
    result = add_warning_comment(cube)
    assert result.attributes["comment"] == expected


class test_get_cube_from_directory(IrisTest):
    """Test that the get_cube_from_directory function returns a cube."""

    get_cube_from_directory

    def setUp(self):
        """Set up variables for use in testing."""
        self.directory = Path(mkdtemp())
        self.empty_directory = Path(mkdtemp())
        self.filepath = os.path.join(self.directory, "temp.nc")
        self.filepath2 = os.path.join(self.directory, "temp2.nc")
        time = datetime(2017, 11, 10, 4, 0)
        time2 = datetime(2017, 11, 10, 5, 0)
        frt = datetime(2017, 11, 10, 0, 0)
        self.cube = set_up_variable_cube(
            np.ones((3, 3, 3), dtype=np.float32), time=time, frt=frt
        )
        self.cube2 = set_up_variable_cube(
            np.zeros((3, 3, 3), dtype=np.float32), time=time2, frt=frt
        )
        save_netcdf(self.cube, self.filepath)
        save_netcdf(self.cube2, self.filepath2)
        self.realization_points = self.cube.coord("realization").points
        self.time_points = self.cube.coord("time").points
        self.time_points2 = self.cube2.coord("time").points
        self.latitude_points = self.cube.coord("latitude").points
        self.longitude_points = self.cube.coord("longitude").points

    def tearDown(self):
        """Remove temporary directories created for testing."""
        os.remove(self.filepath)
        os.remove(self.filepath2)
        os.rmdir(self.directory)
        os.rmdir(self.empty_directory)

    def test_no_files(self):
        """Test that nothing is returned for no inputs"""
        result = get_cube_from_directory(self.empty_directory)
        self.assertIsNone(result)

    def test_old_files(self):
        """Test that nothing is returned for old files"""
        result = get_cube_from_directory(
            self.directory, cycle_point="20180101T0000Z", max_days_offset=30
        )
        self.assertIsNone(result)

    def test_relevant_files(self):
        result = get_cube_from_directory(
            self.directory, cycle_point="20171112T0000Z", max_days_offset=30
        )

        self.assertArrayAlmostEqual(
            result.coord("realization").points, self.realization_points
        )
        self.assertArrayAlmostEqual(
            result.coord("time").points, [self.time_points[0], self.time_points2[0]]
        )
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, self.latitude_points
        )
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, self.longitude_points
        )


if __name__ == "__main__":
    unittest.main()
