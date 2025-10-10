# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for calibration.__init__"""

import importlib
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import iris
import joblib
import numpy as np
import pandas as pd
import pytest
from iris.cube import CubeList

from improver.calibration import (
    add_warning_comment,
    get_common_wmo_ids,
    identify_parquet_type,
    split_cubes_for_samos,
    split_forecasts_and_bias_files,
    split_forecasts_and_coeffs,
    split_forecasts_and_truth,
    split_netcdf_parquet_pickle,
    validity_time_check,
)
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_spot_variable_cube,
    set_up_variable_cube,
)
from improver.utilities.save import save_netcdf
from improver_tests import ImproverTest

pyarrow_installed = True
if not importlib.util.find_spec("pyarrow"):
    pyarrow_installed = False


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
            [self.realization_forecast, self.coefficient_cubelist],
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
            [self.percentile_forecast, self.coefficient_cubelist],
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
            [self.probability_forecast, self.coefficient_cubelist],
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
            [
                self.realization_forecast,
                self.coefficient_cubelist,
                self.additional_predictors,
            ],
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
            [
                self.realization_forecast,
                self.coefficient_cubelist,
                self.land_sea_mask,
            ],
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
            [self.realization_forecast, self.coefficient_cubelist]
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
            [
                self.realization_forecast,
                self.coefficient_cubelist,
                self.probability_forecast,
            ],
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
            [
                self.realization_forecast,
                self.coefficient_cubelist,
                self.additional_predictors,
                self.land_sea_mask,
                self.probability_forecast,
            ],
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
                [
                    self.probability_forecast,
                    self.coefficient_cubelist,
                    self.probability_forecast,
                ],
                self.land_sea_mask_name,
            )

    def test_no_coefficients(self):
        """Test if no EMOS coefficients are provided."""
        _, coeffs, _, _, _ = split_forecasts_and_coeffs(
            [self.percentile_forecast], self.land_sea_mask_name
        )
        self.assertIsNone(coeffs)

    def test_no_forecast(self):
        """Test if no forecast is present."""
        msg = "No forecast is present"
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_coeffs(
                [self.coefficient_cubelist], self.land_sea_mask_name
            )

    def test_duplicate_forecasts(self):
        """Test if a duplicate forecast is provided."""
        msg = "Multiple items have been provided"
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_coeffs(
                [
                    self.percentile_forecast,
                    self.coefficient_cubelist,
                    self.land_sea_mask,
                    self.probability_forecast,
                    self.percentile_forecast,
                ],
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
def forecast_cubes_multi_time():
    output_cubes = CubeList()
    for day in [10, 11]:
        cube = set_up_variable_cube(
            data=np.array(
                [[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [1.0, 3.0, 3.0]], dtype=np.float32
            ),
            name="wind_speed",
            units="m/s",
            time=datetime(2017, 11, day, 0, 0),
            frt=datetime(2017, 11, day - 1, 0, 0),
        )
        output_cubes.append(cube.copy())

    return output_cubes


@pytest.fixture
def probability_forecast_cubes_multi_time():
    output_cubes = CubeList()
    for day in [10, 11]:
        prob_template_cube = set_up_probability_cube(
            data=np.ones((2, 3, 3), dtype=np.float32),
            thresholds=[1.5, 2.5],
            time=datetime(2017, 11, day, 0, 0),
            frt=datetime(2017, 11, day - 1, 0, 0),
            variable_name="wind_speed",
        )
        output_cubes.append(prob_template_cube.copy())

    return output_cubes


@pytest.fixture
def truth_cubes_multi_time():
    output_cubes = CubeList()
    for day in [10, 11]:
        cube = set_up_variable_cube(
            data=np.array(
                [[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [1.0, 3.0, 3.0]], dtype=np.float32
            ),
            name="wind_speed",
            units="m/s",
            time=datetime(2017, 11, int(day), 0, 0),
        )
        cube.remove_coord("forecast_reference_time")
        cube.remove_coord("forecast_period")
        cube.attributes["truth_attribute"] = "truth"
        output_cubes.append(cube.copy())

    return output_cubes


@pytest.fixture
def emos_coefficient_cubes():
    # Set-up coefficient cubes
    fp_names = ["wind_speed"]
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


@pytest.fixture
def emos_additional_fields():
    altitude = set_up_variable_cube(
        np.ones((3, 3), dtype=np.float32), name="surface_altitude", units="m"
    )
    d2o = set_up_variable_cube(
        np.ones((3, 3), dtype=np.float32), name="distance_to_ocean", units="m"
    )

    output_cubes = CubeList([altitude, d2o])
    for cube in output_cubes:
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            cube.remove_coord(coord)

    return output_cubes


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


def create_netcdf_file(tmp_path):
    """Create a netcdf file with forecast data."""
    cube = set_up_variable_cube(
        data=np.array(
            [[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [1.0, 3.0, 3.0]], dtype=np.float32
        ),
        name="wind_speed",
        units="m/s",
    )

    output_dir = tmp_path / "netcdf_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "forecast.nc")
    save_netcdf(cubelist=cube, filename=output_path, compression_level=0)

    return cube, output_path


def create_multi_site_forecast_parquet_file(tmp_path, include_forecast_period=True):
    """Create a parquet file with multi-site forecast data."""

    data_dict = {
        "percentile": np.repeat(50, 5),
        "forecast": [281, 272, 287, 280, 290],
        "altitude": [10, 83, 56, 23, 2],
        "blend_time": [pd.Timestamp("2017-01-02 00:00:00")] * 5,
        "forecast_reference_time": [pd.Timestamp("2017-01-02 00:00:00")] * 5,
        "latitude": [60.1, 59.9, 59.7, 58, 57],
        "longitude": [1, 2, -1, -2, -3],
        "time": [pd.Timestamp("2017-01-02 06:00:00")] * 5,
        "wmo_id": ["03001", "03002", "03003", "03004", "03005"],
        "station_id": ["03001", "03002", "03003", "03004", "03005"],
        "cf_name": ["air_temperature"] * 5,
        "units": ["K"] * 5,
        "experiment": ["latestblend"] * 5,
        "period": [pd.NaT] * 5,
        "height": [1.5] * 5,
        "diagnostic": ["temperature_at_screen_level"] * 5,
    }

    parquet_type = "truth"
    if include_forecast_period:
        data_dict["forecast_period"] = [6 * 3.6e12] * 5
        parquet_type = "forecast"

    data_df = pd.DataFrame(data_dict)

    output_dir = tmp_path / f"{parquet_type}_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"{parquet_type}.parquet")
    data_df.to_parquet(output_path, index=False, engine="pyarrow")

    return output_dir


def create_pickle_file(tmp_path, filename="data.pkl"):
    """Create a pickle file containing a GAM object."""
    # Import pygam here to avoid a dependency for the entire repository.
    import pygam

    obj = [pygam.GAM(pygam.s(0) + pygam.l(1)), pygam.GAM(pygam.te(0, 1))]
    output_dir = tmp_path / "pickle_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / filename)
    joblib.dump(obj, output_path)

    return obj, output_path


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


@pytest.mark.parametrize("comment", [(None), ("Example comment")])
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


@pytest.mark.parametrize("forecast_type", ["realization", "probability"])
def test_split_cubes_for_samos_basic(
    forecast_cubes_multi_time,
    probability_forecast_cubes_multi_time,
    truth_cubes_multi_time,
    forecast_type,
):
    """Test that the split_cubes_for_samos function correctly separates out input
    forecast and truth cubes."""
    merged_input_cubelist = truth_cubes_multi_time.copy()

    if forecast_type == "realization":
        merged_input_cubelist.extend(forecast_cubes_multi_time)
        expected_name = "wind_speed"
        expected_shape = (2, 3, 3)
    else:
        merged_input_cubelist.extend(probability_forecast_cubes_multi_time)
        expected_name = "probability_of_wind_speed_above_threshold"
        expected_shape = (2, 2, 3, 3)

    (
        result_forecast_cube,
        result_truth_cube,
        result_gam_additional_fields,
        result_emos_coefficients,
        result_emos_additional_fields,
        result_prob_template,
    ) = split_cubes_for_samos(
        merged_input_cubelist, gam_features=[], truth_attribute="truth_attribute=truth"
    )

    # Check that the output forecast cube is as expected.
    assert isinstance(result_forecast_cube, iris.cube.Cube)
    assert result_forecast_cube.name() == expected_name
    assert result_forecast_cube.shape == expected_shape
    assert "forecast_reference_time" in [
        c.name() for c in result_forecast_cube.coords()
    ]
    assert "forecast_period" in [c.name() for c in result_forecast_cube.coords()]

    # Check that the output truth cube is as expected.
    assert isinstance(result_truth_cube, iris.cube.Cube)
    assert result_truth_cube.shape == (2, 3, 3)
    assert "forecast_reference_time" not in [
        c.name() for c in result_truth_cube.coords()
    ]
    assert "forecast_period" not in [c.name() for c in result_truth_cube.coords()]
    assert result_truth_cube.attributes["truth_attribute"] == "truth"

    # Check all other outputs are None.
    assert result_gam_additional_fields is None
    assert result_emos_coefficients is None
    assert result_emos_additional_fields is None
    assert result_prob_template is None


@pytest.mark.parametrize(
    "situation",
    [
        "all_matching",
        "fewer_in_forecast",
        "fewer_in_truth",
        "fewer_in_additional_predictors",
        "no_additional_predictors",
        "mixture",
    ],
)
def test_get_common_wmo_ids(situation):
    """Test the get_common_wmo_ids function."""
    forecast_wmo_ids = [1, 2, 3, 4, 5]
    truth_wmo_ids = [1, 2, 3, 4, 5]
    additional_wmo_ids = [1, 2, 3, 4, 5]

    if situation == "fewer_in_forecast":
        forecast_wmo_ids = [1, 2, 3]
    elif situation == "fewer_in_truth":
        truth_wmo_ids = [1, 2, 3]
    elif situation == "fewer_in_additional_predictors":
        additional_wmo_ids = [1, 2, 3]
    elif situation == "no_additional_predictors":
        additional_wmo_ids = []
    elif situation == "mixture":
        forecast_wmo_ids = [1, 2, 3, 4]
        truth_wmo_ids = [1, 2, 3, 5]
        additional_wmo_ids = [1, 2, 3, 6]

    data = np.ones(len(forecast_wmo_ids), dtype=np.float32)
    forecast_cube = set_up_spot_variable_cube(data, wmo_ids=forecast_wmo_ids)
    data = np.ones(len(truth_wmo_ids), dtype=np.float32)
    truth_cube = set_up_spot_variable_cube(data, wmo_ids=truth_wmo_ids)
    data = np.ones(len(additional_wmo_ids), dtype=np.float32)
    additional_predictors = None
    if additional_wmo_ids:
        additional_predictors = CubeList(
            [set_up_spot_variable_cube(data, wmo_ids=additional_wmo_ids)]
        )

    forecast_result, truth_result, additional_predictor_result = get_common_wmo_ids(
        forecast_cube, truth_cube, additional_predictors
    )

    if situation in ["all_matching", "no_additional_predictors"]:
        expected = [f"{x:05}" for x in [1, 2, 3, 4, 5]]
    else:
        expected = [f"{x:05}" for x in [1, 2, 3]]
    assert forecast_result.coord("wmo_id").points.tolist() == expected
    assert truth_result.coord("wmo_id").points.tolist() == expected
    if additional_predictors is None:
        assert additional_predictor_result is None
    else:
        assert (
            additional_predictor_result[0].coord("wmo_id").points.tolist() == expected
        )


@pytest.mark.parametrize(
    "provide_emos_coefficients,expect_emos_coefficients,provide_emos_additional_fields,expect_emos_additional_fields",
    [
        [True, True, False, False],
        [True, False, False, False],
        [False, False, True, True],
        [False, False, True, False],
        [True, True, True, True],
    ],
)
def test_split_cubes_for_samos_emos_cubes(
    forecast_cubes_multi_time,
    truth_cubes_multi_time,
    emos_coefficient_cubes,
    emos_additional_fields,
    provide_emos_coefficients,
    expect_emos_coefficients,
    provide_emos_additional_fields,
    expect_emos_additional_fields,
):
    """Test that the split_cubes_for_samos function correctly separates out input
    cubes when cubes for EMOS calibration are provided. Also test that the correct
    exceptions are raised if cubes for EMOS are provided but not expected."""
    merged_input_cubelist = truth_cubes_multi_time.copy()
    merged_input_cubelist.extend(forecast_cubes_multi_time)

    if provide_emos_coefficients:
        merged_input_cubelist.extend(emos_coefficient_cubes)
    if provide_emos_additional_fields:
        merged_input_cubelist.extend(emos_additional_fields)

    if provide_emos_coefficients and not expect_emos_coefficients:
        with pytest.raises(
            IOError,
            match="Found EMOS coefficients cubes when they were not expected.",
        ):
            split_cubes_for_samos(
                merged_input_cubelist,
                gam_features=[],
                truth_attribute="truth_attribute=truth",
                expect_emos_coeffs=expect_emos_coefficients,
                expect_emos_fields=expect_emos_additional_fields,
            )
    elif provide_emos_additional_fields and not expect_emos_additional_fields:
        with pytest.raises(
            IOError,
            match="Found additional fields cubes which do not match the features in "
            "gam_features.",
        ):
            split_cubes_for_samos(
                merged_input_cubelist,
                gam_features=[],
                truth_attribute="truth_attribute=truth",
                expect_emos_coeffs=expect_emos_coefficients,
                expect_emos_fields=expect_emos_additional_fields,
            )
    else:
        (
            result_forecast_cube,
            result_truth_cube,
            result_gam_additional_fields,
            result_emos_coefficients,
            result_emos_additional_fields,
            result_prob_template,
        ) = split_cubes_for_samos(
            merged_input_cubelist,
            gam_features=[],
            truth_attribute="truth_attribute=truth",
            expect_emos_coeffs=expect_emos_coefficients,
            expect_emos_fields=expect_emos_additional_fields,
        )

        # Check that the output forecast cube is as expected.
        assert isinstance(result_forecast_cube, iris.cube.Cube)
        assert result_forecast_cube.shape == (2, 3, 3)
        assert "forecast_reference_time" in [
            c.name() for c in result_forecast_cube.coords()
        ]
        assert "forecast_period" in [c.name() for c in result_forecast_cube.coords()]

        # Check that the output truth cube is as expected.
        assert isinstance(result_truth_cube, iris.cube.Cube)
        assert result_truth_cube.shape == (2, 3, 3)
        assert "forecast_reference_time" not in [
            c.name() for c in result_truth_cube.coords()
        ]
        assert "forecast_period" not in [c.name() for c in result_truth_cube.coords()]
        assert result_truth_cube.attributes["truth_attribute"] == "truth"

        if expect_emos_coefficients:
            # Check that the output EMOS coefficient cubes are as expected.
            assert isinstance(result_emos_coefficients, CubeList)
            assert len(result_emos_coefficients) == 4
            expected_names = [
                "emos_coefficients_alpha",
                "emos_coefficients_beta",
                "emos_coefficients_gamma",
                "emos_coefficients_delta",
            ]
            for i, cube in enumerate(result_emos_coefficients):
                assert expected_names[i] == cube.name()

        if expect_emos_additional_fields:
            # Check that the output EMOS additional fields cubes are as expected.
            assert isinstance(result_emos_additional_fields, CubeList)
            assert len(result_emos_additional_fields) == 2
            expected_names = ["surface_altitude", "distance_to_ocean"]
            for i, cube in enumerate(result_emos_additional_fields):
                assert expected_names[i] == cube.name()

        # Check all other outputs are None.
        assert result_gam_additional_fields is None
        assert result_prob_template is None


@pytest.mark.parametrize(
    "gam_features",
    [
        [],
        ["surface_altitude"],
        ["distance_to_ocean"],
        ["surface_altitude", "distance_to_ocean"],
    ],
)
def test_split_cubes_for_samos_gam_and_emos_cubes(
    forecast_cubes_multi_time,
    truth_cubes_multi_time,
    emos_additional_fields,
    gam_features,
):
    """Test that the split_cubes_for_samos function correctly separates out input
    cubes when cubes for both EMOS calibration and GAM are provided."""
    merged_input_cubelist = truth_cubes_multi_time.copy()
    merged_input_cubelist.extend(forecast_cubes_multi_time)
    merged_input_cubelist.extend(emos_additional_fields)

    emos_feature_names = [
        name
        for name in ["surface_altitude", "distance_to_ocean"]
        if name not in gam_features
    ]
    expect_emos_additional_fields = True if len(emos_feature_names) > 0 else False

    (
        result_forecast_cube,
        result_truth_cube,
        result_gam_additional_fields,
        result_emos_coefficients,
        result_emos_additional_fields,
        result_prob_template,
    ) = split_cubes_for_samos(
        merged_input_cubelist,
        gam_features=gam_features,
        truth_attribute="truth_attribute=truth",
        expect_emos_coeffs=False,
        expect_emos_fields=expect_emos_additional_fields,
    )

    # Check that the output forecast cube is as expected.
    assert isinstance(result_forecast_cube, iris.cube.Cube)
    assert result_forecast_cube.shape == (2, 3, 3)
    assert "forecast_reference_time" in [
        c.name() for c in result_forecast_cube.coords()
    ]
    assert "forecast_period" in [c.name() for c in result_forecast_cube.coords()]

    # Check that the output truth cube is as expected.
    assert isinstance(result_truth_cube, iris.cube.Cube)
    assert result_truth_cube.shape == (2, 3, 3)
    assert "forecast_reference_time" not in [
        c.name() for c in result_truth_cube.coords()
    ]
    assert "forecast_period" not in [c.name() for c in result_truth_cube.coords()]
    assert result_truth_cube.attributes["truth_attribute"] == "truth"

    # Check that the output GAM additional fields cubes are as expected.
    if len(gam_features) > 0:
        assert len(result_gam_additional_fields) == len(gam_features)
        for cube in result_gam_additional_fields:
            assert cube.name() in gam_features
    else:
        assert result_gam_additional_fields is None

    # Check that the output EMOS additional fields cubes are as expected.
    if len(emos_feature_names) > 0:
        assert len(result_emos_additional_fields) == len(emos_feature_names)
        for cube in result_emos_additional_fields:
            assert cube.name() in emos_feature_names
    else:
        assert result_emos_additional_fields is None

    # Check all other outputs are None.
    assert result_emos_coefficients is None
    assert result_prob_template is None


@pytest.mark.parametrize("n_prob_templates", [1, 2])
def test_split_cubes_for_samos_prob_template(
    forecast_cubes_multi_time,
    truth_cubes_multi_time,
    probability_forecast_cubes_multi_time,
    n_prob_templates,
):
    """Test that the split_cubes_for_samos function correctly separates out input
    cubes when a probability template cube is provided."""
    merged_input_cubelist = truth_cubes_multi_time.copy()
    merged_input_cubelist.extend(forecast_cubes_multi_time)

    if n_prob_templates == 1:
        merged_input_cubelist.append(probability_forecast_cubes_multi_time[0])

        (
            result_forecast_cube,
            result_truth_cube,
            result_gam_additional_fields,
            result_emos_coefficients,
            result_emos_additional_fields,
            result_prob_template,
        ) = split_cubes_for_samos(
            merged_input_cubelist,
            gam_features=[],
            truth_attribute="truth_attribute=truth",
            expect_emos_coeffs=True,
            expect_emos_fields=True,
        )

        # Check that the output forecast cube is as expected.
        assert isinstance(result_forecast_cube, iris.cube.Cube)
        assert result_forecast_cube.shape == (2, 3, 3)
        assert "forecast_reference_time" in [
            c.name() for c in result_forecast_cube.coords()
        ]
        assert "forecast_period" in [c.name() for c in result_forecast_cube.coords()]

        # Check that the output truth cube is as expected.
        assert isinstance(result_truth_cube, iris.cube.Cube)
        assert result_truth_cube.shape == (2, 3, 3)
        assert "forecast_reference_time" not in [
            c.name() for c in result_truth_cube.coords()
        ]
        assert "forecast_period" not in [c.name() for c in result_truth_cube.coords()]
        assert result_truth_cube.attributes["truth_attribute"] == "truth"

        # Check that the output probability template cube is as expected.
        assert isinstance(result_prob_template, iris.cube.Cube)
        assert result_prob_template.shape == (2, 3, 3)
        assert "wind_speed" in [c.name() for c in result_prob_template.coords()]
        assert "forecast_reference_time" in [
            c.name() for c in result_prob_template.coords()
        ]
        assert "forecast_period" in [c.name() for c in result_prob_template.coords()]

        # Check all other outputs are None.
        assert result_gam_additional_fields is None
        assert result_emos_coefficients is None
        assert result_emos_additional_fields is None

    else:
        probability_cubes = probability_forecast_cubes_multi_time.copy()
        probability_cubes[1].rename("probability_of_air_temperature_above_threshold")
        probability_cubes[1].coord("wind_speed").rename("air_temperature")
        merged_input_cubelist.extend(probability_cubes)

        with pytest.raises(
            IOError, match="Providing multiple probability cubes is not supported."
        ):
            split_cubes_for_samos(
                merged_input_cubelist,
                gam_features=[],
                truth_attribute="truth_attribute=truth",
                expect_emos_coeffs=True,
                expect_emos_fields=True,
            )


@pytest.mark.skipif(not pyarrow_installed, reason="pyarrow not installed")
@pytest.mark.parametrize(
    "include_forecast,include_truth", [(True, False), (False, True), (True, True)]
)
def test_identify_parquet_type_basic(tmp_path, include_forecast, include_truth):
    """Test that this function correctly splits out input parquet files into forecasts
    and truths."""
    merged_input_paths = []
    if include_forecast:
        forecast_dir = create_multi_site_forecast_parquet_file(
            tmp_path=tmp_path,
            include_forecast_period=True,
        )
        merged_input_paths.append(Path(forecast_dir))
    if include_truth:
        truth_dir = create_multi_site_forecast_parquet_file(
            tmp_path=tmp_path,
            include_forecast_period=False,
        )
        merged_input_paths.append(Path(truth_dir))

    (
        result_forecast_paths,
        result_truth_paths,
    ) = identify_parquet_type(merged_input_paths)

    assert result_forecast_paths == (
        merged_input_paths[0] if include_forecast else None
    )
    assert result_truth_paths == (merged_input_paths[-1] if include_truth else None)


@pytest.mark.skipif(not pyarrow_installed, reason="pyarrow not installed")
@pytest.mark.parametrize("include_netcdf", [True, False])
@pytest.mark.parametrize("include_parquet", [True, False])
@pytest.mark.parametrize("n_pickles", [0, 1, 2])
def test_split_netcdf_parquet_pickle_basic(
    tmp_path, forecast_cubes_multi_time, include_netcdf, include_parquet, n_pickles
):
    """Test that this function correctly splits out input files into pickle, parquet
    and netcdf files."""
    pytest.importorskip("pygam")

    merged_input_paths = []

    if include_parquet:
        forecast_dir = create_multi_site_forecast_parquet_file(
            tmp_path=tmp_path,
            include_forecast_period=True,
        )
        merged_input_paths.append(Path(forecast_dir))

        truth_dir = create_multi_site_forecast_parquet_file(
            tmp_path=tmp_path,
            include_forecast_period=False,
        )
        merged_input_paths.append(Path(truth_dir))

    if include_netcdf:
        test_cube, netcdf_path = create_netcdf_file(
            tmp_path=tmp_path,
        )
        merged_input_paths.append(Path(netcdf_path))

    if n_pickles > 0:
        pickle_object, pickle_path = create_pickle_file(
            tmp_path=tmp_path, filename="data1.pkl"
        )
        merged_input_paths.append(Path(pickle_path))

    if n_pickles == 2:
        pickle_object, pickle_path = create_pickle_file(
            tmp_path=tmp_path, filename="data2.pkl"
        )
        merged_input_paths.append(Path(pickle_path))

        msg = "Multiple pickle inputs have been provided. Only one is expected."
        with pytest.raises(ValueError, match=msg):
            split_netcdf_parquet_pickle(merged_input_paths)
    else:
        (
            result_cube,
            result_parquets,
            result_pickles,
        ) = split_netcdf_parquet_pickle(merged_input_paths)

        if include_netcdf:
            assert len(result_cube) == 1
            result_cube = result_cube[0]
            # The save/load process adds a Conventions global attribute to the cube.
            test_cube.attributes.globals["Conventions"] = "CF-1.7"
            assert result_cube == test_cube
        else:
            assert result_cube is None

        if include_parquet:
            # Parquets are returned as filepaths.
            assert result_parquets == merged_input_paths[0:2]
        else:
            assert result_parquets is None

        if n_pickles == 1:
            # The pickled object is a list containing 2 pygam GAMs.
            for i in range(len(result_pickles)):
                for key in ["max_iter", "tol", "fit_intercept"]:
                    assert (
                        result_pickles[i].get_params()[key]
                        == pickle_object[i].get_params()[key]
                    )
                assert result_pickles[i].terms == pickle_object[i].terms
                assert result_pickles[i].distribution == pickle_object[i].distribution
                assert result_pickles[i].link == pickle_object[i].link
        else:
            assert result_pickles is None


if __name__ == "__main__":
    unittest.main()
