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
"""Unit tests for the `ensemble_calibration.ApplyEMOS` class."""

import unittest
from typing import Sequence, Union

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.tests import IrisTest

from improver.calibration.ensemble_calibration import ApplyEMOS
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.cube_manipulation import get_dim_coord_names


def build_coefficients_cubelist(
    template: Cube,
    coeff_values: Union[Sequence, np.ndarray],
    forecast_predictors: CubeList,
) -> CubeList:
    """Make a cubelist of coefficients with expected metadata

    Args:
        template (iris.cube.Cube):
            Cube containing information about the time,
            forecast_reference_time, forecast_period, x coordinate and
            y coordinate that will be used within the EMOS coefficient cube.
        coeff_values (numpy.ndarray or list):
            The values of the coefficients. These values will be used as the
            cube data.
        forecast_predictors (iris.cube.CubeList):
            The forecast predictors used for constructing the coordinates
            required for the beta coefficient.

    Returns:
        cubelist (iris.cube.CubeList) - The resulting EMOS
            coefficients cubelist.

    """
    dim_coords_and_dims = []
    aux_coords_and_dims = []

    # add spatial and temporal coords from forecast to be calibrated
    for coord in ["forecast_period", "forecast_reference_time"]:
        aux_coords_and_dims.append((template.coord(coord).copy(), None))

    for coord in [template.coord(axis="x"), template.coord(axis="y")]:
        coord_diffs = np.diff(coord.points)
        min_bound = min(coord.points) - (coord_diffs[0] / 2)
        max_bound = max(coord.points) + (coord_diffs[-1] / 2)
        bounds = [min_bound, max_bound]
        point = np.median(bounds)
        new_coord = coord.copy(points=[point], bounds=[bounds])
        aux_coords_and_dims.append((new_coord, None))

    attributes = {
        "diagnostic_standard_name": "air_temperature",
        "distribution": "norm",
        "title": "Ensemble Model Output Statistics coefficients",
    }

    coeff_names = ["alpha", "beta", "gamma", "delta"]
    cubelist = iris.cube.CubeList([])
    for optimised_coeff, coeff_name in zip(coeff_values, coeff_names):
        modified_dim_coords_and_dims = dim_coords_and_dims.copy()
        modified_aux_coords_and_dims = aux_coords_and_dims.copy()
        coeff_units = "1"
        if coeff_name in ["alpha", "gamma"]:
            coeff_units = template.units
        if coeff_name == "beta":
            fp_names = [fp.name() for fp in forecast_predictors]
            predictor_index = iris.coords.DimCoord(
                np.array(range(len(fp_names)), dtype=np.int32),
                long_name="predictor_index",
                units="1",
            )
            modified_dim_coords_and_dims.append((predictor_index, 0))
            predictor_name = iris.coords.AuxCoord(
                fp_names, long_name="predictor_name", units="no_unit"
            )
            modified_aux_coords_and_dims.append((predictor_name, 0))
        cube = iris.cube.Cube(
            np.atleast_1d(optimised_coeff) if "beta" == coeff_name else optimised_coeff,
            long_name=f"emos_coefficient_{coeff_name}",
            units=coeff_units,
            dim_coords_and_dims=modified_dim_coords_and_dims,
            aux_coords_and_dims=modified_aux_coords_and_dims,
            attributes=attributes,
        )
        cubelist.append(cube)

    return cubelist


class Test_process(IrisTest):
    """Tests for the ApplyEMOS callable plugin"""

    def setUp(self):
        """Set up some "uncalibrated forecast" inputs"""
        attributes = {
            "title": "MOGREPS-UK Forecast",
            "source": "Met Office Unified Model",
            "institution": "Met Office",
        }

        forecast = np.array(
            [np.full((3, 3), 10.4), np.full((3, 3), 10.8), np.full((3, 3), 10.1)],
            dtype=np.float32,
        )
        self.realizations = set_up_variable_cube(
            forecast, units="degC", attributes=attributes
        )

        percentiles = np.array(
            [np.full((3, 3), 10.2), np.full((3, 3), 10.4), np.full((3, 3), 10.6)],
            dtype=np.float32,
        )
        self.percentiles = set_up_percentile_cube(
            percentiles,
            np.array([25, 50, 75], dtype=np.float32),
            units="degC",
            attributes=attributes,
        )

        probabilities = np.array(
            [np.full((3, 3), 1), np.full((3, 3), 0.9), np.full((3, 3), 0)],
            dtype=np.float32,
        )
        self.probabilities = set_up_probability_cube(
            probabilities,
            np.array([9, 10, 11], dtype=np.float32),
            threshold_units="degC",
            attributes=attributes,
        )

        self.coefficients = build_coefficients_cubelist(
            self.realizations, [0, 1, 0, 1], CubeList([self.realizations])
        )

        self.null_percentiles_expected_mean = np.mean(self.percentiles.data)
        self.null_percentiles_expected = np.array(
            [
                np.full((3, 3), 10.265101),
                np.full((3, 3), 10.4),
                np.full((3, 3), 10.534898),
            ]
        )
        self.alternative_percentiles = [25.0, 50.0, 75.0]

    def test_null_percentiles(self):
        """Test effect of "neutral" emos coefficients in percentile space
        (this is small but non-zero due to limited sampling of the
        distribution)"""
        result = ApplyEMOS()(self.percentiles, self.coefficients, realizations_count=3)
        self.assertIn("percentile", get_dim_coord_names(result))
        self.assertArrayAlmostEqual(result.data, self.null_percentiles_expected)
        self.assertAlmostEqual(
            np.mean(result.data), self.null_percentiles_expected_mean
        )

    def test_null_realizations(self):
        """Test effect of "neutral" emos coefficients in realization space"""
        expected_mean = np.mean(self.realizations.data)
        expected_data = np.array(
            [
                np.full((3, 3), 10.433333),
                np.full((3, 3), 10.670206),
                np.full((3, 3), 10.196461),
            ]
        )
        result = ApplyEMOS()(self.realizations, self.coefficients)
        self.assertIn("realization", get_dim_coord_names(result))
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertAlmostEqual(np.mean(result.data), expected_mean)

    def test_null_probabilities(self):
        """Test effect of "neutral" emos coefficients in probability space.
        Mean, 0 and 1 probabilities are not preserved."""
        expected_data = np.array(
            [
                np.full((3, 3), 0.9999999),
                np.full((3, 3), 0.9452005),
                np.full((3, 3), 0.02274995),
            ]
        )
        result = ApplyEMOS()(
            self.probabilities, self.coefficients, realizations_count=3
        )
        self.assertIn("probability_of", result.name())
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_bias(self):
        """Test emos coefficients that correct a bias"""
        # update the "alpha" value
        self.coefficients[0].data = 1
        expected_mean = np.mean(self.percentiles.data + 1.0)
        expected_data = np.array(
            [
                np.full((3, 3), 11.265101),
                np.full((3, 3), 11.4),
                np.full((3, 3), 11.534898),
            ]
        )
        result = ApplyEMOS()(self.percentiles, self.coefficients, realizations_count=3)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertAlmostEqual(np.mean(result.data), expected_mean)

    def test_spread(self):
        """Test emos coefficients that correct underspread"""
        # update the "gamma" value
        self.coefficients[2].data = 1
        expected_mean = np.mean(self.percentiles.data)
        expected_data = np.array(
            [
                np.full((3, 3), 9.7121525),
                np.full((3, 3), 10.4),
                np.full((3, 3), 11.087847),
            ]
        )
        result = ApplyEMOS()(self.percentiles, self.coefficients, realizations_count=3)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertAlmostEqual(np.mean(result.data), expected_mean)

    def test_error_realizations_count(self):
        """Test an error is raised if the realizations_count is not set"""
        msg = "The 'realizations_count' argument must be defined"
        with self.assertRaisesRegex(ValueError, msg):
            ApplyEMOS()(self.percentiles, self.coefficients)

    def test_additional_predictor(self):
        altitude = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32), name="surface_altitude", units="m"
        )
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            altitude.remove_coord(coord)
        coefficients = build_coefficients_cubelist(
            self.realizations,
            [0, [0.9, 0.1], 0, 1],
            CubeList([self.realizations, altitude]),
        )
        expected_data = np.array(
            [
                np.full((3, 3), 9.325102),
                np.full((3, 3), 9.46),
                np.full((3, 3), 9.594898),
            ]
        )
        result = ApplyEMOS()(
            self.percentiles,
            coefficients,
            additional_fields=CubeList([altitude]),
            realizations_count=3,
        )
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_land_sea_mask(self):
        """Test that coefficients can be effectively applied to "land" points
        only"""
        land_sea_data = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.int32)
        land_sea_mask = set_up_variable_cube(
            land_sea_data, name="land_binary_mask", units="1"
        )
        # update the "gamma" value
        self.coefficients[2].data = 1
        expected_data_slice = np.array(
            [
                [9.7121525, 9.7121525, 10.2],
                [9.7121525, 9.7121525, 10.2],
                [9.7121525, 10.2, 10.2],
            ]
        )
        result = ApplyEMOS()(
            self.percentiles,
            self.coefficients,
            land_sea_mask=land_sea_mask,
            realizations_count=3,
        )
        self.assertArrayAlmostEqual(result.data[0], expected_data_slice)

    def test_null_percentiles_truncnorm_standard_shape_parameters(self):
        """Test effect of "neutral" emos coefficients in percentile space
        (this is small but non-zero due to limited sampling of the
        distribution) for the truncated normal distribution."""
        coefficients = iris.cube.CubeList([])
        for cube in self.coefficients:
            cube.attributes["distribution"] = "truncnorm"
            cube.attributes["shape_parameters"] = np.array([0, np.inf], np.float32)
            coefficients.append(cube)

        result = ApplyEMOS()(self.percentiles, coefficients, realizations_count=3)
        self.assertIn("percentile", get_dim_coord_names(result))
        self.assertArrayAlmostEqual(result.data, self.null_percentiles_expected)
        self.assertAlmostEqual(
            np.mean(result.data), self.null_percentiles_expected_mean
        )

    def test_null_percentiles_truncnorm_alternative_shape_parameters(self):
        """Test effect of "neutral" emos coefficients in percentile space
        (this is small but non-zero due to limited sampling of the
        distribution) for the truncated normal distribution with alternative
        shape parameters to show the truncnorm distribution having an effect."""
        coefficients = iris.cube.CubeList([])
        for cube in self.coefficients:
            cube.attributes["distribution"] = "truncnorm"
            cube.attributes["shape_parameters"] = np.array([10, np.inf], np.float32)
            coefficients.append(cube)

        expected_mean = np.mean(self.percentiles.data)
        expected_data = np.array(
            [
                np.full((3, 3), 10.275656),
                np.full((3, 3), 10.405704),
                np.full((3, 3), 10.5385),
            ]
        )
        result = ApplyEMOS()(self.percentiles, coefficients, realizations_count=3)
        self.assertIn("percentile", get_dim_coord_names(result))
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertNotAlmostEqual(np.mean(result.data), expected_mean)

    def test_null_percentiles_frt_fp_mismatch(self):
        """Test effect of "neutral" emos coefficients in percentile space
        where the forecast is 15 minutes ahead of the coefficients in terms
        of the forecast reference time."""
        percentiles = self.percentiles.copy()
        mins_15_to_secs = 900
        percentiles.coord("forecast_reference_time").points = (
            percentiles.coord("forecast_reference_time").points + mins_15_to_secs
        )
        percentiles.coord("forecast_period").points = (
            percentiles.coord("forecast_period").points - mins_15_to_secs
        )
        expected_frt = percentiles.coord("forecast_reference_time").points
        expected_fp = percentiles.coord("forecast_period").points
        result = ApplyEMOS()(percentiles, self.coefficients, realizations_count=3)
        self.assertAlmostEqual(
            result.coord("forecast_reference_time").points, expected_frt
        )
        self.assertAlmostEqual(result.coord("forecast_period").points, expected_fp)
        self.assertArrayAlmostEqual(result.data, self.null_percentiles_expected)
        self.assertAlmostEqual(
            np.mean(result.data), self.null_percentiles_expected_mean
        )

    def test_alternative_percentiles(self):
        """Test that the calibrated forecast is at a specified set of
        percentiles."""
        result = ApplyEMOS(percentiles=self.alternative_percentiles)(
            self.percentiles, self.coefficients, realizations_count=3
        )
        self.assertArrayEqual(
            result.coord("percentile").points, self.alternative_percentiles
        )

    def test_alternative_string_percentiles(self):
        """Test that the calibrated forecast is at a specified set of
        percentiles where the input percentiles are strings."""
        str_percentiles = list(map(str, self.alternative_percentiles))
        result = ApplyEMOS(percentiles=str_percentiles)(
            self.percentiles, self.coefficients, realizations_count=3
        )
        self.assertArrayEqual(
            result.coord("percentile").points, self.alternative_percentiles
        )

    def test_invalid_attribute(self):
        """Test that an exception is raised if multiple different distribution
        attributes are provided within the coefficients cubelist."""
        self.coefficients[0].attributes["distribution"] = "truncnorm"
        msg = "Coefficients must share the same"
        with self.assertRaisesRegex(AttributeError, msg):
            ApplyEMOS()(self.percentiles, self.coefficients, realizations_count=3)

    def test_missing_attribute(self):
        """Test that an exception is raised if the expected distribution
        attribute is missing from within the coefficients cubelist."""
        self.coefficients[0].attributes.pop("distribution")
        msg = "Coefficients must share the same"
        with self.assertRaisesRegex(AttributeError, msg):
            ApplyEMOS()(self.percentiles, self.coefficients, realizations_count=3)

    def test_completely_missing_attribute(self):
        """Test that an exception is raised if the expected distribution
        attribute is missing from all cubes within the coefficients cubelist."""
        for cube in self.coefficients:
            cube.attributes.pop("distribution")
        msg = "The distribution attribute must be specified on all coefficients cubes."
        with self.assertRaisesRegex(AttributeError, msg):
            ApplyEMOS()(self.percentiles, self.coefficients, realizations_count=3)


if __name__ == "__main__":
    unittest.main()
