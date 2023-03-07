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
"""Unit tests for calibration.__init__"""

import unittest
from datetime import datetime
from typing import List, Tuple

import iris
import numpy as np
import pytest
from iris.cube import CubeList

from improver.calibration import (
    add_warning_comment,
    split_forecasts_and_coeffs,
    split_forecasts_and_truth,
    validity_time_check,
)
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver_tests import ImproverTest


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


def create_input_cubes(forecast_type: str) -> Tuple[List, List, List, List]:
    """Create cubes for testing the split_forecasts_and_truth method.
    Forecast data is all set to 1, and truth data to 0, allowing for a
    simple check that the cubes have been separated as expected."""

    thresholds = [283, 288]
    truth_attributes = {"mosg__model_configuration": "uk_det"}

    if forecast_type == "probability":
        shape = (2, 4, 4)
        data = np.ones(shape, dtype=np.float32)
        forecast_1 = set_up_probability_cube(data, thresholds)
        forecast_2 = set_up_probability_cube(
            data,
            thresholds,
            time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0),
        )
    else:
        shape = (4, 4)
        data = np.ones(shape, dtype=np.float32)
        forecast_1 = set_up_variable_cube(data)
        forecast_2 = set_up_variable_cube(
            data, time=datetime(2017, 11, 11, 4, 0), frt=datetime(2017, 11, 11, 0, 0),
        )

    forecasts = [forecast_1, forecast_2]

    truth_1 = forecast_1.copy(data=np.zeros(shape, dtype=np.float32))
    truth_2 = forecast_2.copy(data=np.zeros(shape, dtype=np.float32))
    truth_1.attributes.update(truth_attributes)
    truth_2.attributes.update(truth_attributes)
    truths = [truth_1, truth_2]

    additional_predictor_1 = set_up_variable_cube(
        data=np.ones((4, 4), dtype=np.float32)
    )
    for coord in ["time", "forecast_reference_time", "forecast_period"]:
        additional_predictor_1.remove_coord(coord)
    additional_predictor_1.rename("kitten_present")
    additional_predictor_2 = additional_predictor_1.copy()
    additional_predictor_2.rename("puppy_present")
    additional_predictors = [additional_predictor_1, additional_predictor_2]

    landsea_mask = additional_predictor_1.copy(data=np.zeros((4, 4), dtype=np.float32))
    landsea_mask.rename("land_binary_mask")
    return forecasts, truths, [landsea_mask], additional_predictors


def cubes_for_exceptions() -> Tuple[List, List, List, str]:
    """Create input cubes for testing exceptions."""
    (realization_forecasts, realization_truths, land_sea_mask, _) = create_input_cubes(
        "realization"
    )
    truth_attribute = "mosg__model_configuration=uk_det"
    return realization_forecasts, realization_truths, land_sea_mask, truth_attribute


@pytest.mark.parametrize("forecast_type", ("probability", "realization"))
@pytest.mark.parametrize("include_land_sea_mask", (True, False))
@pytest.mark.parametrize("n_additional_predictors", (0, 1, 2))
def test_split_forecasts_truth(
    forecast_type, include_land_sea_mask, n_additional_predictors
):
    """Test that when multiple probability forecast cubes and truth cubes
    are provided, the groups are created as expected. Additionally, test that groups
    are created as expected when forecasts are either probabilities or realizations,
    when a landsea mask is or is not provided, and when static additional predictors
    are or are not provided. """

    truth_attribute = "mosg__model_configuration=uk_det"
    (
        input_forecast,
        input_truth,
        input_land_sea_mask,
        input_static_predictors,
    ) = create_input_cubes(forecast_type)
    land_sea_mask_name = input_land_sea_mask[0].name()

    if not include_land_sea_mask:
        input_land_sea_mask = []
    if n_additional_predictors == 0:
        input_static_predictors = []
    if n_additional_predictors == 1:
        input_static_predictors = [input_static_predictors[0]]

    (
        forecast,
        truth,
        land_sea_mask,
        additional_predictors,
    ) = split_forecasts_and_truth(
        input_forecast + input_truth + input_land_sea_mask + input_static_predictors,
        truth_attribute,
        land_sea_mask_name,
    )

    assert isinstance(forecast, iris.cube.Cube)
    assert isinstance(truth, iris.cube.Cube)
    assert np.all(forecast.data)
    assert not np.any(truth.data)
    if forecast_type == "probability":
        assert (2, 2, 4, 4) == forecast.shape
        assert (2, 2, 4, 4) == truth.shape
    else:
        assert (2, 4, 4) == forecast.shape
        assert (2, 4, 4) == truth.shape
    if include_land_sea_mask:
        assert isinstance(land_sea_mask, iris.cube.Cube)
        assert "land_binary_mask" == land_sea_mask.name()
        assert (4, 4) == land_sea_mask.shape
    else:
        assert land_sea_mask is None
    if n_additional_predictors > 0:
        assert len(additional_predictors) == n_additional_predictors
        predictor_names = ["kitten_present", "puppy_present"]
        for index, cube in enumerate(additional_predictors):
            assert isinstance(cube, iris.cube.Cube)
            assert predictor_names[index] == cube.name()
            assert np.all(cube.data)
            assert (4, 4) == cube.shape
    else:
        assert additional_predictors is None


@pytest.mark.parametrize(
    "condition, error_class, msg",
    (
        ("multiple_landmasks", IOError, "Expected at most one cube for land-sea mask."),
        ("name_mismatch", ValueError, "Only forecasts for one diagnostic can be "),
        ("missing_truths", IOError, "Missing truth input."),
        ("missing_forecasts", IOError, "Missing historical forecast input."),
    ),
)
def test_split_forecasts_truth_exceptions(condition, error_class, msg):
    """
    Test that correct exceptions are raised when (1) multiple land-sea masks are
    input, (2) forecasts and truths have different names, (3) truths are not input, and
    (4) forecasts are not input.
    """
    (
        realization_forecasts,
        realization_truths,
        land_sea_mask,
        truth_attribute,
    ) = cubes_for_exceptions()

    land_sea_mask_name = land_sea_mask[0].name()
    if condition == "multiple_landmasks":
        land_sea_mask = land_sea_mask + land_sea_mask
    elif condition == "name_mismatch":
        realization_truths[0].rename("kitten_density")
    elif condition == "missing_truths":
        realization_truths = []
    elif condition == "missing_forecasts":
        realization_forecasts = []

    with pytest.raises(error_class, match=msg):
        split_forecasts_and_truth(
            realization_forecasts + realization_truths + land_sea_mask,
            truth_attribute,
            land_sea_mask_name,
        )


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


if __name__ == "__main__":
    unittest.main()
