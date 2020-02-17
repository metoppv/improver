# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
`ensemble_calibration.ApplyCoefficientsForEnsembleCalibration`
class.

"""
import unittest

import iris
import numpy as np
from iris.tests import IrisTest
from numpy.testing import assert_array_almost_equal

from improver.calibration.ensemble_calibration import \
    ApplyCoefficientsFromEnsembleCalibration as Plugin
from improver.calibration.ensemble_calibration import (
    EstimateCoefficientsForEnsembleCalibration)
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.utilities.warnings_handler import ManageWarnings

from .helper_functions import EnsembleCalibrationAssertions, SetupCubes
from ...set_up_test_cubes import set_up_variable_cube
from .test_EstimateCoefficientsForEnsembleCalibration import (
    SetupExpectedCoefficients)


class SetupCoefficientsCubes(SetupCubes, SetupExpectedCoefficients):

    """Set up coefficients cubes for testing."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence",
                          "can't resolve package from",
                          "The statsmodels can not be imported"],
        warning_types=[UserWarning, DeprecationWarning, ImportWarning,
                       ImportWarning])
    def setUp(self):
        """Set up coefficients cubes for when either the ensemble mean or the
        ensemble realizations have been used as the predictor. The coefficients
        have been constructed from the same underlying set of ensemble
        realizations, so application of these coefficients would be expected
        to give similar results. The values for the coefficients used to
        construct the coefficients cubes are taken from the
        SetupExpectedCoefficients class. These coefficients are the
        expected outputs from the tests to estimate the coefficients."""
        super().setUp()
        # Set up a coefficients cube when using the ensemble mean as the
        # predictor.
        current_cycle = "20171110T0000Z"
        estimator = (
            EstimateCoefficientsForEnsembleCalibration(
                "gaussian", current_cycle, desired_units="Celsius"))
        self.coeffs_from_mean = (
            estimator.create_coefficients_cube(
                self.expected_mean_predictor_gaussian,
                self.current_temperature_forecast_cube))

        # Set up a coefficients cube when using the ensemble realization as the
        # predictor and the coefficients have been estimated using statsmodels.
        estimator = (
            EstimateCoefficientsForEnsembleCalibration(
                "gaussian", current_cycle, desired_units="Celsius",
                predictor="realizations"))
        self.coeffs_from_statsmodels_realizations = (
            estimator.create_coefficients_cube(
                self.expected_realizations_gaussian_statsmodels,
                self.current_temperature_forecast_cube))

        # Set up a coefficients cube when using the ensemble realization as the
        # predictor and the coefficients have been estimated without using
        # statsmodels.
        estimator = (
            EstimateCoefficientsForEnsembleCalibration(
                "gaussian", current_cycle, desired_units="Celsius",
                predictor="realizations"))
        self.coeffs_from_no_statsmodels_realizations = (
            estimator.create_coefficients_cube(
                self.expected_realizations_gaussian_no_statsmodels,
                self.current_temperature_forecast_cube))

        # Some expected data that are used in various tests.
        self.expected_loc_param_mean = (
            np.array([[273.7854, 274.6913, 275.4461],
                      [276.8652, 277.6502, 278.405],
                      [279.492, 280.1562, 280.9715]], dtype=np.float32))
        self.expected_scale_param_mean = (
            np.array([[0.1952, 0.1974, 0.0117],
                      [0.0226, 0.0197, 0.0117],
                      [0.0532, 0.0029, 0.0007]], dtype=np.float32))
        self.expected_loc_param_statsmodels_realizations = (
            np.array([[274.1395, 275.0975, 275.258],
                      [276.9771, 277.3487, 278.3144],
                      [280.0085, 280.2506, 281.1632]], dtype=np.float32))
        self.expected_loc_param_no_statsmodels_realizations = (
            np.array([[273.4695, 274.4673, 275.3034],
                      [276.8648, 277.733, 278.5632],
                      [279.7562, 280.4913, 281.3889]], dtype=np.float32))

        # Create output cubes with the expected data.
        self.expected_loc_param_mean_cube = set_up_variable_cube(
            self.expected_loc_param_mean, name="location_parameter",
            units="K", attributes=MANDATORY_ATTRIBUTE_DEFAULTS)
        self.expected_scale_param_mean_cube = (
            set_up_variable_cube(
                self.expected_scale_param_mean,
                name="scale_parameter", units="Kelvin^2",
                attributes=MANDATORY_ATTRIBUTE_DEFAULTS))


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test without specifying a predictor."""
        plugin = Plugin()
        self.assertEqual(plugin.predictor, "mean")

    def test_with_predictor(self):
        """Test specifying the predictor."""
        plugin = Plugin(predictor="realizations")
        self.assertEqual(plugin.predictor, "realizations")


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test without the predictor."""
        result = str(Plugin())
        msg = ("<ApplyCoefficientsFromEnsembleCalibration: "
               "predictor: mean>")
        self.assertEqual(result, msg)

    def test_with_predictor(self):
        """Test specifying the predictor."""
        result = str(Plugin(predictor="realizations"))
        msg = ("<ApplyCoefficientsFromEnsembleCalibration: "
               "predictor: realizations>")
        self.assertEqual(result, msg)


class Test__spatial_domain_match(SetupCoefficientsCubes):

    """ Test the _spatial_domain_match method."""

    def setUp(self):
        super().setUp()
        self.plugin = Plugin()

    def test_matching(self):
        """Test case in which spatial domains match."""
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cube = self.coeffs_from_mean
        self.plugin._spatial_domain_match()

    def test_unmatching_x_axis(self):
        """Test case in which the x-dimensions of the domains do not match."""
        self.current_temperature_forecast_cube.coord(axis='x').points = (
            self.current_temperature_forecast_cube.coord(axis='x').points * 2.)
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cube = self.coeffs_from_mean
        msg = "The domain along the x axis given by the current forecast"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._spatial_domain_match()

    def test_unmatching_y_axis(self):
        """Test case in which the y-dimensions of the domains do not match."""
        self.current_temperature_forecast_cube.coord(axis='y').points = (
            self.current_temperature_forecast_cube.coord(axis='y').points * 2.)
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cube = self.coeffs_from_mean
        msg = "The domain along the y axis given by the current forecast"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._spatial_domain_match()


class Test__calculate_location_parameter_from_mean(
        SetupCoefficientsCubes, EnsembleCalibrationAssertions):

    """Test the __calculate_location_parameter_from_mean method."""

    def setUp(self):
        """Set-up coefficients and plugin for testing."""
        super().setUp()

        self.optimised_coeffs = (
            dict(zip(self.coeffs_from_mean.coord("coefficient_name").points,
                     self.coeffs_from_mean.data)))
        self.plugin = Plugin()
        self.plugin.current_forecast = self.current_temperature_forecast_cube

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """Test that the expected values for the location parameter are
        calculated when using the ensemble mean. These expected values are
        compared to the results when using the ensemble realizations to ensure
        that the results are similar."""
        location_parameter = (
            self.plugin._calculate_location_parameter_from_mean(
                self.optimised_coeffs))
        self.assertCalibratedVariablesAlmostEqual(
            location_parameter, self.expected_loc_param_mean)
        assert_array_almost_equal(
            location_parameter,
            self.expected_loc_param_statsmodels_realizations,
            decimal=0)
        assert_array_almost_equal(
            location_parameter,
            self.expected_loc_param_no_statsmodels_realizations,
            decimal=0)


class Test__calculate_location_parameter_from_realizations(
        SetupCoefficientsCubes, EnsembleCalibrationAssertions):

    """Test the _calculate_location_parameter_from_realizations method."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def setUp(self):
        """Set-up coefficients and plugin for testing."""
        super().setUp()

        self.plugin = Plugin()
        self.plugin.current_forecast = self.current_temperature_forecast_cube

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_with_statsmodels(self):
        """Test that the expected values for the location parameter are
        calculated when using the ensemble realizations with statsmodels.
        These expected values are compared to the results when using the
        ensemble mean and when statsmodels is not used to ensure that the
        results are similar."""
        optimised_coeffs = dict(
            zip(self.coeffs_from_statsmodels_realizations.coord(
                "coefficient_name").points,
                self.coeffs_from_statsmodels_realizations.data))
        location_parameter = (
            self.plugin._calculate_location_parameter_from_realizations(
                optimised_coeffs))
        self.assertCalibratedVariablesAlmostEqual(
            location_parameter,
            self.expected_loc_param_statsmodels_realizations)
        assert_array_almost_equal(
            location_parameter,
            self.expected_loc_param_mean,
            decimal=0)
        assert_array_almost_equal(
            location_parameter,
            self.expected_loc_param_no_statsmodels_realizations,
            decimal=0)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_without_statsmodels(self):
        """Test that the expected values for the location parameter are
        calculated when using the ensemble realizations without statsmodels.
        These expected values are compared to the results when using the
        ensemble mean and when statsmodels is used to ensure that the results
        are similar."""
        optimised_coeffs = dict(
            zip(self.coeffs_from_no_statsmodels_realizations.coord(
                "coefficient_name").points,
                self.coeffs_from_no_statsmodels_realizations.data))
        location_parameter = (
            self.plugin._calculate_location_parameter_from_realizations(
                optimised_coeffs))
        self.assertCalibratedVariablesAlmostEqual(
            location_parameter,
            self.expected_loc_param_no_statsmodels_realizations)
        assert_array_almost_equal(
            location_parameter,
            self.expected_loc_param_mean,
            decimal=0)
        assert_array_almost_equal(
            location_parameter,
            self.expected_loc_param_statsmodels_realizations,
            decimal=0)


class Test__calculate_scale_parameter(
        SetupCoefficientsCubes, EnsembleCalibrationAssertions):

    """Test the _calculate_scale_parameter method."""

    def setUp(self):
        """Set-up the plugin for testing."""
        super().setUp()
        self.plugin = Plugin()
        self.plugin.current_forecast = self.current_temperature_forecast_cube

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """Test the scale parameter is calculated correctly."""
        optimised_coeffs = dict(
            zip(self.coeffs_from_mean.coord("coefficient_name").points,
                self.coeffs_from_mean.data))
        scale_parameter = (
            self.plugin._calculate_scale_parameter(optimised_coeffs))
        self.assertCalibratedVariablesAlmostEqual(
            scale_parameter, self.expected_scale_param_mean)


class Test__create_output_cubes(
        SetupCoefficientsCubes, EnsembleCalibrationAssertions):

    """Test the _create_output_cubes method."""

    def setUp(self):
        """Set-up the plugin for testing."""
        super().setUp()
        self.plugin = Plugin()
        self.plugin.current_forecast = self.current_temperature_forecast_cube

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """Test that the cubes created containing the location and scale
        parameter are formatted as expected."""
        location_parameter_cube, scale_parameter_cube = (
            self.plugin._create_output_cubes(
                self.expected_loc_param_mean,
                self.expected_scale_param_mean))

        self.assertEqual(
            location_parameter_cube,
            self.expected_loc_param_mean_cube)
        self.assertEqual(
            scale_parameter_cube,
            self.expected_scale_param_mean_cube)


class Test_process(SetupCoefficientsCubes, EnsembleCalibrationAssertions):

    """Test the process plugin."""

    def setUp(self):
        """Set-up the plugin for testing."""
        super().setUp()
        self.plugin = Plugin()

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_variable_setting(self):
        """Test that the cubes passed into the plugin are allocated to
        plugin variables appropriately."""

        _, _ = self.plugin.process(self.current_temperature_forecast_cube,
                                   self.coeffs_from_mean)
        self.assertEqual(self.current_temperature_forecast_cube,
                         self.plugin.current_forecast)
        self.assertEqual(self.coeffs_from_mean,
                         self.plugin.coefficients_cube)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_end_to_end(self):
        """An example end-to-end calculation. This repeats the test elements
        above but all grouped together."""
        calibrated_forecast_predictor, calibrated_forecast_var = (
            self.plugin.process(self.current_temperature_forecast_cube,
                                self.coeffs_from_mean))

        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data,
            self.expected_loc_param_mean)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data,
            self.expected_scale_param_mean)
        self.assertEqual(calibrated_forecast_predictor.dtype, np.float32)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_end_to_end_with_mask(self):
        """An example end-to-end calculation, but making sure that the
        areas that are masked within the landsea mask, are masked at the
        end."""

        # Construct a mask and encapsulate as a cube.
        mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mask_cube = self.current_temperature_forecast_cube[0].copy(data=mask)
        # Convention for IMPROVER is that land points are ones and sea points
        # are zeros in land-sea masks. In this case we want to mask sea points.
        expected_mask = np.array([[False, True, True],
                                  [True, False, True],
                                  [True, True, False]])

        calibrated_forecast_predictor, calibrated_forecast_var = (
            self.plugin.process(self.current_temperature_forecast_cube,
                                self.coeffs_from_mean, landsea_mask=mask_cube))

        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data.data,
            self.expected_loc_param_mean)
        self.assertArrayEqual(
            calibrated_forecast_predictor.data.mask, expected_mask)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data.data, self.expected_scale_param_mean)
        self.assertArrayEqual(
            calibrated_forecast_var.data.mask, expected_mask)


if __name__ == '__main__':
    unittest.main()
