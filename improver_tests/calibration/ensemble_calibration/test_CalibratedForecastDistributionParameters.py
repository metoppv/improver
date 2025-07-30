# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the
`ensemble_calibration.CalibratedForecastDistributionParameters`
class.

"""

import unittest

import numpy as np
from iris.cube import CubeList
from iris.tests import IrisTest
from numpy.testing import assert_array_almost_equal

from improver.calibration.ensemble_calibration import (
    CalibratedForecastDistributionParameters as Plugin,
)
from improver.calibration.ensemble_calibration import (
    EstimateCoefficientsForEnsembleCalibration,
)
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

from .helper_functions import EnsembleCalibrationAssertions, SetupCubes
from .test_EstimateCoefficientsForEnsembleCalibration import SetupExpectedCoefficients


class SetupCoefficientsCubes(SetupCubes, SetupExpectedCoefficients):
    """Set up coefficients cubes for testing."""

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
        estimator = EstimateCoefficientsForEnsembleCalibration(
            "norm", desired_units="Celsius"
        )
        self.coeffs_from_mean = estimator.create_coefficients_cubelist(
            self.expected_mean_pred_norm,
            self.historic_temperature_forecast_cube,
            CubeList([self.historic_temperature_forecast_cube]),
        )

        # Set up a timeshifted coefficients cube using the ensemble mean as a
        # predictor.
        forecast_timeshift_cube = self.historic_temperature_forecast_cube.copy()
        for coord_name in ["time", "forecast_period"]:
            forecast_timeshift_cube.coord(coord_name).points = [
                _ + 3600 for _ in forecast_timeshift_cube.coord(coord_name).points
            ]

        self.coeffs_from_mean_timeshift = estimator.create_coefficients_cubelist(
            self.expected_mean_pred_norm,
            forecast_timeshift_cube,
            CubeList([forecast_timeshift_cube]),
        )

        # Set up a coefficients cube when using the ensemble mean as the
        # predictor and separate coefficients at each point.
        estimator = EstimateCoefficientsForEnsembleCalibration(
            "norm", point_by_point=True, desired_units="Celsius"
        )
        point_by_point_predictor = np.stack(
            [self.expected_mean_pred_norm] * 9
        ).T.reshape(4, 3, 3)
        self.coeffs_from_mean_point_by_point = estimator.create_coefficients_cubelist(
            point_by_point_predictor,
            self.historic_temperature_forecast_cube,
            CubeList([self.historic_temperature_forecast_cube]),
        )

        # Set up a coefficients cube when using the ensemble realization as the
        # predictor.
        estimator = EstimateCoefficientsForEnsembleCalibration(
            "norm", desired_units="Celsius", predictor="realizations"
        )
        self.coeffs_from_realizations = estimator.create_coefficients_cubelist(
            self.expected_realizations_norm,
            self.historic_temperature_forecast_cube,
            CubeList([self.historic_temperature_forecast_cube]),
        )

        # Set up a coefficients cube when using the ensemble realization as the
        # predictor and separate coefficients at each point.
        expected_realizations_each_site = [
            array if array.ndim == 1 else np.squeeze(array)
            for array in list(self.expected_realizations_each_site.values())
        ]

        estimator = EstimateCoefficientsForEnsembleCalibration(
            "norm", predictor="realizations", point_by_point=True
        )
        self.coeffs_from_realizations_sites = estimator.create_coefficients_cubelist(
            expected_realizations_each_site,
            self.historic_forecast_spot_cube,
            CubeList([self.historic_temperature_forecast_cube]),
        )

        # # Set up a coefficients cube when using an additional predictor.
        self.altitude = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32), name="surface_altitude", units="m"
        )
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.altitude.remove_coord(coord)

        estimator = EstimateCoefficientsForEnsembleCalibration(
            "norm", desired_units="Celsius"
        )
        self.coeffs_from_mean_alt = estimator.create_coefficients_cubelist(
            self.expected_mean_pred_norm_alt,
            self.historic_temperature_forecast_cube,
            CubeList([self.historic_temperature_forecast_cube, self.altitude]),
        )

        # Some expected data that are used in various tests.
        self.expected_loc_param_mean = np.array(
            [
                [273.7014, 274.6534, 275.4469],
                [276.9385, 277.7636, 278.5570],
                [279.6996, 280.1122, 281.2547],
            ],
            dtype=np.float32,
        )
        self.expected_scale_param_mean = np.array(
            [
                [0.4813, 0.4840, 0.1295],
                [0.1647, 0.1538, 0.1295],
                [0.2517, 0.3393, 0.1076],
            ],
            dtype=np.float32,
        )
        self.expected_loc_param_realizations = np.array(
            [
                [274.388, 275.3053, 275.4492],
                [277.1295, 277.3866, 278.4672],
                [280.2007, 280.3929, 281.2602],
            ],
            dtype=np.float32,
        )
        self.expected_loc_param_realizations_sites = np.array(
            [277.7437, 277.4434, 277.5435, 277.2432], dtype=np.float32
        )

        self.expected_scale_param_realizations_sites = np.array(
            [0.0005, 0.0005, 0.0005, 0.0005], dtype=np.float32
        )

        self.expected_loc_param_mean_alt = np.array(
            [
                [275.1603, 276.1604, 276.9938],
                [278.5606, 279.4273, 280.2607],
                [281.4609, 281.8943, 283.0944],
            ],
            dtype=np.float32,
        )

        self.expected_scale_param_mean_alt = np.array(
            [
                [1.0636, 1.0695, 0.2832],
                [0.3617, 0.3376, 0.2832],
                [0.5551, 0.7493, 0.2343],
            ],
            dtype=np.float32,
        )

        # Create output cubes with the expected data.
        self.expected_loc_param_mean_cube = set_up_variable_cube(
            self.expected_loc_param_mean,
            name="location_parameter",
            units="K",
            attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        )
        self.expected_scale_param_mean_cube = set_up_variable_cube(
            self.expected_scale_param_mean,
            name="scale_parameter",
            units="K",
            attributes=MANDATORY_ATTRIBUTE_DEFAULTS,
        )


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
        msg = "<CalibratedForecastDistributionParameters: predictor: mean>"
        self.assertEqual(result, msg)

    def test_with_predictor(self):
        """Test specifying the predictor."""
        result = str(Plugin(predictor="realizations"))
        msg = "<CalibratedForecastDistributionParameters: predictor: realizations>"
        self.assertEqual(result, msg)


class Test__spatial_domain_match(SetupCoefficientsCubes):
    """Test the _spatial_domain_match method."""

    def setUp(self):
        super().setUp()
        self.plugin = Plugin()

    def test_matching(self):
        """Test case in which spatial domains match."""
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cubelist = self.coeffs_from_mean
        self.plugin._spatial_domain_match()

    def test_unmatching_x_axis_points(self):
        """Test when the points of the x dimension do not match."""
        self.current_temperature_forecast_cube.coord(axis="x").bounds = (
            self.current_temperature_forecast_cube.coord(axis="x").bounds + 2.0
        )
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cubelist = self.coeffs_from_mean
        msg = "The points or bounds of the x axis given by the current forecast"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._spatial_domain_match()

    def test_unmatching_x_axis_bounds(self):
        """Test when the bounds of the x dimension do not match."""
        self.current_temperature_forecast_cube.coord(axis="x").bounds = [
            [-35, -5],
            [-5, 5],
            [5, 35],
        ]
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cubelist = self.coeffs_from_mean
        msg = "The points or bounds of the x axis given by the current forecast"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._spatial_domain_match()

    def test_unmatching_y_axis(self):
        """Test case in which the y-dimensions of the domains do not match."""
        self.current_temperature_forecast_cube.coord(axis="y").bounds = (
            self.current_temperature_forecast_cube.coord(axis="y").bounds + 2.0
        )
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cubelist = self.coeffs_from_mean
        msg = "The points or bounds of the y axis given by the current forecast"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._spatial_domain_match()

    def test_skipping_spot_forecast(self):
        """Test passing a spot forecast. In this case, the spatial domain
        is not checked."""
        self.plugin.current_forecast = self.current_forecast_spot_cube
        self.plugin._spatial_domain_match()


class Test__calculate_location_parameter_from_mean(
    SetupCoefficientsCubes, EnsembleCalibrationAssertions
):
    """Test the __calculate_location_parameter_from_mean method."""

    def setUp(self):
        """Set-up coefficients and plugin for testing."""
        super().setUp()

        self.plugin = Plugin()
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cubelist = self.coeffs_from_mean

    def test_basic(self):
        """Test that the expected values for the location parameter are
        calculated when using the ensemble mean. These expected values are
        compared to the results when using the ensemble realizations to ensure
        that the results are similar."""
        location_parameter = self.plugin._calculate_location_parameter_from_mean()
        self.assertCalibratedVariablesAlmostEqual(
            location_parameter, self.expected_loc_param_mean
        )
        assert_array_almost_equal(
            location_parameter, self.expected_loc_param_realizations, decimal=0
        )

    def test_missing_additional_predictor(self):
        """Test that an error is raised if an additional predictor is expected
        based on the contents of the coefficients cube."""
        self.plugin.coefficients_cubelist = self.coeffs_from_mean_alt
        msg = "The number of forecast predictors must equal the number"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._calculate_location_parameter_from_mean()


class Test__calculate_location_parameter_from_realizations(
    SetupCoefficientsCubes, EnsembleCalibrationAssertions
):
    """Test the _calculate_location_parameter_from_realizations method."""

    def setUp(self):
        """Set-up coefficients and plugin for testing."""
        super().setUp()

        self.plugin = Plugin()
        self.plugin.current_forecast = self.current_temperature_forecast_cube

    def test_basic(self):
        """Test that the expected values for the location parameter are
        calculated when using the ensemble realizations. These expected values
        are compared to the results when using the ensemble mean to ensure
        that the results are similar."""
        self.plugin.coefficients_cubelist = self.coeffs_from_realizations
        location_parameter = (
            self.plugin._calculate_location_parameter_from_realizations()
        )
        self.assertCalibratedVariablesAlmostEqual(
            location_parameter, self.expected_loc_param_realizations
        )
        assert_array_almost_equal(
            location_parameter, self.expected_loc_param_mean, decimal=0
        )


class Test__calculate_scale_parameter(
    SetupCoefficientsCubes, EnsembleCalibrationAssertions
):
    """Test the _calculate_scale_parameter method."""

    def setUp(self):
        """Set-up the plugin for testing."""
        super().setUp()
        self.plugin = Plugin()
        self.plugin.current_forecast = self.current_temperature_forecast_cube

    def test_basic(self):
        """Test the scale parameter is calculated correctly."""
        self.plugin.coefficients_cubelist = self.coeffs_from_mean
        scale_parameter = self.plugin._calculate_scale_parameter()
        self.assertCalibratedVariablesAlmostEqual(
            scale_parameter, self.expected_scale_param_mean
        )


class Test__create_output_cubes(SetupCoefficientsCubes, EnsembleCalibrationAssertions):
    """Test the _create_output_cubes method."""

    def setUp(self):
        """Set-up the plugin for testing."""
        super().setUp()
        self.plugin = Plugin()
        self.plugin.current_forecast = self.current_temperature_forecast_cube

    def test_basic(self):
        """Test that the cubes created containing the location and scale
        parameter are formatted as expected."""
        (
            location_parameter_cube,
            scale_parameter_cube,
        ) = self.plugin._create_output_cubes(
            self.expected_loc_param_mean, self.expected_scale_param_mean
        )
        self.assertEqual(location_parameter_cube, self.expected_loc_param_mean_cube)
        self.assertEqual(scale_parameter_cube, self.expected_scale_param_mean_cube)


class Test_process(SetupCoefficientsCubes, EnsembleCalibrationAssertions):
    """Test the process plugin."""

    def setUp(self):
        """Set-up the plugin for testing."""
        super().setUp()
        self.plugin = Plugin()

    def test_diagnostic_match(self):
        """Test that an error is raised if the diagnostic_standard_name does
        not match when comparing a forecast cube and coefficients cubelist."""
        msg = "The forecast diagnostic"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(
                self.current_wind_speed_forecast_cube, self.coeffs_from_mean
            )

    def test_time_match(self):
        """Test that an error is raised if the time coordinates do
        not match when comparing a forecast cube and coefficients cubelist."""
        msg = "rounded forecast_period hours"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process(
                self.current_temperature_forecast_cube, self.coeffs_from_mean_timeshift
            )

    def test_time_match_tolerate(self):
        """Test that no error is raised when using a coefficients file with
        a mismatching forecast_period coordinate, if the
        tolerate_time_mismatch option is enabled."""
        calibrated_forecast_predictor, calibrated_forecast_var = self.plugin.process(
            self.current_temperature_forecast_cube,
            self.coeffs_from_mean_timeshift,
            tolerate_time_mismatch=True,
        )
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data, self.expected_loc_param_mean
        )
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data, self.expected_scale_param_mean
        )
        self.assertEqual(calibrated_forecast_predictor.dtype, np.float32)

    def test_variable_setting(self):
        """Test that the cubes passed into the plugin are allocated to
        plugin variables appropriately."""

        _, _ = self.plugin.process(
            self.current_temperature_forecast_cube, self.coeffs_from_mean
        )
        self.assertEqual(
            self.current_temperature_forecast_cube, self.plugin.current_forecast
        )
        self.assertEqual(self.coeffs_from_mean, self.plugin.coefficients_cubelist)

    def test_end_to_end(self):
        """An example end-to-end calculation. This repeats the test elements
        above but all grouped together."""
        calibrated_forecast_predictor, calibrated_forecast_var = self.plugin.process(
            self.current_temperature_forecast_cube, self.coeffs_from_mean
        )

        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data, self.expected_loc_param_mean
        )
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data, self.expected_scale_param_mean
        )
        self.assertEqual(calibrated_forecast_predictor.dtype, np.float32)

    def test_end_to_end_point_by_point(self):
        """An example end-to-end calculation when a separate set of
        coefficients are computed for each grid point. This repeats the test
        elements above but all grouped together."""
        calibrated_forecast_predictor, calibrated_forecast_var = self.plugin.process(
            self.current_temperature_forecast_cube, self.coeffs_from_mean_point_by_point
        )

        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data, self.expected_loc_param_mean
        )
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data, self.expected_scale_param_mean
        )
        self.assertEqual(calibrated_forecast_predictor.dtype, np.float32)

    def test_end_to_end_point_by_point_sites_realizations(self):
        """An example end-to-end calculation when a separate set of
        coefficients are computed for each site using the realizations as the
        predictor. This repeats the test elements above but all grouped together."""
        plugin = Plugin(predictor="realizations")
        calibrated_forecast_predictor, calibrated_forecast_var = plugin.process(
            self.current_forecast_spot_cube, self.coeffs_from_realizations_sites
        )

        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data,
            self.expected_loc_param_realizations_sites,
        )
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data, self.expected_scale_param_realizations_sites
        )
        self.assertEqual(calibrated_forecast_predictor.dtype, np.float32)

    def test_end_to_end_with_additional_predictor(self):
        """Test that the expected calibrated forecast is generated, if an
        additional predictor is provided."""
        calibrated_forecast_predictor, calibrated_forecast_var = self.plugin.process(
            self.current_temperature_forecast_cube,
            self.coeffs_from_mean_alt,
            additional_fields=CubeList([self.altitude]),
        )

        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data, self.expected_loc_param_mean_alt
        )
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data, self.expected_scale_param_mean_alt
        )
        self.assertEqual(calibrated_forecast_predictor.dtype, np.float32)

    def test_end_to_end_with_mask(self):
        """An example end-to-end calculation, but making sure that the
        areas that are masked within the landsea mask, are masked at the
        end."""

        # Construct a mask and encapsulate as a cube.
        mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mask_cube = self.current_temperature_forecast_cube[0].copy(data=mask)
        # Convention for IMPROVER is that land points are ones and sea points
        # are zeros in land-sea masks. In this case we want to mask sea points.
        expected_mask = np.array(
            [[False, True, True], [True, False, True], [True, True, False]]
        )

        calibrated_forecast_predictor, calibrated_forecast_var = self.plugin.process(
            self.current_temperature_forecast_cube,
            self.coeffs_from_mean,
            landsea_mask=mask_cube,
        )

        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data.data, self.expected_loc_param_mean
        )
        self.assertArrayEqual(calibrated_forecast_predictor.data.mask, expected_mask)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data.data, self.expected_scale_param_mean
        )
        self.assertArrayEqual(calibrated_forecast_var.data.mask, expected_mask)


if __name__ == "__main__":
    unittest.main()
