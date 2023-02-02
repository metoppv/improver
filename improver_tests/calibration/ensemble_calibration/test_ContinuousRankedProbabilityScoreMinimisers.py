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
"""
Unit tests for the
`ensemble_calibration.ContinuousRankedProbabilityScoreMinimisers`
class.
"""
import unittest

import iris
import numpy as np
from iris.cube import CubeList
from iris.tests import IrisTest

from improver.calibration.ensemble_calibration import (
    ContinuousRankedProbabilityScoreMinimisers as Plugin,
)
from improver.calibration.utilities import convert_cube_data_to_2d

from .helper_functions import EnsembleCalibrationAssertions, SetupCubes


class SetupInputs(IrisTest):

    """Set up inputs for testing."""

    def setUp(self):
        """Set up inputs for testing."""
        super().setUp()
        self.tolerance = 1e-4
        self.mean_plugin = Plugin("mean", tolerance=self.tolerance)
        self.realizations_plugin = Plugin("realizations", tolerance=self.tolerance)

        self.sqrt_pi = np.sqrt(np.pi).astype(np.float64)

        self.initial_guess_for_mean = np.array([0, 1, 0, 1], dtype=np.float64)
        self.initial_guess_for_realization = np.array(
            [0, np.sqrt(1 / 3.0), np.sqrt(1 / 3.0), np.sqrt(1 / 3.0), 0, 1],
            dtype=np.float64,
        )
        self.initial_guess_mean_additional_predictor = np.array(
            [0, 0.5, 0.5, 0, 1], dtype=np.float64
        )


class SetupNormalInputs(SetupInputs, SetupCubes):

    """Create a class for setting up cubes for testing."""

    def setUp(self):
        """Set up expected inputs."""
        super().setUp()
        # Set up cubes and associated data arrays for temperature.
        self.forecast_predictor_mean = CubeList(
            [
                self.historic_temperature_forecast_cube.collapsed(
                    "realization", iris.analysis.MEAN
                )
            ]
        )
        self.forecast_predictor_realizations = CubeList(
            [(self.historic_temperature_forecast_cube.copy())]
        )
        self.forecast_predictor_spot = CubeList(
            [
                self.historic_forecast_spot_cube.collapsed(
                    "realization", iris.analysis.MEAN
                )
            ]
        )

        self.fp_additional_predictor_spot = CubeList(
            [self.forecast_predictor_spot[0].copy()]
        )
        self.fp_additional_predictor_spot.extend([self.spot_altitude_cube])

        self.forecast_variance = self.historic_temperature_forecast_cube.collapsed(
            "realization", iris.analysis.VARIANCE
        )
        self.forecast_variance_spot = self.forecast_predictor_spot[0].copy()
        self.forecast_variance_spot.data = self.forecast_variance_spot.data / 10.0

        self.truth = self.historic_temperature_forecast_cube.collapsed(
            "realization", iris.analysis.MAX
        )
        self.forecast_predictor_data = (
            self.forecast_predictor_mean[0].data.flatten().astype(np.float64)
        )
        self.forecast_predictor_data_realizations = convert_cube_data_to_2d(
            self.historic_temperature_forecast_cube.copy()
        ).astype(np.float64)
        self.forecast_variance_data = self.forecast_variance.data.flatten().astype(
            np.float64
        )
        self.truth_data = self.truth.data.flatten().astype(np.float64)

        spatial_product = np.prod(self.truth.shape[-2:])
        self.initial_guess_spot_mean = np.broadcast_to(
            self.initial_guess_for_mean,
            (spatial_product, len(self.initial_guess_for_mean),),
        )
        self.initial_guess_spot_realizations = np.broadcast_to(
            self.initial_guess_for_realization,
            (spatial_product, len(self.initial_guess_for_realization),),
        )
        self.ig_spot_mean_additional_predictor = np.broadcast_to(
            self.initial_guess_mean_additional_predictor,
            (spatial_product, len(self.initial_guess_mean_additional_predictor),),
        )


class Test_calculate_normal_crps(SetupNormalInputs):

    """
    Test minimising the CRPS for a normal distribution.
    Either the ensemble mean or the individual ensemble realizations are
    used as the predictors.
    """

    def setUp(self):
        """Set up plugin."""
        super().setUp()
        self.precision = 4

    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value with the
        mean as the predictor. The result indicates the minimum value for the
        CRPS that was achieved by the minimisation.
        """
        result = self.mean_plugin.calculate_normal_crps(
            self.initial_guess_for_mean,
            self.forecast_predictor_data,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.3006, places=self.precision)

    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy float value with the ensemble
        realizations as the predictor. The result indicates the minimum value
        for the CRPS that was achieved by the minimisation.
        """
        result = self.realizations_plugin.calculate_normal_crps(
            self.initial_guess_for_realization,
            self.forecast_predictor_data_realizations,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.3006, places=self.precision)

    def test_basic_mean_predictor_bad_value(self):
        """
        Test that the plugin returns a numpy float64 value
        and that the value matches the BAD_VALUE, when the appropriate
        condition is found. The ensemble mean is the predictor.
        The initial guess is specifically set to float32 precision for the
        purpose for generating the BAD_VALUE for the unit test.
        """
        initial_guess = np.array([1e65, 1e65, 1e65, 1e65], dtype=np.float32)

        result = self.mean_plugin.calculate_normal_crps(
            initial_guess,
            self.forecast_predictor_data,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, self.mean_plugin.BAD_VALUE, self.precision)


class Test_process_normal_distribution(
    SetupNormalInputs, EnsembleCalibrationAssertions
):

    """
    Test minimising the CRPS for a normal distribution.
    Either the ensemble mean or the individual ensemble realizations are used
    as the predictors.
    """

    def setUp(self):
        """Set up expected output.
        The coefficients are in the order [alpha, beta, gamma, delta].
        """
        super().setUp()
        self.expected_mean_coefficients = np.array(
            [-0.0003, 1.0013, 0.0012, 0.5945], dtype=np.float32
        )
        self.expected_realizations_coefficients = np.array(
            [0.0254, 0.4349, 0.39, 0.8122, -0.0016, 0.2724], dtype=np.float32
        )
        self.expected_mean_coefficients_point_by_point = np.array(
            [
                [
                    [0.0015, 0.0037, -0.002],
                    [-0.0009, 0.0008, 0.0015],
                    [-0.0046, 0.0053, -0.0038],
                ],
                [
                    [1.0039, 1.0035, 1.0009],
                    [1.0013, 1.0011, 1.001],
                    [1.002, 1.0015, 1.0008],
                ],
                [
                    [0.0017, -0.0009, -0.0002],
                    [0.0054, 0.0003, -0.0002],
                    [-0.0001, -0.0018, 0.0002],
                ],
                [
                    [-0.0, 0.0007, -0.0009],
                    [0.0003, -0.0001, -0.001],
                    [-0.0013, 0.0, 0.0006],
                ],
            ],
            dtype=np.float32,
        )

        self.expected_mean_coefficients_point_by_point_sites = np.array(
            [
                [0.0017, 0.0017, 0.0017, 0.0017],
                [1.0036, 1.0036, 1.0036, 1.0036],
                [0.0017, 0.0017, 0.0017, 0.0017],
                [-0.0, -0.0, -0.0, 0.0],
            ],
            dtype=np.float32,
        )

        self.expected_realizations_coefficients_point_by_point = np.array(
            [
                [
                    [0.0001, 0.0001, 0.0001],
                    [0.0001, 0.0001, 0.0],
                    [0.0001, 0.0001, 0.0001],
                ],
                [
                    [0.579, 0.5793, 0.5782],
                    [0.5782, 0.5778, 0.5781],
                    [0.5786, 0.5782, 0.5783],
                ],
                [
                    [0.5795, 0.5786, 0.5782],
                    [0.5783, 0.578, 0.5767],
                    [0.5791, 0.578, 0.5763],
                ],
                [
                    [0.5773, 0.5769, 0.5763],
                    [0.5769, 0.5771, 0.5782],
                    [0.5764, 0.5773, 0.5783],
                ],
                [
                    [0.0001, 0.0001, 0.0001],
                    [0.0001, 0.0001, 0.0001],
                    [0.0001, 0.0001, 0.0],
                ],
                [
                    [1.0194, 1.0143, 1.0199],
                    [1.0199, 1.02, 1.013],
                    [1.0144, 0.9885, 1.0246],
                ],
            ],
            dtype=np.float32,
        )

        self.expected_mean_coefficients_additional_predictor = np.array(
            [-0.0066, 1.0036, 0.0001, 0.0066, 0], dtype=np.float32
        )
        self.expected_point_by_point_sites_additional_predictor = np.array(
            [
                [-0.0064, -0.0119, -0.0011, 0.002],
                [0.9908, 0.9219, 0.939, 0.9003],
                [0.3551, 1.128, 0.5948, 0.7123],
                [0.0087, 0.0138, -0.0021, 0.0004],
                [0.0, 0.0039, 0.0036, 0.006],
            ],
            dtype=np.float32,
        )

    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble mean is the predictor.
        """
        distribution = "norm"
        result = self.mean_plugin(
            self.initial_guess_for_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEMOSCoefficientsAlmostEqual(result, self.expected_mean_coefficients)

    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble realizations are the predictor.
        """
        distribution = "norm"
        result = self.realizations_plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients
        )

    def test_mean_predictor_keyerror(self):
        """
        Test that the minimisation has resulted in a KeyError, if the
        distribution that has been requested was not within the dictionary
        containing the minimisation functions.
        """
        distribution = "foo"

        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            self.mean_plugin.process(
                self.initial_guess_for_mean,
                self.forecast_predictor_mean,
                self.truth,
                self.forecast_variance,
                distribution,
            )

    def test_mean_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble mean is the predictor
        assuming a normal distribution and the value specified for the
        max_iterations is overridden. The coefficients are calculated by
        minimising the CRPS.
        """
        predictor = "mean"
        max_iterations = 400
        distribution = "norm"

        plugin = Plugin(
            predictor, tolerance=self.tolerance, max_iterations=max_iterations
        )
        result = plugin(
            self.initial_guess_for_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(result, self.expected_mean_coefficients)

    def test_realizations_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble realizations are the
        predictor assuming a truncated normal distribution and the value
        specified for the MAX_ITERATIONS is overridden. The coefficients are
        calculated by minimising the CRPS.
        """
        predictor = "realizations"
        max_iterations = 1000
        distribution = "norm"

        plugin = Plugin(
            predictor, tolerance=self.tolerance, max_iterations=max_iterations
        )
        result = plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients
        )

    def test_mean_predictor_point_by_point(self):
        """
        Test that the expected coefficients are generated when the ensemble
        mean is the predictor for a normal distribution and coefficients are
        calculated independently at each grid point. The coefficients are
        calculated by minimising the CRPS.
        """
        predictor = "mean"
        distribution = "norm"

        plugin = Plugin(predictor, tolerance=self.tolerance, point_by_point=True)
        result = plugin.process(
            self.initial_guess_spot_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_mean_coefficients_point_by_point
        )

    def test_mean_predictor_point_by_point_sites(self):
        """
        Test that the expected coefficients are generated when the ensemble
        mean is the predictor for a normal distribution and coefficients are
        calculated independently at each site location. The coefficients are
        calculated by minimising the CRPS.
        """
        predictor = "mean"
        distribution = "norm"

        plugin = Plugin(predictor, tolerance=self.tolerance, point_by_point=True)
        result = plugin.process(
            self.initial_guess_spot_mean,
            self.forecast_predictor_spot,
            self.truth_spot_cube,
            self.forecast_variance_spot,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_mean_coefficients_point_by_point_sites
        )

    def test_realizations_predictor_point_by_point(self):
        """
        Test that the expected coefficients are generated when the ensemble
        realizations are the predictor for a normal distribution and
        coefficients are calculated independently at each grid point. The
        coefficients are calculated by minimising the CRPS.
        """
        predictor = "realizations"
        distribution = "norm"

        # Use a larger value for the tolerance to terminate sooner to avoid
        # minimising in computational noise.
        plugin = Plugin(predictor, tolerance=0.01, point_by_point=True)
        result = plugin.process(
            self.initial_guess_spot_realizations,
            self.forecast_predictor_realizations,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertArrayAlmostEqual(
            result, self.expected_realizations_coefficients_point_by_point, decimal=2
        )

    def test_point_by_point_with_nans(self):
        """
        Test that the expected coefficients are generated when the ensemble
        mean is the predictor for a normal distribution and coefficients are
        calculated independently at each grid point with one grid point
        having NaN values for the truth.
        """
        predictor = "mean"
        distribution = "norm"

        self.truth.data[:, 0, 0] = np.nan
        plugin = Plugin(predictor, tolerance=self.tolerance, point_by_point=True)
        result = plugin.process(
            self.initial_guess_spot_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.expected_mean_coefficients_point_by_point[
            :, 0, 0
        ] = self.initial_guess_for_mean
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_mean_coefficients_point_by_point
        )

    def test_mean_predictor_sites_additional_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble mean and altitude are the predictors.
        """
        distribution = "norm"
        result = self.mean_plugin.process(
            self.initial_guess_mean_additional_predictor,
            self.fp_additional_predictor_spot,
            self.truth_spot_cube,
            self.forecast_variance_spot,
            distribution,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_mean_coefficients_additional_predictor
        )

    def test_mean_predictor_point_by_point_sites_additional_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. Coefficients are calculated independently at each site.
        The ensemble mean and altitude are the predictors.
        """
        predictor = "mean"
        distribution = "norm"

        plugin = Plugin(predictor, tolerance=self.tolerance, point_by_point=True)
        result = plugin.process(
            self.ig_spot_mean_additional_predictor,
            self.fp_additional_predictor_spot,
            self.truth_spot_cube,
            self.forecast_variance_spot,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_point_by_point_sites_additional_predictor
        )


class SetupTruncatedNormalInputs(SetupInputs, SetupCubes):

    """Create a class for setting up cubes for testing."""

    def setUp(self):
        """Set up expected inputs."""
        super().setUp()
        # Set up cubes and associated data arrays for wind speed.
        self.forecast_predictor_mean = CubeList(
            [
                self.historic_wind_speed_forecast_cube.collapsed(
                    "realization", iris.analysis.MEAN
                )
            ]
        )
        self.forecast_predictor_realizations = CubeList(
            [(self.historic_wind_speed_forecast_cube.copy())]
        )
        cube = self.historic_wind_speed_forecast_cube.collapsed(
            "realization", iris.analysis.MEAN
        )

        altitude_cube = cube[0].copy(data=np.reshape(np.arange(0, 45, 5), (3, 3)))
        altitude_cube.rename("altitude")
        altitude_cube.units = "m"
        for coord in [
            "forecast_period",
            "forecast_reference_time",
            "realization",
            "time",
        ]:
            altitude_cube.remove_coord(coord)

        self.forecast_predictor_mean_additional_predictor = CubeList(
            [
                self.historic_wind_speed_forecast_cube.collapsed(
                    "realization", iris.analysis.MEAN
                ),
                altitude_cube,
            ]
        )

        self.forecast_variance = self.historic_wind_speed_forecast_cube.collapsed(
            "realization", iris.analysis.VARIANCE
        )
        self.truth = self.historic_wind_speed_forecast_cube.collapsed(
            "realization", iris.analysis.MAX
        )
        self.forecast_predictor_data = (
            self.forecast_predictor_mean[0].data.flatten().astype(np.float64)
        )
        self.forecast_predictor_data_realizations = convert_cube_data_to_2d(
            self.historic_wind_speed_forecast_cube.copy()
        ).astype(np.float64)
        self.forecast_variance_data = self.forecast_variance.data.flatten().astype(
            np.float64
        )
        self.truth_data = self.truth.data.flatten().astype(np.float64)


class Test_calculate_truncated_normal_crps(SetupTruncatedNormalInputs):

    """
    Test minimising the crps for a truncated normal distribution.
    Either the ensemble mean or the individual ensemble realizations are used
    as the predictors.
    """

    def setUp(self):
        """Set up plugin."""
        super().setUp()
        self.precision = 4

    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value. The ensemble mean
        is the predictor. The result indicates the minimum value
        for the CRPS that was achieved by the minimisation.
        """
        result = self.mean_plugin.calculate_truncated_normal_crps(
            self.initial_guess_for_mean,
            self.forecast_predictor_data,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.2150, self.precision)

    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy float value. The ensemble
        realizations are the predictor. The result indicates the minimum value
        for the CRPS that was achieved by the minimisation.
        """
        result = self.realizations_plugin.calculate_truncated_normal_crps(
            self.initial_guess_for_realization,
            self.forecast_predictor_data_realizations,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.2150, self.precision)

    def test_basic_mean_predictor_bad_value(self):
        """
        Test that the plugin returns a numpy float64 value
        and that the value matches the BAD_VALUE, when the appropriate
        condition is found. The ensemble mean is the predictor. The initial
        guess is specifically set to float32 precision for the purpose for
        generating the BAD_VALUE for the unit test.
        """
        initial_guess = np.array([1e65, 1e65, 1e65, 1e65], dtype=np.float32)

        result = self.mean_plugin.calculate_truncated_normal_crps(
            initial_guess,
            self.forecast_predictor_data,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, self.mean_plugin.BAD_VALUE, self.precision)


class Test_process_truncated_normal_distribution(
    SetupTruncatedNormalInputs, EnsembleCalibrationAssertions
):

    """
    Test minimising the CRPS for a truncated normal distribution.
    Either the ensemble mean or the individual ensemble realizations are used
    as the predictors.
    """

    def setUp(self):
        """Set up expected output."""
        super().setUp()
        self.expected_mean_coefficients = np.array(
            [0.3958, 0.9854, -0.0, 0.621], dtype=np.float32
        )
        self.expected_realizations_coefficients = np.array(
            [0.1898, -0.1558, 0.4452, 0.8877, -0.1331, -0.0002], np.float32
        )
        self.expected_additional_predictors = np.array(
            [0.0014, 0.9084, 0.0279, -0.0021, 0.8591], dtype=np.float32
        )

    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy array. The ensemble mean
        is the predictor.
        """
        distribution = "truncnorm"

        result = self.mean_plugin.process(
            self.initial_guess_for_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEMOSCoefficientsAlmostEqual(result, self.expected_mean_coefficients)

    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble realizations are the predictor.
        """
        distribution = "truncnorm"

        result = self.realizations_plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients
        )

    def test_mean_predictor_keyerror(self):
        """
        Test that an exception is raised when the distribution requested is
        not an available option when the predictor is the
        ensemble mean.
        """
        distribution = "foo"

        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            self.mean_plugin.process(
                self.initial_guess_for_mean,
                self.forecast_predictor_mean,
                self.truth,
                self.forecast_variance,
                distribution,
            )

    def test_mean_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble mean is the predictor
        assuming a truncated normal distribution and the value specified
        for the max_iterations is overridden. The coefficients are
        calculated by minimising the CRPS.
        """
        predictor = "mean"
        max_iterations = 400
        distribution = "truncnorm"

        plugin = Plugin(
            predictor, tolerance=self.tolerance, max_iterations=max_iterations
        )
        result = plugin.process(
            self.initial_guess_for_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(result, self.expected_mean_coefficients)

    def test_realizations_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble realizations are the
        predictor assuming a truncated normal distribution and the value
        specified for the max_iterations is overridden. The coefficients are
        calculated by minimising the CRPS.
        """
        predictor = "realizations"
        max_iterations = 1000
        distribution = "truncnorm"

        plugin = Plugin(
            predictor, tolerance=self.tolerance, max_iterations=max_iterations
        )
        result = plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients
        )

    def test_mean_predictor_additional_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble mean and altitude are the predictors.
        """
        distribution = "truncnorm"
        result = self.mean_plugin.process(
            self.initial_guess_mean_additional_predictor,
            self.forecast_predictor_mean_additional_predictor,
            self.truth,
            self.forecast_variance,
            distribution,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_additional_predictors
        )


if __name__ == "__main__":
    unittest.main()
