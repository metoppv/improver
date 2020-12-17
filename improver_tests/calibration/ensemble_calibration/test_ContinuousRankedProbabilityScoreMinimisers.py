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
Unit tests for the
`ensemble_calibration.ContinuousRankedProbabilityScoreMinimisers`
class.

"""
import datetime
import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.calibration.ensemble_calibration import (
    ContinuousRankedProbabilityScoreMinimisers as Plugin,
)
from improver.calibration.utilities import convert_cube_data_to_2d
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import construct_scalar_time_coords
from improver.utilities.warnings_handler import ManageWarnings

from .helper_functions import EnsembleCalibrationAssertions, SetupCubes


class SetupInputs(IrisTest):

    """Set up inputs for testing."""

    def setUp(self):
        """Set up inputs for testing."""
        super().setUp()
        self.sqrt_pi = np.sqrt(np.pi).astype(np.float64)

        self.initial_guess_for_mean = np.array([0, 1, 0, 1], dtype=np.float64)
        self.initial_guess_for_realization = np.array(
            [0, np.sqrt(1 / 3.0), np.sqrt(1 / 3.0), np.sqrt(1 / 3.0), 0, 1],
            dtype=np.float64,
        )


class SetupNormalInputs(SetupInputs, SetupCubes):

    """Create a class for setting up cubes for testing."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."],
        warning_types=[UserWarning],
    )
    def setUp(self):
        """Set up expected inputs."""
        super().setUp()
        # Set up cubes and associated data arrays for temperature.
        self.forecast_predictor_mean = self.historic_temperature_forecast_cube.collapsed(
            "realization", iris.analysis.MEAN
        )
        self.forecast_predictor_realizations = (
            self.historic_temperature_forecast_cube.copy()
        )
        self.forecast_variance = self.historic_temperature_forecast_cube.collapsed(
            "realization", iris.analysis.VARIANCE
        )
        self.truth = self.historic_temperature_forecast_cube.collapsed(
            "realization", iris.analysis.MAX
        )
        self.forecast_predictor_data = self.forecast_predictor_mean.data.flatten().astype(
            np.float64
        )
        self.forecast_predictor_data_realizations = convert_cube_data_to_2d(
            self.historic_temperature_forecast_cube.copy()
        ).astype(np.float64)
        self.forecast_variance_data = self.forecast_variance.data.flatten().astype(
            np.float64
        )
        self.truth_data = self.truth.data.flatten().astype(np.float64)


class Test_calculate_normal_crps(SetupNormalInputs):

    """
    Test minimising the CRPS for a normal distribution.
    Either the ensemble mean or the individual ensemble realizations are
    used as the predictors.
    """

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value with the
        mean as the predictor. The result indicates the minimum value for the
        CRPS that was achieved by the minimisation.
        """
        predictor = "mean"

        plugin = Plugin()
        result = plugin.calculate_normal_crps(
            self.initial_guess_for_mean,
            self.forecast_predictor_data,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
            predictor,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.2609128)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy float value with the ensemble
        realizations as the predictor. The result indicates the minimum value
        for the CRPS that was achieved by the minimisation.
        """
        predictor = "realizations"

        plugin = Plugin()
        result = plugin.calculate_normal_crps(
            self.initial_guess_for_realization,
            self.forecast_predictor_data_realizations,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
            predictor,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.2609116)

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "invalid value encountered in",
        ],
        warning_types=[UserWarning, RuntimeWarning],
    )
    def test_basic_mean_predictor_bad_value(self):
        """
        Test that the plugin returns a numpy float64 value
        and that the value matches the BAD_VALUE, when the appropriate
        condition is found. The ensemble mean is the predictor.
        The initial guess is specifically set to float32 precision for the
        purpose for generating the BAD_VALUE for the unit test.
        """
        initial_guess = np.array([1e65, 1e65, 1e65, 1e65], dtype=np.float32)

        predictor = "mean"

        plugin = Plugin()
        result = plugin.calculate_normal_crps(
            initial_guess,
            self.forecast_predictor_data,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
            predictor,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, plugin.BAD_VALUE)


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
        self.tolerance = 1e-4
        self.plugin = Plugin(tolerance=self.tolerance)
        self.expected_mean_coefficients = [-0.0005, 1.0009, 0.0016, 0.8069]
        self.expected_realizations_coefficients = [
            0.03,
            0.3388,
            0.3149,
            0.8869,
            -0.0016,
            0.2983,
        ]
        self.expected_mean_coefficients_each_point = np.array(
            [
                [
                    [0.0015, 0.0037, 0.002],
                    [-0.0009, -0.0006, 0.0037],
                    [-0.0046, -0.003, 0.0037],
                ],
                [
                    [1.0039, 1.0035, 1.0005],
                    [1.0013, 1.0011, 1.0009],
                    [1.002, 1.0005, 1.0001],
                ],
                [
                    [0.0017, -0.0009, 0.0013],
                    [0.0054, 0.0003, -0.0001],
                    [-0.0001, -0.0001, -0.0],
                ],
                [
                    [0.0, 0.0007, 0.0001],
                    [0.0003, -0.0001, -0.0012],
                    [-0.0013, -0.0013, -0.0001],
                ],
            ],
            dtype=np.float32,
        )

        self.expected_mean_coefficients_each_point_sites = np.array(
            [[-0.0008, -0.0046, -0.0015, -0.0011],
            [ 1.6255,  1.7728,  1.7154,  1.9101],
            [ 0.0012,  0.    ,  0.0005, -0.0009],
            [-0.    , -0.    , -0.    ,  0.    ]], dtype=np.float32)

        self.expected_realizations_coefficients_each_point = np.array(
            [
                [
                    [0.0011, -0.0016, 0.0042],
                    [0.001, 0.0022, -0.0036],
                    [0.0005, 0.0047, 0.0041],
                ],
                [
                    [0.5746, 0.652, 0.5783],
                    [0.6051, 0.5706, 0.5908],
                    [0.6263, 0.5586, 0.5825],
                ],
                [
                    [0.4436, 0.5455, 0.5659],
                    [0.5856, 0.5724, 0.5598],
                    [0.618, 0.6166, 0.5731],
                ],
                [
                    [0.6901, 0.5304, 0.5881],
                    [0.5408, 0.5897, 0.5818],
                    [0.4777, 0.5552, 0.5765],
                ],
                [
                    [-0.0003, 0.0013, 0.0001],
                    [0.0024, -0.0015, 0.001],
                    [-0.0003, -0.0009, 0.0],
                ],
                [[-0.001, -0.0, -0.0], [0.0, 0.0, 0.0], [-0.0014, 0.0001, 0.0002]],
            ],
            dtype=np.float32,
        )

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "divide by zero encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning],
    )
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble mean is the predictor.
        """
        predictor = "mean"
        distribution = "norm"
        result = self.plugin.process(
            self.initial_guess_for_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEMOSCoefficientsAlmostEqual(result, self.expected_mean_coefficients)

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "divide by zero encountered in",
            "invalid value encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning, RuntimeWarning],
    )
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble realizations are the predictor.
        """
        predictor = "realizations"
        distribution = "norm"
        result = self.plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients
        )

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_mean_predictor_keyerror(self):
        """
        Test that the minimisation has resulted in a KeyError, if the
        distribution that has been requested was not within the dictionary
        containing the minimisation functions.
        """
        predictor = "mean"
        distribution = "foo"

        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            self.plugin.process(
                self.initial_guess_for_mean,
                self.forecast_predictor_mean,
                self.truth,
                self.forecast_variance,
                predictor,
                distribution,
            )

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "divide by zero encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning],
    )
    def test_mean_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble mean is the predictor
        assuming a normal distribution and the value specified for the
        max_iterations is overridden. The coefficients are calculated by
        minimising the CRPS and using a set default value for the
        initial guess.
        """
        predictor = "mean"
        max_iterations = 400
        distribution = "norm"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=max_iterations)
        result = plugin.process(
            self.initial_guess_for_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(result, self.expected_mean_coefficients)

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "divide by zero encountered in",
            "invalid value encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning, RuntimeWarning],
    )
    def test_realizations_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble realizations are the
        predictor assuming a truncated normal distribution and the value
        specified for the MAX_ITERATIONS is overridden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        predictor = "realizations"
        max_iterations = 1000
        distribution = "norm"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=max_iterations)
        result = plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients
        )

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "divide by zero encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning],
    )
    def test_mean_predictor_each_point(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble mean is the predictor
        assuming a normal distribution and the value specified for the
        max_iterations is overridden. The coefficients are calculated by
        minimising the CRPS and using a set default value for the
        initial guess.
        """
        predictor = "mean"
        distribution = "norm"

        initial_guess = np.broadcast_to(
            self.initial_guess_for_mean,
            (
                len(self.truth.coord(axis="y").points)
                * len(self.truth.coord(axis="x").points),
                len(self.initial_guess_for_mean),
            ),
        )

        plugin = Plugin(tolerance=self.tolerance, each_point=True)
        result = plugin.process(
            initial_guess,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_mean_coefficients_each_point
        )

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "divide by zero encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning],
    )
    def test_mean_predictor_each_point_sites(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble mean is the predictor
        assuming a normal distribution and the value specified for the
        max_iterations is overridden. The coefficients are calculated by
        minimising the CRPS and using a set default value for the
        initial guess.
        """
        data = np.array([1.6, 1.3, 1.4, 1.1])
        altitude = np.array([10, 20, 30, 40])
        latitude = np.linspace(58.0, 59.5, 4)
        longitude = np.linspace(-0.25, 0.5, 4)
        wmo_id = ["03001", "03002", "03003", "03004"]
        forecast_spot_cubes = iris.cube.CubeList()
        for day in range(1, 3):
            time_coords = construct_scalar_time_coords(datetime.datetime(2020, 12, day, 4, 0), None, datetime.datetime(2020, 12, day, 0, 0))
            time_coords = [t[0] for t in time_coords]
            forecast_spot_cubes.append(build_spotdata_cube(
                data,
                "air_temperature",
                "degC",
                altitude,
                latitude,
                longitude,
                wmo_id,
                scalar_coords=time_coords
            ))
        forecast_spot_cube = forecast_spot_cubes.merge_cube()

        truth_spot_cube = forecast_spot_cube.copy()
        truth_spot_cube.data = truth_spot_cube.data + 1.
        forecast_var_spot_cube = forecast_spot_cube.copy()
        forecast_var_spot_cube.data = forecast_var_spot_cube.data/10.

        predictor = "mean"
        distribution = "norm"

        initial_guess = np.broadcast_to(
            self.initial_guess_for_mean,
            (
                len(self.truth.coord(axis="y").points)
                * len(self.truth.coord(axis="x").points),
                len(self.initial_guess_for_mean),
            ),
        )

        plugin = Plugin(tolerance=self.tolerance, each_point=True)
        result = plugin.process(
            initial_guess,
            forecast_spot_cube,
            truth_spot_cube,
            forecast_var_spot_cube,
            predictor,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_mean_coefficients_each_point_sites
        )

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "divide by zero encountered in",
            "invalid value encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning, RuntimeWarning],
    )
    def test_realizations_predictor_each_point(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble realizations are the
        predictor assuming a truncated normal distribution and the value
        specified for the MAX_ITERATIONS is overridden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        predictor = "realizations"
        distribution = "norm"

        initial_guess = np.broadcast_to(
            self.initial_guess_for_realization,
            (
                len(self.truth.coord(axis="y").points)
                * len(self.truth.coord(axis="x").points),
                len(self.initial_guess_for_realization),
            ),
        )

        plugin = Plugin(tolerance=self.tolerance, each_point=True)
        result = plugin.process(
            initial_guess,
            self.forecast_predictor_realizations,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients_each_point
        )

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "divide by zero encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning],
    )
    @ManageWarnings(
        record=True, ignored_messages=["Collapsing a non-contiguous coordinate."]
    )
    def test_catch_warnings(self, warning_list=None):
        """
        Test that a warning is generated if the minimisation
        does not result in a convergence. The ensemble mean is the predictor.
        """
        predictor = "mean"
        distribution = "norm"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=10)
        plugin.process(
            self.initial_guess_for_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        warning_msg = "Minimisation did not result in convergence after"
        self.assertTrue(any(item.category == UserWarning for item in warning_list))
        self.assertTrue(any(warning_msg in str(item) for item in warning_list))

    @ManageWarnings(
        record=True, ignored_messages=["Collapsing a non-contiguous coordinate."]
    )
    def test_catch_warnings_percentage_change(self, warning_list=None):
        """
        Test that two warnings are generated if the minimisation
        does not result in a convergence. The first warning reports a that
        the minimisation did not result in convergence, whilst the second
        warning reports that the percentage change in the final iteration was
        greater than the tolerated value. The ensemble mean is the predictor.
        """
        initial_guess = np.array([5000, 1, 0, 1], dtype=np.float64)
        predictor = "mean"
        distribution = "norm"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=5)
        plugin.process(
            initial_guess,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        warning_msg_min = "Minimisation did not result in convergence after"
        warning_msg_iter = "The final iteration resulted in a percentage "
        self.assertTrue(any(item.category == UserWarning for item in warning_list))
        self.assertTrue(any(warning_msg_min in str(item) for item in warning_list))
        self.assertTrue(any(warning_msg_iter in str(item) for item in warning_list))


class SetupTruncatedNormalInputs(SetupInputs, SetupCubes):

    """Create a class for setting up cubes for testing."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."],
        warning_types=[UserWarning],
    )
    def setUp(self):
        """Set up expected inputs."""
        super().setUp()
        # Set up cubes and associated data arrays for wind speed.
        self.forecast_predictor_mean = self.historic_wind_speed_forecast_cube.collapsed(
            "realization", iris.analysis.MEAN
        )
        self.forecast_predictor_realizations = (
            self.historic_wind_speed_forecast_cube.copy()
        )
        self.forecast_variance = self.historic_wind_speed_forecast_cube.collapsed(
            "realization", iris.analysis.VARIANCE
        )
        self.truth = self.historic_wind_speed_forecast_cube.collapsed(
            "realization", iris.analysis.MAX
        )
        self.forecast_predictor_data = self.forecast_predictor_mean.data.flatten().astype(
            np.float64
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

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value. The ensemble mean
        is the predictor. The result indicates the minimum value
        for the CRPS that was achieved by the minimisation.
        """
        predictor = "mean"

        plugin = Plugin()
        result = plugin.calculate_truncated_normal_crps(
            self.initial_guess_for_mean,
            self.forecast_predictor_data,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
            predictor,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.1753538)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy float value. The ensemble
        realizations are the predictor. The result indicates the minimum value
        for the CRPS that was achieved by the minimisation.
        """
        predictor = "realizations"

        plugin = Plugin()
        result = plugin.calculate_truncated_normal_crps(
            self.initial_guess_for_realization,
            self.forecast_predictor_data_realizations,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
            predictor,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.1753538)

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "invalid value encountered in",
        ],
        warning_types=[UserWarning, RuntimeWarning],
    )
    def test_basic_mean_predictor_bad_value(self):
        """
        Test that the plugin returns a numpy float64 value
        and that the value matches the BAD_VALUE, when the appropriate
        condition is found. The ensemble mean is the predictor. The initial
        guess is specifically set to float32 precision for the purpose for
        generating the BAD_VALUE for the unit test.
        """
        initial_guess = np.array([1e65, 1e65, 1e65, 1e65], dtype=np.float32)

        predictor = "mean"

        plugin = Plugin()
        result = plugin.calculate_truncated_normal_crps(
            initial_guess,
            self.forecast_predictor_data,
            self.truth_data,
            self.forecast_variance_data,
            self.sqrt_pi,
            predictor,
        )

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, plugin.BAD_VALUE)


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
        self.tolerance = 1e-4
        self.plugin = Plugin(tolerance=self.tolerance)
        self.expected_mean_coefficients = [0.4511, 0.9503, 0.0113, 0.7424]
        self.expected_realizations_coefficients = [
            0.2976,
            0.0086,
            0.5721,
            0.8099,
            -0.0122,
            0.2736,
        ]

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "The final iteration resulted in",
            "invalid value encountered in",
            "divide by zero encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning, RuntimeWarning],
    )
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy array. The ensemble mean
        is the predictor.
        """
        predictor = "mean"
        distribution = "truncnorm"

        result = self.plugin.process(
            self.initial_guess_for_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEMOSCoefficientsAlmostEqual(result, self.expected_mean_coefficients)

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "invalid value encountered in",
            "divide by zero encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning, RuntimeWarning],
    )
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble realizations are the predictor.
        """
        predictor = "realizations"
        distribution = "truncnorm"

        result = self.plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients
        )

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_mean_predictor_keyerror(self):
        """
        Test that an exception is raised when the distribution requested is
        not an available option when the predictor is the
        ensemble mean.
        """
        predictor = "mean"
        distribution = "foo"

        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            self.plugin.process(
                self.initial_guess_for_mean,
                self.forecast_predictor_mean,
                self.truth,
                self.forecast_variance,
                predictor,
                distribution,
            )

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "The final iteration resulted in",
            "invalid value encountered in",
            "divide by zero encountered in",
        ],
        warning_types=[
            UserWarning,
            UserWarning,
            UserWarning,
            RuntimeWarning,
            RuntimeWarning,
        ],
    )
    def test_mean_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble mean is the predictor
        assuming a truncated normal distribution and the value specified
        for the max_iterations is overridden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        predictor = "mean"
        max_iterations = 400
        distribution = "truncnorm"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=max_iterations)
        result = plugin.process(
            self.initial_guess_for_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(result, self.expected_mean_coefficients)

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate.",
            "Minimisation did not result in convergence",
            "invalid value encountered in",
            "divide by zero encountered in",
        ],
        warning_types=[UserWarning, UserWarning, RuntimeWarning, RuntimeWarning],
    )
    def test_realizations_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble realizations are the
        predictor assuming a truncated normal distribution and the value
        specified for the max_iterations is overridden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        predictor = "realizations"
        max_iterations = 1000
        distribution = "truncnorm"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=max_iterations)
        result = plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients
        )

    @ManageWarnings(
        record=True, ignored_messages=["Collapsing a non-contiguous coordinate."]
    )
    def test_catch_warnings(self, warning_list=None):
        """
        Test that a warning is generated if the minimisation
        does not result in a convergence. The ensemble mean is the predictor.
        """
        predictor = "mean"
        distribution = "truncnorm"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=10)
        plugin.process(
            self.initial_guess_for_mean,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        warning_msg = "Minimisation did not result in convergence after"
        self.assertTrue(any(item.category == UserWarning for item in warning_list))
        self.assertTrue(any(warning_msg in str(item) for item in warning_list))

    @ManageWarnings(
        record=True, ignored_messages=["Collapsing a non-contiguous coordinate."]
    )
    def test_catch_warnings_percentage_change(self, warning_list=None):
        """
        Test that two warnings are generated if the minimisation
        does not result in a convergence. The first warning reports a that
        the minimisation did not result in convergence, whilst the second
        warning reports that the percentage change in the final iteration was
        greater than the tolerated value.
        The ensemble mean is the predictor.
        """
        initial_guess = np.array([0, 1, 5000, 1], dtype=np.float64)

        predictor = "mean"
        distribution = "truncnorm"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=5)

        plugin.process(
            initial_guess,
            self.forecast_predictor_mean,
            self.truth,
            self.forecast_variance,
            predictor,
            distribution,
        )
        warning_msg_min = "Minimisation did not result in convergence after"
        warning_msg_iter = "The final iteration resulted in a percentage "
        self.assertTrue(any(item.category == UserWarning for item in warning_list))
        self.assertTrue(any(warning_msg_min in str(item) for item in warning_list))
        self.assertTrue(any(warning_msg_iter in str(item) for item in warning_list))


if __name__ == "__main__":
    unittest.main()
