# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
`ensemble_copula_coupling.GeneratePercentilesFromMeanAndVariance`

"""
import unittest

import iris
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    GeneratePercentilesFromMeanAndVariance as Plugin)
from improver.tests.ensemble_calibration.ensemble_calibration. \
    helper_functions import (set_up_spot_temperature_cube,
                             set_up_temperature_cube,
                             add_forecast_reference_time_and_forecast_period)
from improver.utilities.warnings_handler import ManageWarnings


class Test__mean_and_variance_to_percentiles(IrisTest):

    """Test the _mean_and_variance_to_percentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))
        self.current_temperature_spot_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_spot_temperature_cube()))

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube matching the expected
        data values when a cube containing mean and variance is passed in.
        The resulting data values are the percentiles, which have been
        generated.
        """
        data = np.array([[[[225.56812863, 236.81812863, 248.06812863],
                           [259.31812863, 270.56812863, 281.81812863],
                           [293.06812863, 304.31812863, 315.56812863]]],
                         [[[229.48333333, 240.73333333, 251.98333333],
                           [263.23333333, 274.48333333, 285.73333333],
                           [296.98333333, 308.23333333, 319.48333333]]],
                         [[[233.39853804, 244.64853804, 255.89853804],
                           [267.14853804, 278.39853804, 289.64853804],
                           [300.89853804, 312.14853804, 323.39853804]]]])

        cube = self.current_temperature_forecast_cube
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        percentiles = [10, 50, 90]
        plugin = Plugin()
        result = plugin._mean_and_variance_to_percentiles(
            current_forecast_predictor, current_forecast_variance,
            percentiles)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_simple_data(self):
        """
        Test that the plugin returns the expected values for the generated
        percentiles when an idealised set of data values between 1 and 3
        is used to create the mean and the variance.
        """
        data = np.array([[[[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]],
                         [[[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 2]]],
                         [[[3, 3, 3],
                           [3, 3, 3],
                           [3, 3, 3]]]])

        result_data = np.array([[[[0.71844843, 0.71844843, 0.71844843],
                                  [0.71844843, 0.71844843, 0.71844843],
                                  [0.71844843, 0.71844843, 0.71844843]]],
                                [[[2., 2., 2.],
                                  [2., 2., 2.],
                                  [2., 2., 2.]]],
                                [[[3.28155157, 3.28155157, 3.28155157],
                                  [3.28155157, 3.28155157, 3.28155157],
                                  [3.28155157, 3.28155157, 3.28155157]]]])

        cube = self.current_temperature_forecast_cube
        cube.data = data
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        percentiles = [10, 50, 90]
        plugin = Plugin()
        result = plugin._mean_and_variance_to_percentiles(
            current_forecast_predictor, current_forecast_variance,
            percentiles)
        self.assertArrayAlmostEqual(result.data, result_data)

    @ManageWarnings(
        ignored_messages=["invalid value encountered",
                          "Collapsing a non-contiguous coordinate."],
        warning_types=[RuntimeWarning, UserWarning])
    def test_if_identical_data(self):
        """
        Test that the plugin returns the expected values, if every
        percentile has an identical value. This causes an issue because
        the default for the underlying scipy function is to yield a NaN for
        tied values. For this application, any NaN values are overwritten with
        the predicted mean value for all probability thresholds.
        """
        data = np.array([[1, 1, 1],
                         [2, 2, 2],
                         [3, 3, 3]])
        # Repeat data in the realization dimension.
        data = np.repeat(data[np.newaxis, np.newaxis, :, :], 3, axis=0)

        result_data = np.array([[[[1., 1., 1.],
                                  [2., 2., 2.],
                                  [3., 3., 3.]]],
                                [[[1., 1., 1.],
                                  [2., 2., 2.],
                                  [3., 3., 3.]]],
                                [[[1., 1., 1.],
                                  [2., 2., 2.],
                                  [3., 3., 3.]]]])

        cube = self.current_temperature_forecast_cube
        cube.data = data
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        percentiles = [10, 50, 90]
        plugin = Plugin()
        result = plugin._mean_and_variance_to_percentiles(
            current_forecast_predictor, current_forecast_variance,
            percentiles)
        self.assertArrayAlmostEqual(result.data, result_data)

    @ManageWarnings(
        ignored_messages=["invalid value encountered",
                          "Collapsing a non-contiguous coordinate."],
        warning_types=[RuntimeWarning, UserWarning])
    def test_if_nearly_identical_data(self):
        """
        Test that the plugin returns the expected values, if every
        percentile has an identical value. This causes an issue because
        the default for the underlying scipy function is to yield a NaN for
        tied values. For this application, any NaN values are overwritten with
        the predicted mean value for all probability thresholds.
        """
        data = np.array([[[[1., 1., 1.],
                           [4., 2., 2.],
                           [3., 3., 3.]]],
                         [[[1., 1., 1.],
                           [2., 2., 2.],
                           [3., 3., 3.]]],
                         [[[1., 1., 1.],
                           [2., 2., 2.],
                           [3., 3., 3.]]]])

        result_data = np.array([[[[1., 1., 1.],
                                  [1.18685838, 2., 2.],
                                  [3., 3., 3.]]],
                                [[[1., 1., 1.],
                                  [2.66666667, 2., 2.],
                                  [3., 3., 3.]]],
                                [[[1., 1., 1.],
                                  [4.14647495, 2., 2.],
                                  [3., 3., 3.]]]])

        cube = self.current_temperature_forecast_cube
        cube.data = data
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        percentiles = [10, 50, 90]
        plugin = Plugin()
        result = plugin._mean_and_variance_to_percentiles(
            current_forecast_predictor, current_forecast_variance,
            percentiles)
        self.assertArrayAlmostEqual(result.data, result_data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_many_percentiles(self):
        """
        Test that the plugin returns an iris.cube.Cube if many percentiles
        are requested.
        """
        cube = self.current_temperature_forecast_cube
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        percentiles = np.linspace(1, 99, num=1000, endpoint=True)
        plugin = Plugin()
        result = plugin._mean_and_variance_to_percentiles(
            current_forecast_predictor, current_forecast_variance, percentiles)
        self.assertIsInstance(result, Cube)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_negative_percentiles(self):
        """
        Test that the plugin returns the expected values for the
        percentiles if negative probabilities are requested.
        """
        cube = self.current_temperature_forecast_cube
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        percentiles = [-10, 10]
        plugin = Plugin()
        msg = "NaNs are present within the result for the"
        with self.assertRaisesRegexp(ValueError, msg):
            plugin._mean_and_variance_to_percentiles(
                current_forecast_predictor, current_forecast_variance,
                percentiles)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_spot_forecasts_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube matching the expected
        data values when a cube containing mean and variance is passed in.
        The resulting data values are the percentiles, which have been
        generated for a spot forecast.
        """
        data = np.array([[[225.56812863, 236.81812863, 248.06812863,
                           259.31812863, 270.56812863, 281.81812863,
                           293.06812863, 304.31812863, 315.56812863]],
                         [[229.48333333, 240.73333333, 251.98333333,
                           263.23333333, 274.48333333, 285.73333333,
                           296.98333333, 308.23333333, 319.48333333]],
                         [[233.39853804, 244.64853804, 255.89853804,
                           267.14853804, 278.39853804, 289.64853804,
                           300.89853804, 312.14853804, 323.39853804]]])

        cube = self.current_temperature_spot_forecast_cube
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        percentiles = [10, 50, 90]
        plugin = Plugin()
        result = plugin._mean_and_variance_to_percentiles(
            current_forecast_predictor, current_forecast_variance,
            percentiles)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, data)


class Test_process(IrisTest):

    """Test the process plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences",
                          "Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube."""
        cube = self.current_temperature_forecast_cube
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        raw_forecast = cube.copy()

        predictor_and_variance = CubeList(
            [current_forecast_predictor, current_forecast_variance])
        no_of_percentiles = len(raw_forecast.coord("realization").points)

        plugin = Plugin()
        result = plugin.process(predictor_and_variance, no_of_percentiles)
        self.assertIsInstance(result, Cube)

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences",
                          "Collapsing a non-contiguous coordinate."])
    def test_number_of_percentiles(self):
        """
        Test that the plugin returns a cube with the expected number of
        percentiles.
        """
        cube = self.current_temperature_forecast_cube
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        raw_forecast = cube.copy()

        predictor_and_variance = CubeList(
            [current_forecast_predictor, current_forecast_variance])

        no_of_percentiles = len(raw_forecast.coord("realization").points)

        plugin = Plugin()
        result = plugin.process(predictor_and_variance, no_of_percentiles)
        self.assertEqual(
            len(raw_forecast.coord("realization").points),
            len(result.coord("percentile_over_realization").points))


if __name__ == '__main__':
    unittest.main()
