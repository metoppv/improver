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
`ensemble_copula_coupling.ConvertLocationAndScaleParametersToPercentiles`
"""
import unittest

import iris
import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertLocationAndScaleParametersToPercentiles as Plugin)
from improver.utilities.warnings_handler import ManageWarnings

from ...calibration.ensemble_calibration.helper_functions import (
    add_forecast_reference_time_and_forecast_period,
    set_up_spot_temperature_cube,
    set_up_temperature_cube)


class Test__repr__(IrisTest):

    """Test string representation of plugin."""

    def test_basic(self):
        """Test string representation"""
        expected_string = ("<ConvertLocationAndScaleParametersToPercentiles: "
                           "distribution: norm; shape_parameters: []>")
        result = str(Plugin())
        self.assertEqual(result, expected_string)


class Test__location_and_scale_parameters_to_percentiles(IrisTest):

    """Test the _location_and_scale_parameters_to_percentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.temperature_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))
        self.temperature_spot_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_spot_temperature_cube()))
        self.data = np.array([[[[225.568115, 236.818115, 248.068115],
                                [259.318115, 270.568115, 281.818115],
                                [293.068115, 304.318115, 315.568115]]],
                              [[[229.483322, 240.733322, 251.983322],
                                [263.233307, 274.483307, 285.733307],
                                [296.983307, 308.233307, 319.483307]]],
                              [[[233.398529, 244.648529, 255.898529],
                                [267.148499, 278.398499, 289.648499],
                                [300.898499, 312.148499, 323.398499]]]],
                             dtype=np.float32)
        self.location_parameter = self.temperature_cube.collapsed(
            "realization", iris.analysis.MEAN)
        self.scale_parameter = self.temperature_cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        self.percentiles = [10, 50, 90]

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube matching the expected
        data values when a cubes containing location and scale parameters are
        passed in, which are equivalent to the ensemble mean and ensemble
        variance. The resulting data values are the percentiles, which have
        been generated.
        """
        plugin = Plugin()
        result = plugin._location_and_scale_parameters_to_percentiles(
            self.location_parameter, self.scale_parameter,
            self.temperature_cube, self.percentiles)
        self.assertIsInstance(result, Cube)
        np.testing.assert_allclose(result.data, self.data, rtol=1.e-4)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_masked_location_parameter(self):
        """
        Test that the plugin returns the correctly masked data when
        given a location parameter that is masked.
        """
        mask = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]])
        expected_mask = np.broadcast_to(mask, (3, 3, 3))
        expected_data = np.ma.masked_array(self.data, mask=expected_mask)
        self.location_parameter.data = np.ma.masked_array(
            self.location_parameter.data, mask=mask)
        plugin = Plugin()
        result = plugin._location_and_scale_parameters_to_percentiles(
            self.location_parameter, self.scale_parameter,
            self.temperature_cube, self.percentiles)
        np.testing.assert_allclose(result.data, expected_data, rtol=1.e-4)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_masked_scale_parameter(self):
        """
        Test that the plugin returns the correctly masked data when
        given a scale parameter that is masked.
        """
        mask = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 1]])
        expected_mask = np.broadcast_to(mask, (3, 3, 3))
        expected_data = np.ma.masked_array(self.data, mask=expected_mask)
        self.scale_parameter.data = np.ma.masked_array(
            self.scale_parameter.data, mask=mask)
        plugin = Plugin()
        result = plugin._location_and_scale_parameters_to_percentiles(
            self.location_parameter, self.scale_parameter,
            self.temperature_cube, self.percentiles)
        np.testing.assert_allclose(result.data, expected_data, rtol=1.e-4)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_both_masked(self):
        """
        Test that the plugin returns the correctly masked data when
        both the scale and location parameters are masked.
        """
        mask1 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        mask2 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        expected_mask = np.broadcast_to(mask1+mask2, (3, 3, 3))
        expected_data = np.ma.masked_array(self.data, mask=expected_mask)
        self.location_parameter.data = np.ma.masked_array(
            self.location_parameter.data, mask=mask1)
        self.scale_parameter.data = np.ma.masked_array(
            self.scale_parameter.data, mask=mask2)
        plugin = Plugin()
        result = plugin._location_and_scale_parameters_to_percentiles(
            self.location_parameter, self.scale_parameter,
            self.temperature_cube, self.percentiles)
        np.testing.assert_allclose(result.data, expected_data, rtol=1.e-4)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_simple_data_truncnorm_distribution(self):
        """
        Test that the plugin returns an iris.cube.Cube matching the expected
        data values when cubes containing the location parameter and scale
        parameter are passed in. In this test, the ensemble mean and variance
        is used as a proxy for the location and scale parameter. The resulting
        data values are the percentiles, which have been generated using a
        truncated normal distribution.
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

        result_data = np.array([[[[1.3042759, 1.3042759, 1.3042759],
                                  [1.3042759, 1.3042759, 1.3042759],
                                  [1.3042759, 1.3042759, 1.3042759]]],
                                [[[3.0300407, 3.0300407, 3.0300407],
                                  [3.0300407, 3.0300407, 3.0300407],
                                  [3.0300407, 3.0300407, 3.0300407]]],
                                [[[4.8261294, 4.8261294, 4.8261294],
                                  [4.8261294, 4.8261294, 4.8261294],
                                  [4.8261294, 4.8261294, 4.8261294]]]])

        cube = self.temperature_cube
        cube.data = data
        # Use an adjusted version of the ensemble mean as a proxy for the
        # location parameter for the truncated normal distribution.
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_predictor.data = current_forecast_predictor.data + 1
        # Use an adjusted version of the ensemble variance as a proxy for the
        # scale parameter for the truncated normal distribution.
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        current_forecast_variance.data = current_forecast_variance.data + 1
        plugin = Plugin(distribution="truncnorm", shape_parameters=[0, np.inf])
        result = plugin._location_and_scale_parameters_to_percentiles(
            current_forecast_predictor, current_forecast_variance, cube,
            self.percentiles)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, result_data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_simple_data(self):
        """
        Test that the plugin returns the expected values for the generated
        percentiles when an idealised set of data values between 1 and 3
        is used to create the mean (location parameter) and the variance
        (scale parameter).
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

        cube = self.temperature_cube
        cube.data = data
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        plugin = Plugin()
        result = plugin._location_and_scale_parameters_to_percentiles(
            current_forecast_predictor, current_forecast_variance, cube,
            self.percentiles)
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

        cube = self.temperature_cube
        cube.data = data
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        plugin = Plugin()
        result = plugin._location_and_scale_parameters_to_percentiles(
            current_forecast_predictor, current_forecast_variance, cube,
            self.percentiles)
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

        cube = self.temperature_cube
        cube.data = data
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        plugin = Plugin()
        result = plugin._location_and_scale_parameters_to_percentiles(
            current_forecast_predictor, current_forecast_variance, cube,
            self.percentiles)
        self.assertArrayAlmostEqual(result.data, result_data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_many_percentiles(self):
        """
        Test that the plugin returns an iris.cube.Cube if many percentiles
        are requested.
        """
        cube = self.temperature_cube
        percentiles = np.linspace(1, 99, num=1000, endpoint=True)
        plugin = Plugin()
        result = plugin._location_and_scale_parameters_to_percentiles(
            self.location_parameter, self.scale_parameter, cube,
            percentiles)
        self.assertIsInstance(result, Cube)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_negative_percentiles(self):
        """
        Test that the plugin returns the expected values for the
        percentiles if negative probabilities are requested.
        """
        cube = self.temperature_cube
        percentiles = [-10, 10]
        plugin = Plugin()
        msg = "NaNs are present within the result for the"
        with self.assertRaisesRegex(ValueError, msg):
            plugin._location_and_scale_parameters_to_percentiles(
                self.location_parameter, self.scale_parameter, cube,
                percentiles)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_spot_forecasts_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube matching the expected
        data values when a cube containing mean (location parameter) and
        variance (scale parameter) is passed in. The resulting data values are
        the percentiles, which have been generated for a spot forecast.
        """
        data = np.reshape(self.data, (3, 1, 9))
        cube = self.temperature_spot_cube
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        plugin = Plugin()
        result = plugin._location_and_scale_parameters_to_percentiles(
            current_forecast_predictor, current_forecast_variance, cube,
            self.percentiles)
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

        no_of_percentiles = len(raw_forecast.coord("realization").points)

        plugin = Plugin()
        result = plugin.process(
            current_forecast_predictor, current_forecast_variance, cube,
            no_of_percentiles=no_of_percentiles)
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

        no_of_percentiles = len(raw_forecast.coord("realization").points)
        expected = np.array(
            [[[[227.42273, 238.67273, 249.92273],
               [261.1727, 272.4227, 283.6727],
               [294.9227, 306.1727, 317.4227]]],
             [[[229.48332, 240.73332, 251.98332],
               [263.2333, 274.4833, 285.7333],
               [296.9833, 308.2333, 319.4833]]],
             [[[231.54391, 242.79391, 254.04391],
               [265.2939, 276.5439, 287.7939],
               [299.0439, 310.2939, 321.5439]]]])

        plugin = Plugin()
        result = plugin.process(
            current_forecast_predictor, current_forecast_variance, cube,
            no_of_percentiles=no_of_percentiles)

        self.assertEqual(
            len(raw_forecast.coord("realization").points),
            len(result.coord("percentile").points))
        self.assertArrayAlmostEqual(expected, result.data, decimal=4)

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences",
                          "Collapsing a non-contiguous coordinate."])
    def test_list_of_percentiles(self):
        """
        Test that the plugin returns a cube with the expected percentiles
        when a specific list of percentiles is provided.
        """
        cube = self.current_temperature_forecast_cube
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)

        percentiles = [10, 50, 90]
        expected = np.array(
            [[[[225.56812, 236.81812, 248.06812],
               [259.3181, 270.5681, 281.8181],
               [293.0681, 304.3181, 315.5681]]],
             [[[229.48332, 240.73332, 251.98332],
               [263.2333, 274.4833, 285.7333],
               [296.9833, 308.2333, 319.4833]]],
             [[[233.39853, 244.64853, 255.89853],
               [267.1485, 278.3985, 289.6485],
               [300.8985, 312.1485, 323.3985]]]])

        plugin = Plugin()
        result = plugin.process(
            current_forecast_predictor, current_forecast_variance, cube,
            percentiles=percentiles)

        self.assertEqual(
            len(percentiles),
            len(result.coord("percentile").points))
        self.assertArrayAlmostEqual(
            percentiles, result.coord("percentile").points)
        self.assertArrayAlmostEqual(expected, result.data, decimal=4)

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences",
                          "Collapsing a non-contiguous coordinate."])
    def test_multiple_keyword_arguments_error(self):
        """
        Test that the plugin raises an error when both the no_of_percentiles
        keyword argument and the percentiles keyword argument are provided.
        """
        cube = self.current_temperature_forecast_cube
        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        raw_forecast = cube.copy()

        no_of_percentiles = len(raw_forecast.coord("realization").points)
        percentiles = [10, 25, 50, 75, 90]

        plugin = Plugin()
        msg = "Please specify either the number of percentiles or"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(
                current_forecast_predictor, current_forecast_variance, cube,
                no_of_percentiles=no_of_percentiles, percentiles=percentiles)


if __name__ == '__main__':
    unittest.main()
