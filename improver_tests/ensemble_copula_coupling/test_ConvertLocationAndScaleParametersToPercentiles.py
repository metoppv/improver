# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
    ConvertLocationAndScaleParametersToPercentiles as Plugin,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

from .ecc_test_data import ECC_TEMPERATURE_REALIZATIONS, set_up_spot_test_cube


class Test__repr__(IrisTest):

    """Test string representation of plugin."""

    def test_basic(self):
        """Test string representation"""
        expected_string = (
            "<ConvertLocationAndScaleParametersToPercentiles: "
            "distribution: norm; shape_parameters: []>"
        )
        result = str(Plugin())
        self.assertEqual(result, expected_string)


class Test__location_and_scale_parameters_to_percentiles(IrisTest):

    """Test the _location_and_scale_parameters_to_percentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.temperature_cube = set_up_variable_cube(
            ECC_TEMPERATURE_REALIZATIONS.copy()
        )
        self.data = np.array(
            [
                [
                    [225.5681, 236.8181, 248.0681],
                    [259.3181, 270.5681, 281.8181],
                    [293.0681, 304.3181, 315.5681],
                ],
                [
                    [229.4833, 240.7333, 251.9833],
                    [263.2333, 274.4833, 285.7333],
                    [296.9833, 308.2333, 319.4833],
                ],
                [
                    [233.3985, 244.6485, 255.8985],
                    [267.1485, 278.3985, 289.6485],
                    [300.8985, 312.1485, 323.3985],
                ],
            ],
            dtype=np.float32,
        )

        self.location_parameter = self.temperature_cube.collapsed(
            "realization", iris.analysis.MEAN
        )
        self.scale_parameter = self.temperature_cube.collapsed(
            "realization", iris.analysis.STD_DEV,
        )
        self.percentiles = [10, 50, 90]

    def test_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube matching the expected
        data values when a cubes containing location and scale parameters are
        passed in, which are equivalent to the ensemble mean and ensemble
        standard deviation. The resulting data values are the percentiles, which
        have been generated.
        """
        result = Plugin()._location_and_scale_parameters_to_percentiles(
            self.location_parameter,
            self.scale_parameter,
            self.temperature_cube,
            self.percentiles,
        )
        self.assertIsInstance(result, Cube)
        np.testing.assert_allclose(result.data, self.data, rtol=1.0e-4)

    def test_masked_location_parameter(self):
        """
        Test that the plugin returns the correctly masked data when
        given a location parameter that is masked.
        """
        mask = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]])
        expected_mask = np.broadcast_to(mask, (3, 3, 3))
        expected_data = np.ma.masked_array(self.data, mask=expected_mask)
        self.location_parameter.data = np.ma.masked_array(
            self.location_parameter.data, mask=mask
        )
        result = Plugin()._location_and_scale_parameters_to_percentiles(
            self.location_parameter,
            self.scale_parameter,
            self.temperature_cube,
            self.percentiles,
        )
        np.testing.assert_allclose(result.data, expected_data, rtol=1.0e-4)

    def test_masked_scale_parameter(self):
        """
        Test that the plugin returns the correctly masked data when
        given a scale parameter that is masked.
        """
        mask = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 1]])
        expected_mask = np.broadcast_to(mask, (3, 3, 3))
        expected_data = np.ma.masked_array(self.data, mask=expected_mask)
        self.scale_parameter.data = np.ma.masked_array(
            self.scale_parameter.data, mask=mask
        )
        result = Plugin()._location_and_scale_parameters_to_percentiles(
            self.location_parameter,
            self.scale_parameter,
            self.temperature_cube,
            self.percentiles,
        )
        np.testing.assert_allclose(result.data, expected_data, rtol=1.0e-4)

    def test_both_masked(self):
        """
        Test that the plugin returns the correctly masked data when
        both the scale and location parameters are masked.
        """
        mask1 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        mask2 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        expected_mask = np.broadcast_to(mask1 + mask2, (3, 3, 3))
        expected_data = np.ma.masked_array(self.data, mask=expected_mask)
        self.location_parameter.data = np.ma.masked_array(
            self.location_parameter.data, mask=mask1
        )
        self.scale_parameter.data = np.ma.masked_array(
            self.scale_parameter.data, mask=mask2
        )
        result = Plugin()._location_and_scale_parameters_to_percentiles(
            self.location_parameter,
            self.scale_parameter,
            self.temperature_cube,
            self.percentiles,
        )
        np.testing.assert_allclose(result.data, expected_data, rtol=1.0e-4)

    def test_simple_data_truncnorm_distribution(self):
        """
        Test that the plugin returns an iris.cube.Cube matching the expected
        data values when cubes containing the location parameter and scale
        parameter are passed in. In this test, the ensemble mean and standard
        deviation is used as a proxy for the location and scale parameter.
        The resulting data values are the percentiles, which have been
        generated using a truncated normal distribution.
        """
        data = np.array(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            ]
        )
        self.temperature_cube.data = data

        expected_data = np.array(
            [
                [
                    [1.0121, 1.0121, 1.0121],
                    [1.0121, 1.0121, 1.0121],
                    [1.0121, 1.0121, 1.0121],
                ],
                [
                    [3.1677, 3.1677, 3.1677],
                    [3.1677, 3.1677, 3.1677],
                    [3.1677, 3.1677, 3.1677],
                ],
                [
                    [5.6412, 5.6412, 5.6412],
                    [5.6412, 5.6412, 5.6412],
                    [5.6412, 5.6412, 5.6412],
                ],
            ]
        )

        # Use an adjusted version of the ensemble mean as a proxy for the
        # location parameter for the truncated normal distribution.
        current_forecast_predictor = self.temperature_cube.collapsed(
            "realization", iris.analysis.MEAN
        )
        current_forecast_predictor.data = current_forecast_predictor.data + 1
        # Use an adjusted version of the ensemble standard deviation as a proxy for the
        # scale parameter for the truncated normal distribution.
        current_forecast_stddev = self.temperature_cube.collapsed(
            "realization", iris.analysis.STD_DEV,
        )
        current_forecast_stddev.data = current_forecast_stddev.data + 1
        plugin = Plugin(
            distribution="truncnorm",
            shape_parameters=np.array([0, np.inf], dtype=np.float32),
        )
        result = plugin._location_and_scale_parameters_to_percentiles(
            current_forecast_predictor,
            current_forecast_stddev,
            self.temperature_cube,
            self.percentiles,
        )
        self.assertIsInstance(result, Cube)
        np.testing.assert_allclose(result.data, expected_data, rtol=1.0e-4)

    def test_simple_data(self):
        """
        Test that the plugin returns the expected values for the generated
        percentiles when an idealised set of data values between 1 and 3
        is used to create the mean (location parameter) and the standard deviation
        (scale parameter).
        """
        data = np.array(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            ]
        )
        self.temperature_cube.data = data

        expected_data = np.array(
            [
                [
                    [0.71844843, 0.71844843, 0.71844843],
                    [0.71844843, 0.71844843, 0.71844843],
                    [0.71844843, 0.71844843, 0.71844843],
                ],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                [
                    [3.28155157, 3.28155157, 3.28155157],
                    [3.28155157, 3.28155157, 3.28155157],
                    [3.28155157, 3.28155157, 3.28155157],
                ],
            ]
        )

        current_forecast_predictor = self.temperature_cube.collapsed(
            "realization", iris.analysis.MEAN
        )
        current_forecast_stddev = self.temperature_cube.collapsed(
            "realization", iris.analysis.STD_DEV
        )
        result = Plugin()._location_and_scale_parameters_to_percentiles(
            current_forecast_predictor,
            current_forecast_stddev,
            self.temperature_cube,
            self.percentiles,
        )
        np.testing.assert_allclose(result.data, expected_data, rtol=1.0e-4)

    def test_if_identical_data(self):
        """
        Test that the plugin returns the expected values, if every
        percentile has an identical value. This causes an issue because
        the default for the underlying scipy function is to yield a NaN for
        tied values. For this application, any NaN values are overwritten with
        the predicted mean value for all probability thresholds.
        """
        data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        # Repeat data in the realization dimension.
        data = np.repeat(data[np.newaxis, :, :], 3, axis=0)
        self.temperature_cube.data = data

        expected_data = np.array(
            [
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            ]
        )

        current_forecast_predictor = self.temperature_cube.collapsed(
            "realization", iris.analysis.MEAN
        )
        current_forecast_stddev = self.temperature_cube.collapsed(
            "realization", iris.analysis.STD_DEV
        )
        result = Plugin()._location_and_scale_parameters_to_percentiles(
            current_forecast_predictor,
            current_forecast_stddev,
            self.temperature_cube,
            self.percentiles,
        )
        np.testing.assert_allclose(result.data, expected_data, rtol=1.0e-4)

    def test_if_nearly_identical_data(self):
        """
        Test that the plugin returns the expected values, if every
        percentile has an identical value. This causes an issue because
        the default for the underlying scipy function is to yield a NaN for
        tied values. For this application, any NaN values are overwritten with
        the predicted mean value for all probability thresholds.
        """
        data = np.array(
            [
                [[1.0, 1.0, 1.0], [4.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            ]
        )
        self.temperature_cube.data = data

        expected_data = np.array(
            [
                [[1.0, 1.0, 1.0], [1.18685838, 2.0, 2.0], [3.0, 3.0, 3.0]],
                [[1.0, 1.0, 1.0], [2.66666667, 2.0, 2.0], [3.0, 3.0, 3.0]],
                [[1.0, 1.0, 1.0], [4.14647495, 2.0, 2.0], [3.0, 3.0, 3.0]],
            ]
        )

        current_forecast_predictor = self.temperature_cube.collapsed(
            "realization", iris.analysis.MEAN
        )
        current_forecast_stddev = self.temperature_cube.collapsed(
            "realization", iris.analysis.STD_DEV
        )
        result = Plugin()._location_and_scale_parameters_to_percentiles(
            current_forecast_predictor,
            current_forecast_stddev,
            self.temperature_cube,
            self.percentiles,
        )
        np.testing.assert_allclose(result.data, expected_data, rtol=1.0e-4)

    def test_many_percentiles(self):
        """
        Test that the plugin returns an iris.cube.Cube if many percentiles
        are requested.
        """
        percentiles = np.linspace(1, 99, num=1000, endpoint=True)
        result = Plugin()._location_and_scale_parameters_to_percentiles(
            self.location_parameter,
            self.scale_parameter,
            self.temperature_cube,
            percentiles,
        )
        self.assertIsInstance(result, Cube)

    def test_negative_percentiles(self):
        """
        Test that the plugin returns the expected values for the
        percentiles if negative probabilities are requested.
        """
        percentiles = [-10, 10]
        msg = "NaNs are present within the result for the"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._location_and_scale_parameters_to_percentiles(
                self.location_parameter,
                self.scale_parameter,
                self.temperature_cube,
                percentiles,
            )

    def test_spot_forecasts_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube matching the expected
        data values when a cube containing mean (location parameter) and
        standard deviation (scale parameter) is passed in. The resulting data values are
        the percentiles, which have been generated for a spot forecast.
        """
        data = np.reshape(self.data, (3, 9))
        cube = set_up_spot_test_cube()

        current_forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        current_forecast_stddev = cube.collapsed("realization", iris.analysis.STD_DEV)
        result = Plugin()._location_and_scale_parameters_to_percentiles(
            current_forecast_predictor, current_forecast_stddev, cube, self.percentiles,
        )
        self.assertIsInstance(result, Cube)
        np.testing.assert_allclose(result.data, data, rtol=1.0e-4)

    def test_scalar_realisation_percentile(self):
        """
        Test that the plugin returns the expected values when providing a cube
        with a scalar realization or percentile coordinate.
        """

        result = Plugin()._location_and_scale_parameters_to_percentiles(
            self.location_parameter,
            self.scale_parameter,
            self.temperature_cube[0],
            self.percentiles,
        )
        self.assertIsInstance(result, Cube)


class Test_process(IrisTest):

    """Test the process plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.cube = set_up_variable_cube(ECC_TEMPERATURE_REALIZATIONS.copy())
        self.forecast_predictor = self.cube.collapsed("realization", iris.analysis.MEAN)
        self.forecast_stddev = self.cube.collapsed("realization", iris.analysis.STD_DEV)
        self.no_of_percentiles = len(self.cube.coord("realization").points)

    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube."""
        result = Plugin().process(
            self.forecast_predictor,
            self.forecast_stddev,
            self.cube,
            no_of_percentiles=self.no_of_percentiles,
        )
        self.assertIsInstance(result, Cube)

    def test_number_of_percentiles(self):
        """
        Test that the plugin returns a cube with the expected number of
        percentiles.
        """
        expected = np.array(
            [
                [
                    [227.42273, 238.67273, 249.92273],
                    [261.1727, 272.4227, 283.6727],
                    [294.9227, 306.1727, 317.4227],
                ],
                [
                    [229.48332, 240.73332, 251.98332],
                    [263.2333, 274.4833, 285.7333],
                    [296.9833, 308.2333, 319.4833],
                ],
                [
                    [231.54391, 242.79391, 254.04391],
                    [265.2939, 276.5439, 287.7939],
                    [299.0439, 310.2939, 321.5439],
                ],
            ]
        )

        result = Plugin().process(
            self.forecast_predictor,
            self.forecast_stddev,
            self.cube,
            no_of_percentiles=self.no_of_percentiles,
        )

        self.assertEqual(len(result.coord("percentile").points), self.no_of_percentiles)
        self.assertArrayAlmostEqual(expected, result.data, decimal=4)

    def test_list_of_percentiles(self):
        """
        Test that the plugin returns a cube with the expected percentiles
        when a specific list of percentiles is provided.
        """
        percentiles = [10, 50, 90]
        expected = np.array(
            [
                [
                    [225.56812, 236.81812, 248.06812],
                    [259.3181, 270.5681, 281.8181],
                    [293.0681, 304.3181, 315.5681],
                ],
                [
                    [229.48332, 240.73332, 251.98332],
                    [263.2333, 274.4833, 285.7333],
                    [296.9833, 308.2333, 319.4833],
                ],
                [
                    [233.39853, 244.64853, 255.89853],
                    [267.1485, 278.3985, 289.6485],
                    [300.8985, 312.1485, 323.3985],
                ],
            ]
        )

        result = Plugin().process(
            self.forecast_predictor,
            self.forecast_stddev,
            self.cube,
            percentiles=percentiles,
        )

        self.assertEqual(len(percentiles), len(result.coord("percentile").points))
        self.assertArrayAlmostEqual(percentiles, result.coord("percentile").points)
        self.assertArrayAlmostEqual(expected, result.data, decimal=4)

    def test_multiple_keyword_arguments_error(self):
        """
        Test that the plugin raises an error when both the no_of_percentiles
        keyword argument and the percentiles keyword argument are provided.
        """
        percentiles = [10, 25, 50, 75, 90]
        msg = "Please specify either the number of percentiles or"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin().process(
                self.forecast_predictor,
                self.forecast_stddev,
                self.cube,
                no_of_percentiles=self.no_of_percentiles,
                percentiles=percentiles,
            )


if __name__ == "__main__":
    unittest.main()
