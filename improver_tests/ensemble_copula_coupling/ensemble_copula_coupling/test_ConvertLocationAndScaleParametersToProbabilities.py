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
Unit tests for ConvertLocationAndScaleParametersToProbabilities
"""
import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertLocationAndScaleParametersToProbabilities as Plugin)
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.cube_manipulation import enforce_coordinate_ordering

from ...calibration.ensemble_calibration.helper_functions import (
    set_up_probability_above_threshold_temperature_cube)


class Test__repr__(IrisTest):

    """Test string representation of plugin."""

    def test_basic(self):
        """Test string representation"""
        expected_string = (
            "<ConvertLocationAndScaleParametersToProbabilities: "
            "distribution: norm; shape_parameters: []>")
        result = str(Plugin())
        self.assertEqual(result, expected_string)


class Test__check_template_cube(IrisTest):

    """Test the _check_template_cube function."""

    def setUp(self):
        """Set up temperature cube."""
        self.cube = (
            set_up_probability_above_threshold_temperature_cube())

    def test_valid_cube(self):
        """Pass in a valid cube that raises no exception. Cube should be
        unchanged by being passed into the function."""
        cube = iris.util.squeeze(self.cube)
        expected = iris.util.squeeze(self.cube.copy())
        Plugin()._check_template_cube(cube)
        self.assertEqual(expected, cube)

    def test_valid_cube_reordered(self):
        """Pass in a cube with the expected dimensions, but with threshold not
        the leading dimension. Check that threshold is moved to be leading."""
        cube = iris.util.squeeze(self.cube)
        enforce_coordinate_ordering(cube, 'latitude')
        expected = ['air_temperature', 'latitude', 'longitude']
        Plugin()._check_template_cube(cube)
        result = [coord.name() for coord in cube.coords(dim_coords=True)]
        self.assertListEqual(expected, result)

    def test_fail_on_cube_with_additional_dim_coord(self):
        """Pass in a cube with an additional dimensional coordinate. This will
        raise an exception."""

        msg = ("ConvertLocationAndScaleParametersToProbabilities expects a "
               "cube with")
        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._check_template_cube(self.cube)

    def test_fail_with_missing_spatial_coordinate(self):
        """Pass in a cube with a missing spatial coordinate. This will raise an
        exception."""

        cube = self.cube[:, 0, :, 0]
        msg = "The cube does not contain the expected"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._check_template_cube(cube)


class Test__check_unit_compatibility(IrisTest):

    """Test the _check_unit_compatibility function."""

    def setUp(self):
        """Set up temperature cube."""
        self.template_cube = (
            set_up_probability_above_threshold_temperature_cube())
        self.template_cube = iris.util.squeeze(self.template_cube)
        self.location_parameters = self.template_cube[0, :, :].copy()
        self.location_parameters.units = 'Celsius'
        self.scale_parameters = self.template_cube[0, :, :].copy()
        self.scale_parameters.units = 'Celsius2'

    def test_compatible_units(self):
        """Pass in compatible cubes that should not raise an exception. No
        assert statement required as any other input will raise an
        exception."""
        Plugin()._check_unit_compatibility(
            self.location_parameters, self.scale_parameters,
            self.template_cube)

    def test_convertible_units(self):
        """Pass in cubes with units that can be made equivalent by modification
        to match the threshold units."""
        self.location_parameters.units = 'Fahrenheit'
        self.scale_parameters.units = 'Fahrenheit2'
        Plugin()._check_unit_compatibility(
            self.location_parameters, self.scale_parameters,
            self.template_cube)
        self.assertEqual(self.location_parameters.units, "Celsius")

    def test_incompatible_units(self):
        """Pass in cubes of incompatible units that should raise an
        exception."""
        self.scale_parameters.units = 'm s-1'
        msg = 'This is likely because the mean'
        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._check_unit_compatibility(
                self.location_parameters, self.scale_parameters,
                self.template_cube)


class Test__location_and_scale_parameters_to_probabilities(IrisTest):

    """Test the _location_and_scale_parameters_to_probabilities function."""

    def setUp(self):
        """Set up temperature cube."""
        self.template_cube = (
            set_up_probability_above_threshold_temperature_cube())
        self.template_cube = iris.util.squeeze(self.template_cube)

        # Thresholds such that we obtain probabilities of 75%, 50%, and 25% for
        # the location and scale parameter values set here.
        threshold_coord = find_threshold_coordinate(self.template_cube)
        threshold_coord.points = [0.65105, 2., 3.34895]
        location_parameter_values = np.ones((3, 3)) * 2
        scale_parameter_values = np.ones((3, 3)) * 4
        self.expected = (np.ones((3, 3, 3)) * [0.75, 0.5, 0.25]).T
        self.location_parameter_values = (
            self.template_cube[0, :, :].copy(data=location_parameter_values))
        self.location_parameter_values.units = 'Celsius'
        self.scale_parameter_values = (
            self.template_cube[0, :, :].copy(data=scale_parameter_values))
        self.scale_parameter_values.units = 'Celsius2'

    def test_threshold_above_cube(self):
        """Test that the expected probabilities are returned for a cube in
        which they are calculated above the thresholds."""

        result = Plugin()._location_and_scale_parameters_to_probabilities(
            self.location_parameter_values, self.scale_parameter_values,
            self.template_cube)
        np.testing.assert_allclose(result.data, self.expected, rtol=1.e-4)

    def test_with_location_mask(self):
        """Test that the plugin gets the correct data out when you put in
        masked location parameter data."""
        mask = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        expected_mask = np.broadcast_to(mask, (3, 3, 3))
        self.location_parameter_values.data = np.ma.masked_array(
            self.location_parameter_values.data, mask=mask)
        expected_with_mask = np.ma.masked_array(
            self.expected, mask=expected_mask)
        result = Plugin()._location_and_scale_parameters_to_probabilities(
            self.location_parameter_values, self.scale_parameter_values,
            self.template_cube)
        np.testing.assert_allclose(
            result.data, expected_with_mask, rtol=1.e-4)

    def test_with_scale_mask(self):
        """Test that the plugin gets the correct data out when you put in
        masked scale parameter data."""
        mask = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 1]])
        expected_mask = np.broadcast_to(mask, (3, 3, 3))
        self.scale_parameter_values.data = np.ma.masked_array(
            self.scale_parameter_values.data, mask=mask)
        expected_with_mask = np.ma.masked_array(
            self.expected, mask=expected_mask)
        result = Plugin()._location_and_scale_parameters_to_probabilities(
            self.location_parameter_values, self.scale_parameter_values,
            self.template_cube)
        np.testing.assert_allclose(
            result.data, expected_with_mask, rtol=1.e-4)

    def test_both_masked(self):
        """Test that the plugin returns the correct  data when you put in
        masked data for both the scale and location parameters."""
        mask1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        mask2 = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        expected_mask = np.broadcast_to(mask1+mask2, (3, 3, 3))
        self.location_parameter_values.data = np.ma.masked_array(
            self.location_parameter_values.data, mask=mask1)
        self.scale_parameter_values.data = np.ma.masked_array(
            self.scale_parameter_values.data, mask=mask2)
        expected_with_mask = np.ma.masked_array(
            self.expected, mask=expected_mask)
        result = Plugin()._location_and_scale_parameters_to_probabilities(
            self.location_parameter_values, self.scale_parameter_values,
            self.template_cube)
        np.testing.assert_allclose(
            result.data, expected_with_mask, rtol=1.e-4)

    def test_threshold_above_cube_truncnorm(self):
        """Test that the expected probabilities are returned for a cube in
        which they are calculated above the thresholds using a truncated normal
        distribution."""

        expected = (np.ones((3, 3, 3)) * [0.8914245, 0.5942867, 0.2971489]).T
        plugin = Plugin(distribution="truncnorm", shape_parameters=[0, np.inf])
        result = plugin._location_and_scale_parameters_to_probabilities(
            self.location_parameter_values, self.scale_parameter_values,
            self.template_cube)
        np.testing.assert_allclose(result.data, expected, rtol=1.e-4)

    def test_threshold_below_cube(self):
        """Test that the expected probabilites are returned for a cube in which
        they are calculated below the thresholds."""

        self.template_cube.coord(
            var_name="threshold").attributes['spp__relative_to_threshold'] = (
                'below')
        expected = (np.ones((3, 3, 3)) * [0.25, 0.5, 0.75]).T
        result = Plugin()._location_and_scale_parameters_to_probabilities(
            self.location_parameter_values, self.scale_parameter_values,
            self.template_cube)
        np.testing.assert_allclose(result.data, expected, rtol=1.e-4)


class Test_process(IrisTest):

    """Test the process function."""

    def setUp(self):
        """Set up temperature cube."""
        self.template_cube = (
            set_up_probability_above_threshold_temperature_cube())
        self.template_cube = iris.util.squeeze(self.template_cube)

        threshold_coord = find_threshold_coordinate(self.template_cube)
        threshold_coord.points = [8.65105, 10., 11.34895]
        location_parameter_values = np.ones((3, 3)) * 10
        scale_parameter_values = np.ones((3, 3)) * 4
        self.location_parameter_values = (
            self.template_cube[0, :, :].copy(data=location_parameter_values))
        self.location_parameter_values.units = 'Celsius'
        self.scale_parameter_values = (
            self.template_cube[0, :, :].copy(data=scale_parameter_values))
        self.scale_parameter_values.units = 'Celsius2'

    def test_metadata_matches_template(self):
        """Test that the returned cube's metadata matches the template cube."""

        result = Plugin().process(
            self.location_parameter_values, self.scale_parameter_values,
            self.template_cube)
        self.assertTrue(result.metadata == self.template_cube.metadata)
        self.assertTrue(result.name() == self.template_cube.name())

    def test_template_data_disregarded(self):
        """Test that the returned cube does not contain data from the template
        cube."""

        self.template_cube.data = np.ones((3, 3, 3))
        result = Plugin().process(
            self.location_parameter_values, self.scale_parameter_values,
            self.template_cube)
        self.assertTrue((result.data != self.template_cube.data).all())


if __name__ == '__main__':
    unittest.main()
