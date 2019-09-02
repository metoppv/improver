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
Unit tests for the `SplitHistoricForecastAndTruth` plugin.

"""
import unittest

from iris.tests import IrisTest

from improver.ensemble_calibration.ensemble_calibration_utilities import (
    SplitHistoricForecastAndTruth)
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import SetupCubes


class SetupDicts(IrisTest):
    """Set up historical forecast and truth cubes and associated dictionaries
    for use in testing."""

    def setUp(self):
        """Set up dictionaries for testing."""
        if hasattr(super(), "setUp"):
            super().setUp()
        # Set up dictionaries specifying the metadata to identify the
        # historical forecasts and truth
        self.historic_forecast_dict = {
            "attributes": {
                "mosg__model_configuration": "uk_ens"
            }
        }
        self.truth_dict = {
            "attributes": {
                "mosg__model_configuration": "uk_det"
            }
        }

        # Initialise the plugin
        self.plugin = SplitHistoricForecastAndTruth(
            self.historic_forecast_dict, self.truth_dict)


class Test__init__(SetupCubes, SetupDicts):
    """Test class initialisation"""

    def test_basic(self):
        """Test that the historic forecast and truth dictionaries are specified
        correctly."""
        self.assertEqual(
            self.plugin.historic_forecast_dict, self.historic_forecast_dict)
        self.assertEqual(self.plugin.truth_dict, self.truth_dict)

    def test_non_attributes(self):
        """Test that the if metadata other than attributes are specified then
        a NotImplementedError is raised."""
        input_dict = {
            "coord": "forecast_period==0"
        }
        msg = "'attributes' is the only supported"
        with self.assertRaisesRegex(NotImplementedError, msg):
            SplitHistoricForecastAndTruth(input_dict, input_dict)


class Test__repr__(SetupCubes, SetupDicts):
    """Test class representation"""

    def test_basic(self):
        """Test string representation"""
        result = str(self.plugin)
        expected_result = (
            "<SplitHistoricForecastsAndTruth: "
            "historic_forecast_dict={'attributes': "
            "{'mosg__model_configuration': 'uk_ens'}}, "
            "truth_dict={'attributes': "
            "{'mosg__model_configuration': 'uk_det'}}>")
        self.assertEqual(result, expected_result)


class Test__find_required_cubes_using_metadata(SetupCubes, SetupDicts):
    """Test the _find_required_cubes_using_metadata method."""

    def test_attributes(self):
        """Test that the desired cube is returned if the required attributes
        are specified."""
        result = self.plugin._find_required_cubes_using_metadata(
            self.combined, self.historic_forecast_dict)
        self.assertEqual(result, self.historic_forecasts)

    def test_non_matching_attributes(self):
        """Test that the if attributes are specified but these do not match
        any attributes on the cube then a ValueError is raised."""
        input_dict = {
            "attributes": {
                "mosg__model_configuration": "mars_det"
            }
        }
        msg = "The metadata to identify"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._find_required_cubes_using_metadata(
                self.combined, input_dict)


class Test_process(SetupCubes, SetupDicts):
    """Test the process method."""

    def test_basic(self):
        """Test that the input cubelist combining historic forecasts and truth
        can be split using the metadata dictionaries provided."""
        hf_result, truth_result = self.plugin.process(self.combined)
        self.assertEqual(hf_result, self.historic_temperature_forecast_cube)
        self.assertEqual(truth_result, self.temperature_truth_cube)


if __name__ == '__main__':
    unittest.main()
