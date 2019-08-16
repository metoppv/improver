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
import datetime
import unittest

import iris
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_calibration.ensemble_calibration_utilities import (
    SplitHistoricForecastAndTruth)
from improver.tests.set_up_test_cubes import set_up_variable_cube


class SetupCubesAndDicts(IrisTest):
    """Set up historical forecast and truth cubes and associated dictionaries
    for use in testing."""

    def setUp(self):
        """Set up dictionaries for testing."""
        # Create historic forecasts and truth cubelists
        data = np.array([[1., 2.],
                         [3., 4.]], dtype=np.float32)
        frt_dt = datetime.datetime(2017, 11, 10, 0, 0)
        time_dt = datetime.datetime(2017, 11, 10, 4, 0)
        self.historic_forecasts = iris.cube.CubeList([])
        self.truth = iris.cube.CubeList([])
        for day in range(6):
            new_frt_dt = frt_dt + datetime.timedelta(days=day)
            new_time_dt = time_dt + datetime.timedelta(days=day)
            self.historic_forecasts.append(
                set_up_variable_cube(data, time=new_time_dt, frt=new_frt_dt,
                                     standard_grid_metadata="uk_ens"))
            self.truth.append(
                set_up_variable_cube(data, time=new_time_dt, frt=new_time_dt,
                                     standard_grid_metadata="uk_det"))

        self.combined = self.historic_forecasts + self.truth

        # Create the historic and truth cubes
        self.historic_forecasts_cube = self.historic_forecasts.merge_cube()
        self.truth_cube = self.truth.merge_cube()

        # Create historical forecasts and truth cubes and cubelists where
        # some items are missing.
        self.partial_historic_forecasts = (
            self.historic_forecasts[:2] + self.historic_forecasts[3:])
        self.partial_historic_forecasts_cube = (
            self.partial_historic_forecasts.merge_cube())
        self.partial_truth = self.truth[:2] + self.truth[3:]
        self.partial_truth_cube = self.partial_truth.merge_cube()

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


class Test__init__(SetupCubesAndDicts):
    """Test class initialisation"""

    def test_basic(self):
        """Test that the historic forecast and truth dictionaries are specified
        correctly."""
        self.assertEqual(
            self.plugin.historic_forecast_dict, self.historic_forecast_dict)
        self.assertEqual(self.plugin.truth_dict, self.truth_dict)


class Test__repr__(SetupCubesAndDicts):
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


class Test__find_required_cubes_using_metadata(SetupCubesAndDicts):
    """Test the _find_required_cubes_using_metadata method."""

    def test_attributes(self):
        """Test that the desired cube is returned if the required attributes
        are specified."""
        result = self.plugin._find_required_cubes_using_metadata(
            self.combined, self.historic_forecast_dict)
        self.assertEqual(result, self.historic_forecasts)

    def test_non_attributes(self):
        """Test that the if metadata other than attributes are specified then
        a NotImplementedError is raised."""
        input_dict = {
            "coord": "forecast_period==0"
        }
        msg = "only constraining on attributes is supported"
        with self.assertRaisesRegex(NotImplementedError, msg):
            self.plugin._find_required_cubes_using_metadata(
                self.combined, input_dict)

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


class Test__filter_non_matching_cubes(SetupCubesAndDicts):
    """Test the _filter_non_matching_cubes method."""

    def test_all_matching(self):
        """Test for when the historic forecast and truth cubes all match."""
        hf_result, truth_result = self.plugin._filter_non_matching_cubes(
            self.historic_forecasts, self.truth)
        self.assertEqual(hf_result, self.historic_forecasts)
        self.assertEqual(truth_result, self.truth)

    def test_fewer_historic_forecasts(self):
        """Test for when there are fewer historic forecasts than truths,
        for example, if there is a missing forecast cycle."""
        hf_result, truth_result = self.plugin._filter_non_matching_cubes(
            self.partial_historic_forecasts, self.truth)
        self.assertEqual(hf_result, self.partial_historic_forecasts)
        self.assertEqual(truth_result, self.partial_truth)

    def test_fewer_truths(self):
        """Test for when there are fewer truths than historic forecasts,
        for example, if there is a missing analysis."""
        hf_result, truth_result = self.plugin._filter_non_matching_cubes(
            self.historic_forecasts, self.partial_truth)
        self.assertEqual(hf_result, self.partial_historic_forecasts)
        self.assertEqual(truth_result, self.partial_truth)


class Test_process(SetupCubesAndDicts):
    """Test the process method."""

    def test_basic(self):
        """Test that the input cubelist combining historic forecasts and truth
        can be split using the metadata dictionaries provided."""
        hf_result, truth_result = self.plugin.process(self.combined)
        self.assertEqual(hf_result, self.historic_forecasts_cube)
        self.assertEqual(truth_result, self.truth_cube)

    def test_fewer_historic_forecasts(self):
        """Test that the input cubelist combining historic forecasts and truth
        can be split using the metadata dictionaries provided, when there are
        fewer historic forecasts than truths."""
        combined = self.partial_historic_forecasts + self.truth

        hf_result, truth_result = self.plugin.process(combined)
        self.assertEqual(hf_result, self.partial_historic_forecasts_cube)
        self.assertEqual(truth_result, self.partial_truth_cube)

    def test_fewer_truths(self):
        """Test that the input cubelist combining historic forecasts and truth
        can be split using the metadata dictionaries provided, when there are
        fewer truths than historic forecasts."""
        combined = self.historic_forecasts + self.partial_truth

        hf_result, truth_result = self.plugin.process(combined)
        self.assertEqual(hf_result, self.partial_historic_forecasts_cube)
        self.assertEqual(truth_result, self.partial_truth_cube)


if __name__ == '__main__':
    unittest.main()
