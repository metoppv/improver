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
"""Unit tests for the ApplyReliabilityCalibration plugin."""

import unittest

import iris
import numpy as np
from cf_units import Unit
from numpy.testing import assert_allclose, assert_array_equal

from improver.calibration.reliability_calibration import (
    ApplyReliabilityCalibration as Plugin,
)
from improver.calibration.reliability_calibration import (
    ConstructReliabilityCalibrationTables as CalPlugin,
)
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube
from improver.utilities.warnings_handler import ManageWarnings


class Test_ReliabilityCalibrate(unittest.TestCase):

    """Test class for the Test_ApplyReliabilityCalibration tests,
    setting up cubes to use as inputs."""

    def setUp(self):
        """Create reliability calibration table and forecast cubes for
        testing."""

        forecast_probabilities_0 = np.linspace(0.5, 1, 9, dtype=np.float32)
        forecast_probabilities_1 = np.linspace(0, 0.4, 9, dtype=np.float32)
        thresholds = [275.0, 280.0]
        forecast_probabilities = np.stack(
            [forecast_probabilities_0, forecast_probabilities_1]
        ).reshape((2, 3, 3))
        self.forecast = set_up_probability_cube(forecast_probabilities, thresholds)
        self.forecast_thresholds = iris.cube.CubeList()
        for forecast_slice in self.forecast.slices_over("air_temperature"):
            self.forecast_thresholds.append(forecast_slice)

        reliability_cube_format = CalPlugin()._create_reliability_table_cube(
            self.forecast[0], self.forecast.coord(var_name="threshold")
        )
        reliability_cube_format = reliability_cube_format.collapsed(
            [
                reliability_cube_format.coord(axis="x"),
                reliability_cube_format.coord(axis="y"),
            ],
            iris.analysis.SUM,
        )
        # Over forecasting exceeding 275K.
        reliability_data_0 = np.array(
            [
                [0, 0, 250, 500, 750],  # Observation count
                [0, 250, 500, 750, 1000],  # Sum of forecast probability
                [1000, 1000, 1000, 1000, 1000],  # Forecast count
            ],
            dtype=np.float32,
        )

        # Under forecasting exceeding 280K.
        reliability_data_1 = np.array(
            [
                [250, 500, 750, 1000, 1000],  # Observation count
                [0, 250, 500, 750, 1000],  # Sum of forecast probability
                [1000, 1000, 1000, 1000, 1000],  # Forecast count
            ],
            dtype=np.float32,
        )

        r0 = reliability_cube_format.copy(data=reliability_data_0)
        r1 = reliability_cube_format.copy(data=reliability_data_1)
        r1.replace_coord(self.forecast[1].coord(var_name="threshold"))

        self.reliability_cubelist = iris.cube.CubeList([r0, r1])
        self.reliability_cube = self.reliability_cubelist.merge_cube()

        self.threshold = self.forecast.coord(var_name="threshold")
        self.plugin = Plugin()
        self.plugin.threshold_coord = self.threshold


class Test__init__(unittest.TestCase):

    """Test the __init__ method."""

    def test_using_defaults(self):
        """Test without providing any arguments."""

        plugin = Plugin()
        self.assertIsNone(plugin.threshold_coord, None)


class Test__extract_matching_reliability_table(Test_ReliabilityCalibrate):

    """Test the _extract_matching_reliability_table method."""

    def test_matching_coords(self):
        """Test that no exception is raised in the case that the forecast
        and reliability table cubes have equivalent threshold coordinates."""

        result = self.plugin._extract_matching_reliability_table(
            self.forecast[0], self.reliability_cube
        )
        self.assertEqual(result.xml(), self.reliability_cube[0].xml())

    def test_matching_coords_cubelist(self):
        """Test that no exception is raised in the case that the forecast
        and reliability table cubes have equivalent threshold coordinates and
        the reliability_table is provided as a cubelist"""

        result = self.plugin._extract_matching_reliability_table(
            self.forecast[0], self.reliability_cubelist
        )
        self.assertEqual(result.xml(), self.reliability_cubelist[0].xml())

    def test_unmatching_coords(self):
        """Test that an exception is raised in the case that the forecast
        and reliability table cubes have different threshold coordinates."""

        msg = "No reliability table found to match threshold"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._extract_matching_reliability_table(
                self.forecast[0], self.reliability_cubelist[1]
            )


class Test__ensure_monotonicity_across_thresholds(Test_ReliabilityCalibrate):

    """Test the _ensure_monotonicity_across_thresholds method."""

    @ManageWarnings(record=True)
    def test_monotonic_case(self, warning_list=None):
        """Test that a probability cube in which the data is already
        ordered monotonically is unchanged by this method. Additionally, no
        warnings or exceptions should be raised."""

        expected = self.forecast.copy()

        self.plugin._ensure_monotonicity_across_thresholds(self.forecast)

        assert_array_equal(self.forecast.data, expected.data)
        self.assertFalse(warning_list)

    @ManageWarnings(record=True)
    def test_single_disordered_element(self, warning_list=None):
        """Test that if the values are disordered at a single position in the
        array, this position is sorted across the thresholds, whilst the rest
        of the array remains unchanged."""

        expected = self.forecast.copy()
        switch_val = self.forecast.data[0, 1, 1]
        self.forecast.data[0, 1, 1] = self.forecast.data[1, 1, 1]
        self.forecast.data[1, 1, 1] = switch_val

        self.plugin._ensure_monotonicity_across_thresholds(self.forecast)

        assert_array_equal(self.forecast.data, expected.data)
        warning_msg = "Exceedance probabilities are not decreasing"
        self.assertTrue(any(item.category == UserWarning for item in warning_list))
        self.assertTrue(any(warning_msg in str(item) for item in warning_list))

    @ManageWarnings(record=True)
    def test_monotonic_in_wrong_direction(self, warning_list=None):
        """Test that the data is reordered and a warning raised if the
        probabilities in the cube are non-monotonic in the sense defined by
        the relative_to_threshold attribute."""

        expected = self.forecast.copy(data=self.forecast.data[::-1])

        self.forecast.coord(self.threshold).attributes[
            "spp__relative_to_threshold"
        ] = "below"

        self.plugin._ensure_monotonicity_across_thresholds(self.forecast)

        assert_array_equal(self.forecast.data, expected.data)
        warning_msg = "Below threshold probabilities are not increasing"
        self.assertTrue(any(item.category == UserWarning for item in warning_list))
        self.assertTrue(any(warning_msg in str(item) for item in warning_list))

    def test_exception_without_relative_to_threshold(self):
        """Test that an exception is raised if the probability cube's
        threshold coordinate does not include an attribute declaring whether
        probabilities are above or below the threshold. This attribute is
        expected to be called spp__relative_to_threshold."""

        self.forecast.coord(self.threshold).attributes.pop("spp__relative_to_threshold")

        msg = "Cube threshold coordinate does not define whether"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._ensure_monotonicity_across_thresholds(self.forecast)


class Test__calculate_reliability_probabilities(Test_ReliabilityCalibrate):

    """Test the _calculate_reliability_probabilities method."""

    def test_values(self):
        """Test expected values are returned when two or more bins are
        available for interpolation."""

        expected_0 = (
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([0.0, 0.0, 0.25, 0.5, 0.75]),
        )
        expected_1 = (
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([0.25, 0.5, 0.75, 1.0, 1.0]),
        )
        plugin = Plugin()
        threshold_0 = plugin._calculate_reliability_probabilities(
            self.reliability_cube[0]
        )
        threshold_1 = plugin._calculate_reliability_probabilities(
            self.reliability_cube[1]
        )

        assert_array_equal(threshold_0, expected_0)
        assert_array_equal(threshold_1, expected_1)

    def test_fewer_than_two_bins(self):
        """Test that if fewer than two probability bins are provided, no
        calibration is applied."""
        reliability_cube = self.reliability_cube[0, :, 0]
        probability_bin_coord = iris.coords.DimCoord(
            np.array([0.5], dtype=np.float32),
            bounds=np.array([[0.0, 1.0]], dtype=np.float32),
            standard_name=None,
            units=Unit("1"),
            long_name="probability_bin",
        )
        reliability_cube.replace_coord(probability_bin_coord)

        reliability_cube.data = np.array(
            [
                5.0,  # Observation count
                5.0,  # Sum of forecast probability
                10.0,  # Forecast count
            ],
            dtype=np.float32,
        )

        result = self.plugin._calculate_reliability_probabilities(reliability_cube)

        self.assertIsNone(result[0])
        self.assertIsNone(result[1])


class Test__interpolate(unittest.TestCase):

    """Test the _interpolate method."""

    def setUp(self):

        """Set up data for testing the interpolate method."""

        self.reliability_probabilities = np.array([0.0, 0.4, 0.8])
        self.observation_frequencies = np.array([0.2, 0.6, 1.0])
        self.plugin = Plugin()

    def test_unmasked_data(self):
        """Test unmasked data is interpolated and returned as expected."""

        expected = np.array([0.4, 0.6, 0.8])
        forecast_threshold = np.array([0.2, 0.4, 0.6])

        result = self.plugin._interpolate(
            forecast_threshold,
            self.reliability_probabilities,
            self.observation_frequencies,
        )

        assert_allclose(result, expected)

    def test_masked_data(self):
        """Test masked data is interpolated and returned with the original
        mask in place."""

        expected = np.ma.masked_array([np.nan, 0.6, 0.8], mask=[1, 0, 0])
        forecast_threshold = np.ma.masked_array([np.nan, 0.4, 0.6], mask=[1, 0, 0])

        result = self.plugin._interpolate(
            forecast_threshold,
            self.reliability_probabilities,
            self.observation_frequencies,
        )

        assert_allclose(result, expected)

    def test_clipping(self):
        """Test the result, when using data constructed to cause extrapolation
        to a probability outside the range 0 to 1, is clipped. In this case
        an input probability of 0.9 would  return a calibrated probability
        of 1.1 in the absence of clipping."""

        expected = np.array([0.4, 0.6, 1.0])
        forecast_threshold = np.array([0.2, 0.4, 0.9])

        result = self.plugin._interpolate(
            forecast_threshold,
            self.reliability_probabilities,
            self.observation_frequencies,
        )

        assert_allclose(result, expected)

    def test_reshaping(self):
        """Test that the result has the same shape as the forecast_threshold
        input data."""

        expected = np.array([[0.2, 0.325, 0.45], [0.575, 0.7, 0.825], [0.95, 1.0, 1.0]])

        forecast_threshold = np.linspace(0, 1, 9).reshape((3, 3))

        result = self.plugin._interpolate(
            forecast_threshold,
            self.reliability_probabilities,
            self.observation_frequencies,
        )

        self.assertEqual(result.shape, expected.shape)
        assert_allclose(result, expected)


class Test_process(Test_ReliabilityCalibrate):

    """Test the process method."""

    @ManageWarnings(record=True)
    def test_calibrating_forecast_with_reliability_table_cube(self, warning_list=None):
        """Test application of the reliability table cube to the forecast. The
        input probabilities and table values have been chosen such that no
        warnings should be raised by this operation."""

        expected_0 = np.array(
            [[0.25, 0.3125, 0.375], [0.4375, 0.5, 0.5625], [0.625, 0.6875, 0.75]]
        )
        expected_1 = np.array([[0.25, 0.3, 0.35], [0.4, 0.45, 0.5], [0.55, 0.6, 0.65]])

        result = self.plugin.process(self.forecast, self.reliability_cube)

        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)
        self.assertFalse(warning_list)

    @ManageWarnings(record=True)
    def test_calibrating_forecast_with_reliability_table_cubelist(
        self, warning_list=None
    ):
        """Test application of a reliability table cubelist to the forecast.
        The input probabilities and table values have been chosen such that no
        warnings should be raised by this operation."""

        expected_0 = np.array(
            [[0.25, 0.3125, 0.375], [0.4375, 0.5, 0.5625], [0.625, 0.6875, 0.75]]
        )
        expected_1 = np.array([[0.25, 0.3, 0.35], [0.4, 0.45, 0.5], [0.55, 0.6, 0.65]])

        # swap order of cubes in reliabilty_cubelist to ensure order of
        # cubelist doesn't matter
        self.reliability_cubelist = iris.cube.CubeList(
            [self.reliability_cubelist[1], self.reliability_cubelist[0]]
        )
        result = self.plugin.process(self.forecast, self.reliability_cubelist)

        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)
        self.assertFalse(warning_list)

    @ManageWarnings(record=True)
    def test_one_threshold_uncalibrated(self, warning_list=None):
        """Test application of the reliability table to the forecast. In this
        case the reliability table has been altered for the first threshold
        (275K) such that it cannot be used. We expect the first threshold to
        be returned unchanged and a warning to be raised."""

        expected_0 = self.forecast[0].copy().data
        expected_1 = np.array([[0.25, 0.3, 0.35], [0.4, 0.45, 0.5], [0.55, 0.6, 0.65]])

        reliability_cube_0 = self.reliability_cubelist[0][:, 0]
        probability_bin_coord = iris.coords.DimCoord(
            np.array([0.5], dtype=np.float32),
            bounds=np.array([[0.0, 1.0]], dtype=np.float32),
            standard_name=None,
            units=Unit("1"),
            long_name="probability_bin",
        )
        reliability_cube_0.replace_coord(probability_bin_coord)

        reliability_cube_0.data = np.array(
            [
                5.0,  # Observation count
                5.0,  # Sum of forecast probability
                10.0,
            ],  # Forecast coun
            dtype=np.float32,
        )
        reliability_cubelist = iris.cube.CubeList(
            [reliability_cube_0, self.reliability_cubelist[1]]
        )
        result = self.plugin.process(self.forecast, reliability_cubelist)

        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)

        warning_msg = (
            "The following thresholds were not calibrated due to "
            "insufficient forecast counts in reliability table "
            "bins: [275.0]"
        )
        self.assertTrue(any(item.category == UserWarning for item in warning_list))
        self.assertTrue(any(warning_msg in str(item) for item in warning_list))

    @ManageWarnings(record=True)
    def test_calibrating_without_single_value_bins(self, warning_list=None):
        """Test application of the reliability table to the forecast. In this
        case the single_value_bins have been removed, requiring that that the
        results for some of the points are calculated using extrapolation. The
        input probabilities and table values have been chosen such that no
        warnings should be raised by this operation."""

        reliability_cube = self.reliability_cube[..., 1:4]
        crd = reliability_cube.coord("probability_bin")
        bounds = crd.bounds.copy()
        bounds[0, 0] = 0.0
        bounds[-1, -1] = 1.0

        new_crd = crd.copy(points=crd.points, bounds=bounds)
        reliability_cube.replace_coord(new_crd)

        expected_0 = np.array(
            [[0.25, 0.3125, 0.375], [0.4375, 0.5, 0.5625], [0.625, 0.6875, 0.75]]
        )
        expected_1 = np.array([[0.25, 0.3, 0.35], [0.4, 0.45, 0.5], [0.55, 0.6, 0.65]])

        result = self.plugin.process(self.forecast, reliability_cube)

        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)
        self.assertFalse(warning_list)

    def test_calibrating_point_by_point(self, warning_list=None):
        """Test application of reliability table to the forecast. In this case
        the forecast and reliability table have been altered so that point_by_point
        functionality can be tested."""

        y_name = self.forecast.coord(axis="y").name()
        x_name = self.forecast.coord(axis="x").name()

        # create reliability table in same format as that output from
        # ManipulateReliabilityTable when point_by_point=True
        reliability_table = self.reliability_cubelist.copy()
        reliability_cube_list = iris.cube.CubeList()
        for threshold_cube in reliability_table:
            threshold_cube.remove_coord(y_name)
            threshold_cube.remove_coord(x_name)
            threshold_list = iris.cube.CubeList()
            for forecast_point in self.forecast.slices_over([y_name, x_name]):
                reliability_cube_spatial = threshold_cube.copy()
                reliability_cube_spatial.add_aux_coord(forecast_point.coord(y_name))
                reliability_cube_spatial.add_aux_coord(forecast_point.coord(x_name))
                threshold_list.append(reliability_cube_spatial)
            reliability_cube_list.append(threshold_list)

        expected_0 = np.array(
            [[0.25, 0.3125, 0.375], [0.4375, 0.5, 0.5625], [0.625, 0.6875, 0.75]]
        )
        expected_1 = np.array([[0.25, 0.3, 0.35], [0.4, 0.45, 0.5], [0.55, 0.6, 0.65]])

        result = self.plugin.process(
            self.forecast, reliability_cube_list, point_by_point=True,
        )

        # check that data is as expected
        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)

        # check that coordinates match
        coords_table = [c.name() for c in self.forecast.coords()]
        coords_result = [c.name() for c in result.coords()]
        assert coords_table == coords_result

        self.assertFalse(warning_list)


if __name__ == "__main__":
    unittest.main()
