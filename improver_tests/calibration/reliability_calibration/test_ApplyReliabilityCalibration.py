# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ApplyReliabilityCalibration plugin."""

import datetime
import unittest
import warnings

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.cube import Cube, CubeList
from numpy.testing import assert_allclose, assert_array_equal

from improver.calibration.reliability_calibration import (
    ApplyReliabilityCalibration as Plugin,
)
from improver.calibration.reliability_calibration import (
    ConstructReliabilityCalibrationTables as CalPlugin,
)
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    construct_scalar_time_coords,
    set_up_probability_cube,
)


def create_point_by_point_reliability_table(
    forecast: Cube, reliability_table: Cube
) -> CubeList:
    """
    Duplicate the input reliability table for each spatial point in the input forecast

    Args:
        forecast:
            The forecast to be calibrated.
        reliability_table:
            The reliability table to duplicate

    Returns:
        CubeList with each cube a copy of the input reliability table with spatial
        coordinates added corresponding to one of the spatial points in the input
        cube
    """
    y_name = forecast.coord(axis="y").name()
    x_name = forecast.coord(axis="x").name()

    # create reliability table in same format as that output from
    # ManipulateReliabilityTable when point_by_point=True
    reliability_cube_list = iris.cube.CubeList()
    for threshold_cube in reliability_table:
        threshold_cube.remove_coord(y_name)
        threshold_cube.remove_coord(x_name)
        for forecast_point in forecast.slices_over([y_name, x_name]):
            reliability_cube_spatial = threshold_cube.copy()
            reliability_cube_spatial.add_aux_coord(forecast_point.coord(y_name))
            reliability_cube_spatial.add_aux_coord(forecast_point.coord(x_name))
            reliability_cube_list.append(reliability_cube_spatial)

    return reliability_cube_list


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

        # create spot data forecast cube
        altitude = np.array([10, 20, 30])
        latitude = np.linspace(58.0, 59, 3)
        longitude = np.linspace(-0.25, 0.25, 3)
        wmo_id = ["03001", "03002", "03003"]
        i = -1
        forecast_spot_cubes = iris.cube.CubeList()
        for threshold in thresholds:
            i = i + 1
            threshold_coord = [
                self.forecast.coord("air_temperature").copy(
                    points=np.array(threshold, dtype="float32")
                )
            ]
            time_coords = construct_scalar_time_coords(
                datetime.datetime(2017, 11, 1, 4, 0),
                None,
                datetime.datetime(2017, 11, 1, 0, 0),
            )
            time_coords = [t[0] for t in time_coords]
            forecast_spot_cubes.append(
                build_spotdata_cube(
                    forecast_probabilities[i, :, 0],
                    "probability_of_air_temperature_above_threshold",
                    "1",
                    altitude,
                    latitude,
                    longitude,
                    wmo_id,
                    scalar_coords=time_coords + threshold_coord,
                )
            )
        self.forecast_spot_cube = forecast_spot_cubes.merge_cube()

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
        self.plugin_point_by_point = Plugin(point_by_point=True)
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

    def test_monotonic_case(self):
        """Test that a probability cube in which the data is already
        ordered monotonically is unchanged by this method. Additionally, no
        warnings or exceptions should be raised."""

        expected = self.forecast.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.plugin._ensure_monotonicity_across_thresholds(self.forecast)

        assert_array_equal(self.forecast.data, expected.data)

    def test_single_disordered_element(self):
        """Test that if the values are disordered at a single position in the
        array, this position is sorted across the thresholds, whilst the rest
        of the array remains unchanged."""

        expected = self.forecast.copy()
        switch_val = self.forecast.data[0, 1, 1]
        self.forecast.data[0, 1, 1] = self.forecast.data[1, 1, 1]
        self.forecast.data[1, 1, 1] = switch_val
        warning_msg = "Exceedance probabilities are not decreasing"

        with pytest.warns(UserWarning, match=warning_msg):
            self.plugin._ensure_monotonicity_across_thresholds(self.forecast)

        assert_array_equal(self.forecast.data, expected.data)

    def test_monotonic_in_wrong_direction(self):
        """Test that the data is reordered and a warning raised if the
        probabilities in the cube are non-monotonic in the sense defined by
        the relative_to_threshold attribute."""

        expected = self.forecast.copy(data=self.forecast.data[::-1])
        self.forecast.coord(self.threshold).attributes["spp__relative_to_threshold"] = (
            "below"
        )
        warning_msg = "Below threshold probabilities are not increasing"

        with pytest.warns(UserWarning, match=warning_msg):
            self.plugin._ensure_monotonicity_across_thresholds(self.forecast)

        assert_array_equal(self.forecast.data, expected.data)

    def test_exception_without_relative_to_threshold(self):
        """Test that an exception is raised if the probability cube's
        threshold coordinate does not include an attribute declaring whether
        probabilities are above or below the threshold. This attribute is
        expected to be called spp__relative_to_threshold."""

        self.forecast.coord(self.threshold).attributes.pop("spp__relative_to_threshold")

        msg = "Cube threshold coordinate does not define whether"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._ensure_monotonicity_across_thresholds(self.forecast)

    def test_single_threshold(self):
        """Test on a probability cube with only a single threshold.
        The data should be unchanged as it is already monotonic.
        Additionally no warnings or exceptions should be raised."""

        expected = self.forecast[0].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.plugin._ensure_monotonicity_across_thresholds(self.forecast[0])

        assert_array_equal(self.forecast[0].data, expected.data)


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

    def test_calibrating_forecast_with_reliability_table_cube(self):
        """Test application of the reliability table cube to the forecast. The
        input probabilities and table values have been chosen such that no
        warnings should be raised by this operation."""

        expected_0 = np.array(
            [[0.25, 0.3125, 0.375], [0.4375, 0.5, 0.5625], [0.625, 0.6875, 0.75]]
        )
        expected_1 = np.array([[0.25, 0.3, 0.35], [0.4, 0.45, 0.5], [0.55, 0.6, 0.65]])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = self.plugin.process(self.forecast, self.reliability_cube)

        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)

    def test_calibrating_forecast_with_reliability_table_cubelist(self):
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
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = self.plugin.process(self.forecast, self.reliability_cubelist)

        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)

    def test_one_threshold_uncalibrated(self):
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
        warning_msg = (
            "The following thresholds were not calibrated due to "
            "insufficient forecast counts in reliability table "
            "bins: \\[275.0\\]"
        )
        with pytest.warns(UserWarning, match=warning_msg):
            result = self.plugin.process(self.forecast, reliability_cubelist)

        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)

    def test_calibrating_without_single_value_bins(self):
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

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = self.plugin.process(self.forecast, reliability_cube)

        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)

    def test_calibrating_point_by_point(self):
        """Test application of reliability table to the forecast. In this case
        the forecast and reliability table have been altered so that point_by_point
        functionality can be tested."""

        # add additional auxiliary coordinate to test specific handling of auxiliary
        # coordinates implemented as part of point_by_point functionality
        test_forecast = self.forecast.copy()
        x_name = test_forecast.coord(axis="x").name()
        test_coord = iris.coords.AuxCoord(
            points=np.arange(len(test_forecast.coord(x_name).points)),
            long_name="test_coord",
        )
        test_forecast.add_aux_coord(test_coord, data_dims=1)

        reliability_cube_list = create_point_by_point_reliability_table(
            test_forecast, self.reliability_cubelist
        )

        expected_0 = np.array(
            [[0.25, 0.3125, 0.375], [0.4375, 0.5, 0.5625], [0.625, 0.6875, 0.75]]
        )
        expected_1 = np.array([[0.25, 0.3, 0.35], [0.4, 0.45, 0.5], [0.55, 0.6, 0.65]])

        result = self.plugin_point_by_point.process(
            test_forecast, reliability_cube_list
        )

        # check that data is as expected
        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)

        # check that coordinates match
        coords_table = [c.name() for c in test_forecast.coords()]
        coords_result = [c.name() for c in result.coords()]
        assert coords_table == coords_result

    def test_calibrating_spot(self):
        """Test application of reliability table to spot forecasts."""

        expected_0 = [0.25, 0.4375, 0.625]
        expected_1 = [0.25, 0.4, 0.55]

        result = self.plugin.process(self.forecast_spot_cube, self.reliability_cube)

        # check that data is as expected
        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)

        # check that coordinates match
        coords_table = [c.name() for c in self.forecast_spot_cube.coords()]
        coords_result = [c.name() for c in result.coords()]
        assert coords_table == coords_result

    def test_calibrating_spot_point_by_point(self):
        """Test application of reliability table to spot forecasts using
        point_by_point functionality."""

        reliability_cube_list = create_point_by_point_reliability_table(
            self.forecast_spot_cube, self.reliability_cubelist
        )

        expected_0 = [0.25, 0.4375, 0.625]
        expected_1 = [0.25, 0.4, 0.55]

        result = self.plugin_point_by_point.process(
            self.forecast_spot_cube, reliability_cube_list
        )

        # check that data is as expected
        assert_allclose(result[0].data, expected_0)
        assert_allclose(result[1].data, expected_1)

        # check that coordinates match
        coords_table = [c.name() for c in self.forecast_spot_cube.coords()]
        coords_result = [c.name() for c in result.coords()]
        assert coords_table == coords_result

    def test_calibrating_forecast_single_threshold(self):
        """Test application of reliability tables on a probability cube
        that only contains a single threshold."""

        expected_0 = np.array(
            [[0.25, 0.3125, 0.375], [0.4375, 0.5, 0.5625], [0.625, 0.6875, 0.75]]
        )
        expected_1 = np.array(
            [[0.25, 0.3, 0.35], [0.4, 0.45, 0.5], [0.55, 0.6, 0.65]]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result_0 = self.plugin.process(self.forecast[0], self.reliability_cube)
            result_1 = self.plugin.process(self.forecast[1], self.reliability_cube)

        assert_allclose(result_0.data, expected_0)
        assert_allclose(result_1.data, expected_1)


if __name__ == "__main__":
    unittest.main()
