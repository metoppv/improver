# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
   weighted_blend.WeightedBlendAcrossWholeDimension plugin."""


import unittest
from datetime import datetime

import iris
import numpy as np
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.blending.weighted_blend import WeightedBlendAcrossWholeDimension
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.cube_manipulation import enforce_coordinate_ordering

from .test_PercentileBlendingAggregator import (
    BLENDED_PERCENTILE_DATA,
    BLENDED_PERCENTILE_DATA_EQUAL_WEIGHTS,
    BLENDED_PERCENTILE_DATA_SPATIAL_WEIGHTS,
    PERCENTILE_DATA,
)


def percentile_cube(frt_points, time, frt):
    """Create a percentile cube for testing."""
    cube = set_up_percentile_cube(
        np.zeros((6, 2, 2), dtype=np.float32),
        np.arange(0, 101, 20).astype(np.float32),
        name="air_temperature",
        units="C",
        time=time,
        frt=frt,
    )
    cube = add_coordinate(
        cube,
        frt_points,
        "forecast_reference_time",
        is_datetime=True,
        order=(1, 0, 2, 3),
    )
    cube.data = PERCENTILE_DATA
    return cube


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = WeightedBlendAcrossWholeDimension("time")
        self.assertEqual(plugin.blend_coord, "time")

    def test_threshold_blending_unsupported(self):
        """Test that the __init__ raises an error if trying to blend over
        thresholds (which is neither supported nor sensible)."""
        msg = "Blending over thresholds is not supported"
        with self.assertRaisesRegex(ValueError, msg):
            WeightedBlendAcrossWholeDimension("threshold")


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeightedBlendAcrossWholeDimension("time"))
        msg = "<WeightedBlendAcrossWholeDimension: coord = time, timeblending: False>"
        self.assertEqual(result, msg)


class Test_weighted_blend(IrisTest):

    """A shared setup for tests in the WeightedBlendAcrossWholeDimension
    plugin."""

    def setUp(self):
        """Create data cubes and weights for testing"""
        self.coord = "forecast_reference_time"
        self.plugin = WeightedBlendAcrossWholeDimension(self.coord)

        frt_points = [
            datetime(2015, 11, 19, 0),
            datetime(2015, 11, 19, 1),
            datetime(2015, 11, 19, 2),
        ]

        cube = set_up_variable_cube(
            np.zeros((2, 2), dtype=np.float32),
            name="precipitation_amount",
            units="kg m^-2 s^-1",
            time=datetime(2015, 11, 19, 2),
            frt=datetime(2015, 11, 19, 0),
            standard_grid_metadata="gl_det",
            attributes={"title": "Operational ENGL Model Forecast"},
        )
        self.cube = add_coordinate(
            cube, frt_points, "forecast_reference_time", is_datetime=True
        )
        self.cube.data[0][:][:] = 1.0
        self.cube.data[1][:][:] = 2.0
        self.cube.data[2][:][:] = 3.0

        cube_threshold = set_up_probability_cube(
            np.zeros((2, 2, 2), dtype=np.float32),
            np.array([0.4, 1], dtype=np.float32),
            variable_name="precipitation_amount",
            threshold_units="kg m^-2 s^-1",
            time=datetime(2015, 11, 19, 2),
            frt=datetime(2015, 11, 19, 0),
            standard_grid_metadata="gl_det",
            attributes={"title": "Operational ENGL Model Forecast"},
        )

        self.cube_threshold = add_coordinate(
            cube_threshold,
            frt_points,
            "forecast_reference_time",
            is_datetime=True,
            order=(1, 0, 2, 3),
        )
        self.cube_threshold.data[0, 0, :, :] = 0.2
        self.cube_threshold.data[0, 1, :, :] = 0.4
        self.cube_threshold.data[0, 2, :, :] = 0.6
        self.cube_threshold.data[1, 0, :, :] = 0.4
        self.cube_threshold.data[1, 1, :, :] = 0.6
        self.cube_threshold.data[1, 2, :, :] = 0.8

        # percentile cube dimensions that would be input to process
        self.perc_cube = percentile_cube(
            frt_points, datetime(2015, 11, 19, 2), datetime(2015, 11, 19, 0)
        )
        # plugin internals assume leading blend coord (enforced in process)
        self.reordered_perc_cube = self.perc_cube.copy()
        enforce_coordinate_ordering(self.reordered_perc_cube, [self.coord])

        # Weights cubes
        # 3D varying in space and forecast reference time.
        weights3d = np.array(
            [
                [[0.1, 0.3], [0.2, 0.4]],
                [[0.1, 0.3], [0.2, 0.4]],
                [[0.8, 0.4], [0.6, 0.2]],
            ],
            dtype=np.float32,
        )
        self.weights3d = self.cube.copy(data=weights3d)
        self.weights3d.rename("weights")
        self.weights3d.units = "no_unit"
        self.weights3d.attributes = {}

        # 1D varying with forecast reference time.
        weights1d = np.array([0.6, 0.3, 0.1], dtype=np.float32)
        self.weights1d = self.weights3d[:, 0, 0].copy(data=weights1d)
        self.weights1d.remove_coord("latitude")
        self.weights1d.remove_coord("longitude")


class Test_check_percentile_coord(Test_weighted_blend):

    """Test the percentile coord checking function."""

    def test_basic(self):
        """Tests the basic use of check_percentile_coord"""
        cube = self.perc_cube[:2]
        self.assertTrue(WeightedBlendAcrossWholeDimension.check_percentile_coord(cube))

    def test_fails_perc_coord_not_dim(self):
        """Test it raises a Value Error if percentile coord not a dim."""
        new_cube = self.cube.copy()
        new_cube.add_aux_coord(AuxCoord([10.0], long_name="percentile"))
        msg = "The percentile coord must be a dimension of the cube."
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.check_percentile_coord(new_cube)

    def test_fails_only_one_percentile_value(self):
        """Test it raises a Value Error if there is only one percentile."""
        new_cube = Cube([[0.0]])
        new_cube.add_dim_coord(DimCoord([10.0], long_name="percentile"), 0)
        new_cube.add_dim_coord(DimCoord([10.0], long_name="forecast_reference_time"), 1)
        msg = (
            "Percentile coordinate does not have enough points"
            " in order to blend. Must have at least 2 percentiles."
        )
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.check_percentile_coord(new_cube)


class Test_check_compatible_time_points(Test_weighted_blend):

    """Test the time point compatibility checking function."""

    def test_basic(self):
        """Test that no errors are raised when calling check_compatible_time_point"""
        plugin = WeightedBlendAcrossWholeDimension("time")
        plugin.check_compatible_time_points(self.cube)

    def test_unmatched_validity_time_exception(self):
        """Test that a ValueError is raised if the validity time of the
        slices over the blending coordinate differ. We should only be blending
        data at equivalent times unless we are triangular time blending."""
        self.cube.remove_coord("time")
        self.cube.coord("forecast_reference_time").rename("time")
        cube = self.cube[:2]
        plugin = WeightedBlendAcrossWholeDimension(self.coord)
        msg = "Attempting to blend data for different validity times."
        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_compatible_time_points(cube)

    def test_unmatched_validity_time_exemption(self):
        """Test that no ValueError is raised for unmatched validity times if
        we use the timeblending=True flag. As long as no exception is raised
        this test passes."""
        self.cube.remove_coord("time")
        self.cube.coord("forecast_reference_time").rename("time")
        plugin = WeightedBlendAcrossWholeDimension(self.coord, timeblending=True)
        plugin.check_compatible_time_points(self.cube)


class Test_shape_weights(Test_weighted_blend):

    """Test the shape weights function is able to create a valid a set of
    weights, or raises an error."""

    def test_1D_weights_3D_cube_weighted_mean(self):
        """Test a 1D cube of weights results in a 3D array of weights of the
        same shape as the data cube."""
        result = self.plugin.shape_weights(self.cube, self.weights1d)
        self.assertEqual(self.cube.shape, result.shape)
        self.assertArrayEqual(self.weights1d.data, result[:, 0, 0])

    def test_3D_weights_3D_cube_weighted_mean(self):
        """Test a 3D cube of weights results in a 3D array of weights of the
        same shape as the data cube."""
        result = self.plugin.shape_weights(self.cube, self.weights3d)
        self.assertEqual(self.cube.shape, result.shape)
        self.assertArrayEqual(self.weights3d.data, result)

    def test_3D_weights_3D_cube_weighted_mean_wrong_order(self):
        """Test a 3D cube of weights results in a 3D array of weights of the
        same shape as the data cube. In this test the weights cube has the
        same coordinates but slightly differently ordered. These should be
        reordered to match the cube."""
        expected = self.weights3d.copy().data
        self.weights3d.transpose([1, 0, 2])
        result = self.plugin.shape_weights(self.cube, self.weights3d)
        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(expected.data, result)

    def test_3D_weights_4D_cube_weighted_mean_wrong_order(self):
        """Test a 4D cube of weights results in a 3D array of weights of the
        same shape as the data cube. In this test the input cube has the
        same coordinates but slightly differently ordered. The weights cube
        should be reordered to match the cube."""
        # Add a new axis to input cube to make it 4D
        cube = iris.util.new_axis(self.cube, scalar_coord="time")
        cube.transpose([3, 2, 0, 1])
        # Create an expected array which has been transposed to match
        # input cube with the extra axis added.
        expected = self.weights3d.copy()
        expected.transpose([2, 1, 0])
        expected = np.expand_dims(expected.data, axis=2)
        result = self.plugin.shape_weights(cube, self.weights3d)
        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(expected, result)

    def test_3D_weights_3D_cube_weighted_mean_unmatched_coordinate(self):
        """Test a 3D cube of weights results in a 3D array of weights of the
        same shape as the data cube. In this test the weights cube has the
        same shape but different coordinates to the diagnostic cube, raising
        a ValueError as a result."""
        weights = self.weights3d.copy()
        weights.coord("longitude").rename("projection_x_coordinate")
        msg = (
            "projection_x_coordinate is a coordinate on the weights cube "
            "but it is not found on the cube we are trying to collapse."
        )
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.shape_weights(self.cube, weights)

    def test_incompatible_weights_and_data_cubes(self):
        """Test an exception is raised if the weights cube and the data cube
        have incompatible coordinates."""
        self.weights1d.coord(self.coord).rename("threshold")
        msg = (
            "threshold is a coordinate on the weights cube but it "
            "is not found on the cube we are trying to collapse."
        )
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.shape_weights(self.cube, self.weights1d)

    def test_incompatible_weights_and_data_cubes_shape(self):
        """Test an exception is raised if the weights cube and the data cube
        have incompatible shapes."""
        msg = "Weights cube is not a compatible shape"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.shape_weights(self.cube[:1], self.weights1d)


class Test_get_weights_array(Test_weighted_blend):

    """Test the get_weights_array function."""

    def test_no_weights_cube_provided(self):
        """Test that if a weights cube is not provided, the function generates
        a weights array that will equally weight all fields along the blending
        coordinate."""
        result = self.plugin.get_weights_array(self.cube, None)
        (blending_coord_length,) = self.cube.coord(self.coord).shape
        expected = (np.ones(blending_coord_length) / blending_coord_length).astype(
            np.float32
        )
        self.assertEqual(self.cube.shape, result.shape)
        self.assertArrayEqual(expected, result[:, 0, 0])

    def test_1D_weights(self):
        """Test a 1D cube of weights results in a 3D array of weights of an
        appropriate shape."""
        expected = np.empty_like(self.cube.data)
        result = self.plugin.get_weights_array(self.cube, self.weights1d)
        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(self.weights1d.data, result[:, 0, 0])

    def test_3D_weights(self):
        """Test a 3D cube of weights results in a 3D array of weights of the
        same shape as the data cube."""
        expected = np.empty_like(self.cube.data)
        result = self.plugin.get_weights_array(self.cube, self.weights3d)
        self.assertEqual(expected.shape, result.shape)
        self.assertArrayEqual(self.weights3d.data, result[:, :, :])


class Test__normalise_weights(Test_weighted_blend):

    """Test the _normalise_weights function."""

    def test_noop(self):
        """Test the function has no impact if a weights cube is provided
        which is properly normalised, i.e. the weights sum to one over the
        leading blend dimension."""
        expected_data = self.weights3d.data.copy()
        result = self.plugin._normalise_weights(self.weights3d.data)
        self.assertArrayAlmostEqual(result, expected_data)

    def test_noop_with_zero(self):
        """Test the function has no impact if a weights cube is provided
        which is either properly normalised or where the weights sum to zero."""
        weights = np.array(
            [[[0, 0.3], [0.2, 0.4]], [[0, 0.3], [0.2, 0.4]], [[0, 0.4], [0.6, 0.2]]]
        )
        expected_data = weights.copy()
        result = self.plugin._normalise_weights(weights)
        self.assertArrayAlmostEqual(result, expected_data)

    def test_normalisation(self):
        """Test that if a weights cube is provided which is not properly
        normalised, the output weights are normalised."""
        weights = self.weights3d.data
        weights[0, 0, 1] = 0.9
        expected_data = np.array(
            [
                [[0.1, 0.5625], [0.2, 0.4]],
                [[0.1, 0.1875], [0.2, 0.4]],
                [[0.8, 0.25], [0.6, 0.2]],
            ],
        )
        result = self.plugin._normalise_weights(weights)
        self.assertArrayAlmostEqual(result, expected_data)


class Test_percentile_weighted_mean(Test_weighted_blend):

    """Test the percentile_weighted_mean function."""

    def test_with_weights(self):
        """Test function when a data cube and a weights cube are provided."""
        result = self.plugin.percentile_weighted_mean(
            self.reordered_perc_cube, self.weights1d
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, BLENDED_PERCENTILE_DATA)

    def test_with_spatially_varying_weights(self):
        """Test function when a data cube and a multi dimensional weights cube
        are provided. This tests spatially varying weights, where each x-y
        position is weighted differently in each slice along the blending
        coordinate."""
        result = self.plugin.percentile_weighted_mean(
            self.reordered_perc_cube, self.weights3d
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.data, BLENDED_PERCENTILE_DATA_SPATIAL_WEIGHTS
        )

    def test_without_weights(self):
        """Test function when a data cube is provided, but no weights cube
        which should result in equal weightings."""
        result = self.plugin.percentile_weighted_mean(self.reordered_perc_cube, None)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, BLENDED_PERCENTILE_DATA_EQUAL_WEIGHTS)


class Test_weighted_mean(Test_weighted_blend):

    """Test the weighted_mean function."""

    def test_with_weights(self):
        """Test function when a data cube and a weights cube are provided."""
        result = self.plugin.weighted_mean(self.cube, self.weights1d)
        expected = np.full((2, 2), 1.5)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_with_spatially_varying_weights(self):
        """Test function when a data cube and a multi dimensional weights cube
        are provided. This tests spatially varying weights, where each x-y
        position is weighted differently in each slice along the blending
        coordinate."""
        result = self.plugin.weighted_mean(self.cube, self.weights3d)
        expected = np.array([[2.7, 2.1], [2.4, 1.8]])

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_without_weights(self):
        """Test function when a data cube is provided, but no weights cube
        which should result in equal weightings."""
        result = self.plugin.weighted_mean(self.cube, None)
        expected = np.full((2, 2), 2.0)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_wind_directions(self):
        """Test function when a wind direction data cube is provided, and
        the directions don't cross the 0/360° boundary."""
        frt_points = [
            datetime(2015, 11, 19, 0),
            datetime(2015, 11, 19, 1),
        ]
        cube = set_up_variable_cube(
            np.zeros((2, 2), dtype=np.float32),
            name="wind_from_direction",
            units="degrees",
            time=datetime(2015, 11, 19, 2),
            frt=datetime(2015, 11, 19, 0),
            standard_grid_metadata="gl_det",
            attributes={"title": "Operational ENGL Model Forecast"},
        )
        cube = add_coordinate(
            cube, frt_points, "forecast_reference_time", is_datetime=True
        )
        cube.data[0] = 10.0
        cube.data[1] = 30.0
        expected = np.full((2, 2), 20.0)
        result = self.plugin.weighted_mean(cube, weights=None)
        self.assertArrayAlmostEqual(result.data, expected, decimal=4)

    def test_wind_directions_over_north(self):
        """Test function when a wind direction data cube is provided, and
        the directions cross the 0/360° boundary."""
        frt_points = [
            datetime(2015, 11, 19, 0),
            datetime(2015, 11, 19, 1),
        ]
        cube = set_up_variable_cube(
            np.zeros((2, 2), dtype=np.float32),
            name="wind_direction",
            units="degrees",
            time=datetime(2015, 11, 19, 2),
            frt=datetime(2015, 11, 19, 0),
            standard_grid_metadata="gl_det",
            attributes={"title": "Operational ENGL Model Forecast"},
        )
        cube = add_coordinate(
            cube, frt_points, "forecast_reference_time", is_datetime=True
        )
        cube.data[0] = 350.0
        cube.data[1] = 30.0
        expected = np.full((2, 2), 10.0)
        result = self.plugin.weighted_mean(cube, weights=None)
        self.assertArrayAlmostEqual(result.data, expected, decimal=4)

    def test_collapse_dims_with_weights(self):
        """Test function matches when the blend coordinate is first or second."""
        # Create a new axis.
        new_cube = add_coordinate(self.cube, [0.5], "height", coord_units="m")
        new_cube = iris.util.new_axis(new_cube, "height")
        order = np.array([1, 0, 2, 3])
        new_cube.transpose(order)
        expected = np.full((2, 2), 1.5)
        result_blend_coord_first = self.plugin.weighted_mean(new_cube, self.weights1d)
        self.assertIsInstance(result_blend_coord_first, iris.cube.Cube)
        self.assertArrayAlmostEqual(result_blend_coord_first.data, expected)


class Test_process(Test_weighted_blend):

    """Test the process method."""

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube with appropriate metadata"""
        expected_dim_coords = [
            coord.name() for coord in self.cube.coords(dim_coords=True)
        ]
        expected_dim_coords.remove(self.coord)
        expected_scalar_coords = {
            coord.name() for coord in self.cube.coords(dim_coords=False)
        }
        expected_scalar_coords.update({self.coord, self.coord})

        result = self.plugin(self.cube)

        dim_coords = [coord.name() for coord in result.coords(dim_coords=True)]
        aux_coords = {coord.name() for coord in result.coords(dim_coords=False)}

        self.assertIsInstance(result, Cube)
        self.assertSequenceEqual(dim_coords, expected_dim_coords)
        self.assertSetEqual(aux_coords, expected_scalar_coords)

    def test_perc(self):
        """Test that the plugin returns a percentile cube"""
        self.perc_cube.attributes = self.cube.attributes
        result = self.plugin(self.perc_cube)
        self.assertIn("percentile", [x.name() for x in result.dim_coords])

    def test_fails_coord_not_in_cube(self):
        """Test it raises CoordinateNotFoundError if the blending coord is not
        found in the cube."""
        coord = "notset"
        plugin = WeightedBlendAcrossWholeDimension(coord)
        msg = "Coordinate to be collapsed not found in cube."
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            plugin(self.cube)

    def test_fails_coord_not_in_weights_cube(self):
        """Test it raises CoordinateNotFoundError if the blending coord is not
        found in the weights cube."""
        self.weights1d.remove_coord(self.coord)
        msg = "Coordinate to be collapsed not found in weights cube."
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            self.plugin(self.cube, self.weights1d)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Type Error if not supplied with a cube."""
        notacube = 0.0
        msg = "The first argument must be an instance of iris.cube.Cube"
        with self.assertRaisesRegex(TypeError, msg):
            self.plugin(notacube)

    def test_scalar_coord(self):
        """Test plugin throws an error if trying to blending across a scalar
        coordinate."""
        coord = "dummy_scalar_coord"
        new_scalar_coord = AuxCoord(1, long_name=coord, units="no_unit")
        self.cube.add_aux_coord(new_scalar_coord)
        plugin = WeightedBlendAcrossWholeDimension(coord)
        weights = [1.0]
        msg = "has no associated dimension"
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.cube, weights)

    def test_threshold_cube_with_weights_weighted_mean(self):
        """Test weighted_mean method works collapsing a cube with a threshold
        dimension when the blending is over a different coordinate. Note that
        this test is in process to include the slicing."""
        result = self.plugin(self.cube_threshold, self.weights1d)
        expected_result_array = np.ones((2, 2, 2)) * 0.3
        expected_result_array[1, :, :] = 0.5
        self.assertArrayAlmostEqual(result.data, expected_result_array)


if __name__ == "__main__":
    unittest.main()
