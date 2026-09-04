# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the wind_direction.WindDirection plugin."""

import unittest

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.wind_calculations.wind_direction import WindDirection, deg_to_complex

# Data to test complex/degree handling functions.
# Complex angles equivalent to np.arange(0., 360, 10) degrees.
COMPLEX_ANGLES = np.array(
    [
        1.0 + 0j,
        0.984807753 + 0.173648178j,
        0.939692621 + 0.342020143j,
        0.866025404 + 0.5j,
        0.766044443 + 0.642787610j,
        0.642787610 + 0.766044443j,
        0.5 + 0.866025404j,
        0.342020143 + 0.939692621j,
        0.173648178 + 0.984807753j,
        0.0 + 1.0j,
        -0.173648178 + 0.984807753j,
        -0.342020143 + 0.939692621j,
        -0.5 + 0.866025404j,
        -0.642787610 + 0.766044443j,
        -0.766044443 + 0.642787610j,
        -0.866025404 + 0.5j,
        -0.939692621 + 0.342020143j,
        -0.984807753 + 0.173648178j,
        -1.0 + 0.0j,
        -0.984807753 - 0.173648178j,
        -0.939692621 - 0.342020143j,
        -0.866025404 - 0.5j,
        -0.766044443 - 0.642787610j,
        -0.642787610 - 0.766044443j,
        -0.5 - 0.866025404j,
        -0.342020143 - 0.939692621j,
        -0.173648178 - 0.984807753j,
        -0.0 - 1.0j,
        0.173648178 - 0.984807753j,
        0.342020143 - 0.939692621j,
        0.5 - 0.866025404j,
        0.642787610 - 0.766044443j,
        0.766044443 - 0.642787610j,
        0.866025404 - 0.5j,
        0.939692621 - 0.342020143j,
        0.984807753 - 0.173648178j,
    ]
)

# Data to test the ensemble averaging codes.
WIND_DIR_COMPLEX = np.array(
    [
        [
            [6.12323400e-17 + 1.0j, 0.642787610 + 0.76604444j],
            [-1.83697020e-16 - 1.0j, 0.984807753 - 0.17364818j],
        ],
        [
            [-1.83697020e-16 - 1.0j, 0.5 + 0.8660254j],
            [0.342020143 - 0.93969262j, 0.984807753 + 0.17364818j],
        ],
    ]
)


def make_wdir_cube_534():
    """Make a 5x3x4 wind direction cube for testing this plugin"""
    data = np.array(
        [
            [
                [170.0, 50.0, 90.0, 90.0],
                [170.0, 170.0, 47.0, 350.0],
                [10.0, 309.0, 10.0, 10.0],
            ],
            [
                [170.0, 50.0, 90.0, 90.0],
                [170.0, 170.0, 47.0, 47.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
            [
                [10.0, 50.0, 90.0, 90.0],
                [170.0, 170.0, 47.0, 47.0],
                [310.0, 309.0, 10.0, 10.0],
            ],
            [
                [190.0, 40.0, 270.0, 90.0],
                [170.0, 170.0, 47.0, 47.0],
                [310.0, 309.0, 10.0, 10.0],
            ],
            [
                [190.0, 40.0, 270.0, 270.0],
                [170.0, 170.0, 47.0, 47.0],
                [310.0, 309.0, 10.0, 10.0],
            ],
        ],
        dtype=np.float32,
    )

    cube = set_up_variable_cube(
        data, name="wind_from_direction", units="degrees", spatial_grid="equalarea"
    )

    return cube


def make_wdir_cube_222():
    """Make a 2x2x2 wind direction cube for testing this plugin"""
    data = np.array(
        [[[90.0, 50.0], [270.0, 350.0]], [[270.0, 60.0], [290.0, 10.0]]],
        dtype=np.float32,
    )
    cube = set_up_variable_cube(
        data, name="wind_from_direction", units="degrees", spatial_grid="equalarea"
    )
    return cube


def pad_wdir_cube_222():
    """Make a padded wind direction cube using the same data as make_wdir_cube().
    Original data: 2x2x2; padded data 2x10x10"""
    data = np.array(
        [[[90.0, 50.0], [270.0, 350.0]], [[270.0, 60.0], [290.0, 10.0]]],
        dtype=np.float32,
    )
    padded_data = np.pad(
        data, ((0, 0), (4, 4), (4, 4)), "constant", constant_values=(0.0, 0.0)
    )
    cube = set_up_variable_cube(
        padded_data.astype(np.float32),
        name="wind_from_direction",
        units="degrees",
        spatial_grid="equalarea",
    )
    cube.coord(axis="x").points = np.arange(-50000.0, -31000.0, 2000.0)
    cube.coord(axis="y").points = np.arange(0.0, 19000.0, 2000.0)

    return cube


class Test__init__(unittest.TestCase):
    """Test the init method."""

    def test_basic(self):
        """Test that the __init__ does not fail."""
        result = WindDirection()
        self.assertIsInstance(result, WindDirection)

    def test_backup_method(self):
        """Test that the __init__ accepts this keyword."""
        result = WindDirection(backup_method="neighbourhood")
        self.assertIsInstance(result, WindDirection)

    def test_invalid_method(self):
        """Test that the __init__ fails when an unrecognised option is given"""
        msg = "Invalid option for keyword backup_method "
        with self.assertRaisesRegex(ValueError, msg):
            WindDirection(backup_method="invalid")


class Test__repr__(unittest.TestCase):
    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WindDirection())
        msg = (
            '<WindDirection: backup_method "neighbourhood"; neighbourhood '
            'radius "6000.0"m>'
        )
        self.assertEqual(result, msg)


class Test_calc_wind_dir_mean(unittest.TestCase):
    """Test the calc_wind_dir_mean function."""

    def setUp(self):
        """Initialise plugin and supply data for tests"""
        self.plugin = WindDirection()
        # 5x3x4 3D Array containing wind direction in angles.
        cube = make_wdir_cube_534()
        self.plugin.wdir_complex = deg_to_complex(cube.data)
        self.plugin.wdir_slice_mean = next(cube.slices_over("realization"))
        self.plugin.realization_axis = 0

        self.expected_wind_mean = np.array(
            [
                [176.636276, 46.002445, 90.0, 90.0],
                [170.0, 170.0, 47.0, 36.544231],
                [333.413239, 320.035217, 10.0, 10.0],
            ],
            dtype=np.float32,
        )

    def test_complex(self):
        """Test that the function defines correct complex mean."""
        self.plugin.calc_wind_dir_mean()
        result = self.plugin.wdir_mean_complex
        expected_complex = deg_to_complex(
            self.expected_wind_mean, radius=np.absolute(result)
        )
        np.testing.assert_array_almost_equal(result, expected_complex)

    def test_degrees(self):
        """Test that the function defines correct degrees cube."""
        self.plugin.calc_wind_dir_mean()
        result = self.plugin.wdir_slice_mean
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(result.data, np.ndarray)
        np.testing.assert_array_almost_equal(
            result.data, self.expected_wind_mean, decimal=4
        )


class Test_find_r_values(unittest.TestCase):
    """Test the find_r_values function."""

    def setUp(self):
        """Initialise plugin and supply data for tests"""
        self.plugin = WindDirection()

    def test_converts_single(self):
        """Tests that r-value is correctly extracted from complex value."""
        # Attach a cube for the plugin to copy in creating the resulting cube:
        self.plugin.wdir_slice_mean = make_wdir_cube_222()[0][0][0]
        expected_out = 2.0
        # Set-up complex values for angle=45 and r=2
        self.plugin.wdir_mean_complex = 1.4142135624 + 1.4142135624j
        self.plugin.find_r_values()
        self.assertAlmostEqual(self.plugin.r_vals_slice.data, expected_out)

    def test_converts_array(self):
        """Test that code can find r-values from array of complex numbers."""
        longitude = DimCoord(
            np.linspace(-180, 180, 36), standard_name="longitude", units="degrees"
        )

        cube = Cube(
            COMPLEX_ANGLES,
            standard_name="wind_from_direction",
            dim_coords_and_dims=[(longitude, 0)],
            units="degree",
        )
        # Attach a cube for the plugin to copy in creating the resulting cube:
        self.plugin.wdir_slice_mean = cube
        self.plugin.wdir_mean_complex = COMPLEX_ANGLES
        expected_out = np.ones(COMPLEX_ANGLES.shape, dtype=np.float32)
        self.plugin.find_r_values()
        result = self.plugin.r_vals_slice.data
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, expected_out)


class Test_wind_dir_decider(unittest.TestCase):
    """Test the wind_dir_decider function."""

    def test_runs_function_1st_member(self):
        """First element has two angles directly opposite (90 & 270 degs).
        Therefore the calculated mean angle of 180 degs is basically
        meaningless with an r value of nearly zero. So the code substitutes the
        wind direction taken from the first ensemble value in its place."""
        cube = make_wdir_cube_222()
        self.plugin = WindDirection(backup_method="first_realization")
        self.plugin.wdir_complex = WIND_DIR_COMPLEX
        self.plugin.realization_axis = 0
        self.plugin.wdir_slice_mean = cube[0].copy(
            data=np.array([[180.0, 55.0], [280.0, 0.0]])
        )
        self.plugin.wdir_mean_complex = deg_to_complex(self.plugin.wdir_slice_mean.data)
        expected_out = np.array([[90.0, 55.0], [280.0, 0.0]])
        where_low_r = np.array([[True, False], [False, False]])
        self.plugin.wind_dir_decider(where_low_r, cube)
        result = self.plugin.wdir_slice_mean.data

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, expected_out)

    def test_runs_function_nbhood(self):
        """First element has two angles directly opposite (90 & 270 degs).
        Therefore the calculated mean angle of 180 degs is basically
        meaningless with an r value of nearly zero. So the code substitutes the
        wind direction taken using the neighbourhood method."""
        expected_out = np.array([[354.91, 55.0], [280.0, 0.0]])

        cube = pad_wdir_cube_222()
        where_low_r = np.pad(
            np.array([[True, False], [False, False]]),
            ((4, 4), (4, 4)),
            "constant",
            constant_values=(True, True),
        )

        wind_dir_deg_mean = np.array([[180.0, 55.0], [280.0, 0.0]])

        self.plugin = WindDirection(backup_method="neighbourhood")
        self.plugin.realization_axis = 0
        self.plugin.n_realizations = 1
        self.plugin.wdir_mean_complex = np.pad(
            deg_to_complex(wind_dir_deg_mean),
            ((4, 4), (4, 4)),
            "constant",
            constant_values=(0.0, 0.0),
        )
        self.plugin.wdir_complex = np.pad(
            WIND_DIR_COMPLEX,
            ((0, 0), (4, 4), (4, 4)),
            "constant",
            constant_values=(0.0 + 0.0j),
        )
        self.plugin.wdir_slice_mean = cube[0].copy(
            data=np.pad(
                wind_dir_deg_mean, ((4, 4), (4, 4)), "constant", constant_values=0.0
            )
        )
        self.plugin.wind_dir_decider(where_low_r, cube)
        result = self.plugin.wdir_slice_mean.data
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result[4:6, 4:6], expected_out, decimal=2)


class Test_process(unittest.TestCase):
    """Test entire code handles a cube correctly."""

    def setUp(self):
        """Create a cube with collapsable coordinates."""
        self.cube = make_wdir_cube_534()

        self.expected_wind_mean = np.array(
            [
                [176.63627625, 46.00244522, 90.0, 90.0],
                [170.0, 170.0, 47.0, 36.54423141],
                [333.41320801, 320.03521729, 10.0, 10.0],
            ],
            dtype=np.float32,
        )

    def test_basic(self):
        """Test that the plugin returns expected data types."""
        result_cube = WindDirection().process(self.cube)

        self.assertIsInstance(result_cube, Cube)

    def test_fails_if_data_is_not_cube(self):
        """Test code raises a Type Error if input cube is not a cube."""
        input_data = 50.0
        msg = "Wind direction input is not a cube, but {0}".format(type(input_data))
        with self.assertRaisesRegex(TypeError, msg):
            WindDirection().process(input_data)

    def test_fails_if_data_is_not_convertible_to_degrees(self):
        """Test code raises a ValueError if input cube is not convertible to
        degrees."""
        data = np.array([[300.0, 270.0], [270.0, 300.0]], dtype=np.float32)
        cube = set_up_variable_cube(data, name="air_temperature", units="K")

        msg = "Input cube cannot be converted to degrees"
        with self.assertRaisesRegex(ValueError, msg):
            WindDirection().process(cube)

    def test_return_single_precision(self):
        """Test that the function returns data of float32."""

        result_cube = WindDirection().process(self.cube)

        self.assertEqual(result_cube.dtype, np.float32)

    def test_returns_expected_values(self):
        """Test that the function returns correct 2D arrays of floats."""

        result_cube = WindDirection().process(self.cube)

        result = result_cube.data

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, self.expected_wind_mean, decimal=4)

    def test_with_backup(self):
        """Test that wind_dir_decider is invoked to select a better value for
        a low-confidence point."""
        # create a low-confidence point
        self.cube.data[:, 1, 1] = [0.0, 72.0, 144.0, 216.0, 288.0]

        # set up a larger cube using a "neutral" pad value so that
        # neighbourhood processing does not fail
        data = np.full((5, 10, 10), 30.0, dtype=np.float32)
        data[:, 3:6, 3:7] = self.cube.data[:, :, :].copy()

        cube = set_up_variable_cube(
            data, name="wind_from_direction", units="degrees", spatial_grid="equalarea"
        )
        cube.coord(axis="x").points = np.arange(-50000.0, -31000.0, 2000.0)
        cube.coord(axis="y").points = np.arange(0.0, 19000.0, 2000.0)

        self.expected_wind_mean[1, 1] = 30.0870

        result_cube = WindDirection().process(cube)

        result = result_cube.data[3:6, 3:7]
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, self.expected_wind_mean, decimal=4)


if __name__ == "__main__":
    unittest.main()
