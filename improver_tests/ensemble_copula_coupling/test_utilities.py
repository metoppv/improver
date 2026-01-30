# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the
`ensemble_copula_coupling.EnsembleCopulaCouplingUtilities` class.
"""

import importlib
import unittest
import unittest.mock as mock
from datetime import datetime
from unittest.case import skipIf
from unittest.mock import patch

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from improver.ensemble_copula_coupling.utilities import (
    CalculatePercentilesFromIntensityDistribution,
    choose_set_of_percentiles,
    concatenate_2d_array_with_2d_array_endpoints,
    create_cube_with_percentiles,
    get_bounds_of_distribution,
    insert_lower_and_upper_endpoint_to_1d_array,
    interpolate_multiple_rows_same_x,
    interpolate_multiple_rows_same_y,
    restore_non_percentile_dimensions,
    slow_interp_same_x,
    slow_interp_same_y,
)
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_spot_probability_cube,
    set_up_spot_variable_cube,
    set_up_variable_cube,
)

from .ecc_test_data import ECC_TEMPERATURE_REALIZATIONS, set_up_spot_test_cube


class Test_concatenate_2d_array_with_2d_array_endpoints(unittest.TestCase):
    """Test the concatenate_2d_array_with_2d_array_endpoints."""

    def test_basic(self):
        """Test that result is a numpy array with the expected contents."""
        expected = np.array([[0, 20, 50, 80, 100]])
        input_array = np.array([[20, 50, 80]])
        result = concatenate_2d_array_with_2d_array_endpoints(input_array, 0, 100)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, expected)

    def test_1d_input(self):
        """Test 1D input results in the expected error"""
        input_array = np.array([-40, 200, 1000])
        msg = "Expected 2D input"
        with self.assertRaisesRegex(ValueError, msg):
            concatenate_2d_array_with_2d_array_endpoints(input_array, -100, 10000)

    def test_3d_input(self):
        """Test 3D input results in expected error"""
        input_array = np.array([[[-40, 200, 1000]]])
        msg = "Expected 2D input"
        with self.assertRaisesRegex(ValueError, msg):
            concatenate_2d_array_with_2d_array_endpoints(input_array, -100, 10000)


class Test_choose_set_of_percentiles(unittest.TestCase):
    """Test the choose_set_of_percentiles plugin."""

    def test_basic(self):
        """
        Test that the plugin returns a list with the expected number of
        percentiles.
        """
        no_of_percentiles = 3
        result = choose_set_of_percentiles(no_of_percentiles)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), no_of_percentiles)

    def test_data(self):
        """
        Test that the plugin returns a list with the expected data values
        for the percentiles.
        """
        data = np.array([25, 50, 75])
        no_of_percentiles = 3
        result = choose_set_of_percentiles(no_of_percentiles)
        np.testing.assert_array_almost_equal(result, data)

    def test_random(self):
        """
        Test that the plugin returns a list with the expected number of
        percentiles, if the random sampling option is selected.
        """
        no_of_percentiles = 3
        result = choose_set_of_percentiles(no_of_percentiles, sampling="random")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), no_of_percentiles)

    def test_unknown_sampling_option(self):
        """
        Test that the plugin returns the expected error message,
        if an unknown sampling option is selected.
        """
        no_of_percentiles = 3
        msg = "Unrecognised sampling option"
        with self.assertRaisesRegex(ValueError, msg):
            choose_set_of_percentiles(no_of_percentiles, sampling="unknown")


class Test_create_cube_with_percentiles(unittest.TestCase):
    """Test the _create_cube_with_percentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.cube = set_up_variable_cube(ECC_TEMPERATURE_REALIZATIONS[0])
        self.cube_data = ECC_TEMPERATURE_REALIZATIONS

    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube with suitable units."""
        cube_data = self.cube_data + 2
        percentiles = [10, 50, 90]
        result = create_cube_with_percentiles(percentiles, self.cube, cube_data)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.units, self.cube.units)

    def test_changed_cube_units(self):
        """Test that the plugin returns a cube with chosen units."""
        cube_data = self.cube_data + 2
        percentiles = [10, 50, 90]
        result = create_cube_with_percentiles(
            percentiles, self.cube, cube_data, cube_unit="1"
        )
        self.assertEqual(result.units, Unit("1"))

    def test_many_percentiles(self):
        """Test that the plugin returns an Iris.cube.Cube with many percentiles."""
        percentiles = np.linspace(0, 100, 100)
        cube_data = np.zeros(
            [
                len(percentiles),
                len(self.cube.coord("latitude").points),
                len(self.cube.coord("longitude").points),
            ]
        )
        result = create_cube_with_percentiles(percentiles, self.cube, cube_data)
        self.assertEqual(cube_data.shape, result.data.shape)

    def test_incompatible_percentiles(self):
        """
        Test that the plugin fails if the percentile values requested
        are not numbers.
        """
        percentiles = ["cat", "dog", "elephant"]
        cube_data = np.zeros(
            [
                len(percentiles),
                len(self.cube.coord("latitude").points),
                len(self.cube.coord("longitude").points),
            ]
        )
        msg = "could not convert string to float"
        with self.assertRaisesRegex(ValueError, msg):
            create_cube_with_percentiles(percentiles, self.cube, cube_data)

    def test_percentile_points(self):
        """
        Test that the plugin returns an Iris.cube.Cube
        with a percentile coordinate with the desired points.
        """
        cube_data = self.cube_data + 2
        percentiles = [10, 50, 90]
        result = create_cube_with_percentiles(percentiles, self.cube, cube_data)
        self.assertIsInstance(result.coord("percentile"), DimCoord)
        np.testing.assert_array_almost_equal(
            result.coord("percentile").points, percentiles
        )

    def test_spot_forecasts_percentile_points(self):
        """
        Test that the plugin returns a Cube with a percentile dimension
        coordinate and that the percentile dimension has the expected points
        for an input spot forecast.
        """
        cube = set_up_spot_test_cube()
        spot_data = cube.data.copy() + 2
        spot_cube = next(cube.slices_over("realization"))
        spot_cube.remove_coord("realization")

        percentiles = [10, 50, 90]
        result = create_cube_with_percentiles(percentiles, spot_cube, spot_data)
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(result.coord("percentile"), DimCoord)
        np.testing.assert_array_almost_equal(
            result.coord("percentile").points, percentiles
        )

    def test_percentile_length_too_short(self):
        """
        Test that the plugin raises the default ValueError, if the number
        of percentiles is fewer than the length of the zeroth dimension of the
        required cube data.
        """
        cube_data = self.cube_data + 2
        percentiles = [10, 50]
        msg = "Require data with shape"
        with self.assertRaisesRegex(ValueError, msg):
            create_cube_with_percentiles(percentiles, self.cube, cube_data)

    def test_percentile_length_too_long(self):
        """
        Test that the plugin raises the default ValueError, if the number
        of percentiles exceeds the length of the zeroth dimension of the
        required data.
        """
        cube_data = self.cube_data[0, :, :] + 2
        percentiles = [10, 50, 90]
        msg = "Require data with shape"
        with self.assertRaisesRegex(ValueError, msg):
            create_cube_with_percentiles(percentiles, self.cube, cube_data)

    def test_metadata_copy(self):
        """
        Test that the metadata dictionaries within the input cube, are
        also present on the output cube.
        """
        self.cube.attributes = {"source": "ukv"}
        cube_data = self.cube_data + 2
        percentiles = [10, 50, 90]
        result = create_cube_with_percentiles(percentiles, self.cube, cube_data)
        self.assertDictEqual(self.cube.metadata._asdict(), result.metadata._asdict())

    def test_coordinate_copy(self):
        """
        Test that the coordinates within the input cube, are
        also present on the output cube.
        """
        cube_data = self.cube_data + 2
        percentiles = [10, 50, 90]
        result = create_cube_with_percentiles(percentiles, self.cube, cube_data)
        for coord in self.cube.coords():
            if coord not in result.coords():
                msg = "Coordinate: {} not found in cube {}".format(coord, result)
                raise CoordinateNotFoundError(msg)


class Test_get_bounds_of_distribution(unittest.TestCase):
    """Test the get_bounds_of_distribution plugin."""

    def test_basic(self):
        """Test that the result is a numpy array."""
        cube_name = "air_temperature"
        cube_units = Unit("degreesC")
        result = get_bounds_of_distribution(cube_name, cube_units)
        self.assertIsInstance(result, np.ndarray)

    def test_check_data(self):
        """
        Test that the expected results are returned for the bounds_pairing.
        """
        cube_name = "air_temperature"
        cube_units = Unit("degreesC")
        bounds_pairing = (-100, 60)
        result = get_bounds_of_distribution(cube_name, cube_units)
        np.testing.assert_array_almost_equal(result, bounds_pairing)

    def test_check_unit_conversion(self):
        """
        Test that the expected results are returned for the bounds_pairing,
        if the units of the bounds_pairings need to be converted to match
        the units of the forecast.
        """
        cube_name = "air_temperature"
        cube_units = Unit("fahrenheit")
        bounds_pairing = (-148, 140)  # In fahrenheit
        result = get_bounds_of_distribution(cube_name, cube_units)
        np.testing.assert_array_almost_equal(result, bounds_pairing)

    def test_check_exception_is_raised(self):
        """
        Test that the expected results are returned for the bounds_pairing.
        """
        cube_name = "nonsense"
        cube_units = Unit("degreesC")
        msg = "The bounds_pairing_key"
        with self.assertRaisesRegex(KeyError, msg):
            get_bounds_of_distribution(cube_name, cube_units)


class Test_insert_lower_and_upper_endpoint_to_1d_array(unittest.TestCase):
    """Test the insert_lower_and_upper_endpoint_to_1d_array."""

    def test_basic(self):
        """Test that the result is a numpy array with the expected contents."""
        expected = [0, 20, 50, 80, 100]
        percentiles = np.array([20, 50, 80])
        result = insert_lower_and_upper_endpoint_to_1d_array(percentiles, 0, 100)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_example(self):
        """Test 2D input results in expected error"""
        percentiles = np.array([[-40, 200, 1000], [-40, 200, 1000]])
        msg = "Expected 1D input"
        with self.assertRaisesRegex(ValueError, msg):
            insert_lower_and_upper_endpoint_to_1d_array(percentiles, -100, 10000)


class Test_restore_non_percentile_dimensions(unittest.TestCase):
    """Test the restore_non_percentile_dimensions."""

    def setUp(self):
        """Set up template cube and temperature data."""
        self.cube = set_up_variable_cube(282 * np.ones((3, 3), dtype=np.float32))
        # function is designed to reshape an input data array with dimensions of
        # "percentiles x points" - generate suitable input data
        self.expected_data = np.sort(ECC_TEMPERATURE_REALIZATIONS.copy(), axis=0)
        points_data = [self.expected_data[i].flatten() for i in range(3)]
        self.input_data = np.array(points_data)

    def test_multiple_percentiles(self):
        """
        Test the result is an array with the expected shape and contents.
        """
        reshaped_array = restore_non_percentile_dimensions(
            self.input_data, self.cube, 3
        )
        self.assertIsInstance(reshaped_array, np.ndarray)
        np.testing.assert_array_almost_equal(reshaped_array, self.expected_data)

    def test_single_percentile(self):
        """
        Test the array size and contents if the percentile coordinate is scalar.
        """
        expected = np.array(
            [[226.15, 237.4, 248.65], [259.9, 271.15, 282.4], [293.65, 304.9, 316.15]],
            dtype=np.float32,
        )
        reshaped_array = restore_non_percentile_dimensions(
            self.input_data[0], self.cube, 1
        )
        np.testing.assert_array_almost_equal(reshaped_array, expected)

    def test_multiple_timesteps(self):
        """
        Test that the data has been reshaped correctly when there are multiple timesteps.
        The array contents are also checked.  The output cube has only a single percentile,
        which is therefore demoted to a scalar coordinate.
        """
        expected = np.array(
            [
                [[4.0, 4.71428571], [5.42857143, 6.14285714]],
                [[6.85714286, 7.57142857], [8.28571429, 9.0]],
            ]
        )

        cubelist = CubeList([])
        for i, hour in enumerate([7, 8]):
            cubelist.append(
                set_up_percentile_cube(
                    np.array([expected[i, :, :]], dtype=np.float32),
                    np.array([50], dtype=np.float32),
                    units="degC",
                    time=datetime(2015, 11, 23, hour),
                    frt=datetime(2015, 11, 23, 6),
                )
            )
        percentile_cube = cubelist.merge_cube()

        reshaped_array = restore_non_percentile_dimensions(
            percentile_cube.data.flatten(),
            next(percentile_cube.slices_over("percentile")),
            1,
        )
        np.testing.assert_array_almost_equal(reshaped_array, expected)


numba_installed = True
try:
    importlib.util.find_spec("numba")
    from improver.ensemble_copula_coupling.numba_utilities import (
        fast_interp_same_x,
        fast_interp_same_y,
    )
except ImportError:
    numba_installed = False


class Test_interpolate_multiple_rows_same_y(unittest.TestCase):
    """Test interpolate_multiple_rows_same_y"""

    def setUp(self):
        """Set up arrays."""
        np.random.seed(0)
        self.x = np.arange(0, 1, 0.01)
        self.xp = np.sort(np.random.random_sample((100, 100)), axis=1)
        self.fp = np.arange(0, 100, 1).astype(float)

    def test_slow(self):
        """Test slow interp against known result."""
        xp = np.array([[0, 1, 2, 3, 4], [-4, -3, -2, -1, 0]], dtype=np.float32)
        fp = np.array([0, 2, 4, 6, 8], dtype=np.float32)
        x = np.array([-1, 0.5, 2], dtype=np.float32)
        expected = np.array([[0, 1, 4], [6, 8, 8]], dtype=np.float32)
        result = slow_interp_same_y(x, xp, fp)
        np.testing.assert_allclose(result, expected)

    @patch.dict("sys.modules", numba=None)
    @patch("improver.ensemble_copula_coupling.utilities.slow_interp_same_y")
    def test_slow_interp_same_y_called(self, interp_imp):
        """Test that slow_interp_same_y is called if numba is not installed."""
        interpolate_multiple_rows_same_y(
            mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp
        )
        interp_imp.assert_called_once_with(
            mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp
        )

    @skipIf(not (numba_installed), "numba not installed")
    @patch("improver.ensemble_copula_coupling.numba_utilities.fast_interp_same_y")
    def test_fast_interp_same_y_called(self, interp_imp):
        """Test that fast_interp_same_y is called if numba is installed."""
        interpolate_multiple_rows_same_y(
            mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp
        )
        interp_imp.assert_called_once_with(
            mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp
        )

    @skipIf(not (numba_installed), "numba not installed")
    def test_fast(self):
        """Test fast interp against known result."""
        xp = np.array([[0, 1, 2, 3, 4], [-4, -3, -2, -1, 0]], dtype=np.float32)
        fp = np.array([0, 2, 4, 6, 8], dtype=np.float32)
        x = np.array([-1, 0.5, 2], dtype=np.float32)
        expected = np.array([[0, 1, 4], [6, 8, 8]], dtype=np.float32)
        result = fast_interp_same_y(x, xp, fp)
        np.testing.assert_allclose(result, expected)

    @skipIf(not (numba_installed), "numba not installed")
    def test_slow_vs_fast(self):
        """Test that slow and fast versions give same result."""
        result_slow = slow_interp_same_y(self.x, self.xp, self.fp)
        result_fast = fast_interp_same_y(self.x, self.xp, self.fp)
        np.testing.assert_allclose(result_slow, result_fast)

    @skipIf(not (numba_installed), "numba not installed")
    def test_slow_vs_fast_unordered(self):
        """Test that slow and fast versions give same result
        when x is not sorted."""
        shuffled_x = self.x.copy()
        np.random.shuffle(shuffled_x)
        result_slow = slow_interp_same_y(shuffled_x, self.xp, self.fp)
        result_fast = fast_interp_same_y(shuffled_x, self.xp, self.fp)
        np.testing.assert_allclose(result_slow, result_fast)

    @skipIf(not (numba_installed), "numba not installed")
    def test_slow_vs_fast_repeated(self):
        """Test that slow and fast versions give same result when
        rows of xp contain repeats."""
        xp_repeat = self.xp.copy()
        xp_repeat[:, 51] = xp_repeat[:, 50]
        result_slow = slow_interp_same_y(self.x, xp_repeat, self.fp)
        result_fast = fast_interp_same_y(self.x, xp_repeat, self.fp)
        np.testing.assert_allclose(result_slow, result_fast)

    @skipIf(not (numba_installed), "numba not installed")
    def test_slow_vs_multi(self):
        """Test that slow interp gives same result as
        interpolate_multiple_rows_same_y."""
        result_slow = slow_interp_same_y(self.x, self.xp, self.fp)
        result_multiple = interpolate_multiple_rows_same_y(self.x, self.xp, self.fp)
        np.testing.assert_allclose(result_slow, result_multiple)


class TestInterpolateMultipleRowsSameX(unittest.TestCase):
    """Test interpolate_multiple_rows"""

    def setUp(self):
        """Set up arrays."""
        np.random.seed(0)
        self.x = np.arange(0, 1, 0.01)
        self.xp = np.sort(np.random.random_sample(100))
        self.fp = np.random.random((100, 100))

    def test_slow(self):
        """Test slow interp against known result."""
        xp = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        fp = np.array([[0, 0.5, 1, 1.5, 2], [0, 2, 4, 6, 8]], dtype=np.float32)
        x = np.array([-1, 0.5, 2], dtype=np.float32)
        expected = np.array([[0, 0.25, 1], [0, 1, 4]], dtype=np.float32)
        result = slow_interp_same_x(x, xp, fp)
        np.testing.assert_allclose(result, expected)

    @skipIf(not (numba_installed), "numba not installed")
    def test_fast(self):
        """Test fast interp against known result."""
        xp = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        fp = np.array([[0, 0.5, 1, 1.5, 2], [0, 2, 4, 6, 8]], dtype=np.float32)
        x = np.array([-1, 0.5, 2], dtype=np.float32)
        expected = np.array([[0, 0.25, 1], [0, 1, 4]], dtype=np.float32)
        result = fast_interp_same_x(x, xp, fp)
        np.testing.assert_allclose(result, expected)

    @skipIf(not (numba_installed), "numba not installed")
    def test_slow_vs_fast(self):
        """Test that slow and fast versions give same result."""
        result_slow = slow_interp_same_x(self.x, self.xp, self.fp)
        result_fast = fast_interp_same_x(self.x, self.xp, self.fp)
        np.testing.assert_allclose(result_slow, result_fast)

    @skipIf(not (numba_installed), "numba not installed")
    def test_slow_vs_fast_unordered(self):
        """Test that slow and fast versions give same result
        when x is not sorted."""
        shuffled_x = self.x.copy()
        np.random.shuffle(shuffled_x)
        result_slow = slow_interp_same_x(shuffled_x, self.xp, self.fp)
        result_fast = fast_interp_same_x(shuffled_x, self.xp, self.fp)
        np.testing.assert_allclose(result_slow, result_fast)

    @skipIf(not (numba_installed), "numba not installed")
    def test_slow_vs_fast_repeated(self):
        """Test that slow and fast versions give same result when xp
        contains repeats."""
        repeat_xp = self.xp.copy()
        repeat_xp[51] = repeat_xp[50]
        result_slow = slow_interp_same_x(self.x, repeat_xp, self.fp)
        result_fast = fast_interp_same_x(self.x, repeat_xp, self.fp)
        np.testing.assert_allclose(result_slow, result_fast)

    @skipIf(not (numba_installed), "numba not installed")
    def test_slow_vs_multi(self):
        """Test that slow interp gives same result as
        interpolate_multiple_rows_same_x."""
        result_slow = slow_interp_same_x(self.x, self.xp, self.fp)
        result_multiple = interpolate_multiple_rows_same_x(self.x, self.xp, self.fp)
        np.testing.assert_allclose(result_slow, result_multiple)

    @patch.dict("sys.modules", numba=None)
    @patch("improver.ensemble_copula_coupling.utilities.slow_interp_same_x")
    def test_slow_interp_same_x_called(self, interp_imp):
        """Test that slow_interp_same_x is called if numba is not installed."""
        interpolate_multiple_rows_same_x(
            mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp
        )
        interp_imp.assert_called_once_with(
            mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp
        )

    @skipIf(not (numba_installed), "numba not installed")
    @patch("improver.ensemble_copula_coupling.numba_utilities.fast_interp_same_x")
    def test_fast_interp_same_x_called(self, interp_imp):
        """Test that fast_interp_same_x is called if numba is installed."""
        interpolate_multiple_rows_same_x(
            mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp
        )
        interp_imp.assert_called_once_with(
            mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp
        )


def _create_intensity_and_probability_cubes(
    intensity_data,
    thresholds,
    probability_data,
    nan_mask_value=None,
):
    """Helper function to create intensity and probability cubes for testing.

    Args:
        intensity_data: 3D array (realization, y, x) of intensity values.
        thresholds: List of threshold values for probability cube.
        probability_data: 3D array (threshold, y, x) of probability values.
        nan_mask_value: Optional value to set to NaN in intensity data.

    Returns:
        Tuple of (intensity_cube, probability_cube)
    """
    intensity_cube = set_up_variable_cube(
        intensity_data,
        name="precipitation_rate",
        units="mm h-1",
    )

    if nan_mask_value is not None:
        intensity_cube.data[intensity_cube.data == nan_mask_value] = nan_mask_value

    probability_cube = set_up_probability_cube(
        probability_data,
        thresholds=thresholds,
        variable_name="precipitation_rate",
        threshold_units="mm h-1",
        spp__relative_to_threshold="above",
    )

    return intensity_cube, probability_cube


# Tests for CalculatePercentilesFromIntensityDistribution initialisation


def test_init_valid_distribution():
    """Test initialisation with valid distribution."""
    plugin = CalculatePercentilesFromIntensityDistribution(distribution="gamma")
    assert plugin.distribution == "gamma"
    assert plugin.nan_mask_value == 0.0


@pytest.mark.parametrize("distribution", ["normal", "exponential", "invalid"])
def test_init_invalid_distribution(distribution):
    """Test initialisation raises error for invalid distributions."""
    with pytest.raises(ValueError, match="Unrecognised distribution option"):
        CalculatePercentilesFromIntensityDistribution(distribution=distribution)


@pytest.mark.parametrize("nan_mask_value", [0.0, None, 1.0, -999.0])
def test_init_nan_mask_value(nan_mask_value):
    """Test initialisation with different nan_mask_values."""
    plugin = CalculatePercentilesFromIntensityDistribution(
        nan_mask_value=nan_mask_value
    )
    assert plugin.nan_mask_value == nan_mask_value


# Tests for CalculatePercentilesFromIntensityDistribution.process method


@pytest.mark.parametrize("rescale", [True, False])
@pytest.mark.parametrize("nan_mask_value", [0.0, None])
def test_process_basic_3d(rescale, nan_mask_value):
    """Test basic percentile calculation with 3D gamma distribution, with and without
    rescaling the percentiles and with/without NaN masking."""
    # Create test data: 3 realizations, 3x3 grid
    intensity_data = np.array(
        [
            [[0.5, 2.0, 2.0], [0.0, 0.0, 2.0], [1.0, 2.0, 0.0]],
            [[4.0, 8.0, 4.0], [2.0, 8.0, 2.0], [4.0, 2.0, 1.0]],
            [[6.0, 10.0, 6.0], [2.0, 2.0, 6.0], [6.0, 6.0, 2.0]],
        ],
        dtype=np.float32,
    )
    thresholds = [0.5, 1.0, 2.0, 4.0]

    probability_data = np.array(
        [
            [[0.6, 0.9, 1.0], [0.4, 0.7, 1.0], [0.9, 1.0, 0.6]],
            [[0.5, 0.6, 0.6], [0.3, 0.5, 0.9], [0.6, 0.6, 0.3]],
            [[0.4, 0.4, 0.4], [0.2, 0.4, 0.5], [0.4, 0.4, 0.1]],
            [[0.3, 0.3, 0.2], [0.0, 0.1, 0.2], [0.1, 0.1, 0.0]],
        ],
        dtype=np.float32,
    )

    # Expected results for each combination of rescale and nan_mask_value.
    if rescale and nan_mask_value == 0.0:
        # Focusing on with and without rescaling, for the top left corner, the intensity
        # values are [0.5, 4.0, 6.0] with probabilities [0.6, 0.5, 0.4, 0.3].
        # Without rescaling, this maps the 0.5 value to a percentile value of 0.021
        # when using a gamma distribution. With rescaling, the percentiles are scaled
        # between the min and max percentiles i.e. (1 - max_probability (0.6)) = 0.4 to
        # give the min percentile and a max percentile of 1. This gives a rescaled
        # percentile value of 0.413.
        expected = np.array(
            [
                [[0.413, 0.133, 0.084], [0.000, 0.000, 0.263], [0.140, 0.263, 0.000]],
                [[0.799, 0.734, 0.554], [0.801, 0.393, 0.263], [0.671, 0.263, 0.492]],
                [[0.921, 0.861, 0.884], [0.801, 0.896, 0.909], [0.885, 0.909, 0.907]],
            ],
            dtype=np.float32,
        )
    elif not rescale and nan_mask_value == 0.0:
        expected = np.array(
            [
                [[0.021, 0.037, 0.084], [0.000, 0.000, 0.263], [0.044, 0.263, 0.000]],
                [[0.665, 0.704, 0.554], [0.503, 0.133, 0.263], [0.635, 0.263, 0.153]],
                [[0.869, 0.846, 0.884], [0.503, 0.852, 0.909], [0.872, 0.909, 0.845]],
            ],
            dtype=np.float32,
        )
    elif rescale and nan_mask_value is None:
        # Expected results when no NaN masking is applied (i.e., zeros are included in
        # calculations). For the middle left position in realization 0, when zero
        # values are included, the intensity values are [0., 2.0, 2.0] with
        # probabilities [0.4, 0.3, 0.2, 0.0]. Previously, the masking of zeros meant
        # that the mapping of a 0.0 intensity value to a percentile resulted in a
        # percentile value of 0.0. However, when zeros are included, the rescaling means
        # that the min percentile is now (1 - max_probability (0.4)) = 0.6 and the max
        # percentile is 1. This gives a rescaled percentile value of 0.600.
        expected = np.array(
            [
                [[0.413, 0.133, 0.084], [0.600, 0.300, 0.263], [0.140, 0.263, 0.400]],
                [[0.799, 0.734, 0.554], [0.920, 0.620, 0.263], [0.671, 0.263, 0.765]],
                [[0.921, 0.861, 0.884], [0.920, 0.935, 0.909], [0.885, 0.909, 0.933]],
            ],
            dtype=np.float32,
        )
    elif not rescale and nan_mask_value is None:
        # Expected results when no NaN masking is applied and no rescaling. For the
        # central grid point in realization 1, when zero values are included, the
        # intensity values are [0., 8.0, 2.0] with probabilities [0.7, 0.5, 0.4, 0.1].
        # Previously, the masking of zeros meant that the percentile creation using the
        # gamma distribution only included the [8.0, 2.0] intensity values. If rescaling
        # is applied the percentiles are restricted to between 0.3 and 1. This results
        # in the central grid point being scaled to 0.620. Withoutout rescaling, this
        # results in a percentile value of 0.458.
        expected = np.array(
            [
                [[0.021, 0.037, 0.084], [0.000, 0.000, 0.263], [0.044, 0.263, 0.000]],
                [[0.665, 0.704, 0.554], [0.801, 0.458, 0.263], [0.635, 0.263, 0.608]],
                [[0.869, 0.846, 0.884], [0.801, 0.907, 0.909], [0.872, 0.909, 0.888]],
            ],
            dtype=np.float32,
        )

    intensity_cube, probability_cube = _create_intensity_and_probability_cubes(
        intensity_data, thresholds, probability_data, nan_mask_value=nan_mask_value
    )

    plugin = CalculatePercentilesFromIntensityDistribution(
        rescale_percentiles=rescale, nan_mask_value=nan_mask_value
    )
    result = plugin.process(probability_cube, intensity_cube)
    assert isinstance(result, np.ndarray)
    assert result.shape == intensity_data.shape
    assert result.dtype == np.float32
    # Check that the calculation has handled zeros appropriately
    assert np.all(np.isfinite(result))
    np.testing.assert_array_almost_equal(result, expected, decimal=3)


@pytest.mark.parametrize("rescale", [True, False])
@pytest.mark.parametrize("nan_mask_value", [0.0, None])
def test_process_2d_spot_data(rescale, nan_mask_value):
    """Test process method with 2D input (e.g., spot data), with and without
    rescaling the percentiles and with/without NaN masking."""
    # 2D input: 3 realizations, 5 sites
    intensity_data = np.array(
        [
            [0.0, 4.0, 3.0, 4.0, 5.0],
            [2.0, 2.0, 4.0, 0.0, 0.0],
            [8.0, 1.0, 1.0, 6.0, 7.0],
        ],
        dtype=np.float32,
    )

    thresholds = [0.5, 2.0, 4.0]
    probability_data = np.array(
        [
            [0.8, 0.9, 1.0, 0.6, 0.7],
            [0.6, 0.8, 0.8, 0.4, 0.6],
            [0.5, 0.3, 0.2, 0.3, 0.5],
        ],
        dtype=np.float32,
    )

    intensity_cube = set_up_spot_variable_cube(
        intensity_data,
        name="precipitation_rate",
        units="mm h-1",
    )

    probability_cube = set_up_spot_probability_cube(
        probability_data,
        thresholds=thresholds,
        variable_name="precipitation_rate",
        threshold_units="mm h-1",
        spp__relative_to_threshold="above",
    )

    if rescale and nan_mask_value == 0.0:
        expected = np.array(
            [
                [0.0, 0.203, 0.051, 0.0, 0.0],
                [0.306, 0.514, 0.66, 0.494, 0.41],
                [0.881, 0.909, 0.86, 0.906, 0.89],
            ],
            dtype=np.float32,
        )
    elif not rescale and nan_mask_value == 0.0:
        expected = np.array(
            [
                [0.0, 0.115, 0.051, 0.0, 0.0],
                [0.133, 0.46, 0.66, 0.157, 0.157],
                [0.852, 0.899, 0.86, 0.843, 0.842],
            ],
            dtype=np.float32,
        )
    elif rescale and nan_mask_value is None:
        expected = np.array(
            [
                [0.2, 0.203, 0.051, 0.4, 0.3],
                [0.566, 0.514, 0.66, 0.815, 0.798],
                [0.926, 0.909, 0.86, 0.92, 0.901],
            ],
            dtype=np.float32,
        )
    elif not rescale and nan_mask_value is None:
        expected = np.array(
            [
                [0.0, 0.115, 0.051, 0.0, 0.0],
                [0.458, 0.46, 0.66, 0.691, 0.712],
                [0.907, 0.899, 0.86, 0.866, 0.859],
            ],
            dtype=np.float32,
        )

    plugin = CalculatePercentilesFromIntensityDistribution(
        rescale_percentiles=rescale, nan_mask_value=nan_mask_value
    )
    result = plugin.process(probability_cube, intensity_cube)
    assert isinstance(result, np.ndarray)
    assert result.shape == intensity_data.shape
    assert result.dtype == np.float32
    np.testing.assert_array_almost_equal(result, expected, decimal=3)


def test_process_1d_input_raises_error():
    """Test that process method with 1D input raises appropriate error."""
    from iris.cube import Cube

    intensity_data = np.ones(3, dtype=np.float32)
    probability_data = np.ones(2, dtype=np.float32)

    intensity_cube = Cube(intensity_data, units="mm h-1")
    probability_cube = Cube(probability_data, units="1")

    plugin = CalculatePercentilesFromIntensityDistribution()

    with pytest.raises(ValueError, match="Expected at least 2D input"):
        plugin.process(probability_cube, intensity_cube)


if __name__ == "__main__":
    unittest.main()
