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
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.utilities import (
    choose_set_of_percentiles,
    concatenate_2d_array_with_2d_array_endpoints,
    create_cube_with_percentiles,
    get_bounds_of_distribution,
    insert_lower_and_upper_endpoint_to_1d_array,
    interpolate_multiple_rows_same_x,
    interpolate_multiple_rows_same_y,
    interpolate_pointwise,
    restore_non_percentile_dimensions,
    slow_interp_same_x,
    slow_interp_same_y,
    slow_interpolate_pointwise,
)
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_variable_cube,
)

from .ecc_test_data import ECC_TEMPERATURE_REALIZATIONS, set_up_spot_test_cube


class Test_concatenate_2d_array_with_2d_array_endpoints(IrisTest):

    """Test the concatenate_2d_array_with_2d_array_endpoints."""

    def test_basic(self):
        """Test that result is a numpy array with the expected contents."""
        expected = np.array([[0, 20, 50, 80, 100]])
        input_array = np.array([[20, 50, 80]])
        result = concatenate_2d_array_with_2d_array_endpoints(input_array, 0, 100)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)

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


class Test_choose_set_of_percentiles(IrisTest):

    """Test the choose_set_of_percentiles plugin."""

    def test_basic(self):
        """
        Test that the plugin returns a list with the expected number of
        percentiles.
        """
        no_of_percentiles = 3
        result = choose_set_of_percentiles(no_of_percentiles)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), no_of_percentiles)

    def test_data(self):
        """
        Test that the plugin returns a list with the expected data values
        for the percentiles.
        """
        data = np.array([25, 50, 75])
        no_of_percentiles = 3
        result = choose_set_of_percentiles(no_of_percentiles)
        self.assertArrayAlmostEqual(result, data)

    def test_random(self):
        """
        Test that the plugin returns a list with the expected number of
        percentiles, if the random sampling option is selected.
        """
        no_of_percentiles = 3
        result = choose_set_of_percentiles(no_of_percentiles, sampling="random")
        self.assertIsInstance(result, list)
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


class Test_create_cube_with_percentiles(IrisTest):

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
        """Test that the plugin returns an Iris.cube.Cube with many percentiles.
        """
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
        self.assertArrayAlmostEqual(result.coord("percentile").points, percentiles)

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
        self.assertArrayAlmostEqual(result.coord("percentile").points, percentiles)

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


class Test_get_bounds_of_distribution(IrisTest):

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
        self.assertArrayAlmostEqual(result, bounds_pairing)

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
        self.assertArrayAlmostEqual(result, bounds_pairing)

    def test_check_exception_is_raised(self):
        """
        Test that the expected results are returned for the bounds_pairing.
        """
        cube_name = "nonsense"
        cube_units = Unit("degreesC")
        msg = "The bounds_pairing_key"
        with self.assertRaisesRegex(KeyError, msg):
            get_bounds_of_distribution(cube_name, cube_units)


class Test_insert_lower_and_upper_endpoint_to_1d_array(IrisTest):

    """Test the insert_lower_and_upper_endpoint_to_1d_array."""

    def test_basic(self):
        """Test that the result is a numpy array with the expected contents."""
        expected = [0, 20, 50, 80, 100]
        percentiles = np.array([20, 50, 80])
        result = insert_lower_and_upper_endpoint_to_1d_array(percentiles, 0, 100)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)

    def test_2d_example(self):
        """Test 2D input results in expected error"""
        percentiles = np.array([[-40, 200, 1000], [-40, 200, 1000]])
        msg = "Expected 1D input"
        with self.assertRaisesRegex(ValueError, msg):
            insert_lower_and_upper_endpoint_to_1d_array(percentiles, -100, 10000)


class Test_restore_non_percentile_dimensions(IrisTest):

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
        self.assertArrayAlmostEqual(reshaped_array, self.expected_data)

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
        self.assertArrayAlmostEqual(reshaped_array, expected)

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
        self.assertArrayAlmostEqual(reshaped_array, expected)


numba_installed = True
try:
    importlib.util.find_spec("numba")
    from improver.ensemble_copula_coupling.numba_utilities import (
        fast_interp_same_x,
        fast_interp_same_y,
        fast_interpolate_pointwise,
    )
except ImportError:
    numba_installed = False


class Test_interpolate_multiple_rows_same_y(IrisTest):

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


class TestInterpolateMultipleRowsSameX(IrisTest):

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


class TestInterpolatePointwise(IrisTest):

    """Test interpolate_pointwise"""

    def setUp(self):
        """Set up arrays."""
        np.random.seed(0)
        self.x = np.arange(0, 0.1, 0.01)
        self.xp = np.sort(np.random.uniform(0, 0.1, (15, 20)), axis=0)
        self.fp = np.random.random((15, 20))

    def test_slow(self):
        """Test slow interp against known result."""
        xp = np.array(
            [[-4, -3, -2, -1, 0], [0, 1, 2, 3, 4]], dtype=np.float32
        ).transpose()
        fp = np.array(
            [[-2, -1.5, -1, -0.5, 0], [0, 2, 4, 6, 8]], dtype=np.float32
        ).transpose()
        x = np.array([-1, 0.5, 2], dtype=np.float32)
        expected = np.array([[-0.5, 0, 0], [0, 1, 4]], dtype=np.float32).transpose()
        result = slow_interpolate_pointwise(x, xp, fp)
        np.testing.assert_allclose(result, expected)

    @skipIf(not (numba_installed), "numba not installed")
    def test_fast(self):
        """Test fast interp against known result."""
        xp = np.array(
            [[-4, -3, -2, -1, 0], [0, 1, 2, 3, 4]], dtype=np.float32
        ).transpose()
        fp = np.array(
            [[-2, -1.5, -1, -0.5, 0], [0, 2, 4, 6, 8]], dtype=np.float32
        ).transpose()
        x = np.array([-1, 0.5, 2], dtype=np.float32)
        expected = np.array([[-0.5, 0, 0], [0, 1, 4]], dtype=np.float32).transpose()
        result = fast_interpolate_pointwise(x, xp, fp)
        np.testing.assert_allclose(result, expected)

    @skipIf(not (numba_installed), "numba not installed")
    def test_slow_vs_fast(self):
        """Test that slow and fast versions give same result."""
        result_slow = slow_interpolate_pointwise(self.x, self.xp, self.fp)
        result_fast = fast_interpolate_pointwise(self.x, self.xp, self.fp)
        np.testing.assert_allclose(result_slow, result_fast)

    @patch.dict("sys.modules", numba=None)
    @patch("improver.ensemble_copula_coupling.utilities.slow_interpolate_pointwise")
    def test_slow_interpolate_pointwise_called(self, interp_imp):
        """Test that slow_interpolate_pointwise is called if numba is not installed."""
        interpolate_pointwise(mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp)
        interp_imp.assert_called_once_with(
            mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp
        )

    @skipIf(not (numba_installed), "numba not installed")
    @patch(
        "improver.ensemble_copula_coupling.numba_utilities.fast_interpolate_pointwise"
    )
    def test_fast_interpolate_pointwise_called(self, interp_imp):
        """Test that fast_interpolate_pointwise is called if numba is installed."""
        interpolate_pointwise(mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp)
        interp_imp.assert_called_once_with(
            mock.sentinel.x, mock.sentinel.xp, mock.sentinel.fp
        )

    def test_shape_errors_slow(self):
        """Test that an error is raised if inputs have the wrong shape."""
        x = np.arange(1, 5)
        xp = np.ones((2, 3))
        fp = np.ones((3, 2))
        msg = r"xp and fp must have the same shape"
        with self.assertRaisesRegex(ValueError, msg):
            slow_interpolate_pointwise(x, xp, fp)
        xp = np.ones((2,))
        fp = np.ones((2,))
        msg = r"xp and fp must have at least 2 dimensions"
        with self.assertRaisesRegex(ValueError, msg):
            slow_interpolate_pointwise(x, xp, fp)

    @skipIf(not (numba_installed), "numba not installed")
    def test_shape_errors_fast(self):
        """Test that an error is raised if inputs have the wrong shape."""
        x = np.arange(1, 5)
        xp = np.ones((2, 3))
        fp = np.ones((3, 2))
        msg = r"xp and fp must have the same shape"
        with self.assertRaisesRegex(ValueError, msg):
            fast_interpolate_pointwise(x, xp, fp)
        xp = np.ones((2,))
        fp = np.ones((2,))
        msg = r"xp and fp must have at least 2 dimensions"
        with self.assertRaisesRegex(ValueError, msg):
            fast_interpolate_pointwise(x, xp, fp)


if __name__ == "__main__":
    unittest.main()
