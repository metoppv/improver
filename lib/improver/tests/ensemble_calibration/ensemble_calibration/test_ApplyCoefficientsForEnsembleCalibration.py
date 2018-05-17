# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
`ensemble_calibration.ApplyCoefficientsForEnsembleCalibration`
class.

"""
import datetime
import unittest

from cf_units import Unit
import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import CubeList
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_calibration.ensemble_calibration import (
    ApplyCoefficientsFromEnsembleCalibration as Plugin)
from improver.utilities.cube_manipulation import concatenate_cubes
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import (set_up_temperature_cube,
                             add_forecast_reference_time_and_forecast_period)
from improver.utilities.warnings_handler import ManageWarnings


def datetime_from_timestamp(timestamp):
    """Wrapper for timestamp to return a datetime object"""
    return datetime.datetime.utcfromtimestamp(timestamp*3600)


class Test__find_coords_of_length_one(IrisTest):

    """Test the find length_one coords method."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.optimised_coeffs = [4.55819380e-06, -8.02401974e-09,
                                 1.66667055e+00, 1.00000011e+00]
        self.coeff_names = ["gamma", "delta", "a", "beta"]

    def test_basic(self):
        """Test that the plugin returns a list."""
        plugin = Plugin(self.cube, self.optimised_coeffs,
                        self.coeff_names)
        result = plugin._find_coords_of_length_one(self.cube)
        self.assertIsInstance(result, list)

    def test_length_one_coords_list_of_tuples(self):
        """Test that the plugin returns a list inside the tuple."""
        plugin = Plugin(self.cube, self.optimised_coeffs,
                        self.coeff_names)
        result = plugin._find_coords_of_length_one(self.cube)
        self.assertIsInstance(result[0], tuple)

    def test_length_one_coords_list_of_coords(self):
        """Test that the plugin returns a DimCoord inside the list."""
        plugin = Plugin(self.cube, self.optimised_coeffs,
                        self.coeff_names)
        result = plugin._find_coords_of_length_one(
            self.cube, add_dimension=False)
        self.assertIsInstance(result[0], DimCoord)

    def test_check_all_coords(self):
        """Test that the plugin returns a DimCoord inside the list."""
        current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))
        plugin = Plugin(
            current_temperature_forecast_cube,
            self.optimised_coeffs, self.coeff_names)
        results = plugin._find_coords_of_length_one(
            current_temperature_forecast_cube, add_dimension=False)
        coord_names = [result.name() for result in results]
        for coord_name in ["time", "forecast_period",
                           "forecast_reference_time"]:
            self.assertIn(coord_name, coord_names)
        for result in results:
            self.assertIsInstance(result, DimCoord)


class Test__separate_length_one_coords_into_aux_and_dim(IrisTest):

    """
    Test the separate length one coords into aux and dim coordinates method.
    """

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.optimised_coeffs = [4.55819380e-06, -8.02401974e-09,
                                 1.66667055e+00, 1.00000011e+00]
        self.coeff_names = ["gamma", "delta", "a", "beta"]
        self.plugin = Plugin(self.cube, self.optimised_coeffs,
                             self.coeff_names)

    def test_basic_dim_coord(self):
        """Test that the plugin returns a list."""
        length_one_coords = [DimCoord(
            np.array([402192.5]), standard_name='time',
            units=Unit('hours since 1970-01-01 00:00:00',
                       calendar='gregorian'))]
        result = self.plugin._separate_length_one_coords_into_aux_and_dim(
            length_one_coords)
        for aresult in result:
            self.assertIsInstance(aresult, list)

    def test_basic_aux_coord_and_dim_coord(self):
        """Test that the plugin returns a list."""
        length_one_coords = [
            DimCoord(
                np.array([402193]),
                standard_name='time',
                units=Unit('hours since 1970-01-01 00:00:00',
                           calendar='gregorian')),
            DimCoord(
                np.array([402190]),
                standard_name='forecast_reference_time',
                units=Unit('hours since 1970-01-01 00:00:00',
                           calendar='gregorian'))]
        result = self.plugin._separate_length_one_coords_into_aux_and_dim(
            length_one_coords)
        for aresult in result:
            self.assertIsInstance(aresult, list)

    def test_multiple_dim_coords(self):
        """
        Test that the plugin returns a list containing no auxiliary
        coordinates and some dimension coordinates."""
        length_one_coords = [
            DimCoord(
                np.array([402193]),
                standard_name='time',
                units=Unit('hours since 1970-01-01 00:00:00',
                           calendar='gregorian')),
            DimCoord(
                np.array([402190]),
                standard_name='forecast_reference_time',
                units=Unit('hours since 1970-01-01 00:00:00',
                           calendar='gregorian'))]
        result = self.plugin._separate_length_one_coords_into_aux_and_dim(
            length_one_coords, dim_coords=["time", "forecast_reference_time"])
        len_one_coords_for_aux_coords = result[0]
        len_one_coords_for_dim_coords = result[1]
        self.assertFalse(len_one_coords_for_aux_coords)
        self.assertTrue(len_one_coords_for_dim_coords)
        for coord in len_one_coords_for_dim_coords:
            self.assertIsInstance(coord, DimCoord)

    def test_multiple_aux_coord_and_dim_coord(self):
        """
        Test that the plugin returns a list containing no auxiliary
        coordinates and some dimension coordinates."""
        length_one_coords = [
            DimCoord(
                np.array([402193]),
                standard_name='time',
                units=Unit('hours since 1970-01-01 00:00:00',
                           calendar='gregorian')),
            DimCoord(
                np.array([402190]),
                standard_name='forecast_reference_time',
                units=Unit('hours since 1970-01-01 00:00:00',
                           calendar='gregorian')),
            AuxCoord(
                np.array([402187]),
                standard_name='forecast_period',
                units=Unit('hours'))]
        result = self.plugin._separate_length_one_coords_into_aux_and_dim(
            length_one_coords, dim_coords=["time", "forecast_reference_time"])
        len_one_coords_for_aux_coords = result[0]
        len_one_coords_for_dim_coords = result[1]
        self.assertTrue(len_one_coords_for_dim_coords)
        for coord in len_one_coords_for_aux_coords:
            self.assertIsInstance(coord, AuxCoord)
        for coord in len_one_coords_for_dim_coords:
            self.assertIsInstance(coord, DimCoord)

    def test_check_coord_names(self):
        """
        Test that the plugin returns a list with the auxiliary coordinates
        within the auxiliary coordinate list and the dimension coordinate
        within the dimension coordinate list.
        """
        length_one_coords = [
            DimCoord(
                np.array([402193]),
                standard_name='time',
                units=Unit('hours since 1970-01-01 00:00:00',
                           calendar='gregorian')),
            DimCoord(
                np.array([402190]),
                standard_name='forecast_reference_time',
                units=Unit('hours since 1970-01-01 00:00:00',
                           calendar='gregorian')),
            AuxCoord(
                np.array([402187]),
                standard_name='forecast_period',
                units=Unit('hours'))]
        result = self.plugin._separate_length_one_coords_into_aux_and_dim(
            length_one_coords, dim_coords=["time", "forecast_reference_time"])
        aux_coords = result[0]
        dim_coords = result[1]
        aux_coord_names = [aux_coord.name() for aux_coord in aux_coords]
        dim_coord_names = [dim_coord.name() for dim_coord in dim_coords]
        for coord_name in ["time", "forecast_reference_time"]:
            self.assertIn(coord_name, dim_coord_names)
        for coord_name in ["forecast_period"]:
            self.assertIn(coord_name, aux_coord_names)

    def test_check_coord_names_tuple(self):
        """
        Test that the plugin returns a list wi th the auxiliary coordinates
        within the auxiliary coordinate list and the dimension coordinate
        within the dimension coordinate list. In this test, the returned
        value for the dimension coordinates is a tuple.
        """
        length_one_coords = [
            (DimCoord(
                np.array([402193]),
                standard_name='time',
                units=Unit('hours since 1970-01-01 00:00:00',
                           calendar='gregorian')), 0),
            DimCoord(
                np.array([402190]),
                standard_name='forecast_reference_time',
                units=Unit('hours since 1970-01-01 00:00:00',
                           calendar='gregorian')),
            AuxCoord(
                np.array([402187]),
                standard_name='forecast_period',
                units=Unit('hours'))]
        result = self.plugin._separate_length_one_coords_into_aux_and_dim(
            length_one_coords, dim_coords=["time", "forecast_reference_time"])
        aux_coords = result[0]
        dim_coords = result[1]
        aux_coord_names = [aux_coord.name() for aux_coord in aux_coords]
        dim_coord_names = [dim_coord[0].name() for dim_coord in dim_coords]
        for coord_name in ["time", "forecast_reference_time"]:
            self.assertIn(coord_name, dim_coord_names)
        for coord_name in ["forecast_period"]:
            self.assertIn(coord_name, aux_coord_names)


class Test___create_coefficient_cube(IrisTest):

    """Test the __create_coefficient_cube method."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.optimised_coeffs = [4.55819380e-06, -8.02401974e-09,
                                 1.66667055e+00, 1.00000011e+00]
        self.coeff_names = ["gamma", "delta", "a", "beta"]
        self.plugin = Plugin(self.cube, self.optimised_coeffs,
                             self.coeff_names)

    def test_basic(self):
        """Test that the plugin returns a CubeList."""
        result = self.plugin._create_coefficient_cube(
            self.cube, self.optimised_coeffs, self.coeff_names)
        self.assertIsInstance(result, CubeList)

    def test_number_of_coefficients(self):
        """
        Test that the plugin returns the expected number of coefficient names.
        """
        result = self.plugin._create_coefficient_cube(
            self.cube, self.optimised_coeffs, self.coeff_names)
        self.assertEqual(len(result), len(self.coeff_names))

    def test_coefficient_data_in_cube(self):
        """
        Test that the plugin returns the expected data for each coefficient.
        """
        self.optimised_coeffs = [4.55819380e-06, -8.02401974e-09,
                                 1.66667055e+00]
        self.coeff_names = ["cat", "dog", "elephant"]
        results = self.plugin._create_coefficient_cube(
            self.cube, self.optimised_coeffs, self.coeff_names)
        for result, coeff in zip(results, self.optimised_coeffs):
            self.assertEqual(result.data, coeff)

    def test_coefficient_name_in_cube(self):
        """
        Test that the plugin returns the expected coefficient name
        for each coefficient.
        """
        self.optimised_coeffs = [4.55819380e-06, -8.02401974e-09,
                                 1.66667055e+00]
        self.coeff_names = ["cat", "dog", "elephant"]
        results = self.plugin._create_coefficient_cube(
            self.cube, self.optimised_coeffs, self.coeff_names)
        for result, coeff_name in zip(results, self.coeff_names):
            self.assertEqual(result.long_name, coeff_name)


class Test_apply_params_entry(IrisTest):

    """Test the apply_params_entry plugin."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

        self.coeff_names = ["gamma", "delta", "a", "beta"]

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """Test that the plugin returns a tuple."""
        cube = self.current_temperature_forecast_cube
        optimised_coeffs = {}
        the_date = datetime_from_timestamp(cube.coord("time").points)
        optimised_coeffs[the_date] = [4.55819380e-06, -8.02401974e-09,
                                      1.66667055e+00, 1.00000011e+00]
        plugin = Plugin(cube, optimised_coeffs,
                        self.coeff_names)
        result = plugin.apply_params_entry()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_members(self):
        """
        Test that the plugin returns a tuple when using the ensemble
        members as the predictor of the mean.
        """
        cube = self.current_temperature_forecast_cube
        optimised_coeffs = {}
        the_date = datetime_from_timestamp(cube.coord("time").points)
        optimised_coeffs[the_date] = np.array([
            4.55819380e-06, -8.02401974e-09, 1.66667055e+00, 1.00000011e+00,
            1.00000011e+00, 1.00000011e+00])
        plugin = Plugin(cube, optimised_coeffs,
                        self.coeff_names, predictor_of_mean_flag="members")
        result = plugin.apply_params_entry()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_output_is_mean(self):
        """
        Test that the plugin returns a tuple containing cubes with a
        mean cell method.
        """
        cube = self.current_temperature_forecast_cube
        optimised_coeffs = {}
        the_date = datetime_from_timestamp(cube.coord("time").points)
        optimised_coeffs[the_date] = [4.55819380e-06, -8.02401974e-09,
                                      1.66667055e+00, 1.00000011e+00]
        plugin = Plugin(cube, optimised_coeffs,
                        self.coeff_names)
        forecast_predictor, _, _ = plugin.apply_params_entry()
        for cell_method in forecast_predictor[0].cell_methods:
            self.assertEqual(cell_method.method, "mean")

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_output_is_variance(self):
        """
        Test that the plugin returns a tuple containing cubes with a
        variance cell method.
        """
        cube = self.current_temperature_forecast_cube
        optimised_coeffs = {}
        the_date = datetime_from_timestamp(cube.coord("time").points)
        optimised_coeffs[the_date] = [4.55819380e-06, -8.02401974e-09,
                                      1.66667055e+00, 1.00000011e+00]
        plugin = Plugin(cube, optimised_coeffs,
                        self.coeff_names)
        _, forecast_variance, _ = plugin.apply_params_entry()
        for cell_method in forecast_variance[0].cell_methods:
            self.assertEqual(cell_method.method, "variance")

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_output_coefficients(self):
        """
        Test that the plugin returns a tuple containing cubes with the
        expected coefficient names.
        """
        cube = self.current_temperature_forecast_cube
        optimised_coeffs = {}
        the_date = datetime_from_timestamp(cube.coord("time").points)
        optimised_coeffs[the_date] = [4.55819380e-06, -8.02401974e-09,
                                      1.66667055e+00, 1.00000011e+00]
        plugin = Plugin(cube, optimised_coeffs,
                        self.coeff_names)
        _, _, coefficients = plugin.apply_params_entry()
        for result, coeff_name, coeff in zip(
                coefficients, self.coeff_names, optimised_coeffs[the_date]):
            self.assertEqual(result.long_name, coeff_name)
            self.assertArrayAlmostEqual(result.data, coeff)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_output_coefficients_members(self):
        """
        Test that the plugin returns a tuple containing cubes with the
        expected coefficient names when using ensemble members as the
        predictor of the mean.
        """
        cube = self.current_temperature_forecast_cube
        optimised_coeffs = {}
        the_date = datetime_from_timestamp(cube.coord("time").points)
        optimised_coeffs[the_date] = np.array([
            4.55819380e-06, -8.02401974e-09, 1.66667055e+00, 1.00000011e+00,
            1.00000011e+00, 1.00000011e+00])
        plugin = Plugin(cube, optimised_coeffs,
                        self.coeff_names, predictor_of_mean_flag="members")
        _, _, coefficients = plugin.apply_params_entry()
        for result, coeff_name, coeff in zip(
                coefficients, self.coeff_names, optimised_coeffs[the_date]):
            self.assertEqual(result.long_name, coeff_name)
            self.assertArrayAlmostEqual(result.data, coeff)


class Test__apply_params(IrisTest):

    """Test the _apply_params plugin."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

        self.coeff_names = ["gamma", "delta", "a", "beta"]

        self.default_optimised_coeffs = [
            4.55819380e-06, -8.02401974e-09, 1.66667055e+00, 1.00000011e+00]

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """Test that the plugin returns a tuple."""
        optimised_coeffs = {}
        cube = self.current_temperature_forecast_cube
        the_date = datetime_from_timestamp(cube.coord("time").points)
        optimised_coeffs[the_date] = self.default_optimised_coeffs

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        predictor_of_mean_flag = "mean"

        plugin = Plugin(cube, optimised_coeffs,
                        self.coeff_names)
        result = plugin._apply_params(
            predictor_cube, variance_cube, optimised_coeffs,
            self.coeff_names, predictor_of_mean_flag)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_two_dates(self):
        """
        Test that the plugin returns a tuple when two dates are present
        within the input cube.
        """
        cube = self.current_temperature_forecast_cube
        cube1 = cube.copy()
        cube2 = cube.copy()

        cube2.coord("time").points = cube2.coord("time").points + 3
        cube2.data += 3

        cube = concatenate_cubes(CubeList([cube1, cube2]))

        optimised_coeffs = {}

        for time_slice in cube.slices_over("time"):
            the_date = datetime_from_timestamp(time_slice.coord("time").points)
            optimised_coeffs[the_date] = self.default_optimised_coeffs

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        predictor_of_mean_flag = "mean"

        plugin = Plugin(cube, optimised_coeffs,
                        self.coeff_names)
        forecast_predictor, forecast_variance, coefficients = (
            plugin._apply_params(
                predictor_cube, variance_cube, optimised_coeffs,
                self.coeff_names, predictor_of_mean_flag))

        for result in [forecast_predictor, forecast_variance]:
            self.assertEqual(len(result), 2)
        self.assertEqual(len(coefficients), 8)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_calibrated_predictor(self):
        """
        Test that the plugin returns values for the calibrated predictor (the
        calibrated mean), which match the expected values.
        """
        data = np.array([[231.15002913, 242.40003036, 253.6500316],
                         [264.90003284, 276.15003408, 287.40003531],
                         [298.65003655, 309.90003779, 321.15003903]])

        cube = self.current_temperature_forecast_cube
        cube1 = cube.copy()
        cube2 = cube.copy()

        cube2.coord("time").points = cube2.coord("time").points + 3
        cube2.data += 3

        cube = concatenate_cubes(CubeList([cube1, cube2]))

        optimised_coeffs = {}

        for time_slice in cube.slices_over("time"):
            the_date = datetime_from_timestamp(time_slice.coord("time").points)
            optimised_coeffs[the_date] = self.default_optimised_coeffs

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        predictor_of_mean_flag = "mean"

        plugin = Plugin(self.cube, optimised_coeffs,
                        self.coeff_names)
        forecast_predictor, _, _ = plugin._apply_params(
            predictor_cube, variance_cube, optimised_coeffs,
            self.coeff_names, predictor_of_mean_flag)
        self.assertArrayAlmostEqual(forecast_predictor[0].data, data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_calibrated_variance(self):
        """
        Test that the plugin returns values for the calibrated variance,
        which match the expected values.
        """
        data = np.array([[2.07777316e-11, 2.07777316e-11, 2.07777316e-11],
                         [2.07777316e-11, 2.07777316e-11, 2.07777316e-11],
                         [2.07777316e-11, 2.07777316e-11, 2.07777316e-11]])

        cube = self.current_temperature_forecast_cube
        cube1 = cube.copy()
        cube2 = cube.copy()

        cube2.coord("time").points = cube2.coord("time").points + 3
        cube2.data += 3

        cube = concatenate_cubes(CubeList([cube1, cube2]))

        optimised_coeffs = {}

        for time_slice in cube.slices_over("time"):
            the_date = datetime_from_timestamp(time_slice.coord("time").points)
            optimised_coeffs[the_date] = self.default_optimised_coeffs

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        predictor_of_mean_flag = "mean"

        plugin = Plugin(self.cube, optimised_coeffs,
                        self.coeff_names)
        _, forecast_variance, _ = plugin._apply_params(
            predictor_cube, variance_cube, optimised_coeffs,
            self.coeff_names, predictor_of_mean_flag)
        self.assertArrayAlmostEqual(forecast_variance[0].data, data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_coefficients(self):
        """
        Test that the plugin returns values for the coefficients,
        which match the expected values.
        """
        data = np.array([4.55819380e-06])

        cube = self.current_temperature_forecast_cube
        cube1 = cube.copy()
        cube2 = cube.copy()

        cube2.coord("time").points = cube2.coord("time").points + 3
        cube2.data += 3

        cube = concatenate_cubes(CubeList([cube1, cube2]))

        optimised_coeffs = {}

        for time_slice in cube.slices_over("time"):
            the_date = datetime_from_timestamp(time_slice.coord("time").points)
            optimised_coeffs[the_date] = self.default_optimised_coeffs

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        predictor_of_mean_flag = "mean"

        plugin = Plugin(self.cube, optimised_coeffs,
                        self.coeff_names)
        _, _, coefficients = plugin._apply_params(
            predictor_cube, variance_cube, optimised_coeffs,
            self.coeff_names, predictor_of_mean_flag)
        self.assertArrayAlmostEqual(coefficients[0].data, data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_calibrated_predictor_members(self):
        """
        Test that the plugin returns values for the calibrated forecasts,
        which match the expected values when the individual ensemble members
        are used as the predictor.
        """
        data = np.array([[239.904135, 251.65926, 263.414385],
                         [275.16951, 286.924635, 298.67976],
                         [310.434885, 322.19001, 333.945135]])

        cube = self.current_temperature_forecast_cube
        cube1 = cube.copy()
        cube2 = cube.copy()

        cube2.coord("time").points = cube2.coord("time").points + 3
        cube2.data += 3

        cube = concatenate_cubes(CubeList([cube1, cube2]))

        optimised_coeffs = {}

        for time_slice in cube.slices_over("time"):
            the_date = datetime_from_timestamp(time_slice.coord("time").points)
            optimised_coeffs[the_date] = np.array(
                [5, 1, 0, 0.57, 0.6, 0.6])
        self.coeff_names = ["gamma", "delta", "a", "beta"]

        predictor_cube = cube.copy()
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        predictor_of_mean_flag = "members"

        plugin = Plugin(self.cube, optimised_coeffs,
                        self.coeff_names)
        forecast_predictor, _, _ = plugin._apply_params(
            predictor_cube, variance_cube, optimised_coeffs,
            self.coeff_names, predictor_of_mean_flag)
        self.assertArrayAlmostEqual(forecast_predictor[0].data, data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_calibrated_variance_members(self):
        """
        Test that the plugin returns values for the calibrated forecasts,
        which match the expected values when the individual ensemble members
        are used as the predictor.
        """
        data = np.array([[34.333333, 34.333333, 34.333333],
                         [34.333333, 34.333333, 34.333333],
                         [34.333333, 34.333333, 34.333333]])

        cube = self.current_temperature_forecast_cube
        cube1 = cube.copy()
        cube2 = cube.copy()

        cube2.coord("time").points = cube2.coord("time").points + 3
        cube2.data += 3

        cube = concatenate_cubes(CubeList([cube1, cube2]))

        optimised_coeffs = {}

        for time_slice in cube.slices_over("time"):
            the_date = datetime_from_timestamp(time_slice.coord("time").points)
            optimised_coeffs[the_date] = np.array(
                [5, 1, 0, 0.57, 0.6, 0.6])
        self.coeff_names = ["gamma", "delta", "a", "beta"]

        predictor_cube = cube.copy()
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        predictor_of_mean_flag = "members"

        plugin = Plugin(self.cube, optimised_coeffs,
                        self.coeff_names)
        _, forecast_variance, _ = plugin._apply_params(
            predictor_cube, variance_cube, optimised_coeffs,
            self.coeff_names, predictor_of_mean_flag)
        self.assertArrayAlmostEqual(forecast_variance[0].data, data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_coefficients_members(self):
        """
        Test that the plugin returns values for the calibrated forecasts,
        which match the expected values when the individual ensemble members
        are used as the predictor.
        """
        data = np.array([5.0])
        cube = self.current_temperature_forecast_cube
        cube1 = cube.copy()
        cube2 = cube.copy()

        cube2.coord("time").points = cube2.coord("time").points + 3
        cube2.data += 3

        cube = concatenate_cubes(CubeList([cube1, cube2]))

        optimised_coeffs = {}

        for time_slice in cube.slices_over("time"):
            the_date = datetime_from_timestamp(time_slice.coord("time").points)
            optimised_coeffs[the_date] = np.array(
                [5, 1, 0, 0.57, 0.6, 0.6])
        self.coeff_names = ["gamma", "delta", "a", "beta"]

        predictor_cube = cube.copy()
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        predictor_of_mean_flag = "members"

        plugin = Plugin(self.cube, optimised_coeffs,
                        self.coeff_names)
        _, _, coefficients = plugin._apply_params(
            predictor_cube, variance_cube, optimised_coeffs,
            self.coeff_names, predictor_of_mean_flag)
        self.assertArrayAlmostEqual(coefficients[0].data, data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_too_many_coefficients(self):
        """
        Test that the plugin returns values for the coefficients,
        which match the expected values.
        """

        cube = self.current_temperature_forecast_cube
        cube1 = cube.copy()
        cube2 = cube.copy()

        cube2.coord("time").points = cube2.coord("time").points + 3
        cube2.data += 3

        cube = concatenate_cubes(CubeList([cube1, cube2]))

        optimised_coeffs = {}

        for time_slice in cube.slices_over("time"):
            the_date = datetime_from_timestamp(time_slice.coord("time").points)
            optimised_coeffs[the_date] = self.default_optimised_coeffs

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        coeff_names = ["cat", "dog", "elephant", "frog", "giraffe"]
        predictor_of_mean_flag = "mean"

        plugin = Plugin(self.cube, optimised_coeffs,
                        coeff_names)
        msg = "Number of coefficient names"
        with self.assertRaisesRegex(ValueError, msg):
            dummy_result = plugin._apply_params(
                predictor_cube, variance_cube, optimised_coeffs,
                coeff_names, predictor_of_mean_flag)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Ensemble calibration not available"])
    def test_missing_date(self):
        """
        Test that the plugin returns values for the calibrated forecasts,
        if the date to be calibrated can not be found in the available
        dictionary of coefficients. In this situation, the raw forecasts are
        returned.
        """
        data = np.array([[229.48333333, 240.73333333, 251.98333333],
                         [263.23333333, 274.48333333, 285.73333333],
                         [296.98333333, 308.23333333, 319.48333333]])

        cube = self.current_temperature_forecast_cube
        optimised_coeffs = {}
        the_date = datetime_from_timestamp(cube.coord("time").points+3)
        optimised_coeffs[the_date] = self.default_optimised_coeffs

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        predictor_of_mean_flag = "mean"

        plugin = Plugin(cube, optimised_coeffs,
                        self.coeff_names)

        result = plugin._apply_params(
            predictor_cube, variance_cube, optimised_coeffs,
            self.coeff_names, predictor_of_mean_flag)

        self.assertArrayAlmostEqual(result[0][0].data, data)

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_missing_date_catch_warning(self, warning_list=None):
        """
        Test that the plugin returns values for the calibrated forecasts,
        if the date to be calibrated can not be found in the available
        dictionary of coefficients. In this situation, the raw forecasts are
        returned.
        """

        cube = self.current_temperature_forecast_cube
        optimised_coeffs = {}
        the_date = datetime_from_timestamp(cube.coord("time").points+3)
        optimised_coeffs[the_date] = self.default_optimised_coeffs

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        predictor_of_mean_flag = "mean"

        plugin = Plugin(cube, optimised_coeffs,
                        self.coeff_names)

        dummy_result = plugin._apply_params(
            predictor_cube, variance_cube, optimised_coeffs,
            self.coeff_names, predictor_of_mean_flag)

        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("Ensemble calibration not available"
                        in str(warning_list[0]))


if __name__ == '__main__':
    unittest.main()
