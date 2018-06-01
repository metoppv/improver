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
Unit tests for the utilities within the "cube_manipulation" module.

"""
import unittest

from cf_units import Unit
import iris
from iris.coords import AuxCoord, DimCoord
from iris.coord_systems import TransverseMercator
from iris.cube import Cube
from iris.exceptions import (
    ConcatenateError, DuplicateDataError, CoordinateNotFoundError)
from iris.tests import IrisTest
import numpy as np

from improver.utilities.cube_manipulation import (
    _associate_any_coordinate_with_master_coordinate,
    _slice_over_coordinate,
    _strip_var_names,
    concatenate_cubes,
    merge_cubes,
    equalise_cubes,
    _equalise_cube_attributes,
    _equalise_cube_coords,
    _equalise_cell_methods,
    compare_attributes,
    compare_coords,
    build_coordinate,
    add_renamed_cell_method,
    sort_coord_in_cube,
    enforce_coordinate_ordering,
    enforce_float32_precision,
    clip_cube_data)

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import (
        set_up_temperature_cube,
        set_up_probability_above_threshold_temperature_cube,
        add_forecast_reference_time_and_forecast_period)

from improver.tests.utilities.test_mathematical_operations import (
    set_up_height_cube)
from improver.utilities.warnings_handler import ManageWarnings


def set_up_percentile_cube(data, phenomenon_standard_name, phenomenon_units,
                           percentiles=np.array([10, 50, 90]), timesteps=1,
                           y_dimension_length=3, x_dimension_length=3):
    """
    Create a cube containing multiple percentile values
    for the coordinate.
    """
    cube = Cube(data, standard_name=phenomenon_standard_name,
                units=phenomenon_units)
    coord_long_name = "percentile_over_realization"
    cube.add_dim_coord(
        DimCoord(percentiles,
                 long_name=coord_long_name,
                 units='%'), 0)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord(np.linspace(402192.5, 402292.5, timesteps),
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, y_dimension_length),
                                "latitude", units="degrees"), 2)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, x_dimension_length),
                                "longitude", units="degrees"), 3)
    return cube


def set_up_percentile_temperature_cube():
    """ Create a cube with metadata and values suitable for air temperature."""
    data = np.array([[[[0.1, 0.1, 0.1],
                       [0.2, 0.2, 0.2],
                       [0.5, 0.5, 0.5]]],
                     [[[1.0, 1.0, 1.0],
                       [0.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5]]],
                     [[[2.0, 3.0, 4.0],
                       [0.8, 1.2, 1.6],
                       [1.5, 2.0, 3.0]]]])
    return (
        set_up_percentile_cube(data, "air_temperature", "K"))


def _check_coord_type(cube, coord):
    '''Function to test whether coord is classified
       as scalar or auxiliary coordinate.

       Args:
           cube (iris.cube.Cube):
               Iris cube containing coordinates to be checked
           coord (iris.coords.DimCoord or iris.coords.AuxCoord):
               Cube coordinate to check
    '''
    coord_scalar = True
    coord_aux = False
    cube_summary = cube.summary()
    aux_ind = cube_summary.find("Auxiliary")
    if coord in cube_summary[aux_ind:]:
        coord_scalar = False
        coord_aux = True
    return coord_scalar, coord_aux


class Test__associate_any_coordinate_with_master_coordinate(IrisTest):

    """Test the _associate_any_coordinate_with_master_coordinate utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_cube_with_forecast_period_and_forecast_reference_time(self):
        """
        Test that the utility returns an iris.cube.Cube with the
        expected values for the forecast_reference_time and forecast_period
        coordinates. This checks that the auxiliary coordinates that were
        added to the cube are still present.

        """
        cube = self.cube

        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"))

        result = _associate_any_coordinate_with_master_coordinate(
            cube, coordinates=["forecast_reference_time", "forecast_period"])
        self.assertArrayAlmostEqual(
            result.coord("forecast_reference_time").points, [402192.5])
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, [0])

    def test_cube_check_coord_type(self):
        """
        Test that the utility returns an iris.cube.Cube with the
        expected values for the forecast_reference_time and forecast_period
        coordinates. This checks that the auxiliary coordinates that were
        added to the cube are still present.

        """
        cube = self.cube

        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"))
        result = _associate_any_coordinate_with_master_coordinate(
            cube, coordinates=["forecast_reference_time", "forecast_period"])
        scalar, aux = _check_coord_type(result, 'forecast_period')
        self.assertFalse(scalar)
        self.assertTrue(aux)
        scalar, aux = _check_coord_type(result, 'forecast_reference_time')
        self.assertFalse(scalar)
        self.assertTrue(aux)

    def test_cube_with_latitude_and_height(self):
        """
        Test that the utility returns an iris.cube.Cube with a height
        coordinate, if this coordinate is added to the input cube. This checks
        that the height coordinate points, which were added to the cube
        are the same as the after applying the utility.
        """
        cube = self.cube
        for latitude_slice in cube.slices_over("latitude"):
            cube = iris.util.new_axis(latitude_slice, "latitude")

        cube.add_aux_coord(
            DimCoord([10], "height", units="m"))

        result = _associate_any_coordinate_with_master_coordinate(
            cube, master_coord="latitude", coordinates=["height"])
        self.assertArrayAlmostEqual(
            result.coord("height").points, [10])

    def test_coordinate_not_on_cube(self):
        """
        Test that the utility returns an iris.cube.Cube without
        forecast_reference_time and forecast_period coordinates, if these
        have not been added to the cube.
        """
        cube = self.cube

        result = _associate_any_coordinate_with_master_coordinate(
            cube, coordinates=["forecast_reference_time", "forecast_period"])
        self.assertFalse(result.coords("forecast_reference_time"))
        self.assertFalse(result.coords("forecast_period"))

    def test_no_time_dimension(self):
        """
        Test that the plugin returns the expected error message,
        if the input cubes do not contain a time coordinate.
        """
        cube1 = self.cube.copy()
        cube1.remove_coord("time")

        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube1.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube1.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"))

        msg = "The master coordinate for associating other coordinates"
        with self.assertRaisesRegex(ValueError, msg):
            _associate_any_coordinate_with_master_coordinate(
                cube1, master_coord="time",
                coordinates=["forecast_reference_time", "forecast_period"])

    def test_scalar_time_coordinate(self):
        """Test that the output cube retains scalar coordinates for the time,
        forecast_period and forecast_reference_time coordinates, if these
        coordinates are scalar within the input cube."""
        cube = self.cube
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"))
        cube = cube[:, 0, ...]
        result = _associate_any_coordinate_with_master_coordinate(
            cube, coordinates=["forecast_reference_time", "forecast_period"])
        self.assertTrue(result.coords("time", dimensions=[]))
        self.assertTrue(result.coords("forecast_period", dimensions=[]))
        self.assertTrue(
            result.coords("forecast_reference_time", dimensions=[]))


class Test__slice_over_coordinate(IrisTest):

    """Test the _slice_over_coordinate utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the utility returns an iris.cube.CubeList."""
        cubelist = iris.cube.CubeList([self.cube])
        result = _slice_over_coordinate(cubelist, "time")
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_basic_cube(self):
        """Test that the utility returns an iris.cube.CubeList."""
        result = _slice_over_coordinate(self.cube, "time")
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_time_first_dimension(self):
        """
        Test that the first dimension of the output cube within the
        output cubelist has time as the first dimension.
        """
        cubelist = iris.cube.CubeList([self.cube])
        result = _slice_over_coordinate(cubelist, "time")
        dim_coord_names = []
        for cube in result:
            for dim_coord in cube.dim_coords:
                dim_coord_names.append(dim_coord.name())
        self.assertEqual(dim_coord_names[0], "time")

    def test_number_of_slices(self):
        """
        Test that the number of cubes returned, after slicing over the
        given coordinate is as expected.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.5
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube1.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube1.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"))
        cube2.add_aux_coord(
            DimCoord([402195.5], "forecast_reference_time", units=tunit))
        cube2.add_aux_coord(
            DimCoord([3], "forecast_period", units="hours"))

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _slice_over_coordinate(cubelist, "forecast_period")
        self.assertEqual(len(result), 2)

    def test_number_of_slices_from_one_cube(self):
        """
        Test that the number of cubes returned, after slicing over the
        given coordinate is as expected.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.5
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube1.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit),
            data_dims=1)
        cube1.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"), data_dims=1)
        cube2.add_aux_coord(
            DimCoord([402195.5], "forecast_reference_time", units=tunit),
            data_dims=1)
        cube2.add_aux_coord(
            DimCoord([3], "forecast_period", units="hours"), data_dims=1)

        cubelist = iris.cube.CubeList([cube1, cube2])

        cubelist = cubelist.concatenate_cube()

        result = _slice_over_coordinate(cubelist, "forecast_period")
        self.assertEqual(len(result), 2)


class Test__strip_var_names(IrisTest):

    """Test the _slice_var_names utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the utility returns an iris.cube.CubeList."""
        self.cube.var_name = "air_temperature"
        result = _strip_var_names(self.cube)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_cube_var_name_is_none(self):
        """
        Test that the utility returns an iris.cube.Cube with a
        var_name of None.
        """
        self.cube.var_name = "air_temperature"
        result = _strip_var_names(self.cube)
        self.assertIsNone(result[0].var_name, None)

    def test_cube_coord_var_name_is_none(self):
        """
        Test that the coordinates have var_names of None.
        """
        self.cube.coord("time").var_name = "time"
        self.cube.coord("latitude").var_name = "latitude"
        self.cube.coord("longitude").var_name = "longitude"
        result = _strip_var_names(self.cube)
        for cube in result:
            for coord in cube.coords():
                self.assertIsNone(coord.var_name, None)

    def test_cubelist(self):
        """Test that the utility returns an iris.cube.CubeList."""
        cube1 = self.cube
        cube2 = self.cube
        cubes = iris.cube.CubeList([cube1, cube2])
        self.cube.var_name = "air_temperature"
        result = _strip_var_names(cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        for cube in result:
            for coord in cube.coords():
                self.assertIsNone(coord.var_name, None)


class Test_concatenate_cubes(IrisTest):

    """Test the concatenate_cubes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the utility returns an iris.cube.Cube."""
        result = concatenate_cubes(self.cube)
        self.assertIsInstance(result, Cube)

    def test_identical_cubes(self):
        """
        Test that the utility returns the expected error message,
        if an attempt is made to concatenate identical cubes.
        """
        cubes = iris.cube.CubeList([self.cube, self.cube])
        msg = "An unexpected problem prevented concatenation."
        with self.assertRaisesRegex(ConcatenateError, msg):
            concatenate_cubes(cubes)

    def test_cubelist_type_and_data(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        resulting data, if a CubeList containing non-identical cubes
        (different values for the time coordinate) is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()

        cube3 = self.cube.copy()
        cube3.transpose([1, 0, 2, 3])
        expected_result = np.vstack([cube3.data, cube3.data])

        cube2.coord("time").points = 402195.0

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(cubelist)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(expected_result, result.data)

    def test_cubelist_different_number_of_realizations(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        realizations, if a CubeList containing cubes with different numbers
        of realizations are passed in as the input.
        """
        cube1 = self.cube.copy()

        cube3 = iris.cube.CubeList([])
        for cube in cube1.slices_over("realization"):
            if cube.coord("realization").points == 0:
                cube2 = cube
            elif cube.coord("realization").points in [1, 2]:
                cube3.append(cube)
        cube3 = cube3.merge_cube()

        cubelist = iris.cube.CubeList([cube2, cube3])

        result = concatenate_cubes(cubelist)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, [0, 1, 2])

    def test_cubelist_different_number_of_realizations_time(self):
        """
        Test that the utility returns the expected error message, if a
        CubeList containing cubes with different numbers of realizations are
        passed in as the input, and the slicing done, in order to help the
        concatenation is only done over time.
        """
        cube1 = self.cube.copy()

        cube3 = iris.cube.CubeList([])
        for cube in cube1.slices_over("realization"):
            if cube.coord("realization").points == 0:
                cube2 = cube
            elif cube.coord("realization").points in [1, 2]:
                cube3.append(cube)
        cube3 = cube3.merge_cube()

        cubelist = iris.cube.CubeList([cube2, cube3])
        msg = "failed to concatenate into a single cube"
        with self.assertRaisesRegex(ConcatenateError, msg):
            concatenate_cubes(cubelist, coords_to_slice_over=["time"])

    def test_cubelist_slice_over_time_only(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        time coordinate, if a CubeList containing cubes with different
        timesteps is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()

        cube2.coord("time").points = 402195.0

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(
            cubelist, coords_to_slice_over=["time"])
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("time").points, [402192.5, 402195.0])

    def test_cubelist_slice_over_realization_only(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        realization coordinate, if a CubeList containing cubes with different
        realizations is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()

        cube2.coord("time").points = 402195.0

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(
            cubelist, coords_to_slice_over=["realization"])
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, [0, 1, 2])

    def test_cubelist_with_forecast_reference_time_only(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        resulting data, if a CubeList containing cubes with different
        forecast_reference_time coordinates is passed in as the input.
        This makes sure that the forecast_reference_time from the input cubes
        is maintained within the output cube, after concatenation.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.5
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube1.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube2.add_aux_coord(
            DimCoord([402195.5], "forecast_reference_time", units=tunit))

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(cubelist)
        self.assertArrayAlmostEqual(
            result.coord("forecast_reference_time").points,
            [402192.5, 402195.5])

    def test_cubelist_different_var_names(self):
        """
        Test that the utility returns an iris.cube.Cube, if a CubeList
        containing non-identical cubes is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.5

        cube1.coord("time").var_name = "time_0"
        cube2.coord("time").var_name = "time_1"

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(cubelist)
        self.assertIsInstance(result, Cube)


class Test_merge_cubes(IrisTest):

    """Test the merge_cubes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.cube_ukv = self.cube.extract(iris.Constraint(realization=1))
        self.cube_ukv.remove_coord('realization')
        self.cube_ukv.attributes.update({'grid_id': 'ukvx_standard_v1'})
        self.cube_ukv.attributes.update({'title':
                                         'Operational UKV Model Forecast'})
        self.cube_ukv_T1 = self.cube_ukv.copy()
        self.cube_ukv_T2 = self.cube_ukv.copy()
        add_forecast_reference_time_and_forecast_period(self.cube_ukv,
                                                        fp_point=4.0)
        add_forecast_reference_time_and_forecast_period(self.cube_ukv_T1,
                                                        fp_point=5.0)
        add_forecast_reference_time_and_forecast_period(self.cube_ukv_T2,
                                                        fp_point=6.0)
        add_forecast_reference_time_and_forecast_period(self.cube,
                                                        fp_point=7.0)
        self.cube.attributes.update({'grid_id': 'enukx_standard_v1'})
        self.cube.attributes.update({'title':
                                     'Operational Mogreps UK Model Forecast'})
        self.prob_ukv = set_up_probability_above_threshold_temperature_cube()
        self.prob_ukv.attributes.update({'grid_id': 'ukvx_standard_v1'})
        self.prob_ukv.attributes.update({'title':
                                         'Operational UKV Model Forecast'})
        self.prob_enuk = set_up_probability_above_threshold_temperature_cube()
        self.prob_enuk.attributes.update({'grid_id': 'enukx_standard_v1'})
        self.prob_enuk.attributes.update(
            {'title':
             'Operational Mogreps UK Model Forecast'})

    @ManageWarnings(record=True)
    def test_basic(self, warning_list=None):
        """Test that the utility returns an iris.cube.Cube."""
        result = merge_cubes(self.cube)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, Cube)

    def test_identical_cubes(self):
        """Test that merging identical cubes fails."""
        cubes = iris.cube.CubeList([self.cube, self.cube])
        msg = "failed to merge into a single cube"
        with self.assertRaisesRegex(DuplicateDataError, msg):
            merge_cubes(cubes)

    def test_lagged_ukv(self):
        """Test Lagged ukv merge OK"""
        cubes = iris.cube.CubeList([self.cube_ukv,
                                    self.cube_ukv_T1,
                                    self.cube_ukv_T2])
        result = merge_cubes(cubes)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, [6.0, 5.0, 4.0])

    def test_multi_model(self):
        """Test Multi models merge OK"""
        cubes = iris.cube.CubeList([self.cube, self.cube_ukv])
        result = merge_cubes(cubes)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("model_realization").points, [0., 1., 2., 1000.])

    def test_threshold_data(self):
        """Test threshold data merges OK"""
        cubes = iris.cube.CubeList([self.prob_ukv, self.prob_enuk])
        result = merge_cubes(cubes)
        self.assertArrayAlmostEqual(
            result.coord("model_id").points, [0, 1000])


class Test_equalise_cubes(IrisTest):

    """Test the_equalise_cubes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.cube_ukv = self.cube.extract(iris.Constraint(realization=1))
        self.cube_ukv.remove_coord('realization')
        self.cube_ukv.attributes.update({'grid_id': 'ukvx_standard_v1'})
        self.cube_ukv.attributes.update({'title':
                                         'Operational UKV Model Forecast'})
        add_forecast_reference_time_and_forecast_period(self.cube_ukv,
                                                        fp_point=4.0)
        add_forecast_reference_time_and_forecast_period(self.cube,
                                                        fp_point=7.0)
        self.cube.attributes.update({'grid_id': 'enukx_standard_v1'})
        self.cube.attributes.update({'title':
                                     'Operational Mogreps UK Model Forecast'})
        self.cube.attributes["history"] = (
            "2017-01-18T08:59:53: StaGE Decoupler")
        self.cube_ukv.attributes["history"] = (
            "2017-01-19T08:59:53: StaGE Decoupler")

    @ManageWarnings(record=True)
    def test_basic(self, warning_list=None):
        """Test that the utility returns an iris.cube.CubeList."""
        cubes = self.cube
        if isinstance(cubes, iris.cube.Cube):
            cubes = iris.cube.CubeList([cubes])
        result = equalise_cubes(cubes)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_equalise_attributes(self):
        """Test that the utility equalises the attributes as expected"""
        cubelist = iris.cube.CubeList([self.cube_ukv, self.cube])
        result = equalise_cubes(cubelist)
        self.assertArrayAlmostEqual(result[0].coord("model_id").points,
                                    np.array([0]))
        self.assertEqual(result[0].coord("model").points[0],
                         'Operational UKV Model Forecast')
        self.assertArrayAlmostEqual(result[1].coord("model_id").points,
                                    np.array([1000]))
        self.assertEqual(result[1].coord("model").points[0],
                         'Operational Mogreps UK Model Forecast')
        self.assertNotIn("title", result[0].attributes)
        self.assertNotIn("title", result[1].attributes)
        self.assertAlmostEqual(result[0].attributes["grid_id"],
                               result[1].attributes["grid_id"])
        self.assertEqual(result[0].attributes["grid_id"],
                         'ukx_standard_v1')
        self.assertNotIn("history", result[0].attributes.keys())
        self.assertNotIn("history", result[1].attributes.keys())

    def test_strip_var_names(self):
        """Test that the utility removes var names"""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube1.coord("time").var_name = "time_0"
        cube2.coord("time").var_name = "time_1"
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = equalise_cubes(cubelist)
        self.assertIsNone(result[0].coord("time").var_name)
        self.assertIsNone(result[1].coord("time").var_name)

    def test_coords_not_equalised_if_not_merging(self):
        """Test that the coords are not equalised if not merging"""
        cubelist = iris.cube.CubeList([self.cube_ukv, self.cube])
        result = equalise_cubes(cubelist, merging=False)
        self.assertEqual(len(result),
                         len(cubelist))

    def test_coords_are_equalised_if_merging(self):
        """Test that the coords are equalised if merging"""
        cubelist = iris.cube.CubeList([self.cube_ukv, self.cube])
        result = equalise_cubes(cubelist)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[3].coord('model_realization').points,
                               1002.0)


class Test__equalise_cube_attributes(IrisTest):

    """Test the equalise_cube_attributes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.cube_ukv = self.cube.extract(iris.Constraint(realization=1))
        self.cube_ukv.remove_coord('realization')

    def test_cubelist_history_removal(self):
        """Test that the utility removes history attribute,
        if they are different.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.0
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["history"] = "2017-01-19T08:59:53: StaGE Decoupler"

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)
        self.assertNotIn("history", result[0].attributes.keys())
        self.assertNotIn("history", result[1].attributes.keys())

    def test_cubelist_no_history_removal(self):
        """Test that the utility does not remove history attribute,
        if they are the same.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.0
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)
        self.assertIn("history", result[0].attributes.keys())
        self.assertIn("history", result[1].attributes.keys())

    def test_cubelist_grid_id_same(self):
        """Test that the utility updates grid_id if in list and not matching"""

        cube1 = self.cube_ukv.copy()
        cube2 = self.cube.copy()
        cube1.attributes.update({'grid_id': 'ukvx_standard_v1'})
        cube2.attributes.update({'grid_id': 'ukvx_standard_v1'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)

        self.assertEqual(result[0].attributes["grid_id"],
                         result[1].attributes["grid_id"])

    def test_cubelist_grid_id_in_list(self):
        """Test that the utility updates grid_id if in list and not matching"""

        cube1 = self.cube_ukv.copy()
        cube2 = self.cube.copy()
        cube1.attributes.update({'grid_id': 'ukvx_standard_v1'})
        cube2.attributes.update({'grid_id': 'enukx_standard_v1'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)

        self.assertEqual(result[0].attributes["grid_id"],
                         result[1].attributes["grid_id"])
        self.assertEqual(cubelist[0].attributes["grid_id"],
                         'ukx_standard_v1')

    def test_cubelist_grid_id_in_list2(self):
        """Test that the utility updates grid_id if in list and not matching
        where grid_id has already been updated to ukv_standard_v1"""

        cube1 = self.cube_ukv.copy()
        cube2 = self.cube.copy()
        cube1.attributes.update({'grid_id': 'ukvx_standard_v1'})
        cube2.attributes.update({'grid_id': 'ukx_standard_v1'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)

        self.assertEqual(result[0].attributes["grid_id"],
                         result[1].attributes["grid_id"])
        self.assertEqual(result[0].attributes["grid_id"],
                         'ukx_standard_v1')

    def test_cubelist_grid_id_not_in_list(self):
        """Test leaves grid_id alone if grid_id not matching and not in list
        In this case the cubes would not merge.
        """

        cube1 = self.cube_ukv.copy()
        cube2 = self.cube.copy()
        cube1.attributes.update({'grid_id': 'ukx_standard_v1'})
        cube2.attributes.update({'grid_id': 'unknown_grid'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)

        self.assertIn("grid_id", result[0].attributes.keys())
        self.assertEqual(result[0].attributes["grid_id"],
                         'ukx_standard_v1')
        self.assertIn("grid_id", result[1].attributes.keys())
        self.assertEqual(result[1].attributes["grid_id"],
                         'unknown_grid')

    def test_cubelist_title_identical(self):
        """Test that the utility does nothing to title if they match"""

        cube1 = self.cube_ukv.copy()
        cube2 = self.cube.copy()
        cube1.attributes.update({'title':
                                 'Operational UKV Model Forecast'})
        cube2.attributes.update({'title':
                                 'Operational UKV Model Forecast'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)
        self.assertEqual(result[0].attributes["title"],
                         result[1].attributes["title"])
        self.assertEqual(result[0].attributes["title"],
                         'Operational UKV Model Forecast')

    def test_cubelist_title(self):
        """Test that the utility adds coords for model if not matching"""

        cube1 = self.cube_ukv.copy()
        cube2 = self.cube.copy()
        cube1.attributes.update({'title':
                                 'Operational UKV Model Forecast'})
        cube2.attributes.update({'title':
                                 'Operational Mogreps UK Model Forecast'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)

        self.assertArrayAlmostEqual(result[0].coord("model_id").points,
                                    np.array([0]))
        self.assertEqual(result[0].coord("model").points[0],
                         'Operational UKV Model Forecast')
        self.assertArrayAlmostEqual(result[1].coord("model_id").points,
                                    np.array([1000]))
        self.assertEqual(result[1].coord("model").points[0],
                         'Operational Mogreps UK Model Forecast')
        self.assertNotIn("title", result[0].attributes.keys())
        self.assertNotIn("title", result[1].attributes.keys())

    @ManageWarnings(record=True)
    def test_unknown_attribute(self, warning_list=None):
        """Test that the utility returns warning and removes unknown
        mismatching attribute."""
        cube1 = self.cube_ukv.copy()
        cube2 = self.cube.copy()
        cube1.attributes.update({'unknown_attribute':
                                 '1'})
        cube2.attributes.update({'unknown_attribute':
                                 '2'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Do not know what to do with "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertNotIn("unknown_attribute",
                         result[0].attributes.keys())
        self.assertNotIn("unknown_attribute",
                         result[1].attributes.keys())


class Test__equalise_cube_coords(IrisTest):

    """Test the_equalise_cube_coords utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    @ManageWarnings(record=True)
    def test_basic(self, warning_list=None):
        """Test that the utility returns an iris.cube.CubeList."""
        result = _equalise_cube_coords(iris.cube.CubeList([self.cube]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube so no differences will be found "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_threshold_exception(self):
        """Test that an exception is raised if a threshold coordinate is
        unmatched."""
        cube = set_up_probability_above_threshold_temperature_cube()
        cube1 = cube.copy()
        cube2 = cube.copy()
        cube2.remove_coord("threshold")
        cubes = iris.cube.CubeList([cube1, cube2])
        msg = "threshold coordinates must match to merge"
        with self.assertRaisesRegex(ValueError, msg):
            _equalise_cube_coords(cubes)

    def test_model_id_without_realization(self):
        """Test that if model_id is an unmatched coordinate, and the cubes
        do not have a realization coordinate the code does not try and
        add realization coordinate."""
        cube1 = self.cube.copy()[0]
        cube2 = self.cube.copy()[0]
        cube1.remove_coord("realization")
        cube2.remove_coord("realization")
        model_id_coord = DimCoord(
            np.array([1000*1], np.int), long_name='model_id')
        cube1.add_aux_coord(model_id_coord)
        cube1 = iris.util.new_axis(cube1)
        cubes = iris.cube.CubeList([cube1, cube2])
        result = _equalise_cube_coords(cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0].coords("realization"))
        self.assertFalse(result[1].coords("realization"))
        self.assertTrue(result[0].coords("model_id"))

    def test_model_id_with_realization_exception(self):
        """Test that an exception is raised if a cube has multiple model_id
        points."""
        cube1 = self.cube.copy()
        model_id_coord = DimCoord(
            np.array([1000], np.int), long_name='model_id')
        cube1.add_aux_coord(model_id_coord)
        cube1 = iris.util.new_axis(cube1, "model_id")
        cube2 = cube1.copy()
        cube2.coord("model_id").points = 200
        cube1 = iris.cube.CubeList([cube1, cube2]).concatenate_cube()
        cube2 = self.cube.copy()[0]
        cube2.remove_coord("realization")
        cubes = iris.cube.CubeList([cube1, cube2])
        msg = "Model_id has more than one point"
        with self.assertRaisesRegex(ValueError, msg):
            _equalise_cube_coords(cubes)

    def test_model_id_with_realization_in_cube(self):
        """Test if model_id is an unmatched coordinate, a cube has a
        realization coordinate and the cube being inspected has a realization
        coordinate."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()[0]
        cube2.remove_coord("realization")
        model_id_coord = DimCoord(
            np.array([1000*1], np.int), long_name='model_id')
        cube1.add_aux_coord(model_id_coord)
        cube1 = iris.util.new_axis(cube1, "model_id")
        cubes = iris.cube.CubeList([cube1, cube2])
        result = _equalise_cube_coords(cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 4)
        self.assertTrue(result[0].coords("realization"))
        self.assertFalse(result[3].coords("realization"))
        self.assertTrue(result[0].coords("model_id"))

    def test_model_id_with_realization_not_in_cube(self):
        """Test if model_id is an unmatched coordinate, a cube has a
        realization coordinate and the cube being inspected does not have a
        realization coordinate."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube1.remove_coord("realization")
        model_id_coord = DimCoord(
            np.array([1000*1], np.int), long_name='model_id')
        cube2.add_aux_coord(model_id_coord)
        cube2 = iris.util.new_axis(cube2, "model_id")
        cubes = iris.cube.CubeList([cube1, cube2])
        result = _equalise_cube_coords(cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 4)
        self.assertFalse(result[0].coords("realization"))
        self.assertTrue(result[1].coords("realization"))
        self.assertTrue(result[1].coords("model_id"))


class Test__equalise_cell_methods(IrisTest):

    """Test the_equalise_cube_coords utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.cell_method1 = iris.coords.CellMethod("mean", "realization")
        self.cell_method2 = iris.coords.CellMethod("mean", "time")
        self.cell_method3 = iris.coords.CellMethod("max", "neighbourhood")

    def test_basic(self):
        """Test returns an iris.cube.CubeList."""
        result = _equalise_cell_methods(iris.cube.CubeList([self.cube,
                                                            self.cube]))
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].is_compatible(result[1]))

    @ManageWarnings(record=True)
    def test_single_cube_in_cubelist(self, warning_list=None):
        """Test single cube in CubeList returns CubeList and raises warning."""
        result = _equalise_cell_methods(iris.cube.CubeList([self.cube]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = ("Only a single cube so no differences "
                       "will be found in cell methods")
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_different_cell_methods(self):
        """Test returns an iris.cube.CubeList with matching cell methods."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube3 = self.cube.copy()
        cube1.cell_methods = tuple([self.cell_method1, self.cell_method2])
        cube2.cell_methods = tuple([self.cell_method1, self.cell_method2,
                                    self.cell_method3])
        cube3.cell_methods = tuple([self.cell_method1, self.cell_method3])
        result = _equalise_cell_methods(iris.cube.CubeList([cube1,
                                                            cube2,
                                                            cube3]))
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result[0].cell_methods), 1)
        check = result[1].cell_methods[0] == self.cell_method1
        self.assertTrue(check)


class Test_compare_attributes(IrisTest):
    """Test the compare_attributes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.cube_ukv = self.cube.extract(iris.Constraint(realization=1))
        self.cube_ukv.remove_coord('realization')
        self.cube_ukv.attributes.update({'grid_id': 'ukvx_standard_v1'})
        self.cube_ukv.attributes.update({'title':
                                         'Operational UKV Model Forecast'})
        self.cube.attributes.update({'grid_id': 'enukx_standard_v1'})
        self.cube.attributes.update({'title':
                                     'Operational Mogreps UK Model Forecast'})

    def test_basic(self):
        """Test that the utility returns a list and have no differences."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_attributes(cubelist)
        self.assertIsInstance(result, list)
        self.assertAlmostEqual(result, [{}, {}])

    @ManageWarnings(record=True)
    def test_warning(self, warning_list=None):
        """Test that the utility returns warning if only one cube supplied."""
        result = (
            compare_attributes(iris.cube.CubeList([self.cube])))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube so no differences will be found "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertAlmostEqual(result, [])

    def test_history_attribute(self):
        """Test that the utility returns diff when history do not match"""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["history"] = "2017-01-19T08:59:53: StaGE Decoupler"
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_attributes(cubelist)
        self.assertEqual(result,
                         [{'history':
                           '2017-01-18T08:59:53: StaGE Decoupler'},
                          {'history':
                           '2017-01-19T08:59:53: StaGE Decoupler'}])

    def test_multiple_differences(self):
        """Test that the utility returns multiple differences"""
        cube_no_attributes = set_up_temperature_cube()
        cubelist = iris.cube.CubeList([cube_no_attributes,
                                       self.cube, self.cube_ukv])
        result = compare_attributes(cubelist)
        self.assertAlmostEqual(result,
                               [{},
                                {'grid_id': 'enukx_standard_v1',
                                 'title':
                                 'Operational Mogreps UK Model Forecast'},
                                {'grid_id': 'ukvx_standard_v1',
                                 'title':
                                 'Operational UKV Model Forecast'}])


class Test_compare_coords(IrisTest):
    """Test the compare_coords utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the utility returns a list."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_coords(cubelist)
        self.assertIsInstance(result, list)

    @ManageWarnings(record=True)
    def test_catch_warning(self, warning_list=None):
        """Test warning is raised if the input is cubelist of length 1."""
        cube = self.cube.copy()
        result = compare_coords(iris.cube.CubeList([cube]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube so no differences will be found "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertAlmostEqual(result, [])

    def test_first_cube_has_extra_dimension_coordinates(self):
        """Test for comparing coordinate between cubes, where the first
        cube in the list has extra dimension coordinates."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        height_coord = DimCoord([5.0], standard_name="height", units="m")
        cube1.add_aux_coord(height_coord)
        cube1 = iris.util.new_axis(cube1, "height")
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_coords(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 0)
        self.assertEqual(result[0]["height"]["coord"].points, np.array([5.]))
        self.assertEqual(result[0]["height"]["coord"].standard_name, "height")
        self.assertEqual(result[0]["height"]["coord"].units, Unit("m"))
        self.assertEqual(result[0]["height"]["data_dims"], 0)
        self.assertEqual(result[0]["height"]["aux_dims"], None)

    def test_second_cube_has_extra_dimension_coordinates(self):
        """Test for comparing coordinate between cubes, where the second
        cube in the list has extra dimension coordinates."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        height_coord = DimCoord([5.0], standard_name="height", units="m")
        cube2.add_aux_coord(height_coord)
        cube2 = iris.util.new_axis(cube2, "height")
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_coords(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 0)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(result[1]["height"]["coord"].points, np.array([5.]))
        self.assertEqual(result[1]["height"]["coord"].standard_name, "height")
        self.assertEqual(result[1]["height"]["coord"].units, Unit("m"))
        self.assertEqual(result[1]["height"]["data_dims"], 0)
        self.assertEqual(result[1]["height"]["aux_dims"], None)

    def test_first_cube_has_extra_auxiliary_coordinates(self):
        """Test for comparing coordinate between cubes, where the first
        cube in the list has extra auxiliary coordinates."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        fp_coord = AuxCoord(
            [3.0], standard_name="forecast_period", units="hours")
        cube1.add_aux_coord(fp_coord, data_dims=1)
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_coords(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 0)
        self.assertEqual(result[0]["forecast_period"]["coord"].points,
                         np.array([3.0]))
        self.assertEqual(result[0]["forecast_period"]["coord"].standard_name,
                         "forecast_period")
        self.assertEqual(result[0]["forecast_period"]["coord"].units,
                         Unit("hours"))
        self.assertEqual(result[0]["forecast_period"]["data_dims"], None)
        self.assertEqual(result[0]["forecast_period"]["aux_dims"], 1)

    def test_second_cube_has_extra_auxiliary_coordinates(self):
        """Test for comparing coordinate between cubes, where the second
        cube in the list has extra auxiliary coordinates."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        fp_coord = AuxCoord(
            [3.0], standard_name="forecast_period", units="hours")
        cube2.add_aux_coord(fp_coord, data_dims=1)
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_coords(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 0)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(result[1]["forecast_period"]["coord"].points,
                         np.array([3.0]))
        self.assertEqual(result[1]["forecast_period"]["coord"].standard_name,
                         "forecast_period")
        self.assertEqual(result[1]["forecast_period"]["coord"].units,
                         Unit("hours"))
        self.assertEqual(result[1]["forecast_period"]["data_dims"], None)
        self.assertEqual(result[1]["forecast_period"]["aux_dims"], 1)


class Test_build_coordinate(IrisTest):
    """Test the build_coordinate utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the utility returns a coord."""
        result = build_coordinate([1.0], long_name='testing')
        self.assertIsInstance(result, DimCoord)

    def test_use_many_keyword_arguments(self):
        """Test that a coordinate is built when most of the keyword arguments
        are specified."""
        standard_name = "height"
        long_name = "height"
        var_name = "height"
        coord_type = AuxCoord
        data_type = np.int64
        units = "m"
        bounds = np.array([0.5, 1.5])
        coord_system = TransverseMercator
        result = build_coordinate(
            [1.0], standard_name=standard_name, long_name=long_name,
            var_name=var_name, coord_type=coord_type, data_type=data_type,
            units=units, bounds=bounds, coord_system=coord_system)
        self.assertIsInstance(result, AuxCoord)
        self.assertEqual(result.standard_name, "height")
        self.assertEqual(result.long_name, "height")
        self.assertEqual(result.var_name, "height")
        self.assertIsInstance(result.points[0], np.int64)
        self.assertEqual(result.units, Unit("m"))
        self.assertArrayAlmostEqual(result.bounds, np.array([[0.5, 1.5]]))
        self.assertArrayAlmostEqual(result.points, np.array([1.0]))
        self.assertEqual(
            result.coord_system, TransverseMercator)

    def test_template_coord(self):
        """Test that a coordinate can be built from a template coordinate."""
        template_coord = DimCoord([2.0], standard_name="height", units="m")
        result = build_coordinate([5.0, 10.0], template_coord=template_coord)
        self.assertIsInstance(result, DimCoord)
        self.assertEqual(result.standard_name, "height")
        self.assertEqual(result.units, Unit("m"))
        self.assertArrayAlmostEqual(result.points, np.array([5.0, 10.0]))

    def test_custom_function(self):
        """Test that a coordinate can be built when using a custom function."""
        def divide_data(data):
            """Basic custom function for testing in build_coordinate"""
            return data/2
        result = build_coordinate(
            [1.0], long_name="realization", custom_function=divide_data)
        self.assertArrayAlmostEqual(result.points, np.array([0.5]))

    def test_build_latitude_coordinate(self):
        """Test building a latitude coordinate."""
        latitudes = np.linspace(-90, 90, 20)
        coord_system = iris.coord_systems.GeogCS(6371229.0)
        result = build_coordinate(latitudes, long_name='latitude',
                                  units='degrees',
                                  coord_system=coord_system)
        self.assertArrayEqual(result.points, latitudes)
        self.assertEqual(result.name(), 'latitude')
        self.assertIsInstance(result, DimCoord)
        self.assertEqual(result.units, 'degrees')


class Test_add_renamed_cell_method(IrisTest):
    """Class to test the add_renamed_cell_method function"""

    def setUp(self):
        """Set up input cube for tests"""
        self.cube = set_up_temperature_cube()
        self.cell_method = iris.coords.CellMethod(method='mean', coords='time')
        self.cube.add_cell_method(self.cell_method)

    def test_basic(self):
        """Basic test for one cell method on input cube"""
        add_renamed_cell_method(self.cube, self.cell_method, 'weighted_mean')
        expected_cell_method = iris.coords.CellMethod(method='weighted_mean',
                                                      coords='time')
        self.assertEqual(self.cube.cell_methods, (expected_cell_method,))

    def test_only_difference_is_name(self):
        """Testing that the input cell method and the new cell method only
        differ by name"""
        add_renamed_cell_method(self.cube, self.cell_method, 'weighted_mean')
        expected_cell_method = iris.coords.CellMethod(method='weighted_mean',
                                                      coords='time')
        self.assertEqual(self.cube.cell_methods, (expected_cell_method,))
        new_cell_method = self.cube.cell_methods[0]
        self.assertEqual(self.cell_method.coord_names,
                         new_cell_method.coord_names)
        self.assertEqual(self.cell_method.intervals, new_cell_method.intervals)
        self.assertEqual(self.cell_method.comments, new_cell_method.comments)

    def test_no_cell_method_in_input_cube(self):
        """Testing that when there are no cell methods on the input cube then
        the new cell method still gets added as expected."""
        self.cube.cell_methods = ()
        add_renamed_cell_method(self.cube, self.cell_method, 'weighted_mean')
        expected_cell_method = iris.coords.CellMethod(method='weighted_mean',
                                                      coords='time')
        self.assertEqual(self.cube.cell_methods, (expected_cell_method,))

    def test_wrong_input(self):
        """Test a sensible error is raised when the wrong input is passed in"""
        self.cube.cell_methods = ()
        message = ('Input Cell_method is not an instance of '
                   'iris.coord.CellMethod')
        with self.assertRaisesRegex(TypeError, message):
            add_renamed_cell_method(self.cube, 'not_a_cell_method',
                                    'weighted_mean')

    def test_multiple_cell_methods_in_input_cube(self):
        """Test that other cell methods are preserved."""
        extra_cell_method = iris.coords.CellMethod(method='max',
                                                   coords='realization')
        self.cube.cell_methods = (self.cell_method, extra_cell_method)
        add_renamed_cell_method(self.cube, self.cell_method, 'weighted_mean')
        expected_cell_method = iris.coords.CellMethod(method='weighted_mean',
                                                      coords='time')
        self.assertEqual(self.cube.cell_methods,
                         (extra_cell_method, expected_cell_method,))


class Test_sort_coord_in_cube(IrisTest):
    """Class to test the sort_coord_in_cube function."""

    def setUp(self):
        """Set up a cube."""
        self.ascending_height_points = np.array([5., 10., 20.])
        cube = set_up_height_cube(self.ascending_height_points)[:, 0, :, :, :]
        data = np.zeros(cube.shape)
        data[0] = np.ones(cube[0].shape, dtype=np.int32)
        data[1] = np.full(cube[1].shape, 2, dtype=np.int32)
        data[2] = np.full(cube[2].shape, 3, dtype=np.int32)
        cube.data = data
        self.ascending_cube = cube
        descending_cube = cube.copy()
        self.descending_height_points = np.array([20., 10., 5.])
        descending_cube.coord("height").points = self.descending_height_points
        self.descending_cube = descending_cube

    def test_ascending_then_ascending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in ascending order."""
        expected_data = np.array(
            [[[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00]]],
             [[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00]]]])
        coord_name = "height"
        result = sort_coord_in_cube(self.ascending_cube, coord_name)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(self.ascending_cube.coord_dims(coord_name),
                         result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            self.ascending_height_points,
            result.coord(coord_name).points)
        self.assertDictEqual(
            self.ascending_cube.coord(coord_name).attributes,
            {"positive": "up"})
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_ascending_then_descending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in descending order."""
        expected_data = np.array(
            [[[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00]]],
             [[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00]]]])
        coord_name = "height"
        result = sort_coord_in_cube(
            self.ascending_cube, coord_name, order="descending")
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(self.descending_cube.coord_dims(coord_name),
                         result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            self.descending_height_points, result.coord(coord_name).points)
        self.assertDictEqual(
            result.coord(coord_name).attributes, {"positive": "down"})
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_descending_then_ascending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in ascending order."""
        expected_data = np.array(
            [[[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00]]],
             [[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00]]]])
        coord_name = "height"
        result = sort_coord_in_cube(self.descending_cube, coord_name)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(self.ascending_cube.coord_dims(coord_name),
                         result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            self.ascending_height_points, result.coord(coord_name).points)
        self.assertDictEqual(
            result.coord(coord_name).attributes, {"positive": "up"})
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_descending_then_descending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in descending order."""
        expected_data = np.array(
            [[[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00]]],
             [[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00]]]])
        coord_name = "height"
        result = sort_coord_in_cube(
            self.descending_cube, coord_name, order="descending")
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(self.descending_cube.coord_dims(coord_name),
                         result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            self.descending_height_points, result.coord(coord_name).points)
        self.assertDictEqual(
            result.coord(coord_name).attributes, {"positive": "down"})
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_latitude(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate (latitude).
        The points in the resulting cube should now be in descending order."""
        expected_data = np.array(
            [[[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [6.00, 1.00, 1.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [6.00, 2.00, 2.00]]],
             [[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [6.00, 3.00, 3.00]]]])
        self.ascending_cube.data[:, :, 0, 0] = 6.0
        expected_points = np.array([45., 0., -45])
        coord_name = "latitude"
        result = sort_coord_in_cube(
            self.ascending_cube, coord_name, order="descending")
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(
            self.ascending_cube.coord_dims(coord_name),
            result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            expected_points, result.coord(coord_name).points)
        self.assertArrayAlmostEqual(result.data, expected_data)

    @ManageWarnings(record=True)
    def test_warn_raised_for_circular_coordinate(self, warning_list=None):
        """Test that a warning is successfully raised when circular
        coordinates are sorted."""
        self.ascending_cube.data[:, :, 0, 0] = 6.0
        coord_name = "latitude"
        self.ascending_cube.coord(coord_name).circular = True
        result = sort_coord_in_cube(
            self.ascending_cube, coord_name, order="descending")
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "The latitude coordinate is circular."
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, iris.cube.Cube)


class Test_enforce_coordinate_ordering(IrisTest):

    """Test the enforce_coordinate_ordering utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the function returns an iris.cube.Cube."""
        result = enforce_coordinate_ordering(self.cube, "realization")
        self.assertIsInstance(result, Cube)

    def test_move_coordinate_to_start_when_already_at_start(self):
        """Test that a cube with the expected data contents is returned when
        the coordinate to be reordered is already in the desired position."""
        result = enforce_coordinate_ordering(self.cube, "realization")
        self.assertEqual(result.coord_dims("realization")[0], 0)
        self.assertArrayAlmostEqual(result.data, self.cube.data)

    def test_move_coordinate_to_start(self):
        """Test that a cube with the expected data contents is returned when
        the time coordinate is reordered to be the first coordinate in the
        cube."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(cube, "time")
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_move_coordinate_to_end(self):
        """Test that a cube with the expected data contents is returned when
        the realization coordinate is reordered to be the last coordinate in
        the cube."""
        expected = self.cube.copy()
        expected.transpose([1, 2, 3, 0])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(cube, "realization", anchor="end")
        self.assertEqual(result.coord_dims("realization")[0], 3)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_move_coordinate_to_start_with_list(self):
        """Test that a cube with the expected data contents is returned when
        the time coordinate is reordered to be the first coordinate in the
        cube. The coordinate name to be reordered is specified as a list."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(cube, ["time"])
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_move_multiple_coordinate_to_start_with_list(self):
        """Test that a cube with the expected data contents is returned when
        the time and realization coordinates are reordered to be the first
        coordinates in the cube. The coordinate name to be reordered is
        specified as a list."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(cube, ["time", "realization"])
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertEqual(result.coord_dims("realization")[0], 1)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_move_multiple_coordinate_to_end_with_list(self):
        """Test that a cube with the expected data contents is returned when
        the time and realization coordinates are reordered to be the last
        coordinates in the cube. The coordinate name to be reordered is
        specified as a list."""
        expected = self.cube.copy()
        expected.transpose([2, 3, 1, 0])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(
            cube, ["time", "realization"], anchor="end")

        self.assertEqual(result.coord_dims("time")[0], 2)
        self.assertEqual(result.coord_dims("realization")[0], 3)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_full_reordering(self):
        """Test that a cube with the expected data contents is returned when
        all the coordinates within the cube are reordered into the order
        specified by the names within the input list."""
        expected = self.cube.copy()
        expected.transpose([2, 0, 3, 1])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(
            cube, ["latitude", "realization", "longitude", "time"])
        self.assertEqual(result.coord_dims("latitude")[0], 0)
        self.assertEqual(result.coord_dims("realization")[0], 1)
        self.assertEqual(result.coord_dims("longitude")[0], 2)
        self.assertEqual(result.coord_dims("time")[0], 3)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_partial_names(self):
        """Test that a cube with the expected data contents is returned when
        the names provided are partial matches of the names of the coordinates
        within the cube."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(cube, ["tim", "realiz"])
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertEqual(result.coord_dims("realization")[0], 1)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_partial_names_multiple_matches_exception(self):
        """Test that the expected exception is raised when the names provided
        are partial matches of the names of multiple coordinates within the
        cube."""
        expected = self.cube.copy()
        expected.transpose([2, 3, 0, 1])
        cube = self.cube.copy()
        msg = "More than 1 coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            enforce_coordinate_ordering(cube, ["l", "e"])

    def test_include_extra_coordinates(self):
        """Test that a cube with the expected data contents is returned when
        extra coordinates are passed in for reordering but these coordinates
        are not present within the cube."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(
            cube, ["time", "realization", "nonsense"])
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertEqual(result.coord_dims("realization")[0], 1)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_force_promotion_of_scalar(self):
        """Test that a cube with the expected data contents is returned when
        the probabilistic dimension is a scalar coordinate, which is promoted
        to a dimension coordinate."""
        cube = self.cube[0, :, :, :]
        result = enforce_coordinate_ordering(
            cube, "realization", promote_scalar=True)
        self.assertEqual(result.coord_dims("realization")[0], 0)
        self.assertArrayAlmostEqual(result.data, [cube.data])

    def test_do_not_promote_scalar(self):
        """Test that a cube with the expected data contents is returned when
        the probabilistic dimension is a scalar coordinate, which is not
        promoted to a dimension coordinate."""
        cube = self.cube[0, :, :, :]
        result = enforce_coordinate_ordering(cube, "realization")
        self.assertFalse(result.coord_dims("realization"))
        self.assertArrayAlmostEqual(result.data, cube.data)

    def test_coordinate_raise_exception(self):
        """Test that the expected error message is raised when the required
        probabilistic dimension is not available in the cube."""
        cube = self.cube[0, :, :, :]
        cube.remove_coord("realization")
        msg = "The requested coordinate"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            enforce_coordinate_ordering(
                cube, "realization", raise_exception=True)


class Test_enforce_float32_precision(IrisTest):
    """ Test the enforce_float32_precision utility."""

    def setUp(self):
        """Create two temperature cubes to test with."""
        self.cube1 = set_up_temperature_cube()
        self.cube2 = set_up_temperature_cube()

    def test_basic(self):
        """Test that the function will return a single iris.cube.Cube with
           float32 precision."""
        result1 = self.cube1
        enforce_float32_precision(result1)
        self.assertEqual(result1.dtype, np.float32)

    def test_process_list(self):
        """Test that the function will return a list of cubes with
           float32 precision."""
        result1 = self.cube1
        result2 = self.cube2
        enforce_float32_precision([result1, result2])
        self.assertEqual(result1.dtype, np.float32)
        self.assertEqual(result2.dtype, np.float32)

    def test_process_none(self):
        """Test that the function ignores None types."""
        result1 = self.cube1
        result2 = None
        enforce_float32_precision([result1, result2])
        self.assertEqual(result1.dtype, np.float32)
        self.assertIsNone(result2)


class Test_clip_cube_data(IrisTest):
    """Test the clip_cube_data utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.minimum_value = self.cube.data.min()
        self.maximum_value = self.cube.data.max()
        self.processed_cube = self.cube.copy(
            data=self.cube.data*2.0 - self.cube.data.mean())

    def test_basic(self):
        """Test that the utility returns a cube."""
        result = clip_cube_data(self.processed_cube,
                                self.minimum_value, self.maximum_value)
        self.assertIsInstance(result, Cube)

    def test_clipping(self):
        """Test that the utility clips the processed cube to the same limits
        as the input cube."""
        result = clip_cube_data(self.processed_cube,
                                self.minimum_value, self.maximum_value)
        self.assertEqual(result.data.min(), self.minimum_value)
        self.assertEqual(result.data.max(), self.maximum_value)

    def test_clipping_slices(self):
        """Test that the utility clips the processed cube to the same limits
        as the input cube, and that it does this when slicing over multiple
        x-y planes."""
        cube = set_up_probability_above_threshold_temperature_cube()
        minimum_value = cube.data.min()
        maximum_value = cube.data.max()
        processed_cube = cube.copy(data=cube.data*2.0 - cube.data.mean())
        result = clip_cube_data(processed_cube, minimum_value, maximum_value)
        self.assertEqual(result.data.min(), minimum_value)
        self.assertEqual(result.data.max(), maximum_value)
        self.assertEqual(result.attributes, processed_cube.attributes)
        self.assertEqual(result.cell_methods, processed_cube.cell_methods)


if __name__ == '__main__':
    unittest.main()
