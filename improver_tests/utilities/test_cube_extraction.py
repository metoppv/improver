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
""" Unit tests for cube extraction utilities """

import collections
import unittest

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.cube_extraction import (
    apply_extraction, create_constraint, create_range_constraint,
    extract_subcube, is_complex_parsing_required, parse_constraint_list)

from ..set_up_test_cubes import set_up_probability_cube


def islambda(function):
    """
    Test function to determine whether an object is a lambda function.

    Args:
        function (object):
            The object to be tested to determine if it is a lambda function.
    Returns:
        bool:
            True if the input object is a lambda function, False if not.
    """
    return (isinstance(function, collections.Callable) and
            function.__name__ == '<lambda>')


def set_up_precip_probability_cube():
    """
    Set up a cube with spatial probabilities of precipitation at three
    exceedance thresholds
    """
    data = np.array([[[0.85, 0.95, 0.73],
                      [0.75, 0.85, 0.65],
                      [0.70, 0.80, 0.62]],
                     [[0.18, 0.20, 0.15],
                      [0.11, 0.16, 0.09],
                      [0.10, 0.14, 0.03]],
                     [[0.03, 0.04, 0.01],
                      [0.02, 0.02, 0.00],
                      [0.01, 0.00, 0.00]]], dtype=np.float32)
    MMH_TO_MS = 0.001 / 3600.

    cube = set_up_probability_cube(
        data, MMH_TO_MS * np.array([0.03, 0.1, 1.0], dtype=np.float32),
        variable_name="precipitation_rate",
        threshold_units="m s-1", spatial_grid="equalarea")
    cube.coord(axis='x').points = np.arange(3, dtype=np.float32)
    cube.coord(axis='y').points = np.arange(3, dtype=np.float32)

    return cube


class Test_create_range_constraint(IrisTest):
    """Test that the desired constraint is created from
    create_range_constraint."""

    def setUp(self):
        """Set up cube for testing range constraint."""
        self.precip_cube = set_up_precip_probability_cube()
        self.coord_name = find_threshold_coordinate(
            self.precip_cube).name()
        self.precip_cube.coord(self.coord_name).convert_units("mm h-1")
        self.expected_data = self.precip_cube[:2].data

    def test_basic(self):
        """Test that a constraint is formed correctly."""
        values = "[0.03:0.1]"
        result = create_range_constraint(self.coord_name, values)
        self.assertIsInstance(result, iris.Constraint)
        self.assertEqual(list(result._coord_values.keys()), [self.coord_name])
        result_cube = self.precip_cube.extract(result)
        self.assertArrayAlmostEqual(result_cube.data, self.expected_data)

    def test_without_square_brackets(self):
        """Test that a constraint is formed correctly when square brackets
        are not within the input."""
        values = "0.03:0.1"
        result = create_range_constraint(self.coord_name, values)
        self.assertIsInstance(result, iris.Constraint)
        self.assertEqual(list(result._coord_values.keys()), [self.coord_name])
        result_cube = self.precip_cube.extract(result)
        self.assertArrayAlmostEqual(result_cube.data, self.expected_data)


class Test_is_complex_parsing_required(IrisTest):
    """Test if the string requires complex parsing."""

    def test_basic_with_colon(self):
        """Test that a value with a colon is parsed correctly."""
        value = "1230:1240"
        result = is_complex_parsing_required(value)
        self.assertTrue(result)

    def test_basic_without_colon(self):
        """Test that a value without a colon is parsed correctly."""
        value = "12301240"
        result = is_complex_parsing_required(value)
        self.assertFalse(result)


class Test_create_constraint(IrisTest):
    """Test the creation of constraints that allow for floating point
    comparisons."""

    def setUp(self):
        """Set up coordinates that can be modified and used to test the
        lambda functions that are created for floating point values. One
        coordinate contains an integer point, the other a float point."""
        self.i_crd = iris.coords.AuxCoord(np.array([10], dtype=np.int32),
                                          long_name='a_coordinate')
        self.f_crd = iris.coords.AuxCoord(np.array([10.0], dtype=np.float32),
                                          long_name='a_coordinate')

    def test_with_string_type(self):
        """Test that an extraction that is to match a string is not changed
        by passing through this function, other than to make a single entry
        into a list."""
        value = "kittens"
        result = create_constraint(value)
        self.assertEqual(result, [value])

    def test_with_int_type(self):
        """Test that an extraction that is to match an integer results in the
        creation of a lambda function. This is done in case unit conversion
        is applied, in which case the cube data may be converted imprecisely,
        e.g. 273.15K might become 1.0E-8C, which will not match a 0C constraint
        unless we use the lambda function to add some tolerance."""
        value = 10
        result = create_constraint(value)
        self.assertTrue(islambda(result))

        crd = self.i_crd.copy(points=value)
        self.assertTrue(result(crd.cell(0)))
        crd = self.f_crd.copy(points=value)
        self.assertTrue(result(crd.cell(0)))

    def test_with_float_type(self):
        """Test that an extraction that is to match a float results in the
        creation of a lambda function which matches the expected values."""
        value = 10.0
        result = create_constraint(value)
        self.assertTrue(islambda(result))

        crd = self.i_crd.copy(points=value)
        self.assertTrue(result(crd.cell(0)))
        crd = self.f_crd.copy(points=value)
        self.assertTrue(result(crd.cell(0)))
        crd = self.i_crd.copy(points=20)
        self.assertFalse(result(crd.cell(0)))

    def test_with_float_type_multiple_values(self):
        """Test that an extraction that is to match multiple floats results in
        the creation of a lambda function which matches the expected values."""
        value = [10.0, 20.0]
        result = create_constraint(value)
        self.assertTrue(islambda(result))

        crd = self.i_crd.copy(points=value)
        self.assertTrue(result(crd.cell(0)))
        self.assertTrue(result(crd.cell(1)))
        crd = self.f_crd.copy(points=value)
        self.assertTrue(result(crd.cell(0)))
        self.assertTrue(result(crd.cell(1)))
        crd = self.f_crd.copy(points=30.0)
        self.assertFalse(result(crd.cell(0)))


class Test_parse_constraint_list(IrisTest):
    """ Test function to parse constraints and units into dictionaries """

    def setUp(self):
        """ Set up some constraints to parse """
        self.constraints = ["percentile=10", "threshold=0.1"]
        self.units = ["none", "mm h-1"]
        self.p_crd = iris.coords.AuxCoord(np.array([10], dtype=np.int32),
                                          long_name='a_coordinate')
        self.t_crd = iris.coords.AuxCoord(np.array([0.1], dtype=np.float32),
                                          long_name='a_coordinate')

    def test_basic_no_units(self):
        """ Test simple key-value splitting with no units """
        result, udict = parse_constraint_list(self.constraints)
        self.assertIsInstance(result, iris.Constraint)
        self.assertCountEqual(
            list(result._coord_values.keys()), ["threshold", "percentile"])
        cdict = result._coord_values
        self.assertTrue(islambda(cdict["percentile"]))
        self.assertTrue(cdict["percentile"](self.p_crd.cell(0)))
        self.assertFalse(cdict["percentile"](self.t_crd.cell(0)))
        self.assertTrue(islambda(cdict["threshold"]))
        self.assertTrue(cdict["threshold"](self.t_crd.cell(0)))
        self.assertFalse(cdict["threshold"](self.p_crd.cell(0)))
        self.assertFalse(udict)

    def test_whitespace(self):
        """ Test constraint parsing with padding whitespace """
        constraints = ["percentile = 10", "threshold = 0.1"]
        result, _ = parse_constraint_list(constraints)
        cdict = result._coord_values
        self.assertTrue(islambda(cdict["percentile"]))
        self.assertTrue(cdict["percentile"](self.p_crd.cell(0)))
        self.assertTrue(islambda(cdict["threshold"]))
        self.assertTrue(cdict["threshold"](self.t_crd.cell(0)))

    def test_string_constraint(self):
        """ Test that a string constraint results in a simple iris constraint,
        not a lambda function. This is created via the literal_eval ValueError.
        """
        constraints = ["percentile=kittens"]
        result, _ = parse_constraint_list(constraints)
        cdict = result._coord_values
        self.assertFalse(islambda(cdict["percentile"]))
        self.assertEqual(cdict["percentile"], "kittens")
        self.assertIsInstance(cdict, dict)

    def test_some_units(self):
        """ Test units list containing "None" elements is correctly parsed """
        _, udict = parse_constraint_list(self.constraints, units=self.units)
        self.assertEqual(udict["threshold"], "mm h-1")
        self.assertNotIn("percentile", udict.keys())

    def test_unmatched_units(self):
        """ Test for ValueError if units list does not match constraints """
        units = ["mm h-1"]
        msg = "units list must match constraints"
        with self.assertRaisesRegex(ValueError, msg):
            parse_constraint_list(self.constraints, units=units)

    def test_list_constraint(self):
        """ Test that a list of constraints is parsed correctly """
        constraints = ["threshold=[0.1,1.0]"]
        result, _ = parse_constraint_list(constraints)
        cdict = result._coord_values
        self.assertTrue(islambda(cdict["threshold"]))

    def test_range_constraint(self):
        """ Test that a constraint passed in as a range is parsed correctly """
        # create input cube
        precip_cube = set_up_precip_probability_cube()
        threshold_coord = find_threshold_coordinate(precip_cube).name()
        precip_cube.coord(threshold_coord).convert_units("mm h-1")
        # generate constraints
        constraints = ["{}=[0.03:0.1]".format(threshold_coord)]
        result, _ = parse_constraint_list(constraints)
        self.assertIsInstance(result, iris._constraints.ConstraintCombination)
        cdict = result.rhs._coord_values
        self.assertEqual(list(cdict.keys()), [threshold_coord])
        # extract from input cube
        result_cube = precip_cube.extract(result)
        self.assertArrayAlmostEqual(
            result_cube.coord(threshold_coord).points, np.array([0.03, 0.1]))


class Test_apply_extraction(IrisTest):
    """ Test function to extract subcube according to constraints """

    def setUp(self):
        """ Set up temporary input cube """
        self.precip_cube = set_up_precip_probability_cube()
        self.threshold_coord = find_threshold_coordinate(
            self.precip_cube).name()
        self.units_dict = {self.threshold_coord: "mm h-1"}

    def test_basic_no_units(self):
        """ Test cube extraction for single constraint without units """
        constraint_dict = {
            "name": "probability_of_precipitation_rate_above_threshold"}
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(self.precip_cube, constr)
        self.assertIsInstance(cube, iris.cube.Cube)
        reference_data = self.precip_cube.data
        self.assertArrayEqual(cube.data, reference_data)

    def test_basic_with_units(self):
        """ Test cube extraction for single constraint with units """
        constraint_dict = {
            self.threshold_coord:
                lambda cell: any(np.isclose(cell.point, [0.1]))}
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(self.precip_cube, constr, self.units_dict)
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertEqual(cube.coord(self.threshold_coord).units, "m s-1")
        reference_data = self.precip_cube.data[1, :, :]
        self.assertArrayEqual(cube.data, reference_data)

    def test_basic_without_reconverting_units(self):
        """ Test cube extraction for single constraint with units,
         where the coordinates are not reconverted into the original units."""
        constraint_dict = {
            self.threshold_coord:
                lambda cell: any(np.isclose(cell.point, [0.1]))}
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(
            self.precip_cube, constr, self.units_dict,
            use_original_units=False)
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertEqual(cube.coord(self.threshold_coord).units, "mm h-1")
        reference_data = self.precip_cube.data[1, :, :]
        self.assertArrayEqual(cube.data, reference_data)

    def test_multiple_constraints_with_units(self):
        """ Test behaviour with a list of constraints and units """
        constraint_dict = {
            "name": "probability_of_precipitation_rate_above_threshold",
                    self.threshold_coord:
                        lambda cell: any(np.isclose(cell.point, [0.03]))}
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(self.precip_cube, constr, self.units_dict)
        self.assertIsInstance(cube, iris.cube.Cube)
        reference_data = self.precip_cube.data[0, :, :]
        self.assertArrayEqual(cube.data, reference_data)

    def test_error_non_coord_units(self):
        """ Test error raised if units are provided for a non-coordinate
        constraint """
        constraint_dict = {
            "name": "probability_of_precipitation_rate_above_threshold"}
        units_dict = {"name": "1"}
        with self.assertRaises(CoordinateNotFoundError):
            apply_extraction(self.precip_cube, constraint_dict, units_dict)

    def test_allow_none(self):
        """ Test function returns None rather than raising an error where
        no subcubes match the required constraints, when unit conversion is
        required """
        constraint_dict = {
            "name": "probability_of_precipitation_rate_above_threshold",
                    self.threshold_coord: 5}
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(self.precip_cube, constr, self.units_dict)
        self.assertFalse(cube)

    def test_list_constraints(self):
        """ Test that a list of constraints behaves correctly """
        constraint_dict = {
            self.threshold_coord:
                lambda cell: any(np.isclose(cell.point, [0.1, 1.0]))}
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(self.precip_cube, constr, self.units_dict)
        reference_data = self.precip_cube.data[1:, :, :]
        self.assertArrayEqual(cube.data, reference_data)

    def test_range_constraints(self):
        """ Test that a list of constraints behaves correctly. This includes
        converting the units to the units that the constraints is
        defined in."""
        lower_bound = 0.03 * (1. - 1.0E-7)
        upper_bound = 0.1 * (1. + 1.0E-7)
        constraint_dict = {
            self.threshold_coord: lambda cell:
                lower_bound <= cell <= upper_bound}
        constr = iris.Constraint(coord_values=constraint_dict)
        cube = apply_extraction(self.precip_cube, constr, self.units_dict)
        reference_data = self.precip_cube.data[:2, :, :]
        self.assertArrayEqual(cube.data, reference_data)


class Test_extract_subcube(IrisTest):
    """Test that a subcube is extracted when the required constraints are
    applied."""

    def setUp(self):
        """ Set up temporary input cube """
        self.precip_cube = set_up_precip_probability_cube()

    def test_single_threshold(self):
        """Test that a single threshold is extracted correctly when using the
        key=value syntax."""
        constraints = ["precipitation_rate=0.03"]
        precip_units = ["mm h-1"]
        expected = self.precip_cube[0]
        result = extract_subcube(self.precip_cube, constraints,
                                 units=precip_units)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_multiple_thresholds(self):
        """Test that multiple thresholds are extracted correctly when using the
        key=[value1,value2] syntax."""
        constraints = ["precipitation_rate=[0.03,0.1]"]
        precip_units = ["mm h-1"]
        expected = self.precip_cube[:2]
        result = extract_subcube(self.precip_cube, constraints,
                                 units=precip_units)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_range_constraint(self):
        """Test that multiple thresholds are extracted correctly when using the
        key=[value1:value2] syntax."""
        constraints = ["projection_y_coordinate=[1:2]"]
        expected = self.precip_cube[:, 1:, :]
        result = extract_subcube(self.precip_cube, constraints)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_multiple_equality_constraints(self):
        """Test that multiple thresholds are extracted correctly when using the
        key=[value1,value2] syntax for more than one quantity (i.e. multiple
        constraints)."""
        constraints = ["precipitation_rate=[0.03,0.1]",
                       "projection_y_coordinate=[1,2]"]
        precip_units = ["mm h-1", "m"]
        expected = self.precip_cube[0:2, 1:, :]
        result = extract_subcube(self.precip_cube, constraints,
                                 units=precip_units)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_multiple_range_constraints(self):
        """Test that multiple range constraints are extracted correctly when
        using the key=[value1:value2] syntax for more than one quantity (i.e.
        multiple constraints)."""
        constraints = ["precipitation_rate=[0.03:0.1]",
                       "projection_y_coordinate=[1:2]"]
        precip_units = ["mm h-1", "m"]
        expected = self.precip_cube[0:2, 1:, :]
        result = extract_subcube(self.precip_cube, constraints,
                                 units=precip_units)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_combination_of_equality_and_range_constraints(self):
        """Test that multiple constraints are extracted correctly when
        using a combination of key=[value1,value2] and key=[value1:value2]
        syntax."""
        constraints = ["precipitation_rate=[0.03,0.1]",
                       "projection_y_coordinate=[1:2]"]
        precip_units = ["mm h-1", "m"]
        expected = self.precip_cube[0:2, 1:, :]
        result = extract_subcube(self.precip_cube, constraints,
                                 units=precip_units)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_single_threshold_use_original_units(self):
        """Test that a single threshold is extracted correctly when using the
        key=value syntax without converting the coordinate units back to the
        original units."""
        constraints = ["precipitation_rate=0.03"]
        precip_units = ["mm h-1"]
        expected = self.precip_cube[0]
        expected.coord("precipitation_rate").convert_units("mm h-1")
        result = extract_subcube(self.precip_cube, constraints,
                                 units=precip_units, use_original_units=False)
        self.assertArrayAlmostEqual(result.data, expected.data)
        self.assertEqual(expected.coord("precipitation_rate"),
                         result.coord("precipitation_rate"))


if __name__ == '__main__':
    unittest.main()
