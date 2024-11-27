# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for cube extraction utilities"""

import collections
import unittest

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.metadata.probabilistic import find_threshold_coordinate
from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.cube_extraction import (
    apply_extraction,
    create_constraint,
    cubelist_extract,
    extract_subcube,
    parse_constraint_list,
)


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
    return (
        isinstance(function, collections.Callable) and function.__name__ == "<lambda>"
    )


def set_up_precip_probability_cube():
    """
    Set up a cube with spatial probabilities of precipitation at three
    exceedance thresholds
    """
    data = np.array(
        [
            [[0.85, 0.95, 0.73], [0.75, 0.85, 0.65], [0.70, 0.80, 0.62]],
            [[0.18, 0.20, 0.15], [0.11, 0.16, 0.09], [0.10, 0.14, 0.03]],
            [[0.03, 0.04, 0.01], [0.02, 0.02, 0.00], [0.01, 0.00, 0.00]],
        ],
        dtype=np.float32,
    )
    MMH_TO_MS = 0.001 / 3600.0

    cube = set_up_probability_cube(
        data,
        MMH_TO_MS * np.array([0.03, 0.1, 1.0], dtype=np.float32),
        variable_name="precipitation_rate",
        threshold_units="m s-1",
        spatial_grid="equalarea",
        x_grid_spacing=1,
        y_grid_spacing=1,
        domain_corner=(0, 0),
    )

    return cube


def set_up_gridded_data():
    return np.arange(56).reshape((7, 8)).astype(np.float32)


def set_up_global_gridded_cube():
    return set_up_variable_cube(
        set_up_gridded_data(),
        name="screen_temperature",
        units="degC",
        spatial_grid="latlon",
        domain_corner=(45, -2),
        x_grid_spacing=2,
        y_grid_spacing=2,
    )


def set_up_uk_gridded_cube():
    return set_up_variable_cube(
        set_up_gridded_data(),
        name="screen_temperature",
        units="degC",
        spatial_grid="equalarea",
        domain_corner=(-5000, -5000),
        x_grid_spacing=2000,
        y_grid_spacing=2000,
    )


def test_cubelist_extract():
    """Test the extraction of a single cube from a cube list."""

    cube1 = set_up_variable_cube(
        np.empty((3, 3), dtype=np.float32), name="temperature", units="K"
    )
    cube2 = set_up_variable_cube(
        np.empty((3, 3), dtype=np.float32), name="precipitation_rate", units="mm h-1"
    )
    cube_list = iris.cube.CubeList([cube1, cube2])
    result = cubelist_extract(cube_list, "temperature")
    assert result == cube1


class Test_create_constraint(IrisTest):
    """Test the creation of constraints that allow for floating point
    comparisons."""

    def setUp(self):
        """Set up coordinates that can be modified and used to test the
        lambda functions that are created for floating point values. One
        coordinate contains an integer point, the other a float point."""
        self.i_crd = iris.coords.AuxCoord(
            np.array([10], dtype=np.int32), long_name="a_coordinate"
        )
        self.f_crd = iris.coords.AuxCoord(
            np.array([10.0], dtype=np.float32), long_name="a_coordinate"
        )

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
    """Test function to parse constraints and units into dictionaries"""

    def setUp(self):
        """Set up some constraints to parse"""
        self.constraints = ["percentile=10", "threshold=0.1"]
        self.units = ["none", "mm h-1"]
        self.p_crd = iris.coords.AuxCoord(
            np.array([10], dtype=np.int32), long_name="a_coordinate"
        )
        self.t_crd = iris.coords.AuxCoord(
            np.array([0.1], dtype=np.float32), long_name="a_coordinate"
        )

    def test_basic_no_units(self):
        """Test simple key-value splitting with no units"""
        result, udict, _, _ = parse_constraint_list(self.constraints)
        self.assertIsInstance(result, iris.Constraint)
        self.assertCountEqual(
            list(result._coord_values.keys()), ["threshold", "percentile"]
        )
        cdict = result._coord_values
        self.assertTrue(islambda(cdict["percentile"]))
        self.assertTrue(cdict["percentile"](self.p_crd.cell(0)))
        self.assertFalse(cdict["percentile"](self.t_crd.cell(0)))
        self.assertTrue(islambda(cdict["threshold"]))
        self.assertTrue(cdict["threshold"](self.t_crd.cell(0)))
        self.assertFalse(cdict["threshold"](self.p_crd.cell(0)))
        self.assertFalse(udict)

    def test_whitespace(self):
        """Test constraint parsing with padding whitespace"""
        constraints = ["percentile = 10", "threshold = 0.1"]
        result, _, _, _ = parse_constraint_list(constraints)
        cdict = result._coord_values
        self.assertTrue(islambda(cdict["percentile"]))
        self.assertTrue(cdict["percentile"](self.p_crd.cell(0)))
        self.assertTrue(islambda(cdict["threshold"]))
        self.assertTrue(cdict["threshold"](self.t_crd.cell(0)))

    def test_string_constraint(self):
        """Test that a string constraint results in a simple iris constraint,
        not a lambda function. This is created via the literal_eval ValueError.
        """
        constraints = ["percentile=kittens"]
        result, _, _, _ = parse_constraint_list(constraints)
        cdict = result._coord_values
        self.assertFalse(islambda(cdict["percentile"]))
        self.assertEqual(cdict["percentile"], "kittens")
        self.assertIsInstance(cdict, dict)

    def test_some_units(self):
        """Test units list containing "None" elements is correctly parsed"""
        _, udict, _, _ = parse_constraint_list(self.constraints, units=self.units)
        self.assertEqual(udict["threshold"], "mm h-1")
        self.assertNotIn("percentile", udict.keys())

    def test_unmatched_units(self):
        """Test for ValueError if units list does not match constraints"""
        units = ["mm h-1"]
        msg = "units list must match constraints"
        with self.assertRaisesRegex(ValueError, msg):
            parse_constraint_list(self.constraints, units=units)

    def test_list_constraint(self):
        """Test that a list of constraints is parsed correctly"""
        constraints = ["threshold=[0.1,1.0]"]
        result, _, _, _ = parse_constraint_list(constraints)
        cdict = result._coord_values
        self.assertTrue(islambda(cdict["threshold"]))

    def test_range_constraint(self):
        """Test that a constraint passed in as a range is parsed correctly"""
        # create input cube
        precip_cube = set_up_precip_probability_cube()
        threshold_coord = find_threshold_coordinate(precip_cube).name()
        precip_cube.coord(threshold_coord).convert_units("mm h-1")
        # generate constraints
        constraints = ["{}=[0.03:0.1]".format(threshold_coord)]
        result, _, _, _ = parse_constraint_list(constraints)
        self.assertIsInstance(result, iris._constraints.ConstraintCombination)
        cdict = result.rhs._coord_values
        self.assertEqual(list(cdict.keys()), [threshold_coord])
        # extract from input cube
        result_cube = precip_cube.extract(result)
        self.assertArrayAlmostEqual(
            result_cube.coord(threshold_coord).points, np.array([0.03, 0.1])
        )

    def test_longitude_constraint(self):
        """Test that the longitude constraint is parsed correctly"""
        constraint = ["longitude=[0:20:2]"]
        _, _, longitude_constraint, thinning_dict = parse_constraint_list(constraint)
        self.assertEqual(longitude_constraint, [0, 20])
        self.assertEqual(thinning_dict, {"longitude": 2})

    def test_longitude_constraint_whitespace(self):
        """Test that the longitude constraint is parsed correctly with whitespace"""
        constraint = ["longitude = [ 0 : 20 : 2 ]"]
        _, _, longitude_constraint, thinning_dict = parse_constraint_list(constraint)
        self.assertEqual(longitude_constraint, [0, 20])
        self.assertEqual(thinning_dict, {"longitude": 2})


class Test_apply_extraction(IrisTest):
    """Test function to extract subcube according to constraints"""

    def setUp(self):
        """Set up temporary input cube"""
        self.precip_cube = set_up_precip_probability_cube()
        self.threshold_coord = find_threshold_coordinate(self.precip_cube).name()
        self.uk_gridded_cube = set_up_uk_gridded_cube()
        self.global_gridded_cube = set_up_global_gridded_cube()
        self.units_dict = {self.threshold_coord: "mm h-1"}

    def test_basic_no_units(self):
        """Test cube extraction for single constraint without units"""
        constraint_dict = {"name": "probability_of_precipitation_rate_above_threshold"}
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(self.precip_cube, constr)
        self.assertIsInstance(cube, iris.cube.Cube)
        reference_data = self.precip_cube.data
        self.assertArrayEqual(cube.data, reference_data)

    def test_basic_with_units(self):
        """Test cube extraction for single constraint with units"""
        constraint_dict = {
            self.threshold_coord: lambda cell: any(np.isclose(cell.point, [0.1]))
        }
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(self.precip_cube, constr, self.units_dict)
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertEqual(cube.coord(self.threshold_coord).units, "m s-1")
        reference_data = self.precip_cube.data[1, :, :]
        self.assertArrayEqual(cube.data, reference_data)

    def test_basic_without_reconverting_units(self):
        """Test cube extraction for single constraint with units,
        where the coordinates are not reconverted into the original units."""
        constraint_dict = {
            self.threshold_coord: lambda cell: any(np.isclose(cell.point, [0.1]))
        }
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(
            self.precip_cube, constr, self.units_dict, use_original_units=False
        )
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertEqual(cube.coord(self.threshold_coord).units, "mm h-1")
        reference_data = self.precip_cube.data[1, :, :]
        self.assertArrayEqual(cube.data, reference_data)

    def test_multiple_constraints_with_units(self):
        """Test behaviour with a list of constraints and units"""
        constraint_dict = {
            "name": "probability_of_precipitation_rate_above_threshold",
            self.threshold_coord: lambda cell: any(np.isclose(cell.point, [0.03])),
        }
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(self.precip_cube, constr, self.units_dict)
        self.assertIsInstance(cube, iris.cube.Cube)
        reference_data = self.precip_cube.data[0, :, :]
        self.assertArrayEqual(cube.data, reference_data)

    def test_error_non_coord_units(self):
        """Test error raised if units are provided for a non-coordinate
        constraint"""
        constraint_dict = {"name": "probability_of_precipitation_rate_above_threshold"}
        units_dict = {"name": "1"}
        with self.assertRaises(CoordinateNotFoundError):
            apply_extraction(self.precip_cube, constraint_dict, units_dict)

    def test_allow_none(self):
        """Test function returns None rather than raising an error where
        no subcubes match the required constraints, when unit conversion is
        required"""
        constraint_dict = {
            "name": "probability_of_precipitation_rate_above_threshold",
            self.threshold_coord: 5,
        }
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(self.precip_cube, constr, self.units_dict)
        self.assertFalse(cube)

    def test_list_constraints(self):
        """Test that a list of constraints behaves correctly"""
        constraint_dict = {
            self.threshold_coord: lambda cell: any(np.isclose(cell.point, [0.1, 1.0]))
        }
        constr = iris.Constraint(**constraint_dict)
        cube = apply_extraction(self.precip_cube, constr, self.units_dict)
        reference_data = self.precip_cube.data[1:, :, :]
        self.assertArrayEqual(cube.data, reference_data)

    def test_range_constraints(self):
        """Test that a list of constraints behaves correctly. This includes
        converting the units to the units that the constraints is
        defined in."""
        lower_bound = 0.03 - 1.0e-7
        upper_bound = 0.1 + 1.0e-7
        constraint_dict = {
            self.threshold_coord: lambda cell: lower_bound <= cell.point <= upper_bound
        }
        constr = iris.Constraint(coord_values=constraint_dict)
        cube = apply_extraction(self.precip_cube, constr, self.units_dict)
        reference_data = self.precip_cube.data[:2, :, :]
        self.assertArrayEqual(cube.data, reference_data)

    def test_subset_uk_grid(self):
        """Test subsetting a gridded cube."""
        expected_data = np.array(
            [
                [27.0, 28.0, 29.0, 30.0],
                [35.0, 36.0, 37.0, 38.0],
                [43.0, 44.0, 45.0, 46.0],
                [51.0, 52.0, 53.0, 54.0],
            ]
        )
        expected_points = np.array([1000.0, 3000.0, 5000.0, 7000.0])
        lower_bound = 1000 - 1.0e-7
        upper_bound = 7000 + 1.0e-7
        constraint_dict = {
            "projection_x_coordinate": lambda cell: lower_bound
            <= cell.point
            <= upper_bound,
            "projection_y_coordinate": lambda cell: lower_bound
            <= cell.point
            <= upper_bound,
        }
        constr = iris.Constraint(**constraint_dict)
        result = apply_extraction(self.uk_gridded_cube, constr)
        self.assertArrayAlmostEqual(result.data, expected_data)
        for axis in ["x", "y"]:
            coord = f"projection_{axis}_coordinate"
            self.assertArrayAlmostEqual(result.coord(coord).points, expected_points)

    def test_subset_global_grid(self):
        """Extract subset of global lat-lon grid"""
        lower_bound = 42 - 1.0e-7
        upper_bound = 52 + 1.0e-7
        constraint_dict = {
            "latitude": lambda cell: lower_bound <= cell.point <= upper_bound
        }
        constr = iris.Constraint(**constraint_dict)
        result = apply_extraction(
            self.global_gridded_cube, constr, longitude_constraint=[0, 7]
        )
        expected_data = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [9.0, 10.0, 11.0, 12.0],
                [17.0, 18.0, 19.0, 20.0],
                [25.0, 26.0, 27.0, 28.0],
            ]
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, np.array([0.0, 2.0, 4.0, 6.0])
        )
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, np.array([45.0, 47.0, 49.0, 51.0])
        )

    def test_subset_global_grid_pacific(self):
        """Extract subset of global lat-lon grid over the international date line"""
        global_pacific_cube = set_up_variable_cube(
            self.global_gridded_cube.data.copy(),
            name="screen_temperature",
            units="degC",
            spatial_grid="latlon",
            domain_corner=(0, 175),
            x_grid_spacing=2,
            y_grid_spacing=2,
        )
        lower_bound = -1.0e-7
        upper_bound = 4 + 1.0e-7
        constraint_dict = {
            "latitude": lambda cell: lower_bound <= cell.point <= upper_bound
        }
        constr = iris.Constraint(**constraint_dict)
        expected_data = np.array(
            [[2.0, 3.0, 4.0], [10.0, 11.0, 12.0], [18.0, 19.0, 20.0]]
        )
        result = apply_extraction(
            global_pacific_cube, constr, longitude_constraint=[179.0, 183.0]
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, [179.0, 181.0, 183.0]
        )
        self.assertArrayAlmostEqual(result.coord("latitude").points, [0.0, 2.0, 4.0])


class Test_extract_subcube(IrisTest):
    """Test that a subcube is extracted when the required constraints are
    applied."""

    def setUp(self):
        """Set up temporary input cube"""
        self.precip_cube = set_up_precip_probability_cube()
        self.global_gridded_cube = set_up_global_gridded_cube()

    def test_single_threshold(self):
        """Test that a single threshold is extracted correctly when using the
        key=value syntax."""
        constraints = ["precipitation_rate=0.03"]
        precip_units = ["mm h-1"]
        expected = self.precip_cube[0]
        result = extract_subcube(self.precip_cube, constraints, units=precip_units)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_multiple_thresholds(self):
        """Test that multiple thresholds are extracted correctly when using the
        key=[value1,value2] syntax."""
        constraints = ["precipitation_rate=[0.03,0.1]"]
        precip_units = ["mm h-1"]
        expected = self.precip_cube[:2]
        result = extract_subcube(self.precip_cube, constraints, units=precip_units)
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
        constraints = ["precipitation_rate=[0.03,0.1]", "projection_y_coordinate=[1,2]"]
        precip_units = ["mm h-1", "m"]
        expected = self.precip_cube[0:2, 1:, :]
        result = extract_subcube(self.precip_cube, constraints, units=precip_units)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_multiple_range_constraints(self):
        """Test that multiple range constraints are extracted correctly when
        using the key=[value1:value2] syntax for more than one quantity (i.e.
        multiple constraints)."""
        constraints = ["precipitation_rate=[0.03:0.1]", "projection_y_coordinate=[1:2]"]
        precip_units = ["mm h-1", "m"]
        expected = self.precip_cube[0:2, 1:, :]
        result = extract_subcube(self.precip_cube, constraints, units=precip_units)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_combination_of_equality_and_range_constraints(self):
        """Test that multiple constraints are extracted correctly when
        using a combination of key=[value1,value2] and key=[value1:value2]
        syntax."""
        constraints = ["precipitation_rate=[0.03,0.1]", "projection_y_coordinate=[1:2]"]
        precip_units = ["mm h-1", "m"]
        expected = self.precip_cube[0:2, 1:, :]
        result = extract_subcube(self.precip_cube, constraints, units=precip_units)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_single_threshold_use_original_units(self):
        """Test that a single threshold is extracted correctly when using the
        key=value syntax without converting the coordinate units back to the
        original units."""
        constraints = ["precipitation_rate=0.03"]
        precip_units = ["mm h-1"]
        expected = self.precip_cube[0]
        expected.coord("precipitation_rate").convert_units("mm h-1")
        result = extract_subcube(
            self.precip_cube, constraints, units=precip_units, use_original_units=False
        )
        self.assertArrayAlmostEqual(result.data, expected.data)
        self.assertEqual(
            expected.coord("precipitation_rate"), result.coord("precipitation_rate")
        )

    def test_thin_global_gridded_cube(self):
        """Subsets a grid from a global grid and thins the data"""
        expected_result = np.array([[1.0, 4.0], [17.0, 20.0]])
        result = extract_subcube(
            self.global_gridded_cube, ["latitude=[42:52:2]", "longitude=[0:7:3]"]
        )
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, np.array([0.0, 6.0])
        )
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, np.array([45.0, 49.0])
        )

    def test_thin_longitude_global_gridded_cube(self):
        """Subsets a grid from a global grid and thins the data"""
        expected_result = np.array(
            [
                [1.0, 4.0],
                [9.0, 12.0],
                [17.0, 20.0],
                [25.0, 28.0],
                [33.0, 36.0],
                [41.0, 44.0],
                [49.0, 52.0],
            ]
        )
        result = extract_subcube(self.global_gridded_cube, ["longitude=[0:7:3]"])
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, np.array([0.0, 6.0])
        )
        self.assertArrayAlmostEqual(
            result.coord("latitude").points,
            np.array([45.0, 47.0, 49.0, 51.0, 53.0, 55.0, 57.0]),
        )


if __name__ == "__main__":
    unittest.main()
