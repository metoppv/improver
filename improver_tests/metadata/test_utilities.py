# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the improver.metadata.utilities module"""

import unittest
from datetime import datetime, timedelta
from typing import Callable, List

import iris
import numpy as np
import pytest
from iris.cube import Cube, CubeList
from numpy.testing import assert_array_equal

from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.metadata.utilities import (
    check_grid_match,
    create_coordinate_hash,
    create_new_diagnostic_cube,
    enforce_time_point_standard,
    generate_hash,
    generate_mandatory_attributes,
    get_model_id_attr,
)
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


class Test_create_new_diagnostic_cube(unittest.TestCase):
    """Test utility to create new diagnostic cubes"""

    def setUp(self):
        """Set up template with data, coordinates, attributes and cell
        methods"""
        self.template_cube = set_up_variable_cube(
            280 * np.ones((3, 5, 5), dtype=np.float32), standard_grid_metadata="uk_det"
        )
        self.template_cube.add_cell_method("time (max): 1 hour")
        self.name = "lwe_precipitation_rate"
        self.units = "mm h-1"
        self.mandatory_attributes = MANDATORY_ATTRIBUTE_DEFAULTS.copy()

    def test_basic(self):
        """Test result is a cube that inherits coordinates only"""
        result = create_new_diagnostic_cube(
            self.name, self.units, self.template_cube, self.mandatory_attributes
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.standard_name, "lwe_precipitation_rate")
        self.assertEqual(result.units, "mm h-1")
        self.assertSequenceEqual(
            result.coords(dim_coords=True), self.template_cube.coords(dim_coords=True)
        )
        self.assertSequenceEqual(
            result.coords(dim_coords=False), self.template_cube.coords(dim_coords=False)
        )
        self.assertFalse(np.allclose(result.data, self.template_cube.data))
        self.assertDictEqual(result.attributes, self.mandatory_attributes)
        self.assertFalse(result.cell_methods)
        self.assertEqual(result.data.dtype, np.float32)

    def test_attributes(self):
        """Test optional attributes can be set on the output cube, and override
        the values in mandatory_attributes"""
        attributes = {"source": "Mars", "mosg__model_configuration": "uk_det"}
        expected_attributes = self.mandatory_attributes
        expected_attributes.update(attributes)
        result = create_new_diagnostic_cube(
            self.name,
            self.units,
            self.template_cube,
            self.mandatory_attributes,
            optional_attributes=attributes,
        )
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_missing_mandatory_attribute(self):
        """Test error is raised if any mandatory attribute is missing"""
        self.mandatory_attributes.pop("source")
        msg = "source attribute is required"
        with self.assertRaisesRegex(ValueError, msg):
            create_new_diagnostic_cube(
                self.name, self.units, self.template_cube, self.mandatory_attributes
            )

    def test_data(self):
        """Test data can be set on the output cube"""
        data = np.arange(3 * 5 * 5).reshape((3, 5, 5)).astype(np.float32)
        result = create_new_diagnostic_cube(
            self.name,
            self.units,
            self.template_cube,
            self.mandatory_attributes,
            data=data,
        )
        self.assertTrue(np.allclose(result.data, data))

    def test_dtype(self):
        """Test dummy data of a different type can be set"""
        result = create_new_diagnostic_cube(
            self.name,
            self.units,
            self.template_cube,
            self.mandatory_attributes,
            dtype=np.int32,
        )
        self.assertEqual(result.data.dtype, np.int32)

    def test_non_standard_name(self):
        """Test cube can be created with a non-CF-standard name"""
        result = create_new_diagnostic_cube(
            "RainRate Composite",
            self.units,
            self.template_cube,
            self.mandatory_attributes,
        )
        self.assertEqual(result.long_name, "RainRate Composite")
        self.assertIsNone(result.standard_name)


class Test_generate_mandatory_attributes(unittest.TestCase):
    """Test the generate_mandatory_attributes utility"""

    def setUp(self):
        """Set up some example input diagnostic cubes"""
        self.attributes = {
            "source": "Met Office Unified Model",
            "institution": "Met Office",
            "title": "UKV Model Forecast on UK 2 km Standard Grid",
        }
        base_data = np.ones((5, 5), dtype=np.float32)
        self.t_cube = set_up_variable_cube(
            285 * base_data,
            spatial_grid="equalarea",
            standard_grid_metadata="uk_det",
            attributes=self.attributes,
        )
        self.p_cube = set_up_variable_cube(
            987 * base_data,
            name="PMSL",
            units="hPa",
            spatial_grid="equalarea",
            standard_grid_metadata="uk_det",
            attributes=self.attributes,
        )
        self.rh_cube = set_up_variable_cube(
            0.8 * base_data,
            name="relative_humidity",
            units="1",
            spatial_grid="equalarea",
            standard_grid_metadata="uk_det",
            attributes=self.attributes,
        )

    def test_consensus(self):
        """Test attributes are inherited if all input fields agree"""
        result = generate_mandatory_attributes([self.t_cube, self.p_cube, self.rh_cube])
        self.assertDictEqual(result, self.attributes)

    def test_no_consensus(self):
        """Test default values if input fields do not all agree"""
        self.t_cube.attributes = {
            "source": "Met Office Unified Model Version 1000",
            "institution": "BOM",
            "title": "UKV Model Forecast on 20 km Global Grid",
        }
        result = generate_mandatory_attributes([self.t_cube, self.p_cube, self.rh_cube])
        self.assertDictEqual(result, MANDATORY_ATTRIBUTE_DEFAULTS)

    def test_missing_attribute(self):
        """Test defaults are triggered if mandatory attribute is missing
        from one input"""
        expected_attributes = self.attributes
        expected_attributes["title"] = MANDATORY_ATTRIBUTE_DEFAULTS["title"]
        self.t_cube.attributes.pop("title")
        result = generate_mandatory_attributes([self.t_cube, self.p_cube, self.rh_cube])
        self.assertDictEqual(result, expected_attributes)

    def test_model_id_consensus(self):
        """Test model ID attribute can be specified and inherited"""
        expected_attributes = self.attributes.copy()
        expected_attributes["mosg__model_configuration"] = "uk_det"
        result = generate_mandatory_attributes(
            [self.t_cube, self.p_cube, self.rh_cube],
            model_id_attr="mosg__model_configuration",
        )
        self.assertDictEqual(result, expected_attributes)

    def test_model_id_no_consensus(self):
        """Test error raised when model ID attributes do not agree"""
        self.t_cube.attributes["mosg__model_configuration"] = "gl_det"
        msg = "is missing or not the same on all input cubes"
        with self.assertRaisesRegex(ValueError, msg):
            generate_mandatory_attributes(
                [self.t_cube, self.p_cube, self.rh_cube],
                model_id_attr="mosg__model_configuration",
            )

    def test_model_id_missing(self):
        """Test error raised when model ID attribute is not present on
        all input diagnostic cubes"""
        self.t_cube.attributes.pop("mosg__model_configuration")
        msg = "is missing or not the same on all input cubes"
        with self.assertRaisesRegex(ValueError, msg):
            generate_mandatory_attributes(
                [self.t_cube, self.p_cube, self.rh_cube],
                model_id_attr="mosg__model_configuration",
            )


class Test_generate_hash(unittest.TestCase):
    """Test utility to generate md5 hash codes from a multitude of inputs."""

    def test_string_input(self):
        """Test the expected hash is returned when input is a string type."""

        hash_input = "this is a test string"
        result = generate_hash(hash_input)
        expected = "7a5a4f1716b08d290d5782da904cc076315376889e9bf641ae527889704fd314"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_numeric_input(self):
        """Test the expected hash is returned when input is a numeric type."""

        hash_input = 1000
        result = generate_hash(hash_input)
        expected = "40510175845988f13f6162ed8526f0b09f73384467fa855e1e79b44a56562a58"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_dictionary_input(self):
        """Test the expected hash is returned when input is a dictionary."""

        hash_input = {"one": 1, "two": 2}
        result = generate_hash(hash_input)
        expected = "c261139b6339f880f4f75a3bf5a08f7c2d6f208e2720760f362e4464735e3845"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_dictionary_order_invariant(self):
        """Test the expected hash is the same for different dict ordering."""

        hash_input1 = {"one": 1, "two": 2}
        hash_input2 = {"two": 2, "one": 1}
        result1 = generate_hash(hash_input1)
        result2 = generate_hash(hash_input2)
        self.assertEqual(result1, result2)

    def test_cube_input(self):
        """Test the expected hash is returned when input is a cube."""

        hash_input = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        result = generate_hash(hash_input)
        expected = "4d82994200559c90234b0186bccc59b9b9d6436284f29bab9a15dc97172d1fb8"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_coordinate_input(self):
        """Test the expected hash is returned when input is a cube
        coordinate."""

        cube = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        hash_input = cube.coord("latitude")
        result = generate_hash(hash_input)
        expected = "ee6a057f5eeef0e94a853cfa98f3c22b121dda31ada3378ce9466e48d06f9887"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_numpy_array_type_variant(self):
        """Test the expected hash is different if the numpy array type is
        different."""

        hash_input32 = np.array([np.sqrt(2.0)], dtype=np.float32)
        hash_input64 = np.array([np.sqrt(2.0)], dtype=np.float64)
        result32 = generate_hash(hash_input32)
        result64 = generate_hash(hash_input64)
        self.assertNotEqual(result32, result64)

    def test_equivalent_input_gives_equivalent_hash(self):
        """Test that creating a hash twice using the same input results in the
        same hash being generated."""

        cube = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        hash_input = cube.coord("latitude")
        result1 = generate_hash(hash_input)
        result2 = generate_hash(hash_input)
        self.assertEqual(result1, result2)


class Test_create_coordinate_hash(unittest.TestCase):
    """Test wrapper to hash generation to return a hash based on the x and y
    coordinates of a given cube."""

    def test_basic(self):
        """Test the expected hash is returned for a given cube."""

        hash_input = set_up_variable_cube(np.zeros((3, 3)).astype(np.float32))
        result = create_coordinate_hash(hash_input)
        expected = "54812a6fed0f92fe75d180d63a6bd6c916407ea1e7e5fd32a5f20f86ea997fac"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_variation(self):
        """Test that two cubes with slightly different coordinates return
        different hashes."""

        hash_input1 = set_up_variable_cube(np.zeros((3, 3)).astype(np.float32))
        hash_input2 = hash_input1.copy()
        latitude = hash_input2.coord("latitude")
        latitude_values = latitude.points * 1.001
        latitude = latitude.copy(points=latitude_values)
        hash_input2.remove_coord("latitude")
        hash_input2.add_dim_coord(latitude, 0)

        result1 = create_coordinate_hash(hash_input1)
        result2 = create_coordinate_hash(hash_input2)
        self.assertNotEqual(result1, result2)


class Test_check_grid_match(unittest.TestCase):
    """Test the check_grid_match function."""

    def setUp(self):
        """Set up cubes for use in testing."""

        data = np.ones(9).reshape(3, 3).astype(np.float32)
        self.reference_cube = set_up_variable_cube(data, spatial_grid="equalarea")
        self.cube1 = self.reference_cube.copy()
        self.cube2 = self.reference_cube.copy()
        self.unmatched_cube = set_up_variable_cube(data, spatial_grid="latlon")

        self.diagnostic_cube_hash = create_coordinate_hash(self.reference_cube)

        neighbours = np.array([[[0.0], [0.0], [0.0]]])
        altitudes = np.array([0])
        latitudes = np.array([0])
        longitudes = np.array([0])
        wmo_ids = np.array([0])
        grid_attributes = ["x_index", "y_index", "vertical_displacement"]
        neighbour_methods = ["nearest"]
        self.neighbour_cube = build_spotdata_cube(
            neighbours,
            "grid_neighbours",
            1,
            altitudes,
            latitudes,
            longitudes,
            wmo_ids,
            grid_attributes=grid_attributes,
            neighbour_methods=neighbour_methods,
        )
        self.neighbour_cube.attributes["model_grid_hash"] = self.diagnostic_cube_hash

    def test_matching_grids(self):
        """Test a case in which the grids match. There is no assert
        statement as this test is successful if no exception is raised."""
        cubes = [self.reference_cube, self.cube1, self.cube2]
        check_grid_match(cubes)

    def test_non_matching_grids(self):
        """Test a case in which a cube with an unmatching grid is included in
        the comparison, raising a ValueError."""
        cubes = [self.reference_cube, self.cube1, self.unmatched_cube]
        msg = (
            "Cubes do not share or originate from the same grid, so cannot "
            "be used together."
        )
        with self.assertRaisesRegex(ValueError, msg):
            check_grid_match(cubes)

    def test_using_model_grid_hash(self):
        """Test a case in which one of the cubes is a spotdata cube without a
        spatial grid. This cube includes a model_grid_hash to indicate on which
        grid the neighbours were found."""
        cubes = [self.reference_cube, self.neighbour_cube, self.cube2]
        check_grid_match(cubes)

    def test_using_model_grid_hash_reordered_cubes(self):
        """Test as above but using the neighbour_cube as the first in the list
        so that it acts as the reference for all the other cubes."""
        cubes = [self.neighbour_cube, self.reference_cube, self.cube2]
        check_grid_match(cubes)

    def test_multiple_model_grid_hash_cubes(self):
        """Test that a check works when all the cubes passed to the function
        have model_grid_hashes."""
        self.cube1.attributes["model_grid_hash"] = self.diagnostic_cube_hash
        cubes = [self.neighbour_cube, self.cube1]
        check_grid_match(cubes)

    def test_mismatched_model_grid_hash_cubes(self):
        """Test that a check works when all the cubes passed to the function
        have model_grid_hashes and these do not match."""
        self.cube1.attributes["model_grid_hash"] = "123"
        cubes = [self.neighbour_cube, self.cube1]
        msg = (
            "Cubes do not share or originate from the same grid, so cannot "
            "be used together."
        )
        with self.assertRaisesRegex(ValueError, msg):
            check_grid_match(cubes)


@pytest.fixture(name="cubes")
def make_cubes() -> List[Cube]:
    """Generates a list of three cubes"""
    cubes = []
    master_cube = set_up_variable_cube(np.zeros((2, 2), dtype=np.float32))
    for name in "temperature", "pressure", "humidity":
        cube = master_cube.copy()
        cube.rename(name)
        cubes.append(cube)
    return cubes


@pytest.mark.parametrize("input_count", (1, 3))
@pytest.mark.parametrize(
    "model_id_attr, model_id_value", (("test_attribute", "test_value"), (None, None))
)
def test_valid_get_model_id_attr(
    cubes: List[Cube], input_count, model_id_attr, model_id_value
):
    """Checks that get_model_id_attr gives the expected result when all input cubes match."""
    for cube in cubes:
        cube.attributes[model_id_attr] = model_id_value
    result = get_model_id_attr(cubes[:input_count], model_id_attr)
    assert result == model_id_value


def attribute_missing_all_cubes(cubes: List[Cube]):
    """Removes the attribute value from all cubes"""
    [cube.attributes.pop("test_attribute") for cube in cubes]


def attribute_missing_one_cube(cubes: List[Cube]):
    """Removes the attribute value from the first cube"""
    cubes[0].attributes.pop("test_attribute")


def attribute_not_unique(cubes: List[Cube]):
    """Changes the attribute value for the first cube so that there is more than one
    model_id_attr in the cube list."""
    cubes[0].attributes["test_attribute"] = "kittens"


@pytest.mark.parametrize(
    "method, message",
    (
        (
            attribute_missing_all_cubes,
            "Model ID attribute test_attribute not present for ",
        ),
        (
            attribute_missing_one_cube,
            "Model ID attribute test_attribute not present for ",
        ),
        (
            attribute_not_unique,
            "Attribute test_attribute must be the same for all input cubes. ",
        ),
    ),
)
def test_errors_get_model_id_attr(cubes: List[Cube], method: Callable, message):
    """Checks that get_model_id_attr raises useful errors when the required conditions are not met."""
    model_id_attr = "test_attribute"
    model_id_value = "test_value"
    for cube in cubes:
        cube.attributes[model_id_attr] = model_id_value
    method(cubes)
    with pytest.raises(ValueError, match=message):
        get_model_id_attr(cubes, model_id_attr)


@pytest.fixture
def data_times():
    """Define the times for the input cubes. The bounds are set to be
    non-conformant with the IMPROVER standards, such that the point
    is the start of the period rather than the end."""
    frt = datetime(2025, 1, 15, 3, 0)
    times = []
    for hour in range(3, 9 + 1, 3):
        time = frt + timedelta(hours=hour)
        bounds = [time, time + timedelta(hours=3)]
        times.append((frt, time, bounds))
    return times


@pytest.fixture
def multi_time_cube(data_times):
    """Create a cube that has a time coordinate with an entry for
    each validity time passed in. Adds bounds to the
    forecast_reference_time purely to demonstrate the functionality."""

    data = 281 * np.ones((3, 3)).astype(np.float32)
    cubes = CubeList()
    for frt, time, time_bounds in data_times:
        cubes.append(
            set_up_variable_cube(data, time=time, time_bounds=time_bounds, frt=frt)
        )
    cube = cubes.merge_cube()
    frt_crd = cube.coord("forecast_reference_time")
    frt_crd.bounds = [[frt, frt + 3600] for frt in frt_crd.points]
    return cube


def test_enforce_time_point_standard(multi_time_cube, data_times):
    """Test that enforce_time_point_standard correctly sets the time coordinate
    points to the upper bound of the periods."""

    enforce_time_point_standard(multi_time_cube)

    for crd in ["time", "forecast_period", "forecast_reference_time"]:
        coord = multi_time_cube.coord(crd)
        assert_array_equal(coord.points, [bound[-1] for bound in coord.bounds])


def test_enforce_time_point_standard_without_bounds(multi_time_cube, data_times):
    """Test that enforce_time_point_standard returns unmodified coordinates if
    there are no bounds on them."""

    # Remove bounds.
    for crd in ["time", "forecast_period", "forecast_reference_time"]:
        multi_time_cube.coord(crd).bounds = None

    # Retain copy that has not been through enforcement step.
    reference = multi_time_cube.copy()

    enforce_time_point_standard(multi_time_cube)

    # Demonstrate that coordinates are unchanged by the enforcement step.
    for crd in ["time", "forecast_period", "forecast_reference_time"]:
        assert multi_time_cube.coord(crd) == reference.coord(crd)


if __name__ == "__main__":
    unittest.main()
