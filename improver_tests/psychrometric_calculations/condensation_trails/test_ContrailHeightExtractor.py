# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ContrailHeightExtractor plugin"""

import iris
import numpy as np
import pytest

from improver.psychrometric_calculations.condensation_trails import (
    ContrailHeightExtractor,
)


def make_cube(
    data,
    name,
    units,
    dims=("pressure", "latitude", "longitude"),
    contrail_type_values=None,
    contrail_type_meaning_values=None,
):
    """
    Simple cube factory for ContrailsHeightExtractor tests.

    Args:
        data (np.ndarray): Data array.
        name (str): Name of the cube.
        units (str): Units of the cube.
        contrail_type_values (list): Values for the contrail_type attribute, e.g. [0, 1, 2]
        contrail_type_meaning_values (str): Meanings for the contrail_type attribute, e.g. "None Non-persistent Persistent"
        dims (tuple): Dimensions of the cube.

    Returns:
        iris.cube.Cube: The constructed cube.
    """
    shape = data.shape
    coords = []
    if "engine_contrail_factor" in dims:
        coords.append(("engine_contrail_factor", np.arange(shape[0])))
        if "pressure" in dims:
            coords.append(("pressure", np.arange(shape[1])))
        if "latitude" in dims:
            coords.append(("latitude", np.arange(shape[2])))
        if "longitude" in dims:
            coords.append(("longitude", np.arange(shape[3])))
    else:
        if "pressure" in dims:
            coords.append(("pressure", np.arange(shape[0])))
        if "latitude" in dims:
            coords.append(("latitude", np.arange(shape[1])))
        if "longitude" in dims:
            coords.append(("longitude", np.arange(shape[2])))
    cube = iris.cube.Cube(data, long_name=name, units=units)
    for dim, (coord_name, points) in enumerate(coords):
        if coord_name == "engine_contrail_factor":
            cube.add_dim_coord(
                iris.coords.DimCoord(points, long_name="engine_contrail_factor"), dim
            )
        elif coord_name == "pressure":
            cube.add_dim_coord(iris.coords.DimCoord(points, long_name="pressure"), dim)
        else:
            cube.add_dim_coord(iris.coords.DimCoord(points, coord_name), dim)
    if contrail_type_values and contrail_type_meaning_values:
        cube.attributes["contrail_type"] = contrail_type_values
        cube.attributes["contrail_type_meaning"] = contrail_type_meaning_values
    return cube


def test_max_extraction():
    """
    Test maximum extraction of contrail formation using ContrailHeightExtractor.
    """

    # 1. Create test input data
    # --> formation data (engine_factor (x2), pressure (x2), latitude (x2), longitude (x2))
    formation = np.array(
        [
            [[[1, 1], [2, 2]], [[1, 1], [2, 0]]],
            [[[2, 0], [1, 1]], [[0, 1], [2, 1]]],
        ]
    )
    # --> height data (pressure (x2), latitude (x2), longitude (x2))
    height = np.array([[[100, 200], [300, 400]], [[200, 300], [400, 500]]])

    # 2. Turn test data into test cubes ready for passing into ContrailHeightExtractor
    formation_cube = make_cube(
        formation,
        "formation",
        "1",
        dims=("engine_factor", "pressure", "latitude", "longitude"),
        contrail_type_values=[0, 1, 2],
        contrail_type_meaning_values="None Non-persistent Persistent",
    )
    height_cube = make_cube(
        height, "height", "m", dims=("pressure", "latitude", "longitude")
    )

    # 3. Create expected results which we would expect from the ContrailHeightExtractor if it uses the test data above.
    # Expected results for *max* heights of contrail formation associated with formation and height data above.
    # Result data arrays have the same dimentions (engine_factor (x2), latitude (x2), longitude (x2))
    expected_non_persistent_max_height = np.array(
        [[[200, 300], [np.nan, np.nan]], [[np.nan, 300], [300, 500]]]
    )
    expected_persistent_max_height = np.array(
        [[[np.nan, np.nan], [400, 400]], [[100, np.nan], [400, np.nan]]]
    )

    # 4. Run ContrailHeightExtractor using the test cubes
    extractor = ContrailHeightExtractor(use_max=True)
    non_persistent, persistent = extractor.process(formation_cube, height_cube)

    # 5. Assert that the results from the extractor match the expected results
    assert np.allclose(
        non_persistent.data, expected_non_persistent_max_height, equal_nan=True
    )
    assert np.allclose(persistent.data, expected_persistent_max_height, equal_nan=True)


def test_min_extraction():
    """
    Test minimum extraction of contrail formation using ContrailHeightExtractor.
    """

    # 1. Create test input data
    # --> formation data (engine_factor (x2), pressure (x2), latitude (x2), longitude (x2))
    formation = np.array(
        [
            [[[1, 1], [2, 2]], [[1, 1], [2, 0]]],
            [[[2, 0], [1, 1]], [[0, 1], [2, 1]]],
        ]
    )
    # --> height data (pressure (x2), latitude (x2), longitude (x2))
    height = np.array([[[100, 200], [300, 400]], [[200, 300], [400, 500]]])

    # 2. Turn test data into test cubes ready for passing into ContrailHeightExtractor
    formation_cube = make_cube(
        formation,
        "formation",
        "1",
        dims=("engine_factor", "pressure", "latitude", "longitude"),
        contrail_type_values=[0, 1, 2],
        contrail_type_meaning_values="None Non-persistent Persistent",
    )
    height_cube = make_cube(
        height, "height", "m", dims=("pressure", "latitude", "longitude")
    )

    # 3. Create expected results which we would expect from the ContrailHeightExtractor if it uses the test data above.
    # Expected results for *max* heights of contrail formation associated with formation and height data above.
    # Result data arrays have the same dimentions (engine_factor (x2), latitude (x2), longitude (x2))
    expected_non_persistent_min_height = np.array(
        [[[100, 200], [np.nan, np.nan]], [[np.nan, 300], [300, 400]]]
    )
    expected_persistent_min_height = np.array(
        [[[np.nan, np.nan], [300, 400]], [[100, np.nan], [400, np.nan]]]
    )

    # 4. Run ContrailHeightExtractor using the test cubes
    extractor = ContrailHeightExtractor(use_max=False)
    non_persistent, persistent = extractor.process(formation_cube, height_cube)

    # 5. Assert that the results from the extractor match the expected results
    assert np.allclose(
        non_persistent.data, expected_non_persistent_min_height, equal_nan=True
    )
    assert np.allclose(persistent.data, expected_persistent_min_height, equal_nan=True)


def test_output_names_and_units():
    """
    Test that output cubes have correct names and units using ContrailHeightExtractor.
    """
    # 1. Create test input data
    # --> formation data (engine_factor (x2), pressure (x2), latitude (x2), longitude (x2))
    formation = np.array(
        [
            [[[1, 1], [2, 2]], [[1, 1], [2, 0]]],
            [[[2, 0], [1, 1]], [[0, 1], [2, 1]]],
        ]
    )
    # --> height data (pressure (x2), latitude (x2), longitude (x2))
    height = np.array([[[100, 200], [300, 400]], [[200, 300], [400, 500]]])

    # 2. Turn test data into test cubes ready for passing into ContrailHeightExtractor
    formation_cube = make_cube(
        formation,
        "formation",
        "1",
        dims=("engine_factor", "pressure", "latitude", "longitude"),
        contrail_type_values=[0, 1, 2],
        contrail_type_meaning_values="None Non-persistent Persistent",
    )
    height_cube = make_cube(
        height, "height", "m", dims=("pressure", "latitude", "longitude")
    )

    # 3. Run ContrailHeightExtractor using the test cubes
    extractor = ContrailHeightExtractor(use_max=True)
    non_persistent, persistent = extractor.process(formation_cube, height_cube)

    # 4. Check names and units of output cubes
    assert "max_height_non_persistent_contrail" in non_persistent.name()
    assert "max_height_persistent_contrail" in persistent.name()
    assert non_persistent.units == height_cube.units
    assert persistent.units == height_cube.units


def test_cube_shape_error_handling():
    """
    Test that an error is raised when input cubes have incompatible shapes using ContrailHeightExtractor.
    """
    # 1. Create test input data
    # --> formation data (engine_factor (x2), pressure (x2), latitude (x2), longitude (x2))
    formation = np.array(
        [
            [[[1, 1], [2, 2]], [[1, 1], [2, 0]]],
            [[[2, 0], [1, 1]], [[0, 1], [2, 1]]],
        ]
    )
    # --> height data (pressure (x2), latitude (x3), longitude (x3))
    height = np.array(
        [[[100, 200, 300], [300, 400, 500]], [[100, 200, 300], [300, 400, 500]]]
    )  # Height has more points

    # 2. Turn test data into test cubes ready for passing into ContrailHeightExtractor
    formation_cube = make_cube(
        formation,
        "formation",
        "1",
        dims=("engine_factor", "pressure", "latitude", "longitude"),
        contrail_type_values=[0, 1, 2],
        contrail_type_meaning_values="None Non-persistent Persistent",
    )
    height_cube = make_cube(
        height, "height", "m", dims=("pressure", "latitude", "longitude")
    )

    # 3. Run ContrailHeightExtractor using the test cubes and check that it raises a ValueError
    extractor = ContrailHeightExtractor(use_max=True)
    with pytest.raises(ValueError):
        extractor.process(formation_cube, height_cube)


def test_attribute_availability_error_handling():
    """
    Test that an error is raised when input cubes are missing required attributes using ContrailHeightExtractor.
    """
    # 1. Create test input data
    # --> formation data (engine_factor (x2), pressure (x2), latitude (x2), longitude (x2))
    formation = np.array(
        [
            [[[1, 1], [2, 2]], [[1, 1], [2, 0]]],
            [[[2, 0], [1, 1]], [[0, 1], [2, 1]]],
        ]
    )
    # --> height data (pressure (x2), latitude (x2), longitude (x2))
    height = np.array([[[100, 200], [300, 400]], [[200, 300], [400, 500]]])

    # 2. Turn test data into test cubes ready for passing into ContrailHeightExtractor
    # Note: formation_cube is created without the required contrail_type and contrail_type_meaning attributes
    formation_cube = make_cube(
        formation,
        "formation",
        "1",
        dims=("engine_factor", "pressure", "latitude", "longitude"),
    )  # Missing attributes
    height_cube = make_cube(
        height, "height", "m", dims=("pressure", "latitude", "longitude")
    )
    # 3. Run ContrailHeightExtractor using the test cubes and check that it raises a ValueError
    extractor = ContrailHeightExtractor(use_max=True)
    with pytest.raises(ValueError):
        extractor.process(formation_cube, height_cube)


def test_attribute_length_error_handling():
    """
    Test that an error is raised when input cubes have mismatched attribute lengths using ContrailHeightExtractor.
    """
    # 1. Create test input data
    # --> formation data (engine_factor (x2), pressure (x2), latitude (x2), longitude (x2))
    formation = np.array(
        [
            [[[1, 1], [2, 2]], [[1, 1], [2, 0]]],
            [[[2, 0], [1, 1]], [[0, 1], [2, 1]]],
        ]
    )
    # --> height data (pressure (x2), latitude (x2), longitude (x2))
    height = np.array([[[100, 200], [300, 400]], [[200, 300], [400, 500]]])

    # 2. Turn test data into test cubes ready for passing into ContrailHeightExtractor
    # Note: formation_cube is created with mismatched contrail_type and contrail_type_meaning attributes
    formation_cube = make_cube(
        formation,
        "formation",
        "1",
        dims=("engine_factor", "pressure", "latitude", "longitude"),
        contrail_type_values=[0, 1],  # Mismatched lengths
        contrail_type_meaning_values="None Non-persistent Persistent",
    )
    height_cube = make_cube(
        height, "height", "m", dims=("pressure", "latitude", "longitude")
    )
    # 3. Run ContrailHeightExtractor using the test cubes and check that it raises a ValueError
    extractor = ContrailHeightExtractor(use_max=True)
    with pytest.raises(ValueError):
        extractor.process(formation_cube, height_cube)
