# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from unittest.mock import patch, sentinel

import pytest
from iris.cube import Cube

from improver.utilities.copy_attributes import CopyAttributes


class HaltExecution(Exception):
    pass


@patch("improver.utilities.copy_attributes.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    """Test that the as_cubelist function is called."""
    mock_as_cubelist.side_effect = HaltExecution
    try:
        CopyAttributes(["attribA", "attribB"])(
            sentinel.cube0, sentinel.template_cube1, sentinel.template_cube2
        )
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(
        sentinel.cube0, sentinel.template_cube1, sentinel.template_cube2
    )


@pytest.mark.parametrize("history", [False, True])
def test_copy_attributes_multi_input(history):
    """
    Test the copy_attributes function for multiple input template cubes.

    Demonstrates copying attributes from the template cube to the input
    cubes and also demonstrates the attributes on the templates cube that
    aren't specified in the attributes list are indeed ignored.

    Also demonstrates that the most recent history attribute is copied correctly if
    present on multiple template cubes.

    Note how we are verifying the object IDs, since CubeAttributes is an
    in-place operation.
    """
    attributes = ["attribA", "attribB"]
    cube0 = Cube([0], attributes={"attribA": "valueA", "attribB": "valueB"})
    template_cube = Cube(
        [0],
        attributes={
            "attribA": "tempA",
            "attribB": "tempB",
            "attribC": "tempC",
            "history": "2024-11-25T00:00:00Z",
        },
    )
    template_cube_2 = Cube(
        [0],
        attributes={
            "attribA": "tempA",
            "attribC": "tempC",
            "history": "2024-11-25T01:43:15Z",
        },
    )
    if history:
        attributes.append("history")

    plugin = CopyAttributes(attributes)
    result = plugin.process(cube0, template_cube_2, template_cube)
    assert type(result) is Cube
    assert result.attributes["attribA"] == "tempA"
    assert result.attributes["attribB"] == "tempB"
    assert "attribC" not in result.attributes
    assert id(result) == id(cube0)
    if history:
        assert result.attributes["history"] == "2024-11-25T01:43:15Z"
    else:
        assert "history" not in result.attributes


def test_copy_attributes_one_history_attribute():
    """Test that the history attribute is copied correctly if only one template cube has a history attribute."""
    attributes = ["attribA", "attribB", "history"]
    cube0 = Cube([0], attributes={"attribA": "valueA", "attribB": "valueB"})
    template_cube = Cube(
        [0],
        attributes={
            "attribA": "tempA",
            "attribB": "tempB",
            "attribC": "tempC",
            "history": "2024-11-25T00:00:00Z",
        },
    )
    template_cube_2 = Cube([0], attributes={"attribA": "tempA", "attribC": "tempC"})

    plugin = CopyAttributes(attributes)
    result = plugin.process(cube0, template_cube_2, template_cube)
    assert type(result) is Cube
    assert result.attributes["attribA"] == "tempA"
    assert result.attributes["attribB"] == "tempB"
    assert "attribC" not in result.attributes
    assert id(result) == id(cube0)
    assert result.attributes["history"] == "2024-11-25T00:00:00Z"


def test_copy_attributes_single_input():
    """
    As per 'test_copy_attributes_multi_input' except only one input cube is provided.
    """
    attributes = ["attribA", "attribB"]
    cube0 = Cube(
        [0], attributes={"attribA": "valueA", "attribB": "valueB", "attribD": "valueD"}
    )
    template_cube = Cube(
        [0], attributes={"attribA": "tempA", "attribB": "tempB", "attribC": "tempC"}
    )

    plugin = CopyAttributes(attributes)
    result = plugin.process(cube0, template_cube)
    assert type(result) is Cube
    assert result.attributes["attribA"] == "tempA"
    assert result.attributes["attribB"] == "tempB"
    assert result.attributes["attribD"] == "valueD"
    assert "attribC" not in result.attributes
    assert id(result) == id(cube0)


def test_copy_attributes_multi_input_mismatching_attributes():
    """Test that an error is raised if the template cubes have mismatching attribute values."""
    attributes = ["attribA", "attribB"]
    cube0 = Cube([0], attributes={"attribA": "valueA", "attribB": "valueB"})
    template_cube = Cube(
        [0], attributes={"attribA": "tempA", "attribB": "tempB", "attribC": "tempC"}
    )
    template_cube_2 = Cube([0], attributes={"attribA": "temp2A", "attribC": "tempC"})

    plugin = CopyAttributes(attributes)
    with pytest.raises(
        ValueError,
        match="Attribute attribA has different values in the provided template cubes",
    ):
        plugin.process(cube0, template_cube_2, template_cube)


def test_copy_attributes_multi_input_missing_attributes():
    """Test that an error is raised if a requested attribute is not present on any of the template cubes."""
    attributes = ["attribA", "attribB"]
    cube0 = Cube([0], attributes={"attribA": "valueA", "attribB": "valueB"})
    template_cube = Cube([0], attributes={"attribB": "tempB", "attribC": "tempC"})
    template_cube_2 = Cube([0], attributes={"attribC": "tempC"})
    plugin = CopyAttributes(attributes)
    with pytest.raises(
        ValueError, match="Attribute attribA not found in any of the template cubes"
    ):
        plugin.process(cube0, template_cube_2, template_cube)


def test_copy_attributes_missing_inputs():
    """Test that an error is raised if the number of input cubes is less than 2."""
    attributes = ["attribA", "attribB"]
    cube0 = Cube([0])

    plugin = CopyAttributes(attributes)
    with pytest.raises(
        RuntimeError, match="At least two cubes are required for this operation, got 1"
    ):
        plugin.process(cube0)
