# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from unittest.mock import patch, sentinel

from iris.cube import Cube, CubeList

from improver.utilities.copy_attributes import CopyAttributes


class HaltExecution(Exception):
    pass


@patch("improver.utilities.copy_attributes.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    mock_as_cubelist.side_effect = HaltExecution
    try:
        CopyAttributes(["attribA", "attribB"])(
            sentinel.cube0, sentinel.cube1, sentinel.template_cube
        )
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(
        sentinel.cube0, sentinel.cube1, sentinel.template_cube
    )


def test_copy_attributes_multi_input():
    """
    Test the copy_attributes function for multiple input cubes.

    Demonstrates copying attributes from the template cube to the input
    cubes and also demonstrates the attributes on the templates cube that
    aren't specified in the attributes list are indeed ignored.

    Note how we are verifying the object IDs, since CubeAttributes is an
    in-place operation.
    """
    attributes = ["attribA", "attribB"]
    cube0 = Cube([0], attributes={"attribA": "valueA", "attribB": "valueB"})
    cube1 = Cube([0], attributes={"attribA": "valueAA", "attribB": "valueBB"})
    template_cube = Cube(
        [0], attributes={"attribA": "tempA", "attribB": "tempB", "attribC": "tempC"}
    )

    plugin = CopyAttributes(attributes)
    result = plugin.process(cube0, cube1, template_cube)
    assert type(result) is CubeList
    for res in result:
        assert res.attributes["attribA"] == "tempA"
        assert res.attributes["attribB"] == "tempB"
        assert "attribC" not in res.attributes
    assert id(result[0]) == id(cube0)
    assert id(result[1]) == id(cube1)


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
