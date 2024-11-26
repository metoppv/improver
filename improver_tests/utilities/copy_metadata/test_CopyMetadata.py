# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from unittest.mock import patch, sentinel

import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from improver.utilities.copy_metadata import CopyMetadata


class HaltExecution(Exception):
    pass


@patch("improver.utilities.copy_metadata.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    mock_as_cubelist.side_effect = HaltExecution
    try:
        CopyMetadata(["attribA", "attribB"])(
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

    plugin = CopyMetadata(attributes)
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

    plugin = CopyMetadata(attributes)
    result = plugin.process(cube0, template_cube)
    assert type(result) is Cube
    assert result.attributes["attribA"] == "tempA"
    assert result.attributes["attribB"] == "tempB"
    assert result.attributes["attribD"] == "valueD"
    assert "attribC" not in result.attributes
    assert id(result) == id(cube0)


@pytest.mark.parametrize("cubelist", [True, False])
def test_auxiliary_coord_modification(cubelist):
    """test adding and altering auxiliary coordinates"""
    data = [[0, 1], [0, 1]]

    auxiliary_coord = ["dummy_0 status_flag", "dummy_1 status_flag"]

    # Create auxiliary coordinates with matching dimensions
    dummy_aux_coord_0 = AuxCoord([0, 0], long_name="dummy_0 status_flag")
    dummy_aux_coord_0_temp = AuxCoord([1, 1], long_name="dummy_0 status_flag")

    dummy_aux_coord_1_temp = AuxCoord([1, 1], long_name="dummy_1 status_flag")

    cube = Cube(data, aux_coords_and_dims=[(dummy_aux_coord_0, 0)])
    # Create the cube with the auxiliary coordinates
    template_cube = Cube(
        data,
        aux_coords_and_dims=[(dummy_aux_coord_0_temp, 0), (dummy_aux_coord_1_temp, 0)],
    )

    cubes = cube
    if cubelist:
        cubes = [cube, cube]

    plugin = CopyMetadata(aux_coord=auxiliary_coord)
    result = plugin.process(cubes, template_cube)
    if cubelist:
        for res in result:
            assert res.coord("dummy_0 status_flag") == dummy_aux_coord_0_temp
            assert res.coord("dummy_1 status_flag") == dummy_aux_coord_1_temp
    else:
        assert result.coord("dummy_0 status_flag") == dummy_aux_coord_0_temp
        assert result.coord("dummy_1 status_flag") == dummy_aux_coord_1_temp
