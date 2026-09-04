# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from unittest.mock import patch, sentinel

import pytest
from iris.coords import AncillaryVariable, AuxCoord
from iris.cube import Cube

from improver.utilities.copy_metadata import CopyMetadata


class HaltExecution(Exception):
    pass


@patch("improver.utilities.copy_metadata.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    """Test that the as_cubelist function is called."""
    mock_as_cubelist.side_effect = HaltExecution
    try:
        CopyMetadata(["attribA", "attribB"])(
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

    plugin = CopyMetadata(attributes)
    result = plugin.process(cube0, template_cube, template_cube_2)
    assert isinstance(result, Cube)
    assert result.attributes["attribA"] == "tempA"
    assert result.attributes["attribB"] == "tempB"
    assert "attribC" not in result.attributes
    assert result == cube0  # Checks cube has been altered in-place
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

    plugin = CopyMetadata(attributes)
    result = plugin.process(cube0, template_cube_2, template_cube)
    assert isinstance(result, Cube)
    assert result.attributes["attribA"] == "tempA"
    assert result.attributes["attribB"] == "tempB"
    assert "attribC" not in result.attributes
    assert result == cube0  # Checks cube has been altered in-place
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

    plugin = CopyMetadata(attributes)
    result = plugin.process(cube0, template_cube)
    assert isinstance(result, Cube)
    assert result.attributes["attribA"] == "tempA"
    assert result.attributes["attribB"] == "tempB"
    assert result.attributes["attribD"] == "valueD"
    assert "attribC" not in result.attributes
    assert result == cube0  # Checks cube has been altered in-place


@pytest.mark.parametrize("cubelist", [True, False])
def test_auxiliary_coord_modification(cubelist):
    """Test adding and altering auxiliary coordinates. We test copying the
    auxilary coordinate 'dummy_0 status_flag' and 'dummy_1 status_flag'.
    'dummy_0 status_flag' is present in the input cube(s) and template cube
    so the aux coordinate on the input cube(s) is replaced. Whereas 'dummy_1
    status_flag' is only present in the template cube so a new aux coord is
    added to the input cube(s).
    """
    data = [[0, 1], [0, 1]]

    auxiliary_coord = ["dummy_0 status_flag", "dummy_1 status_flag"]

    # Create auxiliary coordinates with matching dimensions
    dummy_aux_coord_0 = AuxCoord([0, 0], long_name="dummy_0 status_flag")
    dummy_aux_coord_0_temp = AuxCoord([1, 1], long_name="dummy_0 status_flag")

    dummy_aux_coord_1_temp = AuxCoord([1, 1], long_name="dummy_1 status_flag")

    cube = Cube(data, aux_coords_and_dims=[(dummy_aux_coord_0, 0)])
    # Create the cube with the auxiliary coordinates
    template_cubes = Cube(
        data,
        aux_coords_and_dims=[(dummy_aux_coord_0_temp, 0), (dummy_aux_coord_1_temp, 0)],
    )

    if cubelist:
        template_cubes = [template_cubes, template_cubes]
    plugin = CopyMetadata(aux_coord=auxiliary_coord)
    result = plugin.process(cube, template_cubes)
    assert result.coord("dummy_0 status_flag") == dummy_aux_coord_0_temp
    assert result.coord("dummy_1 status_flag") == dummy_aux_coord_1_temp


@pytest.mark.parametrize("cubelist", [True, False])
def test_ancillary_variable_modification(cubelist):
    """Test adding and altering ancillary variables. We test copying the
    ancillary variable 'status_flag0' and 'status_flag1'.
    'status_flag0' is present in the input cube(s) and template cube
    so the ancillary variable on the input cube(s) is replaced. Whereas 'status_flag1'
    is only present in the template cube so a new ancillary variable is
    added to the input cube(s).
    """
    data = [[0, 1], [0, 1]]

    ancillary_variable = ["status_flag0", "status_flag1"]

    # Create ancillary variables with matching dimensions
    dummy_anc_variable_0 = AncillaryVariable([0, 0], long_name="status_flag0")
    dummy_anc_variable_0_temp = AncillaryVariable([1, 1], long_name="status_flag0")

    dummy_anc_variable_1_temp = AncillaryVariable([1, 1], long_name="status_flag1")

    cube = Cube(data, ancillary_variables_and_dims=[(dummy_anc_variable_0, 0)])
    # Create the cube with the ancillary variables
    template_cubes = Cube(
        data,
        ancillary_variables_and_dims=[
            (dummy_anc_variable_0_temp, 0),
            (dummy_anc_variable_1_temp, 0),
        ],
    )

    if cubelist:
        template_cubes = [template_cubes, template_cubes]
    plugin = CopyMetadata(ancillary_variables=ancillary_variable)
    result = plugin.process(cube, template_cubes)
    assert result.ancillary_variable("status_flag0") == dummy_anc_variable_0_temp
    assert result.ancillary_variable("status_flag1") == dummy_anc_variable_1_temp


@pytest.mark.parametrize("cubelist", [True, False])
def test_named_auxiliary_coordinate_not_found(cubelist):
    """Test that a warning is raised if an auxiliary coordinate to be copied
    is not found in the template cube."""
    data = [[0, 1], [0, 1]]

    auxiliary_coord = ["dummy_0 status_flag", "dummy_1 status_flag"]

    # Create auxiliary coordinate with matching dimensions
    dummy_aux_coord_0_temp = AuxCoord([1, 1], long_name="dummy_0 status_flag")
    # Create the cube with the auxiliary coordinate, use a valid cube name
    template_cubes = Cube(
        data,
        aux_coords_and_dims=[(dummy_aux_coord_0_temp, 0)],
        standard_name="wind_speed",
    )
    if cubelist:
        template_cubes = [template_cubes, template_cubes]
    plugin = CopyMetadata(aux_coord=auxiliary_coord)
    with pytest.warns(
        UserWarning,
        match="Auxiliary Coordinate 'dummy_1 status_flag' not found in cube 'wind_speed'.",
    ):
        plugin.process(Cube(data), template_cubes)


@pytest.mark.parametrize("cubelist", [True, False])
def test_named_ancillary_variable_not_found(cubelist):
    """Test that a warning is raised if an ancillary variable to be copied
    is not found in the template cube."""
    data = [[0, 1], [0, 1]]

    ancillary_variable = ["status_flag0", "status_flag1"]

    # Create ancillary variable with matching dimensions
    dummy_anc_variable_0_temp = AncillaryVariable([1, 1], long_name="status_flag0")

    # Create the cube with the ancillary variable, use a valid cube name
    # to test its name is included in the warning message
    template_cubes = Cube(
        data,
        ancillary_variables_and_dims=[(dummy_anc_variable_0_temp, 0)],
        standard_name="wind_speed",
    )

    if cubelist:
        template_cubes = [template_cubes, template_cubes]
    plugin = CopyMetadata(ancillary_variables=ancillary_variable)
    with pytest.warns(
        UserWarning,
        match="Ancillary Variable 'status_flag1' not found in cube 'wind_speed'.",
    ):
        plugin.process(Cube(data), template_cubes)


def test_copy_attributes_multi_input_mismatching_attributes():
    """Test that an error is raised if the template cubes have mismatching attribute values."""
    attributes = ["attribA", "attribB"]
    cube0 = Cube([0], attributes={"attribA": "valueA", "attribB": "valueB"})
    template_cube = Cube(
        [0], attributes={"attribA": "tempA", "attribB": "tempB", "attribC": "tempC"}
    )
    template_cube_2 = Cube([0], attributes={"attribA": "temp2A", "attribC": "tempC"})

    plugin = CopyMetadata(attributes)
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
    plugin = CopyMetadata(attributes)
    with pytest.raises(
        ValueError, match="Attribute attribA not found in any of the template cubes"
    ):
        plugin.process(cube0, template_cube_2, template_cube)


def test_copy_attributes_missing_inputs():
    """Test that an error is raised if the number of input cubes is less than 2."""
    attributes = ["attribA", "attribB"]
    cube0 = Cube([0])

    plugin = CopyMetadata(attributes)
    with pytest.raises(
        RuntimeError, match="At least two cubes are required for this operation, got 1"
    ):
        plugin.process(cube0)
