# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
from unittest.mock import patch, sentinel

from iris.cube import Cube

from improver.utilities.copy_attributes import CopyAttributes


class HaltExecution(Exception):
    pass


@patch("improver.utilities.copy_attributes.as_cube")
@patch("improver.utilities.copy_attributes.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist, mock_as_cube):
    mock_as_cubelist.return_value = sentinel.cubelist
    mock_as_cube.side_effect = HaltExecution
    try:
        CopyAttributes(["attribA", "attribB"])(
            sentinel.cube0, sentinel.cube1, template_cube=sentinel.template_cube
        )
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(sentinel.cube0, sentinel.cube1)
    mock_as_cube.assert_called_once_with(sentinel.template_cube)


def test_copy_attributes_multi_input():
    """
    Test the copy_attributes function for multiple input cubes.

    Demonstrates copying attributes from the template cube to the input
    cubes and also demonstrates the attributes on the templates cube that
    aren't specified in the attributes list awre indeed ignored.
    """
    attributes = ["attribA", "attribB"]
    cube0 = Cube([0], attributes={"attribA": "valueA", "attribB": "valueB"})
    cube1 = Cube([0], attributes={"attribA": "valueAA", "attribB": "valueBB"})
    template_cube = Cube(
        [0], attributes={"attribA": "tempA", "attribB": "tempB", "attribC": "tempC"}
    )

    plugin = CopyAttributes(attributes)
    result = plugin.process(cube0, cube1, template_cube=template_cube)
    assert type(result) == tuple
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
    cube0 = Cube([0], attributes={"attribA": "valueA", "attribB": "valueB"})
    template_cube = Cube(
        [0], attributes={"attribA": "tempA", "attribB": "tempB", "attribC": "tempC"}
    )

    plugin = CopyAttributes(attributes)
    result = plugin.process(cube0, template_cube=template_cube)
    assert type(result) == Cube
    assert result.attributes["attribA"] == "tempA"
    assert result.attributes["attribB"] == "tempB"
    assert "attribC" not in result.attributes
    assert id(result) == id(cube0)
