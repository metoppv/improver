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
"""Unit tests for the improver.PostProcessingPlugin abstract base class"""

from typing import List, Union

import numpy as np
import pytest
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


@pytest.fixture(name="dummy_plugin")
def dummy_plugin_fixture():
    class DummyPlugin(PostProcessingPlugin):
        """Dummy class inheriting from the abstract base class"""

        def process(self, arg):
            """Local process method has no effect"""
            return arg

    return DummyPlugin


@pytest.fixture(name="plugin_input")
def input_fixture(is_list: bool) -> Union[Cube, List[Cube]]:
    cube = set_up_variable_cube(
        np.ones((3, 3, 3), dtype=np.float32),
        standard_grid_metadata="uk_det",
        attributes={"title": "UKV Model Forecast"},
    )
    if is_list:
        return [cube, cube]
    else:
        return cube


def assert_title_attribute(
    result: Union[Cube, List[Cube]], expected_title: str
) -> None:
    if isinstance(result, Cube):
        assert result.attributes["title"] == expected_title
    else:
        for cube in result:
            assert cube.attributes["title"] == expected_title


@pytest.mark.parametrize("is_list", (True, False))
def test_title_updated(dummy_plugin, plugin_input):
    """Test title is updated as expected"""
    expected_title = "Post-Processed UKV Model Forecast"
    result = dummy_plugin()(plugin_input)
    assert_title_attribute(result, expected_title)


@pytest.mark.parametrize("is_list", (True, False))
def test_title_preserved(dummy_plugin, plugin_input, is_list):
    """Test title is preserved if it contains 'Post-Processed'"""
    expected_title = "IMPROVER Post-Processed Multi-Model Blend"
    if is_list:
        for c in plugin_input:
            c.attributes["title"] = expected_title
    else:
        plugin_input.attributes["title"] = expected_title
    result = dummy_plugin()(plugin_input)
    assert_title_attribute(result, expected_title)


@pytest.mark.parametrize("is_list", (True, False))
def test_title_mandatory_attribute_default(dummy_plugin, plugin_input, is_list):
    """Test title is preserved if it is the same as mandatory_attribute['title']"""
    expected_title = MANDATORY_ATTRIBUTE_DEFAULTS["title"]
    if is_list:
        for c in plugin_input:
            c.attributes["title"] = expected_title
    else:
        plugin_input.attributes["title"] = expected_title
    result = dummy_plugin()(plugin_input)
    assert_title_attribute(result, expected_title)


def test_non_cubes(dummy_plugin):
    """Test non-cube types are returned unchanged"""
    result = dummy_plugin()(None)
    assert result is None

    iterable_input = ["list", "of", "strings"]
    result = dummy_plugin()(iterable_input)
    assert result == iterable_input
