# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the improver.PostProcessingPlugin abstract base class"""

import copy
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


@pytest.mark.parametrize(
    "non_cube", (None, ["list", "of", "strings"], 0, "kittens", {"inputs": "outputs"})
)
def test_non_cubes(dummy_plugin, non_cube):
    """Test non-cube types are returned unchanged"""
    expected = copy.copy(non_cube)
    result = dummy_plugin()(non_cube)
    assert result == expected
