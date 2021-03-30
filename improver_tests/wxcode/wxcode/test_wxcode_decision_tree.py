# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Unit tests for Weather Symbols Trees."""

import pytest

from improver.wxcode.utilities import WX_DICT, get_parameter_names
from improver.wxcode.wxcode_decision_tree import START_NODE, wxcode_decision_tree
from improver.wxcode.wxcode_decision_tree_global import (
    START_NODE_GLOBAL,
    wxcode_decision_tree_global,
)

from . import check_diagnostic_lists_consistency

TREE_NAMES = ["high_resolution", "global"]
TREES = {
    "high_resolution": wxcode_decision_tree(),
    "global": wxcode_decision_tree_global(),
}
START_NODES = {"high_resolution": START_NODE, "global": START_NODE_GLOBAL}

REQUIRED_KEY_WORDS = [
    "succeed",
    "fail",
    "probability_thresholds",
    "threshold_condition",
    "condition_combination",
    "diagnostic_fields",
    "diagnostic_thresholds",
    "diagnostic_conditions",
]

OPTIONAL_KEY_WORDS = ["diagnostic_missing_action", "diagnostic_gamma"]

THRESHOLD_CONDITIONS = ["<=", "<", ">", ">="]
CONDITION_COMBINATIONS = ["AND", "OR"]
DIAGNOSTIC_CONDITIONS = ["below", "above"]

KEYWORDS_DIAGNOSTIC_MISSING_ACTION = ["succeed", "fail"]


@pytest.mark.parametrize("tree_name", TREE_NAMES)
def test_basic(tree_name):
    """Test that the wxcode_decision_tree returns a dictionary."""
    tree = TREES[tree_name]
    assert isinstance(tree, dict)


@pytest.mark.parametrize("tree_name", TREE_NAMES)
def test_keywords(tree_name):
    """Test that the only permissible keywords are used."""
    tree = TREES[tree_name]
    all_key_words = REQUIRED_KEY_WORDS + OPTIONAL_KEY_WORDS
    for node in tree:
        for entry in tree[node]:
            assert entry in all_key_words


@pytest.mark.parametrize("tree_name", TREE_NAMES)
def test_start_node_in_tree(tree_name):
    """Test that the start node is in the tree"""
    tree = TREES[tree_name]
    start_node = START_NODES[tree_name]
    assert start_node in tree


def test_keywords_diagnostic_missing():
    """Test only set keywords are used in diagnostic_missing_action.
    This only exists in the 'high_resolution' tree."""
    tree = TREES["high_resolution"]
    all_key_words = KEYWORDS_DIAGNOSTIC_MISSING_ACTION
    for items in tree.values():
        if "diagnostic_missing_action" in items:
            entry = items["diagnostic_missing_action"]
            assert entry in all_key_words


@pytest.mark.parametrize("tree_name", TREE_NAMES)
def test_condition_combination(tree_name):
    """Test only permissible values are used in condition_combination."""
    tree = TREES[tree_name]
    for node in tree:
        combination = tree[node]["condition_combination"]
        num_diagnostics = len(tree[node]["diagnostic_fields"])
        if num_diagnostics == 2:
            assert combination in CONDITION_COMBINATIONS
        else:
            assert not combination


@pytest.mark.parametrize("tree_name", TREE_NAMES)
def test_threshold_condition(tree_name):
    """Test only permissible values are used in threshold_condition."""
    tree = TREES[tree_name]
    for node in tree:
        threshold = tree[node]["threshold_condition"]
        assert threshold in THRESHOLD_CONDITIONS


@pytest.mark.parametrize("tree_name", TREE_NAMES)
def test_diagnostic_condition(tree_name):
    """Test only permissible values are used in diagnostic_conditions."""
    tree = TREES[tree_name]
    for node in tree:
        diagnostic = tree[node]["diagnostic_conditions"]
        tests_diagnostic = diagnostic
        if isinstance(diagnostic[0], list):
            tests_diagnostic = [item for sublist in diagnostic for item in sublist]
        for value in tests_diagnostic:
            assert value in DIAGNOSTIC_CONDITIONS


@pytest.mark.parametrize("tree_name", TREE_NAMES)
def test_node_points_to_valid_value(tree_name):
    """Test that succeed and fail point to valid values or nodes."""
    valid_codes = list(WX_DICT.keys())
    tree = TREES[tree_name]
    for node in tree:
        for value in tree[node]["succeed"], tree[node]["fail"]:
            if isinstance(value, str):
                assert value in tree.keys()
            else:
                assert value in valid_codes


@pytest.mark.parametrize("tree_name", TREE_NAMES)
def test_diagnostic_len_match(tree_name):
    """Test diagnostic fields, thresholds and conditions are same
    nested-list structure."""
    tree = TREES[tree_name]
    for node in tree:
        query = tree[node]
        check_diagnostic_lists_consistency(query)


@pytest.mark.parametrize("tree_name", TREE_NAMES)
def test_probability_len_match(tree_name):
    """Test probability_thresholds list is right shape."""
    tree = TREES[tree_name]
    for _, query in tree.items():
        check_list = query["probability_thresholds"]
        assert all(isinstance(x, (int, float)) for x in check_list)
        assert len(check_list) == len(get_parameter_names(query["diagnostic_fields"]))
