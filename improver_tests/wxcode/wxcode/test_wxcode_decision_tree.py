# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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

from improver.wxcode.wxcode_decision_tree import (
    START_NODE, wxcode_decision_tree)
from improver.wxcode.wxcode_decision_tree_global import (
    START_NODE_GLOBAL, wxcode_decision_tree_global)
from . import check_diagnostic_lists_consistency

TREES = [wxcode_decision_tree(), wxcode_decision_tree_global()]
START_NODES = [START_NODE, START_NODE_GLOBAL]

REQUIRED_KEY_WORDS = ['succeed',
                      'fail',
                      'probability_thresholds',
                      'threshold_condition',
                      'condition_combination',
                      'diagnostic_fields',
                      'diagnostic_thresholds',
                      'diagnostic_conditions']

OPTIONAL_KEY_WORDS = ['diagnostic_missing_action',
                      'diagnostic_gamma']

THRESHOLD_CONDITIONS = ['<=', '<', '>', '>=']
CONDITION_COMBINATIONS = ['AND', 'OR']
DIAGNOSTIC_CONDITIONS = ['below', 'above']

KEYWORDS_DIAGNOSTIC_MISSING_ACTION = ['succeed',
                                      'fail']


@pytest.mark.parametrize('tree', TREES)
def test_basic(tree):
    """Test that the wxcode_decision_tree returns a dictionary."""
    assert isinstance(tree, dict)

@pytest.mark.parametrize('tree', TREES)
def test_keywords(tree):
    """Test that the only permissible keywords are used."""
    all_key_words = REQUIRED_KEY_WORDS + OPTIONAL_KEY_WORDS
    for node in tree:
        for entry in tree[node]:
            assert entry in all_key_words

@pytest.mark.parametrize('tree,start_node', zip(TREES, START_NODES))
def test_start_node_in_tree(tree, start_node):
    """Test that the start node is in the tree"""
    assert start_node in tree

@pytest.mark.parametrize('tree', TREES)
def test_keywords_diagnostic_missing(tree):
    """Test only set keywords are used in diagnostic_missing_action."""
    all_key_words = KEYWORDS_DIAGNOSTIC_MISSING_ACTION
    for items in tree.values():
        if 'diagnostic_missing_action' in items:
            entry = items['diagnostic_missing_action']
            assert entry in all_key_words

@pytest.mark.parametrize('tree', TREES)
def test_condition_combination(tree):
    """Test only permissible values are used in condition_combination."""
    for node in tree:
        combination = tree[node]['condition_combination']
        num_diagnostics = len(tree[node]['diagnostic_fields'])
        if num_diagnostics == 2:
            assert combination in CONDITION_COMBINATIONS
        else:
            assert not combination

@pytest.mark.parametrize('tree', TREES)
def test_threshold_condition(tree):
    """Test only permissible values are used in threshold_condition."""
    for node in tree:
        threshold = tree[node]['threshold_condition']
        assert threshold in THRESHOLD_CONDITIONS

@pytest.mark.parametrize('tree', TREES)
def test_diagnostic_condition(tree):
    """Test only permissible values are used in diagnostic_conditions."""
    for node in tree:
        diagnostic = tree[node]['diagnostic_conditions']
        tests_diagnostic = diagnostic
        if isinstance(diagnostic[0], list):
            tests_diagnostic = [item for sublist in diagnostic
                                for item in sublist]
        for value in tests_diagnostic:
            assert value in DIAGNOSTIC_CONDITIONS

@pytest.mark.parametrize('tree', TREES)
def test_node_points_to_valid_value(tree):
    """Test that succeed and fail point to valid values or nodes."""
    for node in tree:
        succeed = tree[node]['succeed']
        if isinstance(succeed, str):
            assert succeed in tree.keys()
        fail = tree[node]['fail']
        if isinstance(fail, str):
            assert fail in tree

@pytest.mark.parametrize('tree', TREES)
def test_diagnostic_len_match(tree):
    """Test diagnostic fields, thresholds and conditions are same
    nested-list structure."""
    for node in tree:
        query = tree[node]
        check_diagnostic_lists_consistency(query)

@pytest.mark.parametrize('tree', TREES)
def test_probability_len_match(tree):
    """Test probability_thresholds list is right shape."""
    for _, query in tree.items():
        check_list = query['probability_thresholds']
        assert all([isinstance(x, (int, float)) for x in check_list])
        assert len(check_list) == len(query['diagnostic_fields'])

@pytest.mark.parametrize('tree', TREES)
def test_gamma_len_match(tree):
    """Test diagnostic_gamma list is right shape if present."""
    for _, query in tree.items():
        check_list = query.get('diagnostic_gamma', None)
        if not check_list:
            continue
        assert all([isinstance(x, (int, float)) for x in check_list])
        assert len(check_list) == len(query['diagnostic_fields'])
