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
"""Unit tests for Weather Symbols class."""

import unittest

from iris.tests import IrisTest

from improver.wxcode.wxcode_decision_tree import wxcode_decision_tree
from improver.wxcode.wxcode_utilities import expand_nested_lists

REQUIRED_KEY_WORDS = ['succeed',
                      'fail',
                      'probability_thresholds',
                      'threshold_condition',
                      'condition_combination',
                      'diagnostic_fields',
                      'diagnostic_thresholds',
                      'diagnostic_conditions']

OPTIONAL_KEY_WORDS = ['diagnostic_gamma']

THRESHOLD_CONDITIONS = ['<=', '<', '>', '>=']
CONDITION_COMBINATIONS = ['AND', 'OR']
DIAGNOSTIC_CONDITIONS = ['below', 'above']


class Test_wxcode_decision_tree(IrisTest):

    """Test the wxcode decision tree."""

    def test_basic(self):
        """Test that the wxcode_decision_tree returns a dictionary."""
        result = wxcode_decision_tree()
        self.assertIsInstance(result, dict)

    def test_keywords(self):
        """Test that the only permissible keywords are used."""
        tree = wxcode_decision_tree()
        all_key_words = REQUIRED_KEY_WORDS + OPTIONAL_KEY_WORDS
        for node in tree:
            for entry in tree[node]:
                self.assertEqual(entry in all_key_words, True)

    def test_condition_combination(self):
        """Test only permissible values are used in condition_combination."""
        tree = wxcode_decision_tree()
        for node in tree:
            combination = tree[node]['condition_combination']
            num_diagnostics = len(tree[node]['diagnostic_fields'])
            if num_diagnostics == 2:
                self.assertEqual(combination in CONDITION_COMBINATIONS, True)
            else:
                self.assertEqual(combination, '')

    def test_threshold_condition(self):
        """Test only permissible values are used in threshold_condition."""
        tree = wxcode_decision_tree()
        for node in tree:
            threshold = tree[node]['threshold_condition']
            self.assertEqual(threshold in THRESHOLD_CONDITIONS, True)

    def test_diagnostic_condition(self):
        """Test only permissible values are used in diagnostic_conditions."""
        tree = wxcode_decision_tree()
        for node in tree:
            diagnostic = tree[node]['diagnostic_conditions']
            tests_diagnostic = diagnostic
            if isinstance(diagnostic[0], list):
                tests_diagnostic = [item for sublist in diagnostic
                                    for item in sublist]
            for value in tests_diagnostic:
                self.assertEqual(value in DIAGNOSTIC_CONDITIONS, True)

    def test_node_points_to_valid_value(self):
        """Test that succeed and fail point to valid values or nodes."""
        tree = wxcode_decision_tree()
        for node in tree:
            succeed = tree[node]['succeed']
            if isinstance(succeed, str):
                self.assertEqual(succeed in tree.keys(), True)
            fail = tree[node]['fail']
            if isinstance(fail, str):
                self.assertEqual(fail in tree, True)

    def test_diagnostic_len_match(self):
        """Test diagnostic fields, thresholds and conditions are same len."""
        tree = wxcode_decision_tree()
        for node in tree:
            query = tree[node]
            diag_len = len(expand_nested_lists(query, 'diagnostic_fields'))
            thres_len = len(expand_nested_lists(query,
                                                'diagnostic_thresholds'))
            cond_len = len(expand_nested_lists(query,
                                               'diagnostic_conditions'))
            self.assertEqual(diag_len, thres_len)
            self.assertEqual(diag_len, cond_len)


if __name__ == '__main__':
    unittest.main()
