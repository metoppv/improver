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
"""Utilities for checking decision trees are valid."""

from typing import Any, Dict, List, Optional, Tuple, Union
from improver.wxcode.utilities import (
    WX_DICT,
    get_parameter_names,
    update_tree_units,
    interrogate_decision_tree,
)


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


def check_diagnostic_lists_consistency(query):
    """
    Checks if specific input lists have same nested list
    structure. e.g. ['item'] != [['item']]

    Args:
        query (dict):
            of weather-symbols decision-making information

    Raises:
        ValueError: if diagnostic query lists have different nested list
            structure.

    """
    diagnostic_keys = [
        "diagnostic_fields",
        "diagnostic_conditions",
        "diagnostic_thresholds",
    ]
    values = [
        get_parameter_names(query[key]) if key == "diagnostic_fields" else query[key]
        for key in diagnostic_keys
    ]
    return check_nested_list_consistency(values)


def check_nested_list_consistency(query):
    """
    Return True if all input lists have same nested list
    structure. e.g. ['item'] != [['item']]

    Args:
        query (list of lists):
            Nested lists to check for consistency.

    Returns:
        bool: True if diagnostic query lists have same nested list
            structure.

    """

    def _checker(lists):
        """Return True if all input lists have same nested list
        structure. e.g. ['item'] != [['item']]."""
        type_set = set(map(type, lists))
        if list in type_set:
            return (
                len(type_set) == 1
                and len(set(map(len, lists))) == 1
                and all(map(_checker, zip(*lists)))
            )
        return True

    return _checker(query)


def check_tree(wxtree: Dict) -> str:
    """Perform some checks to ensure the provided decision tree is valid."""

    wxtree = update_tree_units(wxtree)

    issues = []
    valid_codes = list(WX_DICT.keys())
    # Check tree is a dictionary
    if not isinstance(wxtree, dict):
        issues.append("Decision tree is not a dictionary")

    all_key_words = REQUIRED_KEY_WORDS + OPTIONAL_KEY_WORDS
    for node, items in wxtree.items():
        # Check the tree only contains expected keys
        for entry in wxtree[node]:
            if entry not in all_key_words:
                issues.append(f"Node {node} contains unknown key '{entry}'")

        # Check that diagnostic_missing_action point at a succeed or fail node
        if "diagnostic_missing_action" in items:
            entry = items["diagnostic_missing_action"]
            if entry not in KEYWORDS_DIAGNOSTIC_MISSING_ACTION:
                issues.append(
                    f"Node {node} contains a diagnostic_missing_action "
                    f"that targets key '{entry}' which is neither 'succeed' "
                    "nor 'fail'"
                )

        # Check that only permissible values are used in condition_combination
        # this will be AND / OR for multiple diagnostic fields, or blank otherwise
        combination = wxtree[node]["condition_combination"]
        num_diagnostics = len(wxtree[node]["diagnostic_fields"])
        if num_diagnostics == 2 and combination not in CONDITION_COMBINATIONS:
            issues.append(
                f"Node {node} utilises 2 diagnostic fields but "
                f"'{combination}' is not a valid combination condition"
            )
        elif num_diagnostics != 2 and combination:
            issues.append(
                f"Node {node} utilises combination condition "
                f"'{combination}' but does not use 2 diagnostic fields "
                "for combination in this way"
            )

        # Check only permissible values are used in threshold_condition
        threshold = wxtree[node]["threshold_condition"]
        if threshold not in THRESHOLD_CONDITIONS:
            issues.append(
                f"Node {node} uses invalid threshold condition " f"{threshold}"
            )

        # Check diagnostic_conditions are all above or below.
        diagnostic = wxtree[node]["diagnostic_conditions"]
        tests_diagnostic = diagnostic
        if isinstance(diagnostic[0], list):
            tests_diagnostic = [item for sublist in diagnostic for item in sublist]
        for value in tests_diagnostic:
            if value not in DIAGNOSTIC_CONDITIONS:
                issues.append(
                    f"Node {node} uses invalid diagnostic condition "
                    f"{value}; this should be 'above' or 'below'"
                )

        # Check the succeed and fail destinations are valid; that is valid
        # weather codes for leaf nodes, and other tree nodes otherwise
        for result in "succeed", "fail":
            value = wxtree[node][result]
            if isinstance(value, str):
                if value not in wxtree.keys():
                    issues.append(
                        f"Node {node} has an invalid destination "
                        f"of {value} for the {result} condition"
                    )
            else:
                if value not in valid_codes:
                    issues.append(
                        f"Node {node} results in an invalid weather "
                        f"code of {value} for the {result} condition"
                    )

        # Check diagnostic_fields, diagnostic_conditions, and diagnostic_thresholds
        # are all nested equivalently
        if not check_diagnostic_lists_consistency(items):
            issues.append(
                f"Node {node} has inconsistent nesting for the "
                "diagnostic_fields, diagnostic_conditions, and "
                "diagnostic_thresholds fields"
            )

        # Check probabiltiy thresholds are numeric and there are as many of them
        # as there are diagnostics_fields.
        prob_thresholds = items["probability_thresholds"]
        diagnostic_fields = items["diagnostic_fields"]
        if not all(isinstance(x, (int, float)) for x in prob_thresholds):
            issues.append(
                f"Node {node} has a non-numeric probability threshold "
                f"{prob_thresholds}"
            )
        if not len(prob_thresholds) == len(get_parameter_names(diagnostic_fields)):
            issues.append(
                f"Node {node} has a different number of probability thresholds "
                f"and diagnostic_fields: {prob_thresholds}, {diagnostic_fields}"
            )

    if not issues:
        issues.append("Decision tree OK\nRequired inputs are:")
        issues.append(interrogate_decision_tree(wxtree))
    return "\n".join(issues)
