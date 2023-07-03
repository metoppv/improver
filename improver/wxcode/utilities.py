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
"""This module defines the utilities required for wxcode plugin """

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import iris
import numpy as np
from iris.cube import Cube

REQUIRED_KEY_WORDS = [
    "if_true",
    "if_false",
    "probability_thresholds",
    "threshold_condition",
    "condition_combination",
    "diagnostic_fields",
    "diagnostic_thresholds",
    "diagnostic_conditions",
]

OPTIONAL_KEY_WORDS = ["if_diagnostic_missing"]

THRESHOLD_CONDITIONS = ["<=", "<", ">", ">="]
CONDITION_COMBINATIONS = ["AND", "OR"]
DIAGNOSTIC_CONDITIONS = ["below", "above"]

KEYWORDS_DIAGNOSTIC_MISSING = ["if_true", "if_false"]


_WX_DICT_IN = {
    0: "Clear_Night",
    1: "Sunny_Day",
    2: "Partly_Cloudy_Night",
    3: "Partly_Cloudy_Day",
    4: "Dust",
    5: "Mist",
    6: "Fog",
    7: "Cloudy",
    8: "Overcast",
    9: "Light_Shower_Night",
    10: "Light_Shower_Day",
    11: "Drizzle",
    12: "Light_Rain",
    13: "Heavy_Shower_Night",
    14: "Heavy_Shower_Day",
    15: "Heavy_Rain",
    16: "Sleet_Shower_Night",
    17: "Sleet_Shower_Day",
    18: "Sleet",
    19: "Hail_Shower_Night",
    20: "Hail_Shower_Day",
    21: "Hail",
    22: "Light_Snow_Shower_Night",
    23: "Light_Snow_Shower_Day",
    24: "Light_Snow",
    25: "Heavy_Snow_Shower_Night",
    26: "Heavy_Snow_Shower_Day",
    27: "Heavy_Snow",
    28: "Thunder_Shower_Night",
    29: "Thunder_Shower_Day",
    30: "Thunder",
}

WX_DICT = OrderedDict(sorted(_WX_DICT_IN.items(), key=lambda t: t[0]))

DAYNIGHT_CODES = [1, 3, 10, 14, 17, 20, 23, 26, 29]

GROUPED_CODES = {
    "snow": [23, 24, 26, 27],
    "sleet": [17, 18],
    "rain": [10, 12, 14, 15],
    "convection": [20, 21, 29, 30],
}

GROUPED_CODES_NEW = {
    "rain": [10, 12, 14, 15],
    "convection": [10, 14, 17, 20, 21, 29, 30],
    "frozen_precip": [17, 18, 23, 24, 26, 27],
    "visibility": [5, 6],
}

GROUPED_CODES_NEW2 = {
    "snow": [23, 24, 26, 27],
    "sleet": [17, 18],
    "rain": [10, 11, 12, 14, 15],
    "convection": [10, 14, 20, 21, 29, 30],
    "visibility": [5, 6],
}


def update_tree_thresholds(
    tree: Dict[str, Dict[str, Any]], target_period: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Replaces value / unit pairs from tree definition with an Iris AuxCoord
    that encodes the same information. Also scales any threshold values that
    have an associated period (e.g. accumulation in 3600 seconds) by a factor
    to reflect the target period (e.g. a 3-hour, 10800 second, weather symbol).

    Args:
        tree:
            Weather symbols decision tree.
        target_period:
            The period in seconds that the weather symbol being produced should
            represent. This should correspond with any period diagnostics, e.g.
            precipitation accumulation, being used as input. This is used to scale
            any threshold values that are defined with an associated period in
            the decision tree.
    Returns:
        The tree now containing AuxCoords instead of value / unit pairs, with
        period diagnostic threshold values scaled appropriately.

    Raises:
        ValueError: If thresholds are defined with an associated period and no
                    target_period is provided.
    """

    def _make_thresholds_with_units(items):
        if isinstance(items[0], list):
            return [_make_thresholds_with_units(item) for item in items]
        try:
            values, units, period = items
        except ValueError:
            values, units = items
        else:
            if not target_period:
                raise ValueError(
                    "The decision tree contains thresholds defined for a particular "
                    "time period, e.g. the accumulation over 3600 seconds. To "
                    "use such a decision tree a target_period must be provided "
                    "such that the threshold values can be scaled appropriately."
                )
            values = values * target_period / period
        return iris.coords.AuxCoord(values, units=units)

    for query in tree.values():
        query["diagnostic_thresholds"] = _make_thresholds_with_units(
            query["diagnostic_thresholds"]
        )
    return tree


def weather_code_attributes() -> Dict[str, Any]:
    """
    Returns:
        Attributes defining weather code meanings.
    """
    import numpy as np

    attributes = {}
    wx_keys = np.array(list(WX_DICT.keys()))
    attributes.update({"weather_code": wx_keys})
    wxstring = " ".join(WX_DICT.values())
    attributes.update({"weather_code_meaning": wxstring})
    return attributes


def expand_nested_lists(query: Dict[str, Any], key: str) -> List[Any]:
    """
    Produce flat lists from list and nested lists.

    Args:
        query:
            A single query from the decision tree.
        key:
            A string denoting the field to be taken from the dict.

    Returns:
        A 1D list containing all the values for a given key.
    """
    items = []
    for item in query[key]:
        if isinstance(item, list):
            items.extend(item)
        else:
            items.extend([item])
    return items


def update_daynight(cubewx: Cube) -> Cube:
    """ Update weather cube depending on whether it is day or night

    Args:
        cubewx:
            Cube containing only daytime weather symbols.

    Returns:
        Cube containing day and night weather symbols

    Raises:
        CoordinateNotFoundError : cube must have time coordinate.
    """
    import numpy as np
    from iris.exceptions import CoordinateNotFoundError

    import improver.utilities.solar as solar

    if not cubewx.coords("time"):
        msg = "cube must have time coordinate "
        raise CoordinateNotFoundError(msg)

    cubewx_daynight = cubewx.copy()
    daynightplugin = solar.DayNightMask()
    daynight_mask = daynightplugin(cubewx_daynight)

    # Loop over the codes which decrease by 1 if a night time value
    # e.g. 1 - sunny day becomes 0 - clear night.
    for val in DAYNIGHT_CODES:
        index = np.where(cubewx_daynight.data == val)
        # Where day leave as is, where night correct weather
        # code to value  - 1.
        cubewx_daynight.data[index] = np.where(
            daynight_mask.data[index] == daynightplugin.day,
            cubewx_daynight.data[index],
            cubewx_daynight.data[index] - 1,
        )

    return cubewx_daynight


def interrogate_decision_tree(wxtree: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Obtain a list of necessary inputs from the decision tree as it is currently
    defined. Return a formatted string that contains the diagnostic names, the
    thresholds needed, and whether they are thresholded above or below these
    values. This output is used with the --check-tree option in the CLI, informing
    the user of the necessary inputs for a provided decision tree.

    Args:
        wxtree:
            The weather symbol tree that is to be interrogated.

    Returns:
        Returns a formatted string descring the diagnostics required,
        including threshold details.
    """
    # Diagnostic names and threshold values.
    requirements = {}
    for query in wxtree.values():
        diagnostics = get_parameter_names(
            expand_nested_lists(query, "diagnostic_fields")
        )
        thresholds = expand_nested_lists(query, "diagnostic_thresholds")
        for diagnostic, threshold in zip(diagnostics, thresholds):
            requirements.setdefault(diagnostic, set()).add(threshold)

    # Create a list of formatted strings that will be printed as part of the
    # CLI help.
    output = []
    for requirement, uniq_thresh in sorted(requirements.items()):
        (units,) = {u.units for u in uniq_thresh}  # enforces same units
        thresh_str = ", ".join(map(str, sorted({v.points[0] for v in uniq_thresh})))
        output.append("\u26C5 {} ({}): {}".format(requirement, units, thresh_str))

    n_files = len(output)
    formatted_string = "{}\n" * n_files
    formatted_output = formatted_string.format(*output)

    return formatted_output


def is_variable(thing: str) -> bool:
    """
    Identify whether given string is likely to be a variable name by
    identifying the exceptions.

    Args:
        thing:
            The string to operate on

    Returns:
        False if thing is one of ["+", "-", "*", "/"] or if float(
        thing) does not raise a ValueError, else True.
    """
    valid_operators = ["+", "-", "*", "/"]
    try:
        float(thing)
        return False
    except ValueError:
        return thing not in valid_operators


def get_parameter_names(diagnostic_fields: List[List[str]]) -> List[List[str]]:
    """
    For diagnostic fields that can contain operators and values, strips out
    just the parameter names.

    Args:
        diagnostic_fields:

    Returns:
        The parameter names
    """
    parameter_names = []
    for condition in diagnostic_fields:
        if isinstance(condition, list):
            parameter_names.append(get_parameter_names(condition))
        elif is_variable(condition):
            parameter_names.append(condition)
    return parameter_names


def _check_diagnostic_lists_consistency(query: Dict[str, Any]) -> bool:
    """
    Checks if specific input lists have same nested list
    structure. e.g. ['item'] != [['item']]

    Args:
        query:
            of weather-symbols decision-making information
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
    return _check_nested_list_consistency(values)


def _check_nested_list_consistency(query: List[List[Any]]) -> bool:
    """
    Return True if all input lists have same nested list
    structure. e.g. ['item'] != [['item']]

    Args:
        query:
            Nested lists to check for consistency.

    Returns:
        True if diagnostic query lists have same nested list
        structure, else returns False.

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


def check_tree(
    wxtree: Dict[str, Dict[str, Any]], target_period: Optional[int] = None
) -> str:
    """Perform some checks to ensure the provided decision tree is valid.

    Args:
        wxtree:
            Weather symbols decision tree definition, provided as a
            dictionary.
        target_period:
            The period in seconds that the weather symbol being produced should
            represent. This should correspond with any period diagnostics, e.g.
            precipitation accumulation, being used as input. This is used to scale
            any threshold values that are defined with an associated period in
            the decision tree.

    Returns:
        A list of problems found in the decision tree, or if none are found, the
        required input diagnostics.

    Raises:
        ValueError: If wxtree is not a dictionary.
    """
    # Check tree is a dictionary
    if not isinstance(wxtree, dict):
        raise ValueError("Decision tree is not a dictionary")

    issues = []
    start_node = list(wxtree.keys())[0]
    all_targets = np.array(
        [(n["if_true"], n["if_false"]) for n in wxtree.values()]
    ).flatten()
    wxtree = update_tree_thresholds(wxtree, target_period)
    valid_codes = list(WX_DICT.keys())

    all_key_words = REQUIRED_KEY_WORDS + OPTIONAL_KEY_WORDS
    for node, items in wxtree.items():
        # Check the tree only contains expected keys
        for entry in wxtree[node]:
            if entry not in all_key_words:
                issues.append(f"Node {node} contains unknown key '{entry}'")

        # Check that this node is reachable, or is the start_node
        if not ((node == start_node) or node in all_targets):
            issues.append(f"Unreachable node '{node}'")

        # Check that if_diagnostic_missing key points at a if_true or if_false
        # node
        if "if_diagnostic_missing" in items:
            entry = items["if_diagnostic_missing"]
            if entry not in KEYWORDS_DIAGNOSTIC_MISSING:
                issues.append(
                    f"Node {node} contains an if_diagnostic_missing key "
                    f"that targets key '{entry}' which is neither 'if_true' "
                    "nor 'if_false'"
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
            issues.append(f"Node {node} uses invalid threshold condition {threshold}")

        # Check diagnostic_conditions are all above or below.
        diagnostic = wxtree[node]["diagnostic_conditions"]
        tests_diagnostic = diagnostic
        if isinstance(diagnostic[0], list):
            tests_diagnostic = [item for sublist in diagnostic for item in sublist]
        for value in tests_diagnostic:
            if value not in DIAGNOSTIC_CONDITIONS:
                issues.append(
                    f"Node {node} uses invalid diagnostic condition "
                    f"'{value}'; this should be 'above' or 'below'"
                )

        # Check the succeed and fail destinations are valid; that is valid
        # weather codes for leaf nodes, and other tree nodes otherwise
        for result in "if_true", "if_false":
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
        if not _check_diagnostic_lists_consistency(items):
            issues.append(
                f"Node {node} has inconsistent nesting for the "
                "diagnostic_fields, diagnostic_conditions, and "
                "diagnostic_thresholds fields"
            )

        # Check probability thresholds are numeric and there are as many of them
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
