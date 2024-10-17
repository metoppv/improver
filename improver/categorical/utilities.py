# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""This module defines the utilities required for decision tree plugin """

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import iris
import numpy as np
from iris.cube import Cube

META_REQUIRED_KEY_WORDS = ["name"]

REQUIRED_KEY_WORDS = [
    "if_true",
    "if_false",
    "probability_thresholds",
    "thresholds",
    "threshold_condition",
    "condition_combination",
    "diagnostic_fields",
]

LEAF_REQUIRED_KEY_WORDS = ["leaf"]
LEAF_OPTIONAL_KEY_WORDS = ["if_night", "is_unreachable", "group"]

OPTIONAL_KEY_WORDS = [
    "if_diagnostic_missing",
    "deterministic",
    "diagnostic_thresholds",
    "diagnostic_conditions",
]

THRESHOLD_CONDITIONS = ["<=", "<", ">", ">="]
CONDITION_COMBINATIONS = ["AND", "OR"]
DIAGNOSTIC_CONDITIONS = ["below", "above"]

KEYWORDS_DIAGNOSTIC_MISSING = ["if_true", "if_false"]


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
            Decision tree.
        target_period:
            The period in seconds that the categories being produced should
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

    for key, query in tree.items():
        if not is_decision_node(key, query) or query.get("deterministic", False):
            continue
        query["diagnostic_thresholds"] = _make_thresholds_with_units(
            query["diagnostic_thresholds"]
        )
    return tree


def categorical_attributes(decision_tree: Dict, name: str) -> Dict[str, Any]:
    """
    Extracts leaf items from decision_tree and creates cube attributes from them.

    Args:
        decision_tree:
            Decision tree definition, provided as a dictionary.
        name:
            Name of the categorical variable

    Returns:
        Attributes defining category meanings.
    """
    import numpy as np

    attributes = {}
    leaves = {v["leaf"]: k for k, v in decision_tree.items() if "leaf" in v.keys()}
    as_sorted_dict = OrderedDict(sorted(leaves.items(), key=lambda k: k[0]))

    values = np.array(list(as_sorted_dict.keys()))
    attributes.update({name: values})
    meanings = " ".join(as_sorted_dict.values())
    attributes.update({f"{name}_meaning": meanings})
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


def update_daynight(cube: Cube, day_night: Dict) -> Cube:
    """ Update category depending on whether it is day or night

    Args:
        cube:
            Cube containing only daytime categories.
        day_night:
            Dictionary of day codes (keys) and matching night codes (values)

    Returns:
        Cube containing day and night categories

    Raises:
        CoordinateNotFoundError : cube must have time coordinate.
    """
    import numpy as np
    from iris.exceptions import CoordinateNotFoundError

    import improver.utilities.solar as solar

    if not cube.coords("time"):
        msg = "cube must have time coordinate "
        raise CoordinateNotFoundError(msg)

    cube_day_night = cube.copy()
    daynightplugin = solar.DayNightMask()
    daynight_mask = daynightplugin(cube_day_night)

    # Loop over the codes which have a night time value.
    for k, v in day_night.items():
        index = np.where(cube_day_night.data == k)
        # Where day leave as is, where night adjust category to given value.
        cube_day_night.data[index] = np.where(
            daynight_mask.data[index] == daynightplugin.day,
            cube_day_night.data[index],
            v,
        )

    return cube_day_night


def is_decision_node(key: str, query: Dict[str, Any]) -> bool:
    """
    Determine whether a given node is a decision node.
    The meta node has a key of "meta", leaf nodes have a query key of "leaf", everything
    else is a decision node.

    Args:
        key:
            Decision name ("meta" indicates a non-decision node)
        query:
            Dict where key "leaf" indicates a non-decision node

    Returns:
        True if query represents a decision node
    """
    return key != "meta" and "leaf" not in query.keys()


def interrogate_decision_tree(decision_tree: Dict[str, Dict[str, Any]]) -> str:
    """
    Obtain a list of necessary inputs from the decision tree as it is currently
    defined. Return a formatted string that contains the diagnostic names, the
    thresholds needed, and whether they are thresholded above or below these
    values. If the required diagnostic is deterministic then just the diagnostic names are
    outputted. This output is used with the --check-tree option in the CLI, informing
    the user of the necessary inputs for a provided decision tree.

    Args:
        decision_tree:
            The decision tree that is to be interrogated.

    Returns:
        Returns a formatted string describing the diagnostics required,
        including threshold details.
    """
    # Diagnostic names and threshold values.
    requirements = {}
    # Create a list of formatted strings that will be printed as part of the
    # CLI help.
    output = []
    for key, query in decision_tree.items():
        if not is_decision_node(key, query):
            continue
        diagnostics = get_parameter_names(
            expand_nested_lists(query, "diagnostic_fields")
        )
        if query.get("deterministic", False):
            for diagnostic in diagnostics:
                output.append(f"\u26C5 {diagnostic} (deterministic)")
                output.sort()
        else:
            thresholds = expand_nested_lists(query, "diagnostic_thresholds")
            for diagnostic, threshold in zip(diagnostics, thresholds):
                requirements.setdefault(diagnostic, set()).add(threshold)

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
            of categorical decision-making information
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
    decision_tree: Dict[str, Dict[str, Any]], target_period: Optional[int] = None
) -> str:
    """Perform some checks to ensure the provided decision tree is valid.

    Args:
        decision_tree:
            Decision tree definition, provided as a dictionary.
        target_period:
            The period in seconds that the categorical data being produced should
            represent. This should correspond with any period diagnostics, e.g.
            precipitation accumulation, being used as input. This is used to scale
            any threshold values that are defined with an associated period in
            the decision tree.

    Returns:
        A list of problems found in the decision tree, or if none are found, the
        required input diagnostics.

    Raises:
        ValueError: If decision_tree is not a dictionary.
    """
    # Check tree is a dictionary
    if not isinstance(decision_tree, dict):
        raise ValueError("Decision tree is not a dictionary")

    issues = []
    meta = decision_tree.pop("meta", None)
    if meta is None:
        issues.append("Decision tree does not contain a mandatory meta key")
    else:
        missing_keys = set(META_REQUIRED_KEY_WORDS) - set(meta.keys())
        unexpected_keys = set(meta.keys()) - set(META_REQUIRED_KEY_WORDS)
        if missing_keys:
            issues.append(f"Meta node does not contain mandatory keys {missing_keys}")
        if unexpected_keys:
            issues.append(f"Meta node contains unexpected keys {unexpected_keys}")
    start_node = list(decision_tree.keys())[0]
    all_targets = np.array(
        [
            (n.get("if_true"), n.get("if_false"), n.get("if_night"))
            for n in decision_tree.values()
        ]
    ).flatten()
    if not decision_tree.get("deterministic", False):
        decision_tree = update_tree_thresholds(decision_tree, target_period)

    all_key_words = REQUIRED_KEY_WORDS + OPTIONAL_KEY_WORDS
    all_leaf_key_words = LEAF_REQUIRED_KEY_WORDS + LEAF_OPTIONAL_KEY_WORDS

    # Check that all leaves have a unique "leaf" value
    all_leaves = [v["leaf"] for v in decision_tree.values() if "leaf" in v.keys()]
    unique_leaves = set()
    duplicates = [x for x in all_leaves if x in unique_leaves or unique_leaves.add(x)]
    if duplicates:
        issues.append(
            f"These leaf categories are used more than once: {sorted(list(set(duplicates)))}"
        )

    for node, items in decision_tree.items():
        if "leaf" in items.keys():
            # Check the leaf only contains expected keys
            for entry in items.keys():
                if entry not in all_leaf_key_words:
                    issues.append(f"Leaf node '{node}' contains unknown key '{entry}'")

            # Check that this leaf is reachable, or is declared unreachable.
            if not ((items.get("is_unreachable", False)) or node in all_targets):
                issues.append(
                    f"Unreachable leaf '{node}'. Add 'is_unreachable': True to suppress this issue."
                )
            if (items.get("is_unreachable", False)) and node in all_targets:
                issues.append(f"Leaf '{node}' has 'is_unreachable' but can be reached.")

            # If leaf key is present, check it is an int.
            leaf_target = items["leaf"]
            if not isinstance(leaf_target, int):
                issues.append(f"Leaf '{node}' has non-int target: {leaf_target}")

            # If leaf has "if_night", check it points to another leaf
            # AND that the other leaf does NOT have "if_night".
            if "if_night" in items.keys():
                target = decision_tree.get(items["if_night"], None)
                if not target:
                    issues.append(
                        f"Leaf '{node}' does not point to a valid target ({items['if_night']})."
                    )
                elif "leaf" not in target.keys():
                    issues.append(
                        f"Target '{items['if_night']}' of leaf '{node}' is not a leaf."
                    )
                elif "if_night" in target.keys():
                    issues.append(
                        f"Night target '{items['if_night']}' of leaf '{node}' "
                        "also has a night target."
                    )
            # If leaf has "group", check the group contains at least two members.
            if "group" in items.keys():
                members = [
                    k
                    for k, v in decision_tree.items()
                    if v.get("group", None) == items["group"]
                ]
                if len(members) == 1:
                    issues.append(
                        f"Leaf '{node}' is in a group of 1 ({items['group']})."
                    )
        else:
            # Check the tree only contains expected keys
            for entry in items.keys():
                if entry not in all_key_words:
                    issues.append(f"Node {node} contains unknown key '{entry}'")

            # Check that this node is reachable, or is the start_node
            if not ((node == start_node) or node in all_targets):
                issues.append(f"Unreachable node '{node}'")

            # Check that if_diagnostic_missing key points at a if_true or if_false
            # node
            if "if_diagnostic_missing" in items.keys():
                entry = items["if_diagnostic_missing"]
                if entry not in KEYWORDS_DIAGNOSTIC_MISSING:
                    issues.append(
                        f"Node {node} contains an if_diagnostic_missing key "
                        f"that targets key '{entry}' which is neither 'if_true' "
                        "nor 'if_false'"
                    )

            # Check that only permissible values are used in condition_combination
            # this will be AND / OR for multiple diagnostic fields, or blank otherwise
            combination = decision_tree[node]["condition_combination"]
            num_diagnostics = len(decision_tree[node]["diagnostic_fields"])
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
            threshold = items["threshold_condition"]
            if threshold not in THRESHOLD_CONDITIONS:
                issues.append(
                    f"Node {node} uses invalid threshold condition {threshold}"
                )

            # Check the succeed and fail destinations are valid; that is a valid
            # category for leaf nodes, and other tree nodes otherwise
            for result in "if_true", "if_false":
                value = items[result]
                if isinstance(value, str):
                    if value not in decision_tree.keys():
                        issues.append(
                            f"Node {node} has an invalid destination "
                            f"of {value} for the {result} condition"
                        )
                else:
                    issues.append(
                        f"Node {node} results in a bare category "
                        f"of {value} for the {result} condition. Should point to a leaf."
                    )
            if items.get("deterministic", False):
                threshold_name = "thresholds"
            else:
                # Check diagnostic_conditions are all above or below.
                diagnostic = items["diagnostic_conditions"]
                tests_diagnostic = diagnostic
                if isinstance(diagnostic[0], list):
                    tests_diagnostic = [
                        item for sublist in diagnostic for item in sublist
                    ]
                for value in tests_diagnostic:
                    if value not in DIAGNOSTIC_CONDITIONS:
                        issues.append(
                            f"Node {node} uses invalid diagnostic condition "
                            f"'{value}'; this should be 'above' or 'below'"
                        )

                # Check diagnostic_fields, diagnostic_conditions, and diagnostic_thresholds
                # are all nested equivalently
                if not _check_diagnostic_lists_consistency(items):
                    issues.append(
                        f"Node {node} has inconsistent nesting for the "
                        "diagnostic_fields, diagnostic_conditions, and "
                        "diagnostic_thresholds fields"
                    )
                threshold_name = "probability_thresholds"

            # Check thresholds are numeric and there are as many of them
            # as there are diagnostics_fields.
            thresholds = items[threshold_name]
            diagnostic_fields = items["diagnostic_fields"]
            if not all(isinstance(x, (int, float)) for x in thresholds):
                issues.append(
                    f"Node {node} has a non-numeric probability threshold "
                    f"{thresholds}"
                )
            if not len(thresholds) == len(get_parameter_names(diagnostic_fields)):
                issues.append(
                    f"Node {node} has a different number of probability thresholds "
                    f"and diagnostic_fields: {thresholds}, {diagnostic_fields}"
                )

    if not issues:
        issues.append("Decision tree OK\nRequired inputs are:")
        issues.append(interrogate_decision_tree(decision_tree))
    return "\n".join(issues)


def day_night_map(decision_tree: Dict[str, Dict[str, Union[str, List]]]) -> Dict:
    """Returns a dict showing which night values are linked to which day values

    Args:
        decision_tree:
            Decision tree definition, provided as a dictionary.

    Returns:
        dict showing which night categories (values) are linked to which day categories (keys)
    """
    return {
        v["leaf"]: decision_tree[v["if_night"]]["leaf"]
        for k, v in decision_tree.items()
        if "if_night" in v.keys()
    }
