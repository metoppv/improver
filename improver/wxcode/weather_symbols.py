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
"""Module containing weather symbol implementation."""


import copy
import operator
from typing import Any, Dict, List, Optional, Tuple, Union

import iris
import numpy as np
from iris import Constraint
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.amend import update_model_id_attr_attribute
from improver.metadata.probabilistic import (
    find_threshold_coordinate,
    get_threshold_coord_name_from_probability_name,
    probability_is_above_or_below,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.wxcode.utilities import (
    expand_nested_lists,
    get_parameter_names,
    is_variable,
    update_daynight,
    weather_code_attributes,
)
from improver.wxcode.wxcode_decision_tree import START_NODE, wxcode_decision_tree
from improver.wxcode.wxcode_decision_tree_global import (
    START_NODE_GLOBAL,
    wxcode_decision_tree_global,
)


def _define_invertible_conditions() -> Dict[str, str]:
    """Returns a dictionary of boolean comparator strings where the value is the
    logical inverse of the key."""
    invertible_conditions = {
        ">=": "<",
        ">": "<=",
        "OR": "AND",
        "": "",
    }
    # Add reverse {value: key} entries to invertible_conditions
    reverse_inversions = {}
    for k, v in invertible_conditions.items():
        reverse_inversions[v] = k
    invertible_conditions.update(reverse_inversions)
    return invertible_conditions


INVERTIBLE_CONDITIONS = _define_invertible_conditions()


class WeatherSymbols(BasePlugin):
    """
    Definition and implementation of a weather symbol decision tree. This
    plugin uses a variety of diagnostic inputs and the decision tree logic
    to determine the most representative weather symbol for each site
    defined in the input cubes.
    """

    def __init__(
        self, wxtree: str = "high_resolution", model_id_attr: Optional[str] = None
    ) -> None:
        """
        Define a decision tree for determining weather symbols based upon
        the input diagnostics. Use this decision tree to allocate a weather
        symbol to each point.

        Args:
            wxtree:
                Used to choose weather symbol decision tree.
                Default is "high_resolution"
                "global" will load the global weather symbol decision tree.
            model_id_attr:
                Name of attribute recording source models that should be
                inherited by the output cube. The source models are expected as
                a space-separated string.

        float_tolerance defines the tolerance when matching thresholds to allow
        for the difficulty of float comparisons.
        float_abs_tolerance defines the tolerance for when the threshold
        is zero. It has to be sufficiently small that a valid rainfall rate
        or snowfall rate could not trigger it.
        """

        def make_thresholds_with_units(items):
            if isinstance(items, list):
                return [make_thresholds_with_units(item) for item in items]
            values, units = items
            return iris.coords.AuxCoord(values, units=units)

        self.wxtree = wxtree
        self.model_id_attr = model_id_attr
        if wxtree == "global":
            self.queries = wxcode_decision_tree_global()
            self.start_node = START_NODE_GLOBAL
        else:
            self.queries = wxcode_decision_tree()
            self.start_node = START_NODE
        for query in self.queries.values():
            query["diagnostic_thresholds"] = make_thresholds_with_units(
                query["diagnostic_thresholds"]
            )
        self.float_tolerance = 0.01
        self.float_abs_tolerance = 1e-12
        # flag to indicate whether to expect "threshold" as a coordinate name
        # (defaults to False, checked on reading input cubes)
        self.coord_named_threshold = False

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        return "<WeatherSymbols tree={} start_node={}>".format(
            self.wxtree, self.start_node
        )

    def check_input_cubes(self, cubes: CubeList) -> Optional[Dict[str, Any]]:
        """
        Check that the input cubes contain all the diagnostics and thresholds
        required by the decision tree.  Sets self.coord_named_threshold to
        "True" if threshold-type coordinates have the name "threshold" (as
        opposed to the standard name of the diagnostic), for backward
        compatibility.

        Args:
            cubes:
                A CubeList containing the input diagnostic cubes.

        Returns:
            A dictionary of (keyword) nodes names where the diagnostic
            data is missing and (values) node associated with
            diagnostic_missing_action.

        Raises:
            IOError:
                Raises an IOError if any of the required input data is missing.
                The error includes details of which fields are missing.
        """
        optional_node_data_missing = {}
        missing_data = []
        for key, query in self.queries.items():
            diagnostics = get_parameter_names(
                expand_nested_lists(query, "diagnostic_fields")
            )
            thresholds = expand_nested_lists(query, "diagnostic_thresholds")
            conditions = expand_nested_lists(query, "diagnostic_conditions")
            for diagnostic, threshold, condition in zip(
                diagnostics, thresholds, conditions
            ):

                # First we check the diagnostic name and units, performing
                # a conversion is required and possible.
                test_condition = iris.Constraint(name=diagnostic)
                matched_cube = cubes.extract(test_condition)
                if not matched_cube:
                    if "diagnostic_missing_action" in query:
                        optional_node_data_missing.update(
                            {key: query[query["diagnostic_missing_action"]]}
                        )
                    else:
                        missing_data.append([diagnostic, threshold, condition])
                    continue

                cube_threshold_units = find_threshold_coordinate(matched_cube[0]).units
                threshold.convert_units(cube_threshold_units)

                # Then we check if the required threshold is present in the
                # cube, and that the thresholding is relative to it correctly.
                threshold = threshold.points.item()
                threshold_name = find_threshold_coordinate(matched_cube[0]).name()

                # Set flag to check for old threshold coordinate names
                if threshold_name == "threshold" and not self.coord_named_threshold:
                    self.coord_named_threshold = True

                # Check threshold == 0.0
                if abs(threshold) < self.float_abs_tolerance:
                    coord_constraint = {
                        threshold_name: lambda cell: np.isclose(
                            cell.point, 0, rtol=0, atol=self.float_abs_tolerance
                        )
                    }
                else:
                    coord_constraint = {
                        threshold_name: lambda cell: np.isclose(
                            cell.point, threshold, rtol=self.float_tolerance, atol=0
                        )
                    }

                # Checks whether the spp__relative_to_threshold attribute is above
                # or below a threshold and and compares to the diagnostic_condition.
                test_condition = iris.Constraint(
                    coord_values=coord_constraint,
                    cube_func=lambda cube: (
                        probability_is_above_or_below(cube) == condition
                    ),
                )
                matched_threshold = matched_cube.extract(test_condition)
                if not matched_threshold:
                    missing_data.append([diagnostic, threshold, condition])

        if missing_data:
            msg = (
                "Weather Symbols input cubes are missing"
                " the following required"
                " input fields:\n"
            )
            dyn_msg = "name: {}, threshold: {}, " "spp__relative_to_threshold: {}\n"
            for item in missing_data:
                msg = msg + dyn_msg.format(*item)
            raise IOError(msg)

        if not optional_node_data_missing:
            optional_node_data_missing = None
        return optional_node_data_missing

    @staticmethod
    def _invert_comparator(comparator: str) -> str:
        """Inverts a single comparator string."""
        try:
            return INVERTIBLE_CONDITIONS[comparator]
        except KeyError:
            raise KeyError(f"Unexpected condition {comparator}, cannot invert it.")

    def invert_condition(self, condition: Dict) -> Tuple[str, str]:
        """
        Invert a comparison condition to allow positive identification of conditions
        satisfying the negative ('fail') case.

        Args:
            condition:
                A single query from the decision tree.

        Returns:
            - A string representing the inverted comparison.
            - A string representing the inverted combination
        """
        inverted_threshold = self._invert_comparator(condition["threshold_condition"])
        inverted_combination = self._invert_comparator(
            condition["condition_combination"]
        )
        return inverted_threshold, inverted_combination

    def create_condition_chain(self, test_conditions: Dict) -> List:
        """
        Construct a list of all the conditions specified in a single query.

        Args:
            test_conditions:
                A query from the decision tree.

        Returns:
            A valid condition chain is defined recursively:
            (1) If each a_1, ..., a_n is an extract expression (i.e. a
            constraint, or a list of constraints,
            operator strings and floats), and b is either "AND", "OR" or "",
            then [[a1, ..., an], b] is a valid condition chain.
            (2) If a1, ..., an are each valid conditions chain, and b is
            either "AND" or "OR", then [[a1, ..., an], b] is a valid
            condition chain.
        """
        conditions = []
        loop = 0
        for diagnostic, p_threshold, d_threshold in zip(
            test_conditions["diagnostic_fields"],
            test_conditions["probability_thresholds"],
            test_conditions["diagnostic_thresholds"],
        ):

            loop += 1

            if isinstance(diagnostic, list):
                # We have a list which could contain variable names, operators and
                # numbers. The variable names need converting into Iris Constraint
                # syntax while operators and numbers remain unchanged.
                # We expect an entry in p_threshold for each variable name, so
                # d_threshold_index is used to track these.
                d_threshold_index = -1
                extract_constraint = []
                for item in diagnostic:
                    if is_variable(item):
                        # Add a constraint from the variable name and threshold value
                        d_threshold_index += 1
                        extract_constraint.append(
                            self.construct_extract_constraint(
                                item,
                                d_threshold[d_threshold_index],
                                self.coord_named_threshold,
                            )
                        )
                    else:
                        # Add this operator or variable as-is
                        extract_constraint.append(item)
            else:
                # Non-lists are assumed to be constraints on a single variable.
                extract_constraint = self.construct_extract_constraint(
                    diagnostic, d_threshold, self.coord_named_threshold
                )
            conditions.append(
                [
                    extract_constraint,
                    test_conditions["threshold_condition"],
                    p_threshold,
                ]
            )
        condition_chain = [conditions, test_conditions["condition_combination"]]
        return condition_chain

    def construct_extract_constraint(
        self, diagnostic: str, threshold: AuxCoord, coord_named_threshold: bool
    ) -> Constraint:
        """
        Construct an iris constraint.

        Args:
            diagnostic:
                The name of the diagnostic to be extracted from the CubeList.
            threshold:
                The thresholds within the given diagnostic cube that is
                needed, including units.  Note these are NOT coords from the
                original cubes, just constructs to associate units with values.
            coord_named_threshold:
                If true, use old naming convention for threshold coordinates
                (coord.long_name=threshold).  Otherwise extract threshold
                coordinate name from diagnostic name

        Returns:
            A constraint
        """

        if coord_named_threshold:
            threshold_coord_name = "threshold"
        else:
            threshold_coord_name = get_threshold_coord_name_from_probability_name(
                diagnostic
            )

        threshold_val = threshold.points.item()
        if abs(threshold_val) < self.float_abs_tolerance:
            cell_constraint = lambda cell: np.isclose(
                cell.point, threshold_val, rtol=0, atol=self.float_abs_tolerance,
            )
        else:
            cell_constraint = lambda cell: np.isclose(
                cell.point, threshold_val, rtol=self.float_tolerance, atol=0,
            )

        kw_dict = {"{}".format(threshold_coord_name): cell_constraint}
        constraint = iris.Constraint(name=diagnostic, **kw_dict)
        return constraint

    @staticmethod
    def find_all_routes(
        graph: Dict,
        start: str,
        end: int,
        omit_nodes: Optional[Dict] = None,
        route: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Function to trace all routes through the decision tree.

        Args:
            graph:
                A dictionary that describes each node in the tree,
                e.g. {<node_name>: [<succeed_name>, <fail_name>]}
            start:
                The node name of the tree root (currently always
                heavy_precipitation).
            end:
                The weather symbol code to which we are tracing all routes.
            omit_nodes:
                A dictionary of (keyword) nodes names where the diagnostic
                data is missing and (values) node associated with
                diagnostic_missing_action.
            route:
                A list of node names found so far.

        Returns:
            A list of node names that defines the route from the tree root
            to the weather symbol leaf (end of chain).

        References:
            Method based upon Python Patterns - Implementing Graphs essay
            https://www.python.org/doc/essays/graphs/
        """
        if route is None:
            route = []

        if omit_nodes:
            start_not_valid = True
            while start_not_valid:
                if start in omit_nodes:
                    start = omit_nodes[start]
                else:
                    start_not_valid = False
        route = route + [start]
        if start == end:
            return [route]
        if start not in graph.keys():
            return []

        routes = []
        for node in graph[start]:
            if node not in route:
                newroutes = WeatherSymbols.find_all_routes(
                    graph, node, end, omit_nodes=omit_nodes, route=route
                )
                routes.extend(newroutes)
        return routes

    def create_symbol_cube(self, cubes: Union[List[Cube], CubeList]) -> Cube:
        """
        Create an empty weather symbol cube

        Args:
            cubes:
                List of input cubes used to generate weather symbols

        Returns:
            A cube with suitable metadata to describe the weather symbols
            that will fill it and data initiated with the value -1 to allow
            any unset points to be readily identified.
        """
        threshold_coord = find_threshold_coordinate(cubes[0])
        template_cube = next(cubes[0].slices_over([threshold_coord])).copy()
        # remove coordinates and bounds that do not apply to weather symbols
        template_cube.remove_coord(threshold_coord)
        for coord in template_cube.coords():
            if coord.name() in ["forecast_period", "time"]:
                coord.bounds = None

        mandatory_attributes = generate_mandatory_attributes(cubes)
        optional_attributes = weather_code_attributes()
        if self.model_id_attr:
            optional_attributes.update(
                update_model_id_attr_attribute(cubes, self.model_id_attr)
            )

        symbols = create_new_diagnostic_cube(
            "weather_code",
            "1",
            template_cube,
            mandatory_attributes,
            optional_attributes=optional_attributes,
            data=np.ma.masked_all_like(template_cube.data).astype(np.int32),
        )
        return symbols

    @staticmethod
    def compare_array_to_threshold(
        arr: ndarray, comparator: str, threshold: float
    ) -> ndarray:
        """Compare two arrays element-wise and return a boolean array.

        Args:
            arr
            comparator:
                One of  '<', '>', '<=', '>='.
            threshold

        Returns:
            Array of booleans.

        Raises:
            ValueError: If comparator is not one of '<', '>', '<=', '>='.
        """
        if comparator == "<":
            return arr < threshold
        elif comparator == ">":
            return arr > threshold
        elif comparator == "<=":
            return arr <= threshold
        elif comparator == ">=":
            return arr >= threshold
        else:
            raise ValueError(
                "Invalid comparator: {}. ".format(comparator),
                "Comparator must be one of '<', '>', '<=', '>='.",
            )

    def evaluate_extract_expression(
        self, cubes: CubeList, expression: Union[Constraint, List]
    ) -> ndarray:
        """Evaluate a single condition.

        Args:
            cubes:
                A cubelist containing the diagnostics required for the
                weather symbols decision tree, these at co-incident times.
            expression:
                Defined recursively:
                A list consisting of an iris.Constraint or a list of
                iris.Constraint, strings (representing operators) and floats
                is a valid expression.
                A list consisting of valid expressions, strings (representing
                operators) and floats is a valid expression.

        Returns:
            An array or masked array of booleans
        """
        operator_map = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }
        if isinstance(expression, iris.Constraint):
            return cubes.extract(expression)[0].data
        else:
            curr_expression = copy.deepcopy(expression)
            # evaluate sub-expressions first
            for idx, item in enumerate(expression):
                if isinstance(item, list):
                    curr_expression = (
                        curr_expression[:idx]
                        + [self.evaluate_extract_expression(cubes, item)]
                        + curr_expression[idx + 1 :]
                    )
            # evaluate operators in order of precedence
            for op_str in ["/", "*", "+", "-"]:
                while len(curr_expression) > 1:
                    for idx, item in enumerate(curr_expression):
                        if isinstance(item, str) and (item == op_str):
                            left_arg = curr_expression[idx - 1]
                            right_arg = curr_expression[idx + 1]
                            if isinstance(left_arg, iris.Constraint):
                                left_eval = cubes.extract(left_arg)[0].data
                            else:
                                left_eval = left_arg
                            if isinstance(right_arg, iris.Constraint):
                                right_eval = cubes.extract(right_arg)[0].data
                            else:
                                right_eval = right_arg
                            op = operator_map[op_str]
                            res = op(left_eval, right_eval)
                            curr_expression = (
                                curr_expression[: idx - 1]
                                + [res]
                                + curr_expression[idx + 2 :]
                            )
                            break
                    else:
                        break
            if isinstance(curr_expression[0], iris.Constraint):
                res = cubes.extract(curr_expression[0])[0].data
            return res

    def evaluate_condition_chain(
        self, cubes: CubeList, condition_chain: List
    ) -> ndarray:
        """Recursively evaluate the list of conditions.

        We can safely use recursion here since the depth will be small.

        Args:
            cubes:
                A cubelist containing the diagnostics required for the
                weather symbols decision tree, these at co-incident times.
            condition_chain:
                A valid condition chain is defined recursively:
                (1) If each a_1, ..., a_n is an extract expression (i.e. a
                constraint, or a list of constraints,
                operator strings and floats), and b is either "AND", "OR" or "",
                then [[a1, ..., an], b] is a valid condition chain.
                (2) If a1, ..., an are each valid conditions chain, and b is
                either "AND" or "OR", then [[a1, ..., an], b] is a valid
                condition chain.

        Returns:
            An array of masked array of booleans
        """

        def is_chain(item):
            return (
                isinstance(item, list)
                and isinstance(item[1], str)
                and (item[1] in ["AND", "OR", ""])
            )

        items_list, comb = condition_chain
        item = items_list[0]
        if is_chain(item):
            res = self.evaluate_condition_chain(cubes, item)
        else:
            condition, comparator, threshold = item
            res = self.compare_array_to_threshold(
                self.evaluate_extract_expression(cubes, condition),
                comparator,
                threshold,
            )
        for item in items_list[1:]:
            if is_chain(item):
                new_res = self.evaluate_condition_chain(cubes, item)
            else:
                condition, comparator, threshold = item
                new_res = self.compare_array_to_threshold(
                    self.evaluate_extract_expression(cubes, condition),
                    comparator,
                    threshold,
                )
            # If comb is "", then items_list has length 1, so here we can
            # assume comb is either "AND" or "OR"
            if comb == "AND":
                res = res & new_res
            elif comb == "OR":
                res = res | new_res
            else:
                msg = (
                    "Invalid condition chain found. First element has length > 1 ",
                    "but second element is not 'AND' or 'OR'.",
                )
                raise RuntimeError(msg)
        return res

    def process(self, cubes: CubeList) -> Cube:
        """Apply the decision tree to the input cubes to produce weather
        symbol output.

        Args:
            cubes:
                A cubelist containing the diagnostics required for the
                weather symbols decision tree, these at co-incident times.

        Returns:
            A cube of weather symbols.
        """
        # Check input cubes contain required data
        optional_node_data_missing = self.check_input_cubes(cubes)
        # Construct graph nodes dictionary
        graph = {
            key: [self.queries[key]["succeed"], self.queries[key]["fail"]]
            for key in self.queries
        }
        # Search through tree for all leaves (weather code end points)
        defined_symbols = []
        for item in self.queries.values():
            for value in item.values():
                if isinstance(value, int):
                    defined_symbols.append(value)
        # Create symbol cube
        symbols = self.create_symbol_cube(cubes)
        # Loop over possible symbols
        for symbol_code in defined_symbols:

            # In current decision tree
            # start node is heavy_precipitation
            routes = self.find_all_routes(
                graph,
                self.start_node,
                symbol_code,
                omit_nodes=optional_node_data_missing,
            )
            # Loop over possible routes from root to leaf

            for route in routes:
                conditions = []
                for i_node in range(len(route) - 1):
                    current_node = route[i_node]
                    current = copy.copy(self.queries[current_node])
                    try:
                        next_node = route[i_node + 1]
                    except KeyError:
                        next_node = symbol_code

                    if current["fail"] == next_node:
                        (
                            current["threshold_condition"],
                            current["condition_combination"],
                        ) = self.invert_condition(current)

                    conditions.append(self.create_condition_chain(current))
                test_chain = [conditions, "AND"]

                # Set grid locations to suitable weather symbol
                symbols.data[
                    np.ma.where(self.evaluate_condition_chain(cubes, test_chain))
                ] = symbol_code
        # Update symbols for day or night.
        symbols = update_daynight(symbols)
        return symbols
