# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing categorical decision tree implementation."""


import copy
import operator
from typing import Dict, List, Optional, Tuple, Union

import iris
import numpy as np
from iris import Constraint
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray

from improver import BasePlugin
from improver.blending.utilities import (
    record_run_coord_to_attr,
    store_record_run_as_coord,
)
from improver.categorical.utilities import (
    categorical_attributes,
    day_night_map,
    expand_nested_lists,
    get_parameter_names,
    is_decision_node,
    is_variable,
    update_daynight,
    update_tree_thresholds,
)
from improver.metadata.amend import update_model_id_attr_attribute
from improver.metadata.forecast_times import forecast_period_coord
from improver.metadata.probabilistic import (
    find_threshold_coordinate,
    get_threshold_coord_name_from_probability_name,
    probability_is_above_or_below,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.common_input_handle import as_cubelist


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


class ApplyDecisionTree(BasePlugin):
    """
    Definition and implementation of a categorical decision tree. This
    plugin uses a variety of diagnostic inputs and the decision tree logic
    to determine the most representative category for each location
    defined in the input cubes. This can be used for generating categorical data such
    as weather symbols.

    .. See the documentation for information about building a decision tree.
    .. include:: extended_documentation/categorical/build_a_decision_tree.rst
    """

    def __init__(
        self,
        decision_tree: dict,
        model_id_attr: Optional[str] = None,
        record_run_attr: Optional[str] = None,
        target_period: Optional[int] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Define a decision tree for determining a category based upon
        the input diagnostics. Use this decision tree to allocate a category to each location.

        Args:
            decision_tree:
                Decision tree definition, provided as a dictionary.
            model_id_attr:
                Name of attribute recording source models that should be
                inherited by the output cube. The source models are expected as
                a space-separated string.
            record_run_attr:
                Name of attribute used to record models and cycles used in
                constructing the output.
            target_period:
                The period in seconds that the category being produced should
                represent. This should correspond with any period diagnostics, e.g.
                precipitation accumulation, being used as input. This is used to scale
                any threshold values that are defined with an associated period in
                the decision tree. It will only be used if the decision tree
                provided has threshold values defined with an associated period.
            title:
                An optional title to assign to the title attribute of the resulting
                output. This will override the title generated from
                the inputs, where this generated title is only set if all of the
                inputs share a common title.

        float_tolerance defines the tolerance when matching thresholds to allow
        for the difficulty of float comparisons.
        float_abs_tolerance defines the tolerance for when the threshold
        is zero. It has to be sufficiently small that a valid rainfall rate
        or snowfall rate could not trigger it.
        """

        self.model_id_attr = model_id_attr
        self.record_run_attr = record_run_attr
        node_names = list(decision_tree.keys())
        self.start_node = node_names[1] if node_names[0] == "meta" else node_names[0]
        self.target_period = target_period
        self.title = title
        self.meta = decision_tree["meta"]
        self.queries = update_tree_thresholds(
            {k: v for k, v in decision_tree.items() if k != "meta"}, target_period
        )
        self.float_tolerance = 0.01
        self.float_abs_tolerance = 1e-12
        # flag to indicate whether to expect "threshold" as a coordinate name
        # (defaults to False, checked on reading input cubes)
        self.coord_named_threshold = False

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        return "<ApplyDecisionTree start_node={}>".format(self.start_node)

    def prepare_input_cubes(
        self, *cubes: Union[Cube, CubeList]
    ) -> Tuple[CubeList, Optional[List[str]]]:
        """
        Check that the input cubes contain all the diagnostics and thresholds
        required by the decision tree.  Sets self.coord_named_threshold to
        "True" if threshold-type coordinates have the name "threshold" (as
        opposed to the standard name of the diagnostic), for backward
        compatibility. A cubelist containing only cubes of the required
        diagnostic-threshold combinations is returned.

        Args:
            cubes:
                Input diagnostic cubes.

        Returns:
            - A CubeList containing only the required cubes.
            - A list of node names where the diagnostic data is missing and
              this is indicated as allowed by the presence of the if_diagnostic_missing
              key.

        Raises:
            IOError:
                Raises an IOError if any of the required input data is missing.
                The error includes details of which fields are missing.
        """
        cubes = as_cubelist(*cubes)

        # Check that all cubes are valid at or over the same periods
        self.check_coincidence(cubes)

        used_cubes = iris.cube.CubeList()
        optional_node_data_missing = []
        missing_data = []
        for key, query in self.queries.items():
            if not is_decision_node(key, query):
                continue
            diagnostics = get_parameter_names(
                expand_nested_lists(query, "diagnostic_fields")
            )
            if query.get("deterministic", False):
                for diagnostic in diagnostics:
                    test_condition = iris.Constraint(name=diagnostic)
                    matched_cube = cubes.extract(test_condition)
                    if not matched_cube:
                        if "if_diagnostic_missing" in query:
                            optional_node_data_missing.append(key)
                        else:
                            missing_data.append(f"name: {diagnostic} (deterministic)")
                        continue
                    used_cubes.extend(matched_cube)
            else:
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
                        if "if_diagnostic_missing" in query:
                            optional_node_data_missing.append(key)
                        else:
                            missing_data.append(
                                f"name: {diagnostic}, threshold: {threshold}, "
                                f"spp__relative_to_threshold: {condition}\n"
                            )
                        continue

                    cube_threshold_units = find_threshold_coordinate(
                        matched_cube[0]
                    ).units
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
                        missing_data.append(
                            f"name: {diagnostic}, threshold: {threshold}, "
                            f"spp__relative_to_threshold: {condition}\n"
                        )
                    else:
                        used_cubes.extend(matched_threshold)

        if missing_data:
            msg = "Decision Tree input cubes are missing the following required input fields:\n"
            for dyn_msg in missing_data:
                msg += dyn_msg
            raise IOError(msg)

        if not optional_node_data_missing:
            optional_node_data_missing = None
        return used_cubes, optional_node_data_missing

    def check_coincidence(self, cubes: Union[List[Cube], CubeList]):
        """
        Check that all the provided cubes are valid at the same time and if any
        of the input cubes have time bounds, these match.

        The last input cube with bounds (or first input cube if none have bounds)
        is selected as a template_cube for later producing the output
        cube.

        Args:
            cubes:
                List of input cubes used in the decision tree

        Raises:
            ValueError: If validity times differ for diagnostics.
            ValueError: If period diagnostics have different periods.
            ValueError: If period diagnostics do not match target_period.
        """
        times = []
        bounds = []
        diagnostics = []
        self.template_cube = cubes[0]
        for cube in cubes:
            times.extend(cube.coord("time").points)
            time_bounds = cube.coord("time").bounds
            if time_bounds is not None:
                diagnostics.append(cube.name())
                bounds.extend(time_bounds.tolist())
                self.template_cube = cube

        # Check that all validity times are the same
        if len(set(times)) != 1:
            diagnostic_times = [
                f"{diagnostic.name()}: {time}" for diagnostic, time in zip(cubes, times)
            ]
            raise ValueError(
                "Decision Tree input cubes are valid at different times; "
                f"\n{diagnostic_times}"
            )
        # Check that if multiple bounds have been returned, they are all identical.
        if bounds and not bounds.count(bounds[0]) == len(bounds):
            diagnostic_bounds = [
                f"{diagnostic}: {np.diff(bound)[0]}"
                for diagnostic, bound in zip(diagnostics, bounds)
            ]
            raise ValueError(
                "Period diagnostics with different periods have been provided "
                "as input to the decision tree code. Period diagnostics must "
                "all describe the same period to be used together."
                f"\n{diagnostic_bounds}"
            )
        # Check that if time bounded diagnostics are used they match the
        # user specified target_period.
        if bounds and self.target_period:
            if not np.diff(bounds[0])[0] == self.target_period:
                raise ValueError(
                    f"Diagnostic periods ({np.diff(bounds[0])[0]}) do not match "
                    f"the user specified target_period ({self.target_period})."
                )

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
        satisfying the negative case.

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
        if test_conditions.get("deterministic", False):
            coord = "thresholds"
        else:
            coord = "probability_thresholds"

        for index, (diagnostic, p_threshold) in enumerate(
            zip(test_conditions["diagnostic_fields"], test_conditions[coord])
        ):

            d_threshold = test_conditions.get("diagnostic_thresholds")
            d_threshold = d_threshold[index] if d_threshold else None
            loop += 1
            if isinstance(diagnostic, list):
                # We have a list which could contain variable names, operators and
                # numbers. The variable names need converting into Iris Constraint
                # syntax while operators and numbers remain unchanged.
                # We expect an entry in p_threshold for each condition in diagnostic
                # fields. d_threshold_index is used for probabilistic trees to track
                # the diagnostic threshold to extract for each variable in each condition.
                d_threshold_index = -1
                extract_constraint = []
                for item in diagnostic:
                    if is_variable(item):
                        # Add a constraint from the variable name and threshold value
                        d_threshold_index += 1
                        if test_conditions.get("deterministic"):
                            extract_constraint.append(iris.Constraint(item))
                        else:
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
                if d_threshold:
                    extract_constraint = self.construct_extract_constraint(
                        diagnostic, d_threshold, self.coord_named_threshold
                    )
                else:
                    extract_constraint = iris.Constraint(diagnostic)
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

    def remove_optional_missing(self, optional_node_data_missing: List[str]):
        """
        Some decision tree nodes are optional and have an "if_diagnostic_missing"
        key to enable passage through the tree in the absence of the required
        input diagnostic. This code modifies the tree in the following ways:

           - Rewrites the decision tree to skip the missing nodes by connecting
             nodes that proceed them to the node targetted by "if_diagnostic_missing"
           - If the node(s) missing are those at the start of the decision
             tree, the start_node is modified to find the first available node.

        Args:
            optional_node_data_missing:
                List of node names for which data is missing but for which this
                is allowed.
        """
        for missing in optional_node_data_missing:

            # Get the name of the alternative node to bypass the missing one
            target = self.queries[missing]["if_diagnostic_missing"]
            alternative = self.queries[missing][target]

            for query in self.queries.values():
                if query.get("if_true", None) == missing:
                    query["if_true"] = alternative
                if query.get("if_false", None) == missing:
                    query["if_false"] = alternative

            if self.start_node == missing:
                self.start_node = alternative

    @staticmethod
    def find_all_routes(
        graph: Dict, start: str, end: int, route: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Function to trace all routes through the decision tree.

        Args:
            graph:
                A dictionary that describes each node in the tree,
                e.g. {<node_name>: [<if_true_name>, <if_false_name>]}
            start:
                The node name of the tree root (currently always
                lightning).
            end:
                The category code to which we are tracing all routes.
            route:
                A list of node names found so far.

        Returns:
            A list of node names that defines the route from the tree root
            to the category leaf (end of chain).

        References:
            Method based upon Python Patterns - Implementing Graphs essay
            https://www.python.org/doc/essays/graphs/
        """
        if route is None:
            route = []

        route = route + [start]
        if start == end:
            return [route]
        if start not in graph.keys():
            return []

        routes = []
        for node in graph[start]:
            if node not in route:
                newroutes = ApplyDecisionTree.find_all_routes(
                    graph, node, end, route=route
                )
                routes.extend(newroutes)
        return routes

    def create_categorical_cube(self, cubes: Union[List[Cube], CubeList]) -> Cube:
        """
        Create an empty categorical cube taking the cube name and categorical attribute names
        from the meta node, and categorical attribute values from the leaf nodes.
        The reference time is the latest from the set of input cubes and the optional record
        run attribute is a combination from all source cubes. Everything else comes from the
        template cube, which is the first cube with time bounds, or the first cube if none
        have time bounds.

        Args:
            cubes:
                List of input cubes used in the decision tree

        Returns:
            A cube with suitable metadata to describe the categories
            that will fill it and data initiated with the value -1 to allow
            any unset points to be readily identified.
        """
        try:
            threshold_coord = find_threshold_coordinate(self.template_cube)
        except CoordinateNotFoundError:
            template_cube = self.template_cube
        else:
            template_cube = next(
                self.template_cube.slices_over([threshold_coord])
            ).copy()
            # remove coordinates and bounds that do not apply to a categorical cube
            template_cube.remove_coord(threshold_coord)

        mandatory_attributes = generate_mandatory_attributes(cubes)
        if self.title:
            mandatory_attributes.update({"title": self.title})
        optional_attributes = categorical_attributes(self.queries, self.meta["name"])
        if self.model_id_attr:
            optional_attributes.update(
                update_model_id_attr_attribute(cubes, self.model_id_attr)
            )
        if self.record_run_attr and self.model_id_attr:
            store_record_run_as_coord(
                set(cubes), self.record_run_attr, self.model_id_attr
            )

        categories = create_new_diagnostic_cube(
            self.meta["name"],
            "1",
            template_cube,
            mandatory_attributes,
            optional_attributes=optional_attributes,
            data=np.ma.masked_all_like(template_cube.data).astype(np.int32),
        )
        if self.record_run_attr and self.model_id_attr is not None:
            # Use set(cubes) here as the prepare_input_cubes method returns a list
            # of inputs that contains duplicate pointers to the same threshold
            # slices. Using set here ensures each contributing model/cycle/diagnostic
            # is only considered once when creating the record run coordinate.
            record_run_coord_to_attr(
                categories, set(cubes), self.record_run_attr, discard_weights=True
            )
        self._set_reference_time(categories, cubes)

        return categories

    @staticmethod
    def _set_reference_time(cube: Cube, cubes: CubeList):
        """Replace the forecast_reference_time and/or blend_time if present coord point on cube
        with the latest value from cubes. Forecast_period is also updated."""
        coord_names = ["forecast_reference_time", "blend_time"]
        coords_found = []
        for coord_name in coord_names:
            try:
                reference_time = max([c.coord(coord_name).points[0] for c in cubes])
            except CoordinateNotFoundError:
                continue
            coords_found.append(coord_name)
        if not coords_found:
            raise CoordinateNotFoundError(
                f"Could not find {'or '.join(coord_names)} on all input cubes"
            )
        for coord_name in coords_found:
            new_coord = cube.coord(coord_name).copy(reference_time)
            cube.replace_coord(new_coord)
        cube.replace_coord(
            forecast_period_coord(cube, force_lead_time_calculation=True)
        )

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
                f"Invalid comparator: {comparator}. "
                "Comparator must be one of '<', '>', '<=', '>='."
            )

    def evaluate_extract_expression(
        self, cubes: CubeList, expression: Union[Constraint, List]
    ) -> ndarray:
        """Evaluate a single condition.

        Args:
            cubes:
                A cubelist containing the diagnostics required for the
                decision tree, these at co-incident times.
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
            for op_str in [["/", "*"], ["+", "-"]]:
                while len(curr_expression) > 1:
                    for idx, item in enumerate(curr_expression):
                        if isinstance(item, str) and (item in op_str):
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
                            op = operator_map[item]
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
                decision tree, these at co-incident times.
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
                    "Invalid condition chain found. First element has length > 1 "
                    "but second element is not 'AND' or 'OR'."
                )
                raise RuntimeError(msg)
        return res

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """Apply the decision tree to the input cubes to produce categorical output.

        Args:
            cubes:
                Diagnostics required for the decision tree, these at co-incident times.

        Returns:
            A cube of categorical data.
        """
        # Check input cubes contain required data and return only those that
        # are needed to speed up later cube extractions.
        cubes, optional_node_data_missing = self.prepare_input_cubes(*cubes)

        # Reroute the decision tree around missing optional nodes
        if optional_node_data_missing is not None:
            self.remove_optional_missing(optional_node_data_missing)

        # Construct graph nodes dictionary
        graph = {
            key: [self.queries[key]["leaf"]]
            if "leaf" in self.queries[key].keys()
            else [self.queries[key]["if_true"], self.queries[key]["if_false"]]
            for key in self.queries
        }
        # Search through tree for all leaves (category end points)
        defined_categories = []
        for item in self.queries.values():
            for value in item.values():
                if isinstance(value, int):
                    defined_categories.append(value)
        # Create categorical cube
        categories = self.create_categorical_cube(cubes)
        # Loop over possible categories

        for category_code in defined_categories:
            routes = self.find_all_routes(graph, self.start_node, category_code,)
            # Loop over possible routes from root to leaf
            for route in routes:
                conditions = []
                for i_node in range(len(route) - 1):
                    current_node = route[i_node]
                    current = copy.copy(self.queries[current_node])
                    next_node = route[i_node + 1]

                    if current.get("if_false") == next_node:

                        (
                            current["threshold_condition"],
                            current["condition_combination"],
                        ) = self.invert_condition(current)
                    if "leaf" not in current.keys():
                        conditions.append(self.create_condition_chain(current))
                test_chain = [conditions, "AND"]

                # Set grid locations to suitable category
                categories.data[
                    np.ma.where(self.evaluate_condition_chain(cubes, test_chain))
                ] = category_code

        # Update categories for day or night where appropriate.
        categories = update_daynight(categories, day_night_map(self.queries))
        return categories
