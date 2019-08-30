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
"""Module containing weather symbol implementation."""


import copy

import iris
import numpy as np

from improver.utilities.cube_checker import find_threshold_coordinate
from improver.utilities.cube_metadata import extract_diagnostic_name
from improver.wxcode.wxcode_decision_tree import wxcode_decision_tree
from improver.wxcode.wxcode_decision_tree_global import (
    wxcode_decision_tree_global)
from improver.wxcode.wxcode_utilities import (add_wxcode_metadata,
                                              expand_nested_lists,
                                              update_daynight)


class WeatherSymbols(object):
    """
    Definition and implementation of a weather symbol decision tree. This
    plugin uses a variety of diagnostic inputs and the decision tree logic
    to determine the most representative weather symbol for each site
    defined in the input cubes.
    """

    def __init__(self, wxtree='high_resolution'):
        """
        Define a decision tree for determining weather symbols based upon
        the input diagnostics. Use this decision tree to allocate a weather
        symbol to each point.

        Key Args:
            wxtree (str):
                Choose weather symbol decision tree.
                Default is 'high_resolution'
                'global' will load the global weather symbol decision tree.

        float_tolerance defines the tolerance when matching thresholds to allow
        for the difficulty of float comparisons.
        """
        self.wxtree = wxtree
        if wxtree == 'global':
            self.queries = wxcode_decision_tree_global()
        else:
            self.queries = wxcode_decision_tree()
        self.float_tolerance = 0.01
        # flag to indicate whether to expect "threshold" as a coordinate name
        # (defaults to False, checked on reading input cubes)
        self.coord_named_threshold = False
        # dictionary to contain names of threshold coordinates that do not
        # match expected convention
        self.threshold_coord_names = {}

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return '<WeatherSymbols tree={}>'.format(self.wxtree)

    def check_input_cubes(self, cubes):
        """
        Check that the input cubes contain all the diagnostics and thresholds
        required by the decision tree.  Sets self.coord_named_threshold to
        "True" if threshold-type coordinates have the name "threshold" (as
        opposed to the standard name of the diagnostic), for backward
        compatibility.

        Args:
            cubes (iris.cube.CubeList):
                A CubeList containing the input diagnostic cubes.

        Raises:
            IOError:
                Raises an IOError if any of the required input data is missing.
                The error includes details of which fields are missing.
        """
        missing_data = []
        for query in self.queries.values():
            diagnostics = expand_nested_lists(query, 'diagnostic_fields')
            thresholds = expand_nested_lists(query, 'diagnostic_thresholds')
            conditions = expand_nested_lists(query, 'diagnostic_conditions')
            for diagnostic, threshold, condition in zip(
                    diagnostics, thresholds, conditions):

                # First we check the diagnostic name and units, performing
                # a conversion is required and possible.
                test_condition = (iris.Constraint(name=diagnostic))
                matched_cube = cubes.extract(test_condition)
                if not matched_cube:
                    missing_data.append([diagnostic, threshold, condition])
                    continue
                else:
                    cube_threshold_units = (
                        find_threshold_coordinate(matched_cube[0]).units)
                    threshold.convert_units(cube_threshold_units)

                # Then we check if the required threshold is present in the
                # cube, and that the thresholding is relative to it correctly.
                threshold = threshold.points.item()
                threshold_name = find_threshold_coordinate(
                    matched_cube[0]).name()

                # Check cube and threshold coordinate names match according to
                # expected convention.  If not, add to exception dictionary.
                if extract_diagnostic_name(diagnostic) != threshold_name:
                    self.threshold_coord_names[diagnostic] = (
                        threshold_name)

                # Set flag to check for old threshold coordinate names
                if (threshold_name == "threshold" and
                        not self.coord_named_threshold):
                    self.coord_named_threshold = True

                test_condition = (
                    iris.Constraint(
                        coord_values={threshold_name: lambda cell: (
                            threshold * (1. - self.float_tolerance) < cell <
                            threshold * (1. + self.float_tolerance))},
                        cube_func=lambda cube: (
                            find_threshold_coordinate(
                                cube
                            ).attributes['spp__relative_to_threshold'] ==
                            condition)))
                matched_threshold = matched_cube.extract(test_condition)
                if not matched_threshold:
                    missing_data.append([diagnostic, threshold, condition])

        if missing_data:
            msg = ('Weather Symbols input cubes are missing'
                   ' the following required'
                   ' input fields:\n')
            dyn_msg = ('name: {}, threshold: {}, '
                       'spp__relative_to_threshold: {}\n')
            for item in missing_data:
                msg = msg + dyn_msg.format(*item)
            raise IOError(msg)
        return

    @staticmethod
    def invert_condition(test_conditions):
        """
        Invert a comparison condition to select the negative case.

        Args:
            test_conditions (dict):
                A single query from the decision tree.
        Returns:
            (tuple): tuple containing:
                **inverted_threshold** (str):
                    A string representing the inverted comparison.
                **inverted_combination** (str):
                    A string representing the inverted combination
        """
        threshold = test_conditions['threshold_condition']
        inverted_threshold = threshold
        if threshold == '>=':
            inverted_threshold = '<'
        elif threshold == '<=':
            inverted_threshold = '>'
        elif threshold == '<':
            inverted_threshold = '>='
        elif threshold == '>':
            inverted_threshold = '<='
        combination = test_conditions['condition_combination']
        inverted_combination = combination
        if combination == 'OR':
            inverted_combination = 'AND'
        elif combination == 'AND':
            inverted_combination = 'OR'

        return inverted_threshold, inverted_combination

    @staticmethod
    def construct_condition(extract_constraint, condition,
                            probability_threshold, gamma):
        """
        Create a string representing a comparison condition.

        Args:
            extract_constraint (str or list of str):
                A string, or list of strings, encoding iris constraints
                that will be used to extract the correct diagnostic cube
                (by name) from the input cube list and the correct threshold
                from that cube.
            condition (str):
                The condition statement (e.g. greater than, >).
            probability_threshold (float):
                The probability value to use in the comparison.
            gamma (float or None):
                The gamma factor to multiply one field by when performing
                a subtraction. This value will be None in the case that
                extract_constraint is not a list; it will not be used.
        Returns:
            string:
                The formatted condition statement,
                e.g.::

                  cubes.extract(Constraint(
                          name='probability_of_rainfall_rate_above_threshold',
                          coord_values={'threshold': 0.03})
                                )[0].data < 0.5)
        """
        if isinstance(extract_constraint, list):
            return ('(cubes.extract({})[0].data - cubes.extract({})[0].data * '
                    '{}) {} {}'.format(
                        extract_constraint[0], extract_constraint[1], gamma,
                        condition, probability_threshold))
        return 'cubes.extract({})[0].data {} {}'.format(
            extract_constraint, condition, probability_threshold)

    @staticmethod
    def format_condition_chain(conditions, condition_combination='AND'):
        """
        Chain individual condition statements together in a format that
        numpy.where can use to make a series of comparisons.

        Args:
            conditions (list):
                A list of conditions to be combined into a single comparison
                statement.
            condition_combination (str):
                The method by which multiple conditions should be combined,
                either AND or OR.
        Returns:
            string:
                A string formatted as a chain of conditions suitable for use in
                a numpy.where statement.
                e.g. (condition 1) & (condition 2)
        """
        if condition_combination == 'OR':
            return ('({}) | '*len(conditions)).format(*conditions).strip('| ')
        return ('({}) & '*len(conditions)).format(*conditions).strip('& ')

    def create_condition_chain(self, test_conditions):
        """
        A wrapper to call the construct_condition function for all the
        conditions specified in a single query.

        Args:
            test_conditions (dict):
                A query from the decision tree.
        Returns:
            condition_chain (list):
                A list of strings that describe the conditions comprising the
                query.
                e.g.::

                  [
                    "(cubes.extract(Constraint(
                          name='probability_of_rainfall_rate_above_threshold',
                          coord_values={'threshold': 0.03})
                     )[0].data < 0.5) |
                     (cubes.extract(Constraint(
                          name=
                          'probability_of_lwe_snowfall_rate_above_threshold',
                          coord_values={'threshold': 0.03})
                     )[0].data < 0.5)"
                  ]
        """
        conditions = []
        loop = 0
        for diagnostic, p_threshold, d_threshold in zip(
                test_conditions['diagnostic_fields'],
                test_conditions['probability_thresholds'],
                test_conditions['diagnostic_thresholds']):

            gamma = test_conditions.get('diagnostic_gamma')
            if gamma is not None:
                gamma = gamma[loop]
            loop += 1

            extract_constraint = self.construct_extract_constraint(
                diagnostic, d_threshold, self.coord_named_threshold)
            conditions.append(
                WeatherSymbols.construct_condition(
                    extract_constraint, test_conditions['threshold_condition'],
                    p_threshold, gamma))
        condition_chain = WeatherSymbols.format_condition_chain(
            conditions,
            condition_combination=test_conditions['condition_combination'])
        return [condition_chain]

    def construct_extract_constraint(
            self, diagnostics, thresholds, coord_named_threshold):
        """
        Construct an iris constraint.

        Args:
            diagnostics (str or list of str):
                The names of the diagnostics to be extracted from the CubeList.
            thresholds (iris.AuxCoord or list of iris.AuxCoord):
                All thresholds within the given diagnostic cubes that are
                needed, including units.  Note these are NOT coords from the
                original cubes, just constructs to associate units with values.
            coord_named_threshold (bool):
                If true, use old naming convention for threshold coordinates
                (coord.long_name=threshold).  Otherwise extract threshold
                coordinate name from diagnostic name

        Returns:
            (str or list of str):
                String, or list of strings, encoding iris cube constraints.
        """
        def _constraint_string(diagnostic, threshold_name, threshold_val):
            """
            Return iris constraint as a string for deferred creation of the
            lambda functions.
            Args:
                diagnostic (str):
                    Name of diagnostic
                threshold_name (str):
                    Name of threshold coordinate on input cubes
                threshold_val (float):
                    Value of threshold coordinate required
            Returns: (str)
            """
            return ("iris.Constraint(name='{diagnostic}', {threshold_name}="
                    "lambda cell: {threshold_val} * {float_min} < cell < "
                    "{threshold_val} * {float_max})".format(
                        diagnostic=diagnostic, threshold_name=threshold_name,
                        threshold_val=threshold_val,
                        float_min=(1. - WeatherSymbols().float_tolerance),
                        float_max=(1. + WeatherSymbols().float_tolerance)))

        # if input is list, loop over and return a list of strings
        if isinstance(diagnostics, list):
            constraints = []
            for diagnostic, threshold in zip(diagnostics, thresholds):
                if coord_named_threshold:
                    threshold_coord_name = "threshold"
                elif diagnostic in self.threshold_coord_names:
                    threshold_coord_name = (
                        self.threshold_coord_names[diagnostic])
                else:
                    threshold_coord_name = extract_diagnostic_name(diagnostic)
                threshold_val = threshold.points.item()
                constraints.append(
                    _constraint_string(
                        diagnostic, threshold_coord_name, threshold_val))
            return constraints

        # otherwise, return a string
        if coord_named_threshold:
            threshold_coord_name = "threshold"
        elif diagnostics in self.threshold_coord_names:
            threshold_coord_name = self.threshold_coord_names[diagnostics]
        else:
            threshold_coord_name = extract_diagnostic_name(diagnostics)
        threshold_val = thresholds.points.item()
        constraint = _constraint_string(
            diagnostics, threshold_coord_name, threshold_val)
        return constraint

    @staticmethod
    def find_all_routes(graph, start, end, route=None):
        """
        Function to trace all routes through the decision tree.

        Args:
            graph (dict):
                A dictionary that describes each node in the tree,
                e.g. {<node_name>: [<succeed_name>, <fail_name>]}
            start (str):
                The node name of the tree root (currently always
                heavy_precipitation).
            end (int):
                The weather symbol code to which we are tracing all routes.
            route (list):
                A list of node names found so far.

        Returns:
            routes (list):
                A list of node names that defines the route from the tree root
                to the weather symbol leaf (end of chain).

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
                newroutes = WeatherSymbols.find_all_routes(graph, node, end,
                                                           route)
                routes.extend(newroutes)
        return routes

    @staticmethod
    def create_symbol_cube(cube):
        """
        Create an empty weather_symbol cube initialised with -1 across the
        grid.

        Args:
            cube (iris.cube.Cube):
                An x-y slice of one of the input cubes, used to define the
                size of the weather symbol grid.
        Returns:
            symbols (iris.cube.Cube):
                A cube full of -1 values, with suitable metadata to describe
                the weather symbols that will fill it.
        """
        threshold_coord = find_threshold_coordinate(cube)
        cube_format = next(cube.slices_over([threshold_coord]))
        symbols = cube_format.copy(data=np.full(cube_format.data.shape, -1,
                                                dtype=np.int))

        symbols.remove_coord(threshold_coord)
        symbols = add_wxcode_metadata(symbols)

        return symbols

    def process(self, cubes):
        """Apply the decision tree to the input cubes to produce weather
        symbol output.

        Args:
            cubes (iris.cube.CubeList):
                A cubelist containing the diagnostics required for the
                weather symbols decision tree, these at co-incident times.

        Returns:
            symbols (iris.cube.Cube):
                A cube of weather symbols.
        """
        # Check input cubes contain required data
        self.check_input_cubes(cubes)

        # Construct graph nodes dictionary
        graph = {key: [self.queries[key]['succeed'], self.queries[key]['fail']]
                 for key in self.queries.keys()}

        # Search through tree for all leaves (weather code end points)
        defined_symbols = []
        for item in self.queries.values():
            for value in item.values():
                if isinstance(value, int):
                    defined_symbols.append(value)

        # Create symbol cube
        symbols = self.create_symbol_cube(cubes[0])

        # Loop over possible symbols
        for symbol_code in defined_symbols:
            # In current decision tree
            # start node is heavy_precipitation
            routes = self.find_all_routes(graph, 'heavy_precipitation',
                                          symbol_code)

            # Loop over possible routes from root to leaf
            for route in routes:
                conditions = []
                for i_node in range(len(route)-1):
                    current_node = route[i_node]
                    current = copy.copy(self.queries[current_node])
                    try:
                        next_node = route[i_node+1]
                    except KeyError:
                        next_node = symbol_code

                    if current['fail'] == next_node:
                        (current['threshold_condition'],
                         current['condition_combination']) = (
                             self.invert_condition(current))

                    conditions.extend(self.create_condition_chain(current))

                test_chain = self.format_condition_chain(conditions)

                # Set grid locations to suitable weather symbol
                symbols.data[np.where(eval(test_chain))] = symbol_code
        # Update symbols for day or night.
        symbols = update_daynight(symbols)
        return symbols
