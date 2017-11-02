# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Module containing thresholding classes."""


import numpy as np
import copy
import iris
from iris import Constraint
from improver.wxcode.wxcode_utilities import add_wxcode_metadata


class WeatherSymbols(object):
    """
    Definition and implementation of a weather symbol decision tree. This
    plugin uses a variety of diagnostic inputs and the decision tree logic
    to determine the most representative weather symbol for each site
    defined in the input cubes.
    """

    def __init__(self):
        """
        Define a decision tree for determining weather symbols based upon
        the input diagnostics. Use this decision tree to allocate a weather
        symbol to each point.
        """
        self.queries = self._define_decision_tree()

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return '<WeatherSymbols>'

    @staticmethod
    def _define_decision_tree():
        """
        Define queries that comprise the weather symbol decision tree.

        Each queries contains the following elements:
            * succeed: The next query to call if the diagnostic being queried
                  satisfies the current query.
            * fail: The next query to call if the diagnostic being queried
                  does not satisfy the current query.
            * probability_thresholds: A list of probability thresholds that the
                  query requires. One entry is provided for each diagnostic
                  field being tested.
            * threshold_condition: The condition the diagnostic must satisfy
                  relative to the probability threshold (e.g. greater than (>)
                  the probability threshold).
            * condition_combination: The way (AND, OR) in which multiple
                  conditions should be combined;
                  e.g. rainfall > 0.5 AND snowfall > 0.5
            * diagnostics_fields: The diagnostics which are being used in the
                  query. If this is a list of lists, the two fields in a given
                  list are subtracted (1st - (2nd * gamma)) and then compared
                  with the probability threshold.
            * diagnostic_gamma (NOT UNIVERSAL): This is the gamma factor that
                  is used when comparing two fields directly, rather than
                  comparing a single field to a probability threshold.
                  e.g. gamma * P(SnowfallRate) < P(RainfallRate).
            * diagnostic_thresholds: The thresholding that is expected to have
                  been applied to the input data; this is used to extract the
                  approproate data from the input cubes.
            * diagnostic_condition: The condition that is expected to have been
                  applied to the input data; this can be used to ensure the
                  thresholding is as expected.

        Returns:
            queries (dict):
                A dictionary containing the queries that comprise the decision
                tree.
        """
        queries = {
            'significant_precipitation': {
                'succeed': 'heavy_precipitation',
                'fail': 'any_precipitation',
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'condition_combination': 'OR',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate'],
                'diagnostic_thresholds': [0.03, 0.03],
                'diagnostic_condition': 'above'},

            'heavy_precipitation': {
                'succeed': 'heavy_precipitation_cloud',
                'fail': 'light_precipitation',
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'condition_combination': 'OR',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate'],
                'diagnostic_thresholds': [1.0, 1.0],
                'diagnostic_condition': 'above'},

            'heavy_precipitation_cloud': {
                'succeed': 'heavy_sleet_continuous',
                'fail': 'heavy_sleet_shower',
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': ['probability_of_cloud_area_fraction'],
                'diagnostic_thresholds': [0.8125],
                'diagnostic_condition': 'above'},

            'heavy_sleet_continuous': {
                'succeed': 18,
                'fail': 'heavy_rain_or_snow_continuous',
                'probability_thresholds': [0., 0.],
                'threshold_condition': '>=',
                'condition_combination': 'AND',
                'diagnostic_fields': [['probability_of_lwe_snowfall_rate',
                                       'probability_of_rainfall_rate'],
                                      ['probability_of_rainfall_rate',
                                       'probability_of_lwe_snowfall_rate']],
                'diagnostic_gamma': [0.7, 1.0],
                'diagnostic_thresholds': [[1., 1.], [1., 1.]],
                'diagnostic_condition': 'above'},

            'heavy_sleet_shower': {
                'succeed': 17,
                'fail': 'heavy_rain_or_snow_shower',
                'probability_thresholds': [0., 0.],
                'threshold_condition': '>=',
                'condition_combination': 'AND',
                'diagnostic_fields': [['probability_of_lwe_snowfall_rate',
                                       'probability_of_rainfall_rate'],
                                      ['probability_of_rainfall_rate',
                                       'probability_of_lwe_snowfall_rate']],
                'diagnostic_gamma': [0.7, 1.0],
                'diagnostic_thresholds': [[1., 1.], [1., 1.]],
                'diagnostic_condition': 'above'},

            'heavy_rain_or_snow_continuous': {
                'succeed': 27,
                'fail': 15,
                'probability_thresholds': [0.],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': [['probability_of_lwe_snowfall_rate',
                                       'probability_of_rainfall_rate']],
                'diagnostic_gamma': [1.],
                'diagnostic_thresholds': [[1., 1.]],
                'diagnostic_condition': 'above'},

            'heavy_rain_or_snow_shower': {
                'succeed': 26,
                'fail': 14,
                'probability_thresholds': [0.],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': [['probability_of_lwe_snowfall_rate',
                                       'probability_of_rainfall_rate']],
                'diagnostic_gamma': [1.],
                'diagnostic_thresholds': [[1., 1.]],
                'diagnostic_condition': 'above'},

            'light_precipitation': {
                'succeed': 'light_precipitation_cloud',
                'fail': 'drizzle_mist',
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'condition_combination': 'OR',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate'],
                'diagnostic_thresholds': [0.1, 0.1],
                'diagnostic_condition': 'above'},

            'light_precipitation_cloud': {
                'succeed': 'light_sleet_continuous',
                'fail': 'light_sleet_shower',
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': ['probability_of_cloud_area_fraction'],
                'diagnostic_thresholds': [0.8125],
                'diagnostic_condition': 'above'},

            'light_sleet_continuous': {
                'succeed': 18,
                'fail': 'light_rain_or_snow_continuous',
                'probability_thresholds': [0., 0.],
                'threshold_condition': '>=',
                'condition_combination': 'AND',
                'diagnostic_fields': [['probability_of_lwe_snowfall_rate',
                                       'probability_of_rainfall_rate'],
                                      ['probability_of_rainfall_rate',
                                       'probability_of_lwe_snowfall_rate']],
                'diagnostic_gamma': [0.7, 1.0],
                'diagnostic_thresholds': [[0.1, 0.1], [0.1, 0.1]],
                'diagnostic_condition': 'above'},

            'light_rain_or_snow_continuous': {
                'succeed': 24,
                'fail': 12,
                'probability_thresholds': [0.],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': [['probability_of_lwe_snowfall_rate',
                                       'probability_of_rainfall_rate']],
                'diagnostic_gamma': [1.],
                'diagnostic_thresholds': [[0.1, 0.1]],
                'diagnostic_condition': 'above'},

            'light_sleet_shower': {
                'succeed': 17,
                'fail': 'light_rain_or_snow_shower',
                'probability_thresholds': [0., 0.],
                'threshold_condition': '>=',
                'condition_combination': 'AND',
                'diagnostic_fields': [['probability_of_lwe_snowfall_rate',
                                       'probability_of_rainfall_rate'],
                                      ['probability_of_rainfall_rate',
                                       'probability_of_lwe_snowfall_rate']],
                'diagnostic_gamma': [0.7, 1.0],
                'diagnostic_thresholds': [[0.1, 0.1], [0.1, 0.1]],
                'diagnostic_condition': 'above'},

            'light_rain_or_snow_shower': {
                'succeed': 23,
                'fail': 10,
                'probability_thresholds': [0.],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': [['probability_of_lwe_snowfall_rate',
                                       'probability_of_rainfall_rate']],
                'diagnostic_gamma': [1.],
                'diagnostic_thresholds': [[0.1, 0.1]],
                'diagnostic_condition': 'above'},

            'drizzle_mist': {
                'succeed': 11,
                'fail': 'no_precipitation_cloud',
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'condition_combination': 'OR',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_visibility_in_air'],
                'diagnostic_thresholds': [0.03, 5000.],
                'diagnostic_condition': 'above'},

            'drizzle_cloud': {
                'succeed': 11,
                'fail': 'no_precipitation_cloud',
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'condition_combination': 'OR',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      ('probability_of_cloud_area_fraction_'
                                       'assuming_only_consider_surface_to_1000'
                                       '_feet_asl')],
                'diagnostic_thresholds': [0.03, 0.85],
                'diagnostic_condition': 'above'},

            'no_precipitation_cloud': {
                'succeed': 'overcast_cloud',
                'fail': 'partly_cloudy',
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': ['probability_of_cloud_area_fraction'],
                'diagnostic_thresholds': [0.8125],
                'diagnostic_condition': 'above'},

            'overcast_cloud': {
                'succeed': 8,
                'fail': 7,
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': [('probability_of_cloud_area_fraction_'
                                       'assuming_only_consider_surface_to_1000'
                                       '_feet_asl')],
                'diagnostic_thresholds': [0.85],
                'diagnostic_condition': 'above'},

            'partly_cloudy': {
                'succeed': 3,
                'fail': 1,
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': ['probability_of_cloud_area_fraction'],
                'diagnostic_thresholds': [0.1875],
                'diagnostic_condition': 'above'},

            'any_precipitation': {
                'succeed': 'precipitation_in_vicinity',
                'fail': 'mist_conditions',
                'probability_thresholds': [0.05, 0.05],
                'threshold_condition': '>=',
                'condition_combination': 'OR',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate'],
                'diagnostic_thresholds': [0.03, 0.03],
                'diagnostic_condition': 'above'},

            'precipitation_in_vicinity': {
                'succeed': 'sleet_in_vicinity',
                'fail': 'mist_conditions',
                'probability_thresholds': [0.05, 0.05],
                'threshold_condition': '>=',
                'condition_combination': 'OR',
                'diagnostic_fields': [
                    'probability_of_rainfall_rate_in_vicinity',
                    'probability_of_lwe_snowfall_rate_in_vicinity'],
                'diagnostic_thresholds': [0.1, 0.1],
                'diagnostic_condition': 'above'},

            'sleet_in_vicinity': {
                'succeed': 17,
                'fail': 'rain_or_snow_in_vicinity',
                'probability_thresholds': [0., 0.],
                'threshold_condition': '>=',
                'condition_combination': 'AND',
                'diagnostic_fields': [
                    ['probability_of_lwe_snowfall_rate_in_vicinity',
                     'probability_of_rainfall_rate_in_vicinity'],
                    ['probability_of_rainfall_rate_in_vicinity',
                     'probability_of_lwe_snowfall_rate_in_vicinity']],
                'diagnostic_gamma': [0.7, 1.0],
                'diagnostic_thresholds': [[0.1, 0.1], [0.1, 0.1]],
                'diagnostic_condition': 'above'},

            'rain_or_snow_in_vicinity': {
                'succeed': 'snow_in_vicinity_cloud',
                'fail': 'rain_in_vicinity_cloud',
                'probability_thresholds': [0.],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': [
                    ['probability_of_lwe_snowfall_rate_in_vicinity',
                     'probability_of_rainfall_rate_in_vicinity']],
                'diagnostic_gamma': [1.],
                'diagnostic_thresholds': [[0.1, 0.1]],
                'diagnostic_condition': 'above'},

            'snow_in_vicinity_cloud': {
                'succeed': 'heavy_continuous_snow_in_vicinity',
                'fail': 'heavy_snow_shower_in_vicinity',
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': ['probability_of_cloud_area_fraction'],
                'diagnostic_thresholds': [0.8125],
                'diagnostic_condition': 'above'},

            'heavy_snow_continuous_in_vicinity': {
                'succeed': 27,
                'fail': 24,
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': 'AND',
                'diagnostic_fields': [
                    'probability_of_lwe_snowfall_rate_in_vicinity'],
                'diagnostic_thresholds': [1.0],
                'diagnostic_condition': 'above'},

            'heavy_snow_shower_in_vicinity': {
                'succeed': 26,
                'fail': 23,
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': 'AND',
                'diagnostic_fields': [
                    'probability_of_lwe_snowfall_rate_in_vicinity'],
                'diagnostic_thresholds': [1.0],
                'diagnostic_condition': 'above'},

            'rain_in_vicinity_cloud': {
                'succeed': 'heavy_continuous_rain_in_vicinity',
                'fail': 'heavy_rain_shower_in_vicinity',
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': ['probability_of_cloud_area_fraction'],
                'diagnostic_thresholds': [0.8125],
                'diagnostic_condition': 'above'},

            'heavy_rain_continuous_in_vicinity': {
                'succeed': 15,
                'fail': 12,
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': 'AND',
                'diagnostic_fields': [
                    'probability_of_rainfall_rate_in_vicinity'],
                'diagnostic_thresholds': [1.0],
                'diagnostic_condition': 'above'},

            'heavy_rain_shower_in_vicinity': {
                'succeed': 14,
                'fail': 10,
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': 'AND',
                'diagnostic_fields': [
                    'probability_of_rainfall_rate_in_vicinity'],
                'diagnostic_thresholds': [1.0],
                'diagnostic_condition': 'above'},

            'mist_conditions': {
                'succeed': 'fog_conditions',
                'fail': 'no_precipitation_cloud',
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': ['probability_of_visibility_in_air'],
                'diagnostic_thresholds': [5000.],
                'diagnostic_condition': 'below'},

            'fog_conditions': {
                'succeed': 6,
                'fail': 5,
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'condition_combination': '',
                'diagnostic_fields': ['probability_of_visibility_in_air'],
                'diagnostic_thresholds': [1000.],
                'diagnostic_condition': 'below'},


            }
        return queries

    @staticmethod
    def invert_condition(test_conditions):
        """
        Invert a comparison condition to select the negative case.

        Args:
            test_conditions (dict):
                A single query from the decision tree.
        Returns:
            string:
                A string representing the inverted comparison.
        """
        condition = test_conditions['threshold_condition']
        if condition == '>=':
            return '<'
        if condition == '<=':
            return '>'
        if condition == '<':
            return '>='
        if condition == '>':
            return '<='

    @staticmethod
    def construct_condition(extract_constraint, condition,
                            probability_threshold, gamma):
        """
        Create a string representing a comparison condition.

        Args:
            extract_constraint (iris.Constraint or list of Constraints):
                An iris constraint that will be used to extract the correct
                diagnostic cube (by name) from the input cube list and the
                correct threshold from that cube.
            condition (string):
                The condition statement (e.g. greater than, >).
            probability_threshold (float):
                The probability value to use in the comparison.
            gamma (float or unset):
                The gamma factor to multiply one field by when performing
                a subtraction. This value will be unset in the case that
                extract_constraint is not a list; it will not be used.
        Returns:
            string:
                The formatted condition statement,
                e.g. cubes.extract(Constraint(
                         name='probability_of_rainfall_rate',
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
            condition_combination (string):
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

    @staticmethod
    def create_condition_chain(test_conditions):
        """
        A wrapper to call the construct_condition function for all the
        conditions specfied in a single query.

        Args:
            test_conditions (dict):
                A query from the decision tree.
        Returns:
            condition_chain (list):
                A list of strings that describe the conditions comprising the
                query.
                e.g.
                [
                  "(cubes.extract(Constraint(
                        name='probability_of_rainfall_rate',
                        coord_values={'threshold': 0.03})
                   )[0].data < 0.5) |
                   (cubes.extract(Constraint(
                        name='probability_of_lwe_snowfall_rate',
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

            extract_constraint = WeatherSymbols.construct_extract_constraint(
                diagnostic, d_threshold)
            conditions.append(
                WeatherSymbols.construct_condition(
                    extract_constraint, test_conditions['threshold_condition'],
                    p_threshold, gamma))
        condition_chain = WeatherSymbols.format_condition_chain(
            conditions,
            condition_combination=test_conditions['condition_combination'])
        return [condition_chain]

    @staticmethod
    def construct_extract_constraint(diagnostics, thresholds):
        """
        Construct an iris constraint.

        Args:
            diagnostics (string or list of strings):
                The names of the diagnostics to be extracted from the CubeList.
            thresholds (float or list of floats):
                A thresholds within the given diagnostic cubes that are needed.
        Returns:
            iris.Constraint or list of iris.Constraints:
                The constructed iris constraints.
        """

        if isinstance(diagnostics, list):
            constraints = []
            for diagnostic, threshold in zip(diagnostics, thresholds):
                constraints.append(iris.Constraint(
                    name=diagnostic,
                    coord_values={'threshold': threshold}))
            return constraints
        return iris.Constraint(
            name=diagnostics, coord_values={'threshold': thresholds})

    @staticmethod
    def find_all_routes(graph, start, end, route=[]):
        """
        Function to trace all routes through the decision tree.

        Args:
            graph (dict):
                A dictionary that describes each node in the tree,
                e.g. {<node_name>: [<succeed_name>, <fail_name>]}
            start (string):
                The node name of the tree root (currently always
                significant_precipitation).
            end (int):
                The weather symbol code to which we are tracing all routes.

        Returns:
            routes (list):
                A list of node names that defines the route from the tree root
                to the weather symbol leaf (end of chain).

        References:
            Method based upon Python Patterns - Implementing Graphs essay
            https://www.python.org/doc/essays/graphs/
        """
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
                for newroute in newroutes:
                    routes.append(newroute)
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
        cube_format = next(cube.slices([cube.coord(axis='y'),
                                        cube.coord(axis='x')]))
        symbols = cube_format.copy(data=np.full(cube_format.data.shape, -1,
                                                dtype=np.int))
        symbols.remove_coord('threshold')
        symbols.attributes.pop('relative_to_threshold')
        symbols = add_wxcode_metadata(symbols)
        return symbols

    def process(self, cubes):
        """Apply the decision tree to the input cubes to produce weather
        symbol output.

        Args:
            cubes (iris.cube.CubeList):
                A cubelist containing the diagnostics required for the
                weather symbols decision tree, these at conincident times.

        Returns:
            symbols (iris.cube.Cube):
                A cube of weather symbols.
        """
        # Construct graph nodes dictionary
        graph = {key: [self.queries[key]['succeed'], self.queries[key]['fail']]
                 for key in self.queries.keys()}

        # Search through tree for all leaves (weather code end points)
        defined_symbols = []
        for item in self.queries.itervalues():
            for value in item.itervalues():
                if isinstance(value, int):
                    defined_symbols.append(value)

        # Create symbol cube
        symbols = self.create_symbol_cube(cubes[0])

        # Loop over possible symbols
        for symbol_code in defined_symbols:
            routes = self.find_all_routes(graph, 'significant_precipitation',
                                          symbol_code)

            # Loop over possible routes from root to leaf
            for route in routes:
                # print ('--> {}' * len(route)).format(
                #    *[node for node in route])
                conditions = []
                for i_node in range(len(route)-1):
                    current_node = route[i_node]
                    current = copy.copy(self.queries[current_node])
                    try:
                        next_node = route[i_node+1]
                        next_data = copy.copy(self.queries[next_node])
                    except KeyError:
                        next_node = symbol_code

                    if current['fail'] == next_node:
                        current['threshold_condition'] = self.invert_condition(
                            next_data)
                    conditions.extend(self.create_condition_chain(current))

                test_chain = self.format_condition_chain(conditions)
#                print test_chain

                # Set grid locations to suitable weather symbol
                symbols.data[np.where(eval(test_chain))] = symbol_code

        return symbols
