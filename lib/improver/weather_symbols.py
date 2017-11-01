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
from cf_units import Unit
from improver.spotdata.extract_data import ExtractData


class WeatherSymbols(object):

    """Define a decision tree for determining weather symbols based upon the
    input diagnostics. Use this decision tree to allocate a weather symbol to
    each point.
    """

    def __init__(self):
        """
        """
        self.queries = self._define_decision_tree()

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return '<WeatherSymbols>'

    def _define_decision_tree(self):
        """
        Define queries that comprise the weather symbol decision tree.

        Each queries contains the following elements:
            * follows: The query from which this query follows on.
            * succeed: The next query to call if the diagnostic being queried
                  satisfies the current query.
            * fail: The next query to call if the diagnostic being queried
                  does not satisfy the current query.
            * probability_thresholds: The probability thresholds that the query
                  requires.
            * threshold_condition: The condition the diagnostic must satisfy
                  relative to the probability threshold (e.g. greater than, >,
                  the probability threshold).
            * diagnostics_fields: The diagnostics which are being used in the
                  query.
            * diagnostic_thresholds: The thresholding that is expected to have
                  been applied to the input data; this is used to test the
                  input.
            * diagnostic_condition: The condition that is expected to have been
                  applied to the input data; this is used to test the input.

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
                'fail': 'light_rain_or_snow_continuous',
                'probability_thresholds': [0., 0.],
                'threshold_condition': '>=',
                'condition_combination': 'AND',
                'diagnostic_fields': [['probability_of_lwe_snowfall_rate',
                                       'probability_of_rainfall_rate'],
                                      ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate']],
                'diagnostic_gamma': [0.7, 1.0],
                'diagnostic_thresholds': [[1., 1.],[1., 1.]],
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
                'diagnostic_thresholds': [[1., 1.],[1., 1.]],
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
                'diagnostic_thresholds': [[0.1, 0.1],[0.1, 0.1]],
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
                'diagnostic_thresholds': [[0.1, 0.1],[0.1, 0.1]],
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

###################### NOT COMPLETE #############################


            'any_precipitation': {
#                'succeed': 'precipitation_in_vicinity',
                'succeed': 100,
                'fail': 'mist_conditions',
                'probability_thresholds': [0.05, 0.05],
                'threshold_condition': '>=',
                'condition_combination': 'OR',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate'],
                'diagnostic_thresholds': [0.03, 0.03],
                'diagnostic_condition': 'above'},

            'precipitation_in_vicinity': {
                'succeed': 'precipitation_in_vicinity',
                'fail': 'mist_conditions',
                'probability_thresholds': [0.05, 0.05],
                'threshold_condition': '>=',
                'condition_combination': 'OR',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate'],
                'diagnostic_thresholds': [0.03, 0.03],
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
            diagnostic (string):
                The name of the diagnostic to be used in the comparison.
            condition (string):
                The condition statement (e.g. greater than, >).
            probability_threshold (float):
                The probability value to use in the comparison.
        Returns:
            string:
                The formatted condition statement,
                e.g. 'rainfall_rate.data < 0.5'
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
        Returns:
            string:
                A string formatted as a chain of conditions suitable for use in
                a numpy.where statement.
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
                A single query from the decision tree.
        Returns:
            conditions (list):
                A list of strings that describe the conditions comprising the
                query.
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
    def find_all_paths(graph, start, end, path=[]):
        """
        Function to trace all routes through the decision graph.
        Taken from: https://www.python.org/doc/essays/graphs/

        Copyrighted, needs rejigging.
        """
        path = path + [start]
        if start == end:
            return [path]
        if not graph.has_key(start):
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = WeatherSymbols.find_all_paths(graph, node, end,
                                                         path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths


    def process(self, cubes):
        """Apply the decision tree to the input cubes to produce weather
        symbol output.

        Args:
            cubes (iris.cube.CubeList):
                A cubelist containing the diagnostics required for the
                weather symbols decision tree.

        Returns:
            cube : iris.cube.Cube
                A cube of weather symbols that is in the same format as the
                input cubes.

        Raises:
            Various errors if input data is incorrect.

        """
        graph = {key: [self.queries[key]['succeed'], self.queries[key]['fail']]
                 for key in self.queries.keys()}
        defined_symbols = []
        for item in self.queries.itervalues():
            for value in item.itervalues():
                if isinstance(value, int):
                    defined_symbols.append(value)


        symbols_map = np.zeros(cubes[0][0].data.shape)

        for symbol_code in defined_symbols:
            print 'symbol code', symbol_code
            routes = self.find_all_paths(graph, 'significant_precipitation',
                                        symbol_code)

            # Loop over possible routes from root to leaf.
            for route in routes:
                print ('--> {}' * len(route)).format(*[node for node in route])
                conditions = []
                for i_node in range(len(route)-1):
                    current_node = route[i_node]
                    current = copy.copy(self.queries[current_node])
                    try:
                        next_node = route[i_node+1]
                        next = copy.copy(self.queries[next_node])
                    except:
                        next_node = symbol_code

#                    print '{} --> {}'.format(current_node, next_node)
                    if (current['fail'] == next_node):
                        current['threshold_condition'] = self.invert_condition(next)
                    conditions.extend(self.create_condition_chain(current))

                test_chain = self.format_condition_chain(conditions)
#                print test_chain
                symbols_map[np.where(eval(test_chain))] = symbol_code

        return symbols_map

