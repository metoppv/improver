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
                'follows': None,
                'succeed': 'heavy_precipitation',
                'fail': 'any_precipitation',
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate'],
                'diagnostic_thresholds': [0.03, 0.03],
                'diagnostic_condition': 'above'},
            'heavy_precipitation': {
                'follows': 'significant_precipitation',
                'succeed': 'cloud_cover',
                'fail': 'light_precipitation',
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate'],
                'diagnostic_thresholds': [1.0, 1.0],
                'diagnostic_condition': 'above'},
            'light_precipitation': {
                'follows': 'heavy_precipitation',
                'succeed': 'cloud_6.5',
#                'fail': 'drizzle',
                'fail': 11,
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate'],
                'diagnostic_thresholds': [0.1, 0.1],
                'diagnostic_condition': 'above'},
            'cloud_6.5': {
                'follows': 'light_precipitation',
#                'succeed': 'light_sleet_continuous',
                'succeed': 18,
#                'fail': 'light_sleet_shower',
                'fail': 17,
                'probability_thresholds': [0.5],
                'threshold_condition': '>=',
                'diagnostic_fields': ['probability_of_cloud_area_fraction'],
                'diagnostic_thresholds': [0.8125],
                'diagnostic_condition': 'above'},
            'any_precipitation': {
                'follows': 'significant_precipitation',
                'succeed': 'precipitation_in_vicinity',
                'fail': 'mist_conditions',
                'probability_thresholds': [0.05, 0.05],
                'threshold_condition': '>=',
                'diagnostic_fields': ['probability_of_rainfall_rate',
                                      'probability_of_lwe_snowfall_rate'],
                'diagnostic_thresholds': [0.03, 0.03],
                'diagnostic_condition': 'above'},
            'mist_conditions': {
                'follows': 'any_precipitation',
                'succeed': 'fog_conditions',
                'fail': 1,
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'diagnostic_fields': ['probability_of_visibility_in_air'],
                'diagnostic_thresholds': [5000.],
                'diagnostic_condition': 'below'},
            'fog_conditions': {
                'follows': 'mist_conditions',
                'succeed': 6,
                'fail': 5,
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'diagnostic_fields': ['probability_of_visibility_in_air'],
                'diagnostic_thresholds': [1000.],
                'diagnostic_condition': 'below'}
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
                            probability_threshold):
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
        return 'cubes.extract({})[0].data {} {}'.format(
            extract_constraint, condition, probability_threshold)

    @staticmethod
    def format_condition_chain(conditions):
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
        for diagnostic, p_threshold, d_threshold in zip(
                test_conditions['diagnostic_fields'],
                test_conditions['probability_thresholds'],
                test_conditions['diagnostic_thresholds']):
            extract_constraint = WeatherSymbols.construct_extract_constraint(
                diagnostic, d_threshold)
            conditions.append(
                WeatherSymbols.construct_condition(
                    extract_constraint, test_conditions['threshold_condition'],
                    p_threshold))
        return conditions

    @staticmethod
    def construct_extract_constraint(diagnostic, threshold):
        return iris.Constraint(
            name=diagnostic, coord_values={'threshold': threshold})
#    con = (iris.Constraint(
#            name='probability_of_rainfall_rate',
#            coord_values={'threshold': 0.03},
#            cube_func=lambda cube:
#                cube.attributes['relative_to_threshold'] == 'above'))

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
        symbols_map = np.zeros(cubes[0][0].data.shape)
        print symbols_map.shape
        leaves = [key for key in self.queries.keys() if
                  type(self.queries[key]['succeed']) == int]
        leaves.extend([key for key in self.queries.keys() if
                       type(self.queries[key]['fail']) == int])
        leaves.sort()

        previous_leaf = None

        for leaf in leaves:
            conditions = []
            current_node = leaf
            next_node = self.queries[current_node]['follows']
            if (isinstance(self.queries[current_node]['succeed'], int) and
                    leaf != previous_leaf):
                symbol_code = self.queries[current_node]['succeed']
            elif isinstance(self.queries[current_node]['fail'], int):
                symbol_code = self.queries[current_node]['fail']
            print 'symbol code', symbol_code
            while next_node is not None:
                current = copy.copy(self.queries[current_node])
                next_node = current['follows']
                if next_node == None:
                    break
                next = copy.copy(self.queries[next_node])
                # Symbol level
                if (current_node == leaf):
                    print '{} --> {}'.format(symbol_code, current_node)
                    if symbol_code == current['fail']:
#                        print 'INVERT'
                        current['threshold_condition'] = self.invert_condition(
                            current)
                    conditions.extend(self.create_condition_chain(current))
                # Normal process for non-symbol level
                print '{} --> {}'.format(current_node, next_node)
                if (next_node is not None and
                        next['fail'] == current_node):
#                    print 'INVERT'
                    next['threshold_condition'] = self.invert_condition(next)
                conditions.extend(self.create_condition_chain(next))
                current_node = next_node
            test_chain = self.format_condition_chain(conditions)
            print test_chain
            symbols_map[np.where(eval(test_chain))] = symbol_code
            previous_leaf = leaf

        return symbols_map

