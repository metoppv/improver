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
from datetime import datetime as dt
import numpy as np
from cf_units import Unit
import iris
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.metadata.probabilistic import find_threshold_coordinate
from improver.wxcode.utilities import WX_DICT
from improver.wxcode.weather_symbols import WeatherSymbols

from ...set_up_test_cubes import set_up_probability_cube


class Test_WXCode(IrisTest):

    """Test class for the WX code tests, setting up inputs."""

    def setUp(self):
        """Set up cubes and constraints required for Weather Symbols."""

        time = dt(2017, 10, 10, 12, 0)
        frt = dt(2017, 10, 10, 12, 0)

        thresholds = np.array([8.33333333e-09, 2.77777778e-08, 2.77777778e-07],
                              dtype=np.float32)
        data_snow = np.zeros((3, 3, 3), dtype=np.float32)
        snowfall_rate = set_up_probability_cube(
            data_snow, thresholds, variable_name='lwe_snowfall_rate',
            threshold_units='m s-1',
            time=time, frt=frt)

        data_sleet = np.zeros((3, 3, 3), dtype=np.float32)

        sleetfall_rate = set_up_probability_cube(
            data_sleet, thresholds, variable_name='lwe_sleetfall_rate',
            threshold_units='m s-1',
            time=time, frt=frt)

        data_rain = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0], dtype=np.float32).reshape(
                                  (3, 3, 3))

        rainfall_rate = set_up_probability_cube(
            data_rain, thresholds, variable_name='rainfall_rate',
            threshold_units='m s-1',
            time=time, frt=frt)

        # pylint: disable=no-member
        data_precip = np.maximum.reduce([data_snow, data_sleet, data_rain])

        precip_rate = set_up_probability_cube(
            data_precip, thresholds, variable_name='lwe_precipitation_rate',
            threshold_units='m s-1',
            time=time, frt=frt)

        precip_vicinity = set_up_probability_cube(
            data_rain, thresholds,
            variable_name='lwe_precipitation_rate_in_vicinity',
            threshold_units='m s-1',
            time=time, frt=frt)

        thresholds = np.array([0.1875, 0.8125], dtype=np.float32)
        data_cloud = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                               0.0, 0.0, 1.0], dtype=np.float32).reshape(
                                   (2, 3, 3))

        cloud = set_up_probability_cube(
            data_cloud, thresholds, variable_name='cloud_area_fraction',
            threshold_units='1',
            time=time, frt=frt)

        thresholds = np.array([0.85], dtype=np.float32)
        data_cld_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                 0.0, 0.0, 0.0], dtype=np.float32).reshape(
                                     (1, 3, 3))
        cloud_low = set_up_probability_cube(
            data_cld_low, thresholds,
            variable_name='low_type_cloud_area_fraction', threshold_units='1',
            time=time, frt=frt)

        thresholds = np.array([1000.0, 5000.0], dtype=np.float32)
        data_vis = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0], dtype=np.float32).reshape(
                                 (2, 3, 3))
        visibility = set_up_probability_cube(
            data_vis, thresholds, variable_name='visibility_in_air',
            threshold_units='m', spp__relative_to_threshold='below',
            time=time, frt=frt)

        thresholds = np.array([0.0], dtype=np.float32)
        data_lightning = np.zeros((1, 3, 3), dtype=np.float32)
        data_lightning[0, 0, 0] = 0.25
        data_lightning[0, 0, 1] = 0.30

        lightning = set_up_probability_cube(
            data_lightning, thresholds,
            variable_name=('number_of_lightning_flashes_per_unit_area_in_'
                           'vicinity'),
            threshold_units='m-2',
            time=time, frt=frt)

        self.cubes = iris.cube.CubeList([
            snowfall_rate, sleetfall_rate, rainfall_rate, precip_vicinity,
            cloud, cloud_low, visibility, lightning, precip_rate])
        names = [cube.name() for cube in self.cubes]
        self.uk_no_lightning = [name for name in names
                                if 'lightning' not in name]
        self.gbl = [name for name in self.uk_no_lightning[:-1]
                    if 'vicinity' not in name and 'sleet' not in name]


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeatherSymbols())
        msg = '<WeatherSymbols tree=high_resolution start_node=lightning>'
        self.assertEqual(result, msg)

    def test_global(self):
        """Test that the __repr__ returns right string for global tree."""
        result = str(WeatherSymbols(wxtree='global'))
        msg = '<WeatherSymbols tree=global start_node=heavy_precipitation>'
        self.assertEqual(result, msg)


class Test_check_input_cubes(Test_WXCode):

    """Test the check_input_cubes method."""

    def test_basic(self):
        """Test check_input_cubes method raises no error if the data is OK"""
        plugin = WeatherSymbols()
        self.assertEqual(plugin.check_input_cubes(self.cubes), None)

    def test_no_lightning(self):
        """Test check_input_cubes raises no error if lightning missing"""
        plugin = WeatherSymbols()
        cubes = self.cubes.extract(self.uk_no_lightning)
        result = plugin.check_input_cubes(cubes)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        self.assertTrue('lightning' in result)

    def test_raises_error_missing_cubes(self):
        """Test check_input_cubes method raises error if data is missing"""
        plugin = WeatherSymbols()
        cubes = self.cubes.pop()
        msg = 'Weather Symbols input cubes are missing'
        with self.assertRaisesRegex(IOError, msg):
            plugin.check_input_cubes(cubes)

    def test_raises_error_missing_threshold(self):
        """Test check_input_cubes method raises error if data is missing"""
        plugin = WeatherSymbols()
        cubes = self.cubes
        cubes[0] = cubes[0][0]
        msg = 'Weather Symbols input cubes are missing'
        with self.assertRaisesRegex(IOError, msg):
            plugin.check_input_cubes(cubes)

    def test_incorrect_units(self):
        """Test that check_input_cubes method raises an error if the units are
        incompatible between the input cube and the decision tree."""
        plugin = WeatherSymbols()

        msg = "Unable to convert from"
        threshold_coord = find_threshold_coordinate(self.cubes[0])
        self.cubes[0].coord(threshold_coord).units = Unit('mm kg-1')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_input_cubes(self.cubes)

    def test_basic_global(self):
        """Test check_input_cubes method has no error if global data is OK"""
        plugin = WeatherSymbols(wxtree='global')
        cubes = self.cubes.extract(self.gbl)
        self.assertEqual(plugin.check_input_cubes(cubes), None)

    def test_raises_error_missing_cubes_global(self):
        """Test check_input_cubes method raises error if data is missing"""
        plugin = WeatherSymbols(wxtree='global')
        cubes = self.cubes.extract(self.gbl)[0:3]
        msg = 'Weather Symbols input cubes are missing'
        with self.assertRaisesRegex(IOError, msg):
            plugin.check_input_cubes(cubes)


class Test_invert_condition(IrisTest):

    """Test the invert condition method."""

    def test_basic(self):
        """Test that the invert_condition method returns a tuple of strings."""
        plugin = WeatherSymbols()
        tree = plugin.queries
        result = plugin.invert_condition(tree[list(tree.keys())[0]])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], str)

    def test_invert_thresholds_correctly(self):
        """Test invert_condition inverts thresholds correctly."""
        plugin = WeatherSymbols()
        node = {'threshold_condition': '>=', 'condition_combination': ''}
        possible_inputs = ['>=', '<=', '<', '>']
        inverse_outputs = ['<', '>', '>=', '<=']
        for i, val in enumerate(possible_inputs):
            node['threshold_condition'] = val
            result = plugin.invert_condition(node)
            self.assertEqual(result[0], inverse_outputs[i])

    def test_invert_combination_correctly(self):
        """Test invert_condition inverts combination correctly."""
        plugin = WeatherSymbols()
        node = {'threshold_condition': '>=', 'condition_combination': ''}
        possible_inputs = ['AND', 'OR', '']
        inverse_outputs = ['OR', 'AND', '']
        for i, val in enumerate(possible_inputs):
            node['condition_combination'] = val
            result = plugin.invert_condition(node)
            self.assertEqual(result[1], inverse_outputs[i])


class Test_construct_condition(IrisTest):

    """Test the construct condition method."""

    def test_basic(self):
        """Test that the construct_condition method returns a string."""
        plugin = WeatherSymbols()
        constraint_value = iris.Constraint(
            name='probability_of_rainfall_rate_above_threshold',
            coord_values={'threshold': 0.03})
        condition = '<'
        prob_threshold = 0.5
        gamma = None
        expected = ("cubes.extract(Constraint(name="
                    "'probability_of_rainfall_rate_above_threshold',"
                    " coord_values={'threshold': 0.03})"
                    ")[0].data < 0.5")
        result = plugin.construct_condition(constraint_value,
                                            condition,
                                            prob_threshold,
                                            gamma)
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_works_with_lists(self):
        """Test that the construct_condition method works with a list
        of Constraints. """
        plugin = WeatherSymbols()
        constraint_list = [
            iris.Constraint(
                name='probability_of_lwe_snowfall_rate_above_threshold',
                coord_values={'threshold': 0.03}),
            iris.Constraint(
                name='probability_of_rainfall_rate_above_threshold',
                coord_values={'threshold': 0.03})]
        condition = '<'
        prob_threshold = 0.5
        gamma = 0.7
        expected = ("(cubes.extract(Constraint(name="
                    "'probability_of_lwe_snowfall_rate_above_threshold', "
                    "coord_values={'threshold': 0.03}))[0].data - "
                    "cubes.extract(Constraint(name="
                    "'probability_of_rainfall_rate_above_threshold', "
                    "coord_values={'threshold': 0.03}))[0].data * 0.7) < 0.5")
        result = plugin.construct_condition(constraint_list,
                                            condition,
                                            prob_threshold,
                                            gamma)
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)


class Test_format_condition_chain(IrisTest):

    """Test the format_condition_chain method."""

    def test_basic(self):
        """Test that the format_condition_chain method returns a string."""
        plugin = WeatherSymbols()
        conditions = ['condition1', 'condition2']
        expected = '(condition1) & (condition2)'
        result = plugin.format_condition_chain(conditions)
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_works_with_or(self):
        """Test that the format_condition_chain method works with OR."""
        plugin = WeatherSymbols()
        conditions = ['condition1', 'condition2']
        expected = '(condition1) | (condition2)'
        result = plugin.format_condition_chain(conditions,
                                               condition_combination='OR')
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)


class Test_create_condition_chain(IrisTest):
    """Test the create_condition_chain method."""

    def setUp(self):
        """ Set up queries for testing"""
        self.dummy_queries = {
            'significant_precipitation': {
                'succeed': 'heavy_precipitation',
                'fail': 'any_precipitation',
                'probability_thresholds': [0.5, 0.5],
                'threshold_condition': '>=',
                'condition_combination': 'OR',
                'diagnostic_fields':
                    ['probability_of_rainfall_rate_above_threshold',
                     'probability_of_lwe_snowfall_rate_above_threshold'],
                'diagnostic_thresholds': [AuxCoord(0.03, units='mm hr-1'),
                                          AuxCoord(0.03, units='mm hr-1')],
                'diagnostic_conditions': ['above', 'above']}
        }

    def test_basic(self):
        """Test create_condition_chain returns a list of strings."""
        plugin = WeatherSymbols()
        test_condition = self.dummy_queries['significant_precipitation']
        result = plugin.create_condition_chain(test_condition)
        expected = ("(cubes.extract(iris.Constraint(name='probability_of_"
                    "rainfall_rate_above_threshold', rainfall_rate=lambda "
                    "cell: 0.03 * {t_min} < "
                    "cell < 0.03 * {t_max}))[0].data >= 0.5) | (cubes.extract"
                    "(iris.Constraint("
                    "name='probability_of_lwe_snowfall_rate_above_threshold',"
                    " lwe_snowfall_rate=lambda cell: 0.03 * {t_min} < cell < "
                    "0.03 * {t_max}))[0].data >= 0.5)".format(
                        t_min=(1. - WeatherSymbols().float_tolerance),
                        t_max=(1. + WeatherSymbols().float_tolerance)))
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], str)
        self.assertEqual(result[0], expected)

    def test_old_naming_convention(self):
        """Test create_condition_chain can return conditions using old
        threshold coordinate name"""
        plugin = WeatherSymbols()
        plugin.coord_named_threshold = True
        test_condition = self.dummy_queries['significant_precipitation']
        result = plugin.create_condition_chain(test_condition)
        expected = ("(cubes.extract(iris.Constraint(name='probability_of_"
                    "rainfall_rate_above_threshold', threshold=lambda "
                    "cell: 0.03 * {t_min} < "
                    "cell < 0.03 * {t_max}))[0].data >= 0.5) | (cubes.extract"
                    "(iris.Constraint("
                    "name='probability_of_lwe_snowfall_rate_above_threshold',"
                    " threshold=lambda cell: 0.03 * {t_min} < cell < "
                    "0.03 * {t_max}))[0].data >= 0.5)".format(
                        t_min=(1. - WeatherSymbols().float_tolerance),
                        t_max=(1. + WeatherSymbols().float_tolerance)))
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], str)
        self.assertEqual(result[0], expected)


class Test_construct_extract_constraint(IrisTest):

    """Test the construct_extract_constraint method ."""

    def test_basic(self):
        """Test construct_extract_constraint returns a iris.Constraint."""
        plugin = WeatherSymbols()
        diagnostic = 'probability_of_rainfall_rate_above_threshold'
        threshold = AuxCoord(0.03, units='mm hr-1')
        result = plugin.construct_extract_constraint(diagnostic,
                                                     threshold, False)
        expected = ("iris.Constraint("
                    "name='probability_of_rainfall_rate_above_threshold', "
                    "rainfall_rate=lambda cell: 0.03 * {t_min} < cell < "
                    "0.03 * {t_max})".format(
                        t_min=(1. - WeatherSymbols().float_tolerance),
                        t_max=(1. + WeatherSymbols().float_tolerance)))
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_old_naming_convention(self):
        """Test construct_extract_constraint can return a constraint with a
        "threshold" coordinate"""
        plugin = WeatherSymbols()
        diagnostic = 'probability_of_rainfall_rate_above_threshold'
        threshold = AuxCoord(0.03, units='mm hr-1')
        result = plugin.construct_extract_constraint(diagnostic,
                                                     threshold, True)
        expected = ("iris.Constraint("
                    "name='probability_of_rainfall_rate_above_threshold', "
                    "threshold=lambda cell: 0.03 * {t_min} < cell < 0.03 * "
                    "{t_max})".format(
                        t_min=(1. - WeatherSymbols().float_tolerance),
                        t_max=(1. + WeatherSymbols().float_tolerance)))
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_zero_threshold(self):
        """Test construct_extract_constraint when threshold is zero."""
        plugin = WeatherSymbols()
        diagnostic = 'probability_of_rainfall_rate_above_threshold'
        threshold = AuxCoord(0.0, units='mm hr-1')
        result = plugin.construct_extract_constraint(diagnostic,
                                                     threshold, False)
        expected = ("iris.Constraint("
                    "name='probability_of_rainfall_rate_above_threshold', "
                    "rainfall_rate=lambda cell:  -1e-12 < cell < 1e-12)")
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_list_of_constraints(self):
        """Test construct_extract_constraint returns a list
           of iris.Constraint."""
        plugin = WeatherSymbols()
        diagnostics = ['probability_of_rainfall_rate_above_threshold',
                       'probability_of_lwe_snowfall_rate_above_threshold']
        thresholds = [AuxCoord(0.03, units='mm hr-1'),
                      AuxCoord(0.03, units='mm hr-1')]
        result = plugin.construct_extract_constraint(diagnostics,
                                                     thresholds, False)

        expected = ("iris.Constraint("
                    "name='probability_of_lwe_snowfall_rate_above_threshold', "
                    "lwe_snowfall_rate=lambda cell: 0.03 * {t_min} < cell "
                    "< 0.03 * {t_max})".format(
                        t_min=(1. - WeatherSymbols().float_tolerance),
                        t_max=(1. + WeatherSymbols().float_tolerance)))
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[1], str)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1], expected)


class Test_find_all_routes(IrisTest):

    """Test the find_all_routes method ."""

    def setUp(self):
        """ Setup testing graph """
        self.test_graph = {'start_node': ['success_1', 'fail_0'],
                           'success_1': ['success_1_1', 'fail_1_0'],
                           'fail_0': ['success_0_1', 3],
                           'success_1_1': [1, 2],
                           'fail_1_0': [2, 4],
                           'success_0_1': [5, 1]}

    def test_basic(self):
        """Test find_all_routes returns a list of expected nodes."""
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(self.test_graph,
                                        'start_node',
                                        3)
        expected_nodes = [['start_node',
                           'fail_0',
                           3]]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_multiple_routes(self):
        """Test finds multiple routes."""
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(self.test_graph,
                                        'start_node',
                                        1)
        expected_nodes = [['start_node',
                           'success_1',
                           'success_1_1',
                           1],
                          ['start_node',
                           'fail_0',
                           'success_0_1',
                           1]]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_omit_nodes_top_node(self):
        """Test find_all_routes where omit node is top node."""
        omit_nodes = {'start_node': 'success_1'}
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(self.test_graph,
                                        'start_node',
                                        1,
                                        omit_nodes=omit_nodes,)
        expected_nodes = [['success_1',
                           'success_1_1',
                           1]]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_omit_nodes_midtree(self):
        """Test find_all_routes where omit node is mid tree."""
        omit_nodes = {'success_1': 'success_1_1'}
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(self.test_graph,
                                        'start_node',
                                        1,
                                        omit_nodes=omit_nodes,)
        expected_nodes = [['start_node',
                           'success_1_1',
                           1],
                          ['start_node',
                           'fail_0',
                           'success_0_1',
                           1]]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_omit_nodes_blocked(self):
        """Test find_all_routes where omitted node is no longer accessible."""
        omit_nodes = {'fail_0': 3}
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(self.test_graph,
                                        'start_node',
                                        5,
                                        omit_nodes=omit_nodes,)
        expected_nodes = []
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_omit_nodes_multi(self):
        """Test find_all_routes where multiple omitted nodes."""
        omit_nodes = {'fail_0': 3, 'success_1': 'success_1_1'}
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(self.test_graph,
                                        'start_node',
                                        1,
                                        omit_nodes=omit_nodes,)
        expected_nodes = [['start_node',
                           'success_1_1',
                           1]]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)


class Test_create_symbol_cube(IrisTest):

    """Test the create_symbol_cube method ."""

    def setUp(self):
        """Set up cube """
        data = np.array([0.1, 0.3, 0.4, 0.2, 0.6, 0.7, 0.4, 0.2, 0.1,
                         0.2, 0.2, 0.5, 0.1, 0.3, 0.9, 0.8, 0.5, 0.3,
                         0.6, 0.3, 0.5, 0.6, 0.8, 0.2,
                         0.8, 0.1, 0.2], dtype=np.float32).reshape((3, 3, 3))
        self.cube = set_up_probability_cube(
            data, np.array([288, 290, 292], dtype=np.float32))
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())

    def test_basic(self):
        """Test cube is constructed with appropriate metadata"""
        result = WeatherSymbols().create_symbol_cube([self.cube])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)

    def test_removes_bounds(self):
        """Test bounds are removed from time and forecast period coordinate"""
        self.cube.coord("time").bounds = np.array([
            self.cube.coord("time").points[0] - 3600,
            self.cube.coord("time").points[0]], dtype=np.int64)
        self.cube.coord("forecast_period").bounds = np.array([
            self.cube.coord("forecast_period").points[0] - 3600,
            self.cube.coord("forecast_period").points[0]], dtype=np.int32)
        result = WeatherSymbols().create_symbol_cube([self.cube])
        self.assertIsNone(result.coord("time").bounds)
        self.assertIsNone(result.coord("forecast_period").bounds)


class Test_process(Test_WXCode):

    """Test the find_all_routes method ."""

    def setUp(self):
        """ Set up wxcubes for testing. """
        super().setUp()
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())
        self.expected_wxcode = np.array(
            [[1, 29, 5], [6, 7, 8], [10, 11, 12]])
        self.expected_wxcode_night = np.array(
            [[0, 28, 5], [6, 7, 8], [9, 11, 12]])
        self.expected_wxcode_no_lightning = np.array(
            [[1, 3, 5], [6, 7, 8], [10, 11, 12]])
        self.expected_wxcode_alternate = np.array(
            [[14, 15, 17], [18, 23, 24], [26, 27, 27]])

    def test_basic(self):
        """Test process returns a weather code cube with right values and type.
        """
        plugin = WeatherSymbols()
        result = plugin.process(self.cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)
        self.assertArrayEqual(result.data, self.expected_wxcode)
        self.assertEqual(result.dtype, np.int32)

    def test_day_night(self):
        """Test process returns the right values for night. """
        plugin = WeatherSymbols()
        for i, cube in enumerate(self.cubes):
            self.cubes[i].coord('time').points = (cube.coord('time').points +
                                                  3600*12)
        result = plugin.process(self.cubes)
        self.assertArrayEqual(result.data, self.expected_wxcode_night)

    def test_no_lightning(self):
        """Test process returns right values if no lightning. """
        plugin = WeatherSymbols()
        cubes = self.cubes.extract(self.uk_no_lightning)
        result = plugin.process(cubes)
        self.assertArrayEqual(result.data, self.expected_wxcode_no_lightning)

    def test_lightning(self):
        """Test process returns right values if all lightning. """
        plugin = WeatherSymbols()
        data_lightning = np.ones((1, 3, 3))
        cubes = self.cubes
        cubes[7].data = data_lightning
        result = plugin.process(self.cubes)
        expected_wxcode = np.ones((3, 3)) * 29
        expected_wxcode[1, 1:] = 30
        expected_wxcode[2, 2] = 30
        self.assertArrayEqual(result.data, expected_wxcode)

    def test_weather_data(self):
        """Test process returns the right weather values with a different
        set of data to walk the tree differently."""
        plugin = WeatherSymbols()
        data_snow = np.array([[[0.0, 0.0, 1.0],
                               [1.0, 1.0, 1.0],
                               [1.0, 1.0, 0.1]],
                              [[0.0, 0.0, 0.0],
                               [0.0, 1.0, 1.0],
                               [1.0, 1.0, 0.0]],
                              [[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [1.0, 1.0, 0.0]]])
        data_sleet = np.array([[[0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]]])
        data_rain = np.array([[[1.0, 1.0, 1.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[1.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[1.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]])
        # pylint: disable=no-member
        data_precip = np.maximum.reduce([data_snow, data_sleet, data_rain])
        data_precipv = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0]).reshape((3, 3, 3))
        data_cloud = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                               0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                               0.0, 1.0, 1.0]).reshape((2, 3, 3))
        data_cld_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0]).reshape((1, 3, 3))
        data_vis = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0]).reshape((2, 3, 3))
        data_lightning = np.zeros((1, 3, 3))

        cubes = self.cubes
        cubes[0].data = data_snow
        cubes[1].data = data_sleet
        cubes[2].data = data_rain
        cubes[3].data = data_precipv
        cubes[4].data = data_cloud
        cubes[5].data = data_cld_low
        cubes[6].data = data_vis
        cubes[7].data = data_lightning
        cubes[8].data = data_precip
        result = plugin.process(cubes)
        self.assertArrayEqual(result.data, self.expected_wxcode_alternate)

    def test_sleet(self):
        """Test process returns the sleet weather code."""
        plugin = WeatherSymbols()
        data_snow = np.zeros_like(self.cubes[0].data)
        data_sleet = np.ones_like(self.cubes[0].data)
        data_rain = np.zeros_like(self.cubes[0].data)
        # pylint: disable=no-member
        data_precip = np.maximum.reduce([data_snow, data_sleet, data_rain])
        data_precipv = np.ones_like(self.cubes[0].data)
        data_cloud = np.ones_like(self.cubes[4].data)
        data_cld_low = np.ones_like(self.cubes[5].data)
        data_vis = np.zeros_like(self.cubes[6].data)
        data_lightning = np.zeros_like(self.cubes[7].data)
        expected = np.ones_like(self.expected_wxcode_alternate) * 18

        cubes = self.cubes
        cubes[0].data = data_snow
        cubes[1].data = data_sleet
        cubes[2].data = data_rain
        cubes[3].data = data_precipv
        cubes[4].data = data_cloud
        cubes[5].data = data_cld_low
        cubes[6].data = data_vis
        cubes[7].data = data_lightning
        cubes[8].data = data_precip
        result = plugin.process(cubes)
        self.assertArrayEqual(result.data, expected)

    def test_basic_global(self):
        """Test process returns a wxcode cube with right values for global. """
        plugin = WeatherSymbols(wxtree='global')
        cubes = self.cubes.extract(self.gbl)
        result = plugin.process(cubes)
        self.assertArrayEqual(result.data, self.expected_wxcode_no_lightning)

    def test_weather_data_global(self):
        """Test process returns the right weather values global part2 """
        plugin = WeatherSymbols(wxtree='global')

        data_snow = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1,
                              0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                              0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                              1.0, 1.0, 1.0]).reshape((3, 3, 3))
        data_rain = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 0.0]).reshape((3, 3, 3))
        data_cloud = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                               0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                               0.0, 1.0, 1.0]).reshape((2, 3, 3))
        data_cld_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0]).reshape((1, 3, 3))
        data_vis = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0]).reshape((2, 3, 3))
        cubes = self.cubes.extract(self.gbl)
        cubes[0].data = data_snow
        cubes[1].data = data_rain
        cubes[2].data = data_cloud
        cubes[3].data = data_cld_low
        cubes[4].data = data_vis
        result = plugin.process(cubes)
        self.assertArrayEqual(result.data, self.expected_wxcode_alternate)


if __name__ == '__main__':
    unittest.main()
