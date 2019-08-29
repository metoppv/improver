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

import iris
import numpy as np
from cf_units import Unit
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.tests.ensemble_calibration.ensemble_calibration. \
    helper_functions import set_up_probability_threshold_cube
from improver.utilities.cube_checker import find_threshold_coordinate
from improver.wxcode.weather_symbols import WeatherSymbols
from improver.wxcode.wxcode_utilities import WX_DICT


def set_up_wxcubes():
    """Set up cubes required for Weather Symbols """
    data_snow = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0]).reshape(3, 1, 3, 3)
    snowfall_rate = (
        set_up_probability_threshold_cube(
            data_snow,
            'lwe_snowfall_rate',
            'm s-1',
            spp__relative_to_threshold='above',
            forecast_thresholds=np.array([8.33333333e-09,
                                          2.77777778e-08,
                                          2.77777778e-07])))

    data_rain = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0]).reshape(3, 1, 3, 3)
    rainfall_rate = (
        set_up_probability_threshold_cube(
            data_rain,
            'rainfall_rate',
            'm s-1',
            spp__relative_to_threshold='above',
            forecast_thresholds=np.array([8.33333333e-09,
                                          2.77777778e-08,
                                          2.77777778e-07])))

    data_snowv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0]).reshape(3, 1, 3, 3)
    snowfall_vicinity = (
        set_up_probability_threshold_cube(
            data_snowv,
            'lwe_snowfall_rate_in_vicinity',
            'm s-1',
            spp__relative_to_threshold='above',
            forecast_thresholds=np.array([8.33333333e-09,
                                          2.77777778e-08,
                                          2.77777778e-07])))

    data_rainv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0]).reshape(3, 1, 3, 3)
    rainfall_vicinity = (
        set_up_probability_threshold_cube(
            data_rainv,
            'rainfall_rate_in_vicinity',
            'm s-1',
            spp__relative_to_threshold='above',
            forecast_thresholds=np.array([8.33333333e-09,
                                          2.77777778e-08,
                                          2.77777778e-07])))

    data_cloud = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                           0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                           0.0, 0.0, 1.0]).reshape(2, 1, 3, 3)
    cloud = (set_up_probability_threshold_cube(
        data_cloud,
        'cloud_area_fraction',
        '1',
        spp__relative_to_threshold='above',
        forecast_thresholds=np.array([0.1875, 0.8125])))

    data_cld_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                             0.0, 0.0, 0.0]).reshape(1, 1, 3, 3)
    cloud_low = (
        set_up_probability_threshold_cube(
            data_cld_low,
            'low_type_cloud_area_fraction',
            '1',
            spp__relative_to_threshold='above',
            forecast_thresholds=np.array([0.85])))

    data_vis = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0]).reshape(2, 1, 3, 3)
    visibility = (
        set_up_probability_threshold_cube(
            data_vis,
            'visibility_in_air',
            'm',
            spp__relative_to_threshold='below',
            forecast_thresholds=np.array([1000.0, 5000.0])))
    visibility.attributes['relative_to_threshold'] = 'below'

    cubes = iris.cube.CubeList([snowfall_rate, rainfall_rate,
                                snowfall_vicinity, rainfall_vicinity,
                                cloud, cloud_low,
                                visibility])
    return cubes


def set_up_wxcubes_global():
    """Set up cubes required for Weather Symbols """
    data_snow = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0]).reshape(3, 1, 3, 3)
    snowfall_rate = (
        set_up_probability_threshold_cube(
            data_snow,
            'lwe_snowfall_rate',
            'm s-1',
            spp__relative_to_threshold='above',
            forecast_thresholds=np.array([8.33333333e-09,
                                          2.77777778e-08,
                                          2.77777778e-07])))

    data_rain = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0]).reshape(3, 1, 3, 3)
    rainfall_rate = (
        set_up_probability_threshold_cube(
            data_rain,
            'rainfall_rate',
            'm s-1',
            spp__relative_to_threshold='above',
            forecast_thresholds=np.array([8.33333333e-09,
                                          2.77777778e-08,
                                          2.77777778e-07])))

    data_cloud = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                           0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                           0.0, 0.0, 1.0]).reshape(2, 1, 3, 3)
    cloud = (set_up_probability_threshold_cube(
        data_cloud,
        'cloud_area_fraction',
        '1',
        spp__relative_to_threshold='above',
        forecast_thresholds=np.array([0.1875, 0.8125])))

    data_cld_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                            0.0, 0.0, 0.0]).reshape(1, 1, 3, 3)
    cloud_low = (
        set_up_probability_threshold_cube(
            data_cld_low,
            'low_type_cloud_area_fraction',
            '1',
            spp__relative_to_threshold='above',
            forecast_thresholds=np.array([0.85])))

    data_vis = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0]).reshape(2, 1, 3, 3)
    visibility = (
        set_up_probability_threshold_cube(
            data_vis,
            'visibility_in_air',
            'm',
            spp__relative_to_threshold='below',
            forecast_thresholds=np.array([1000.0, 5000.0])))
    visibility.coord(var_name="threshold"
                     ).attributes['spp__relative_to_threshold'] = 'below'

    cubes = iris.cube.CubeList([snowfall_rate, rainfall_rate,
                                cloud, cloud_low,
                                visibility])
    return cubes


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeatherSymbols())
        msg = '<WeatherSymbols tree=high_resolution>'
        self.assertEqual(result, msg)

    def test_global(self):
        """Test that the __repr__ returns right string for global tree."""
        result = str(WeatherSymbols(wxtree='global'))
        msg = '<WeatherSymbols tree=global>'
        self.assertEqual(result, msg)


class Test_check_input_cubes(IrisTest):

    """Test the check_input_cubes method."""

    def setUp(self):
        """ Setup for testing """
        self.cubes = set_up_wxcubes()

    def test_basic(self):
        """Test check_input_cubes method raises no error if the data is OK"""
        plugin = WeatherSymbols()
        self.assertEqual(plugin.check_input_cubes(self.cubes), None)

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
        cubes = set_up_wxcubes_global()
        self.assertEqual(plugin.check_input_cubes(cubes), None)

    def test_raises_error_missing_cubes_global(self):
        """Test check_input_cubes method raises error if data is missing"""
        plugin = WeatherSymbols(wxtree='global')
        cubes = set_up_wxcubes_global()[0:3]
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
        """Test find_all_routes finds multiple routes."""
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


class Test_create_symbol_cube(IrisTest):

    """Test the create_symbol_cube method ."""

    def setUp(self):
        """Set up cube """
        data = np.array([0.1, 0.3, 0.4, 0.2, 0.6, 0.7, 0.4, 0.2, 0.1,
                         0.2, 0.2, 0.5, 0.1, 0.3, 0.9, 0.8, 0.5, 0.3,
                         0.6, 0.3, 0.5, 0.6, 0.8, 0.2,
                         0.8, 0.1, 0.2]).reshape(3, 1, 3, 3)
        self.cube = set_up_probability_threshold_cube(
            data, 'air_temperature', 'K', spp__relative_to_threshold='above')
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())

    def test_basic(self):
        """Test construct_extract_constraint method returns a iris.Constraint.
            or list of iris.Constraint"""
        plugin = WeatherSymbols()

        result = plugin.create_symbol_cube(self.cube[0])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)


class Test_process(IrisTest):

    """Test the find_all_routes method ."""

    def setUp(self):
        """ Set up wxcubes for testing. """
        self.cubes = set_up_wxcubes()
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())

    def test_basic(self):
        """Test process returns a weather code cube with right values. """
        plugin = WeatherSymbols()
        result = plugin.process(self.cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)
        expected_wxcode = np.array([1, 3, 5,
                                    6, 7, 8,
                                    10, 11, 12]).reshape(1, 3, 3)
        self.assertArrayEqual(result.data,
                              expected_wxcode)

    def test_day_night(self):
        """Test process returns the right values for night. """
        plugin = WeatherSymbols()
        for i, cube in enumerate(self.cubes):
            self.cubes[i].coord('time').points = (cube.coord('time').points +
                                                  11.5)
        result = plugin.process(self.cubes)
        expected_wxcode = np.array([0, 2, 5,
                                    6, 7, 8,
                                    9, 11, 12]).reshape(1, 3, 3)
        self.assertArrayEqual(result.data,
                              expected_wxcode)

    def test_weather_data(self):
        """Test process returns the right weather values.part2 """
        plugin = WeatherSymbols()
        data_snow = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1,
                              0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                              0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                              1.0, 1.0, 0.0]).reshape(3, 1, 3, 3)
        data_rain = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 0.0]).reshape(3, 1, 3, 3)
        data_snowv = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0]).reshape(3, 1, 3, 3)
        data_rainv = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 0.0]).reshape(3, 1, 3, 3)
        data_cloud = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                               0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                               0.0, 1.0, 1.0]).reshape(2, 1, 3, 3)
        data_cld_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0]).reshape(1, 1, 3, 3)
        data_vis = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0]).reshape(2, 1, 3, 3)
        cubes = self.cubes
        cubes[0].data = data_snow
        cubes[1].data = data_rain
        cubes[2].data = data_snowv
        cubes[3].data = data_rainv
        cubes[4].data = data_cloud
        cubes[5].data = data_cld_low
        cubes[6].data = data_vis
        result = plugin.process(cubes)
        expected_wxcode = np.array([14, 15, 17,
                                    18, 23, 24,
                                    26, 27, 27]).reshape(1, 3, 3)
        self.assertArrayEqual(result.data,
                              expected_wxcode)

    def test_basic_global(self):
        """Test process returns a wxcode cube with right values for global. """
        plugin = WeatherSymbols(wxtree='global')
        cubes = set_up_wxcubes_global()
        result = plugin.process(cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)
        expected_wxcode = np.array([1, 3, 5,
                                    6, 7, 8,
                                    10, 11, 12]).reshape(1, 3, 3)
        self.assertArrayEqual(result.data,
                              expected_wxcode)

    def test_weather_data_global(self):
        """Test process returns the right weather values global part2 """
        plugin = WeatherSymbols(wxtree='global')

        data_snow = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1,
                              0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                              0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                              1.0, 1.0, 1.0]).reshape(3, 1, 3, 3)
        data_rain = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 0.0]).reshape(3, 1, 3, 3)
        data_cloud = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                               0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                               0.0, 1.0, 1.0]).reshape(2, 1, 3, 3)
        data_cld_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0]).reshape(1, 1, 3, 3)
        data_vis = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0]).reshape(2, 1, 3, 3)
        cubes = set_up_wxcubes_global()
        cubes[0].data = data_snow
        cubes[1].data = data_rain
        cubes[2].data = data_cloud
        cubes[3].data = data_cld_low
        cubes[4].data = data_vis
        result = plugin.process(cubes)
        expected_wxcode = np.array([14, 15, 17,
                                    18, 23, 24,
                                    26, 27, 27]).reshape(1, 3, 3)
        self.assertArrayEqual(result.data,
                              expected_wxcode)


if __name__ == '__main__':
    unittest.main()
