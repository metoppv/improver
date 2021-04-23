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
"""Unit tests for Weather Symbols class."""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
from cf_units import Unit
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.metadata.probabilistic import find_threshold_coordinate
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube
from improver.wxcode.utilities import WX_DICT
from improver.wxcode.weather_symbols import WeatherSymbols


class Test_WXCode(IrisTest):

    """Test class for the WX code tests, setting up inputs."""

    def setUp(self):
        """Set up cubes and constraints required for Weather Symbols."""

        time = dt(2017, 10, 10, 12, 0)
        frt = dt(2017, 10, 10, 12, 0)

        thresholds = np.array(
            [8.33333333e-09, 2.77777778e-08, 2.77777778e-07], dtype=np.float32
        )
        data_snow = np.zeros((3, 3, 3), dtype=np.float32)
        snowfall_rate = set_up_probability_cube(
            data_snow,
            thresholds,
            variable_name="lwe_snowfall_rate",
            threshold_units="m s-1",
            time=time,
            frt=frt,
        )

        data_sleet = np.zeros((3, 3, 3), dtype=np.float32)

        sleetfall_rate = set_up_probability_cube(
            data_sleet,
            thresholds,
            variable_name="lwe_sleetfall_rate",
            threshold_units="m s-1",
            time=time,
            frt=frt,
        )

        data_rain = np.array(
            [
                [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [1.00, 1.00, 1.00]],
                [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [1.00, 0.01, 1.00]],
                [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]],
            ],
            dtype=np.float32,
        )

        rainfall_rate = set_up_probability_cube(
            data_rain,
            thresholds,
            variable_name="rainfall_rate",
            threshold_units="m s-1",
            time=time,
            frt=frt,
        )

        # pylint: disable=no-member
        data_precip = np.maximum.reduce([data_snow, data_sleet, data_rain])

        precip_rate = set_up_probability_cube(
            data_precip,
            thresholds,
            variable_name="lwe_precipitation_rate",
            threshold_units="m s-1",
            time=time,
            frt=frt,
        )

        precip_vicinity = set_up_probability_cube(
            data_precip,
            thresholds,
            variable_name="lwe_precipitation_rate_in_vicinity",
            threshold_units="m s-1",
            time=time,
            frt=frt,
        )

        thresholds = np.array([0.8], dtype=np.float32)
        data_convective_ratio = np.array(
            [[[0.1, 0.1, 0.1], [0.2, 0.2, 1.0], [1.0, 1.0, 0.2]]], dtype=np.float32,
        )

        convective_ratio = set_up_probability_cube(
            data_convective_ratio,
            thresholds,
            variable_name="convective_ratio",
            threshold_units="1",
            time=time,
            frt=frt,
        )

        thresholds = np.array([0.1875, 0.8125], dtype=np.float32)
        data_cloud = np.array(
            [
                [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )

        cloud = set_up_probability_cube(
            data_cloud,
            thresholds,
            variable_name="low_and_medium_type_cloud_area_fraction",
            threshold_units="1",
            time=time,
            frt=frt,
        )

        thresholds = np.array([0.85], dtype=np.float32)
        data_cld_low = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32
        ).reshape((1, 3, 3))
        cloud_low = set_up_probability_cube(
            data_cld_low,
            thresholds,
            variable_name="low_type_cloud_area_fraction",
            threshold_units="1",
            time=time,
            frt=frt,
        )

        thresholds = np.array([1000.0, 5000.0], dtype=np.float32)
        data_vis = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        visibility = set_up_probability_cube(
            data_vis,
            thresholds,
            variable_name="visibility_in_air",
            threshold_units="m",
            spp__relative_to_threshold="below",
            time=time,
            frt=frt,
        )

        thresholds = np.array([0.0], dtype=np.float32)
        data_lightning = np.zeros((1, 3, 3), dtype=np.float32)
        data_lightning[0, 0, 0] = 0.25
        data_lightning[0, 0, 1] = 0.30

        lightning = set_up_probability_cube(
            data_lightning,
            thresholds,
            variable_name="number_of_lightning_flashes_per_unit_area_in_vicinity",
            threshold_units="m-2",
            time=time,
            frt=frt,
        )

        thresholds = np.array([0.05], dtype=np.float32)
        data_cloud_texture = np.array(
            [[[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]], dtype=np.float32,
        )

        cloud_texture = set_up_probability_cube(
            data_cloud_texture,
            thresholds,
            variable_name="texture_of_low_and_medium_type_cloud_area_fraction",
            threshold_units="1",
            time=time,
            frt=frt,
        )

        self.cubes = iris.cube.CubeList(
            [
                snowfall_rate,
                sleetfall_rate,
                rainfall_rate,
                precip_vicinity,
                cloud,
                cloud_low,
                visibility,
                lightning,
                precip_rate,
                cloud_texture,
                convective_ratio,
            ]
        )
        names = [cube.name() for cube in self.cubes]
        self.uk_no_lightning = [name for name in names if "lightning" not in name]
        self.gbl = [
            name
            for name in self.uk_no_lightning
            if "vicinity" not in name and "texture" not in name
        ]

    def assertArrayAndMaskEqual(self, array_a, array_b, **kwargs):
        """
        Checks test output and expected array are equal, using self.assertArrayEqual
        and then checks that if a mask is present on the test array, it matches the
        expected mask.

        Args:
            array_a (np.ndarray):
                Typically, this will be the output from a test
            array_b (np.ndarray):
                Typically, this will be the expected output from a test

        Raises:
            AssertionError:
                if a mask is present on only one argument or masks do not match

        """
        self.assertArrayEqual(array_a, array_b, **kwargs)
        if not np.ma.is_masked(array_a) and not np.ma.is_masked(array_b):
            # Neither array is masked. Test passes.
            return
        if not (np.ma.is_masked(array_a) and np.ma.is_masked(array_b)):
            # Only one array is masked.
            if np.ma.is_masked(array_a):
                msg = f"Only a is masked: {array_a.mask}"
            else:
                msg = f"Only b is masked: {array_b.mask}"
        else:
            if (array_a.mask == array_b.mask).all():
                # Masks match exactly
                return
            msg = f"Masks do not match: {array_a.mask} /= {array_b.mask}"
        raise AssertionError(msg)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeatherSymbols())
        msg = "<WeatherSymbols tree=high_resolution start_node=lightning>"
        self.assertEqual(result, msg)

    def test_global(self):
        """Test that the __repr__ returns right string for global tree."""
        result = str(WeatherSymbols(wxtree="global"))
        msg = "<WeatherSymbols tree=global start_node=heavy_precipitation>"
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
        self.assertTrue("lightning" in result)

    def test_raises_error_missing_cubes(self):
        """Test check_input_cubes method raises error if data is missing"""
        plugin = WeatherSymbols()
        cubes = self.cubes.pop()
        msg = "Weather Symbols input cubes are missing"
        with self.assertRaisesRegex(IOError, msg):
            plugin.check_input_cubes(cubes)

    def test_raises_error_missing_threshold(self):
        """Test check_input_cubes method raises error if data is missing"""
        plugin = WeatherSymbols()
        cubes = self.cubes
        cubes[0] = cubes[0][0]
        msg = "Weather Symbols input cubes are missing"
        with self.assertRaisesRegex(IOError, msg):
            plugin.check_input_cubes(cubes)

    def test_incorrect_units(self):
        """Test that check_input_cubes method raises an error if the units are
        incompatible between the input cube and the decision tree."""
        plugin = WeatherSymbols()

        msg = "Unable to convert from"
        threshold_coord = find_threshold_coordinate(self.cubes[0])
        self.cubes[0].coord(threshold_coord).units = Unit("mm kg-1")
        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_input_cubes(self.cubes)

    def test_basic_global(self):
        """Test check_input_cubes method has no error if global data is OK"""
        plugin = WeatherSymbols(wxtree="global")
        cubes = self.cubes.extract(self.gbl)
        self.assertEqual(plugin.check_input_cubes(cubes), None)

    def test_raises_error_missing_cubes_global(self):
        """Test check_input_cubes method raises error if data is missing"""
        plugin = WeatherSymbols(wxtree="global")
        cubes = self.cubes.extract(self.gbl)[0:3]
        msg = "Weather Symbols input cubes are missing"
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
        node = {"threshold_condition": ">=", "condition_combination": ""}
        possible_inputs = [">=", "<=", "<", ">"]
        inverse_outputs = ["<", ">", ">=", "<="]
        for i, val in enumerate(possible_inputs):
            node["threshold_condition"] = val
            result = plugin.invert_condition(node)
            self.assertEqual(result[0], inverse_outputs[i])

    def test_invert_combination_correctly(self):
        """Test invert_condition inverts combination correctly."""
        plugin = WeatherSymbols()
        node = {"threshold_condition": ">=", "condition_combination": ""}
        possible_inputs = ["AND", "OR", ""]
        inverse_outputs = ["OR", "AND", ""]
        for i, val in enumerate(possible_inputs):
            node["condition_combination"] = val
            result = plugin.invert_condition(node)
            self.assertEqual(result[1], inverse_outputs[i])

    def test_error(self):
        """Test that the _invert_comparator method raises an error when the condition
        cannot be inverted."""
        plugin = WeatherSymbols()
        possible_inputs = ["==", "!=", "NOT", "XOR"]
        for val in possible_inputs:
            with self.assertRaisesRegex(
                KeyError, f"Unexpected condition {val}, cannot invert it."
            ):
                plugin._invert_comparator(val)


class Test_create_condition_chain(Test_WXCode):
    """Test the create_condition_chain method."""

    def setUp(self):
        """ Set up queries for testing"""
        super().setUp()
        self.dummy_queries = {
            "significant_precipitation": {
                "succeed": "heavy_precipitation",
                "fail": "any_precipitation",
                "probability_thresholds": [0.5, 0.5],
                "threshold_condition": ">=",
                "condition_combination": "OR",
                "diagnostic_fields": [
                    "probability_of_rainfall_rate_above_threshold",
                    "probability_of_lwe_snowfall_rate_above_threshold",
                ],
                "diagnostic_thresholds": [
                    AuxCoord(0.03, units="mm hr-1"),
                    AuxCoord(0.03, units="mm hr-1"),
                ],
                "diagnostic_conditions": ["above", "above"],
            }
        }

    def test_basic(self):
        """Test create_condition_chain returns a nested list of iris.Constraint,
        floats, and strings representing operators that extracts the correct data."""
        plugin = WeatherSymbols()
        test_condition = self.dummy_queries["significant_precipitation"]
        for t in test_condition["diagnostic_thresholds"]:
            t.convert_units("m s-1")
        thresholds = [t.points.item() for t in test_condition["diagnostic_thresholds"]]
        result = plugin.create_condition_chain(test_condition)
        expected = [
            [
                [
                    iris.Constraint(
                        name="probability_of_rainfall_rate_above_threshold",
                        rainfall_rate=lambda cell: np.isclose(
                            cell.point,
                            thresholds[0],
                            rtol=plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
                [
                    iris.Constraint(
                        name="probability_of_lwe_snowfall_rate_above_threshold",
                        lwe_snowfall_rate=lambda cell: np.isclose(
                            cell.point,
                            thresholds[1],
                            rtol=plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
            ],
            "OR",
        ]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        for i in range(2):
            constraint_exp = expected[0][i][0]
            constraint_res = result[0][i][0]
            self.assertArrayEqual(
                self.cubes.extract(constraint_res)[0].data,
                self.cubes.extract(constraint_exp)[0].data,
            )
            self.assertEqual(result[0][i][1:], expected[0][i][1:])
        self.assertEqual(result[1], expected[1])

    def test_old_naming_convention(self):
        """Test create_condition_chain can return conditions using old
        threshold coordinate name"""
        plugin = WeatherSymbols()
        plugin.coord_named_threshold = True
        test_condition = self.dummy_queries["significant_precipitation"]
        for t in test_condition["diagnostic_thresholds"]:
            t.convert_units("m s-1")
        thresholds = [t.points.item() for t in test_condition["diagnostic_thresholds"]]
        result = plugin.create_condition_chain(test_condition)
        expected = [
            [
                [
                    iris.Constraint(
                        name="probability_of_rainfall_rate_above_threshold",
                        threshold=lambda cell: np.isclose(
                            cell.point,
                            thresholds[0],
                            rtol=plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
                [
                    iris.Constraint(
                        name="probability_of_lwe_snowfall_rate_above_threshold",
                        threshold=lambda cell: np.isclose(
                            cell.point,
                            thresholds[1],
                            rtol=plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
            ],
            "OR",
        ]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        for i in range(2):
            constraint_res = result[0][i][0]
            self.assertTrue("threshold" in constraint_res.__dict__["_coord_values"])
            self.assertEqual(result[0][i][1:], expected[0][i][1:])
        self.assertEqual(result[1], expected[1])

    def test_complex_condition(self):
        """Test with a condition that uses an operator"""
        query = {
            "rain_or_snow": self.dummy_queries["significant_precipitation"],
        }
        query["rain_or_snow"]["diagnostic_fields"] = [
            [
                "probability_of_lwe_sleetfall_rate_above_threshold",
                "-",
                "probability_of_rainfall_rate_above_threshold",
            ],
            [
                "probability_of_lwe_sleetfall_rate_above_threshold",
                "-",
                "probability_of_lwe_snowfall_rate_above_threshold",
            ],
        ]
        query["rain_or_snow"]["diagnostic_thresholds"] = [
            [AuxCoord(0.1, units="mm hr-1"), AuxCoord(0.1, units="mm hr-1")],
            [AuxCoord(0.1, units="mm hr-1"), AuxCoord(0.1, units="mm hr-1")],
        ]
        query["rain_or_snow"]["diagnostic_conditions"] = [
            ["above", "above"],
            ["above", "above"],
        ]
        plugin = WeatherSymbols()
        test_condition = query["rain_or_snow"]
        for t in (
            test_condition["diagnostic_thresholds"][0]
            + test_condition["diagnostic_thresholds"][1]
        ):
            t.convert_units("m s-1")
        result = plugin.create_condition_chain(test_condition)
        thresholds = [
            t.points.item() for t in test_condition["diagnostic_thresholds"][0]
        ] + [t.points.item() for t in test_condition["diagnostic_thresholds"][1]]
        expected = [
            [
                [
                    [
                        iris.Constraint(
                            name="probability_of_lwe_sleetfall_rate_above_threshold",
                            lwe_sleetfall_rate=lambda cell: np.isclose(
                                cell.point,
                                thresholds[0],
                                rtol=plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                        "-",
                        iris.Constraint(
                            name="probability_of_rainfall_rate_above_threshold",
                            rainfall_rate=lambda cell: np.isclose(
                                cell.point,
                                thresholds[1],
                                rtol=plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                    ],
                    ">=",
                    0.5,
                ],
                [
                    [
                        iris.Constraint(
                            name="probability_of_lwe_sleetfall_rate_above_threshold",
                            lwe_sleetfall_rate=lambda cell: np.isclose(
                                cell.point,
                                thresholds[2],
                                rtol=plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                        "-",
                        iris.Constraint(
                            name="probability_of_lwe_snowfall_rate_above_threshold",
                            lwe_snowfall_rate=lambda cell: np.isclose(
                                cell.point,
                                thresholds[3],
                                rtol=plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                    ],
                    ">=",
                    0.5,
                ],
            ],
            "OR",
        ]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        for i in range(2):
            for k in [0, 2]:
                constraint_exp = expected[0][i][0][k]
                constraint_res = result[0][i][0][k]
                self.assertArrayEqual(
                    self.cubes.extract(constraint_res)[0].data,
                    self.cubes.extract(constraint_exp)[0].data,
                )
            self.assertEqual(result[0][i][0][1], expected[0][i][0][1])
            self.assertEqual(result[0][i][1:], expected[0][i][1:])
        self.assertEqual(result[1], expected[1])


class Test_construct_extract_constraint(Test_WXCode):

    """Test the construct_extract_constraint method ."""

    def test_basic(self):
        """Test construct_extract_constraint returns a iris.Constraint."""
        plugin = WeatherSymbols()
        diagnostic = "probability_of_rainfall_rate_above_threshold"
        threshold = AuxCoord(0.03, units="mm hr-1")
        threshold.convert_units("m s-1")
        result = plugin.construct_extract_constraint(diagnostic, threshold, False)
        expected = iris.Constraint(
            name="probability_of_rainfall_rate_above_threshold",
            rainfall_rate=lambda cell: np.isclose(
                cell.point, threshold.points[0], rtol=plugin.float_tolerance, atol=0,
            ),
        )
        self.assertIsInstance(result, iris.Constraint)
        self.assertArrayEqual(
            self.cubes.extract(result)[0].data, self.cubes.extract(expected)[0].data
        )

    def test_old_naming_convention(self):
        """Test construct_extract_constraint can return a constraint with a
        "threshold" coordinate"""
        plugin = WeatherSymbols()
        diagnostic = "probability_of_rainfall_rate_above_threshold"
        threshold = AuxCoord(0.03, units="mm hr-1")
        result = plugin.construct_extract_constraint(diagnostic, threshold, True)
        self.assertIsInstance(result, iris.Constraint)
        self.assertTrue("threshold" in result.__dict__["_coord_values"])

    def test_zero_threshold(self):
        """Test construct_extract_constraint when threshold is zero."""
        plugin = WeatherSymbols()
        diagnostic = "probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_threshold"
        threshold = AuxCoord(0.0, units="m-2")
        result = plugin.construct_extract_constraint(diagnostic, threshold, False)
        expected = iris.Constraint(
            name="probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_threshold",
            number_of_lightning_flashes_per_unit_area=lambda cell: np.isclose(
                cell.point,
                threshold.points[0],
                rtol=0,
                atol=plugin.float_abs_tolerance,
            ),
        )
        self.assertIsInstance(result, iris.Constraint)
        self.assertArrayEqual(
            self.cubes.extract(result)[0].data, self.cubes.extract(expected)[0].data
        )


class Test_evaluate_extract_expression(Test_WXCode):
    """Test the evaluate_extract_expression method ."""

    def test_basic(self):
        """Test evaluating a basic expression consisting of constraints,
        operators, and constants."""
        plugin = WeatherSymbols()
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        expression = [
            iris.Constraint(
                name="probability_of_lwe_sleetfall_rate_above_threshold",
                lwe_sleetfall_rate=lambda cell: np.isclose(
                    cell.point, t.points[0], rtol=plugin.float_tolerance, atol=0,
                ),
            ),
            "-",
            0.5,
            "*",
            iris.Constraint(
                name="probability_of_rainfall_rate_above_threshold",
                rainfall_rate=lambda cell: np.isclose(
                    cell.point, t.points[0], rtol=plugin.float_tolerance, atol=0,
                ),
            ),
        ]
        result = plugin.evaluate_extract_expression(self.cubes, expression)
        expected = (
            self.cubes.extract(expression[0])[0].data
            - 0.5 * self.cubes.extract(expression[4])[0].data
        )
        self.assertArrayEqual(result, expected)

    def test_sub_expresssions(self):
        """Test evaluating an expression containing sub-expressions."""
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        plugin = WeatherSymbols()
        expression = [
            0.5,
            "*",
            iris.Constraint(
                name="probability_of_lwe_sleetfall_rate_above_threshold",
                lwe_sleetfall_rate=lambda cell: np.isclose(
                    cell.point, t.points[0], rtol=plugin.float_tolerance, atol=0,
                ),
            ),
            "+",
            [
                iris.Constraint(
                    name="probability_of_rainfall_rate_above_threshold",
                    rainfall_rate=lambda cell: np.isclose(
                        cell.point, t.points[0], rtol=plugin.float_tolerance, atol=0,
                    ),
                ),
                "-",
                iris.Constraint(
                    name="probability_of_lwe_snowfall_rate_above_threshold",
                    lwe_snowfall_rate=lambda cell: np.isclose(
                        cell.point, t.points[0], rtol=plugin.float_tolerance, atol=0,
                    ),
                ),
            ],
        ]
        expected = 0.5 * self.cubes.extract(expression[2])[0].data + (
            self.cubes.extract(expression[4][0])[0].data
            - self.cubes.extract(expression[4][2])[0].data
        )
        result = plugin.evaluate_extract_expression(self.cubes, expression)
        self.assertArrayEqual(result, expected)


class Test_evaluate_condition_chain(Test_WXCode):
    """Test the evaluate_condition_chain method ."""

    def test_basic(self):
        """Test a simple condition chain with 2 simple expressions joined by "OR"."""
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        plugin = WeatherSymbols()
        chain = [
            [
                [
                    iris.Constraint(
                        name="probability_of_lwe_sleetfall_rate_above_threshold",
                        lwe_sleetfall_rate=lambda cell: np.isclose(
                            cell.point,
                            t.points[0],
                            rtol=plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
                [
                    iris.Constraint(
                        name="probability_of_rainfall_rate_above_threshold",
                        rainfall_rate=lambda cell: np.isclose(
                            cell.point,
                            t.points[0],
                            rtol=plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
            ],
            "OR",
        ]
        result = plugin.evaluate_condition_chain(self.cubes, chain)
        c1 = chain[0][0][0]
        c2 = chain[0][1][0]
        expected = (self.cubes.extract(c1)[0].data >= 0.5) | (
            self.cubes.extract(c2)[0].data >= 0.5
        )
        self.assertArrayEqual(result, expected)

    def test_error(self):
        """Test that we get an error if first element of the chain has length > 1
        and second element is ""."""
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        plugin = WeatherSymbols()
        chain = [
            [
                [
                    iris.Constraint(
                        name="probability_of_lwe_sleetfall_rate_above_threshold",
                        lwe_sleetfall_rate=lambda cell: np.isclose(
                            cell.point,
                            t.points[0],
                            rtol=plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
                [
                    iris.Constraint(
                        name="probability_of_rainfall_rate_above_threshold",
                        rainfall_rate=lambda cell: np.isclose(
                            cell.point,
                            t.points[0],
                            rtol=plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
            ],
            "",
        ]
        msg = (
            "Invalid condition chain found. First element has length > 1 ",
            "but second element is not 'AND' or 'OR'.",
        )
        with self.assertRaises(RuntimeError, msg=msg):
            plugin.evaluate_condition_chain(self.cubes, chain)

    def test_with_operators(self):
        """Test a condition chain where the expressions contain operators."""
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        plugin = WeatherSymbols()
        chain = [
            [
                [
                    [
                        iris.Constraint(
                            name="probability_of_rainfall_rate_above_threshold",
                            rainfall_rate=lambda cell: np.isclose(
                                cell.point,
                                t.points[0],
                                rtol=plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                        "-",
                        iris.Constraint(
                            name="probability_of_lwe_snowfall_rate_above_threshold",
                            lwe_snowfall_rate=lambda cell: np.isclose(
                                cell.point,
                                t.points[0],
                                rtol=plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                    ],
                    ">=",
                    0.5,
                ],
                [
                    iris.Constraint(
                        name="probability_of_lwe_sleetfall_rate_above_threshold",
                        lwe_sleetfall_rate=lambda cell: np.isclose(
                            cell.point,
                            t.points[0],
                            rtol=plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
            ],
            "OR",
        ]
        result = plugin.evaluate_condition_chain(self.cubes, chain)
        c1 = chain[0][0][0][0]
        c2 = chain[0][0][0][2]
        c3 = chain[0][1][0]
        expected = (
            self.cubes.extract(c1)[0].data - self.cubes.extract(c2)[0].data >= 0.5
        ) | (self.cubes.extract(c3)[0].data >= 0.5)
        self.assertArrayEqual(result, expected)

    def test_with_subconditions(self):
        """Test "AND" condition chain with sub-chain containing "OR"."""
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        plugin = WeatherSymbols()
        chain = [
            [
                [
                    [
                        [
                            iris.Constraint(
                                name="probability_of_lwe_sleetfall_rate_above_threshold",
                                lwe_sleetfall_rate=lambda cell: np.isclose(
                                    cell.point,
                                    t.points[0],
                                    rtol=plugin.float_tolerance,
                                    atol=0,
                                ),
                            ),
                            ">=",
                            0.5,
                        ],
                        [
                            iris.Constraint(
                                name="probability_of_lwe_snowfall_rate_above_threshold",
                                lwe_snowfall_rate=lambda cell: np.isclose(
                                    cell.point,
                                    t.points[0],
                                    rtol=plugin.float_tolerance,
                                    atol=0,
                                ),
                            ),
                            ">=",
                            0.5,
                        ],
                    ],
                    "OR",
                ],
                [
                    iris.Constraint(
                        name="probability_of_rainfall_rate_above_threshold",
                        rainfall_rate=lambda cell: np.isclose(
                            cell.point,
                            t.points[0],
                            rtol=plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
            ],
            "AND",
        ]
        result = plugin.evaluate_condition_chain(self.cubes, chain)
        c1 = chain[0][0][0][0][0]
        c2 = chain[0][0][0][1][0]
        c3 = chain[0][1][0]
        expected = (
            (self.cubes.extract(c1)[0].data >= 0.5)
            | (self.cubes.extract(c2)[0].data >= 0.5)
        ) & (self.cubes.extract(c3)[0].data >= 0.5)
        self.assertArrayEqual(result, expected)

    def test_blank_condition(self):
        """Test a condition chain where the combination condition is ""."""
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        plugin = WeatherSymbols()
        chain = [
            [
                [
                    [
                        iris.Constraint(
                            name="probability_of_lwe_sleetfall_rate_above_threshold",
                            lwe_sleetfall_rate=lambda cell: np.isclose(
                                cell.point,
                                t.points[0],
                                rtol=plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                        "-",
                        0.5,
                        "*",
                        iris.Constraint(
                            name="probability_of_rainfall_rate_above_threshold",
                            rainfall_rate=lambda cell: np.isclose(
                                cell.point,
                                t.points[0],
                                rtol=plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                    ],
                    ">=",
                    0.0,
                ]
            ],
            "",
        ]
        result = plugin.evaluate_condition_chain(self.cubes, chain)
        expression = chain[0][0][0]
        expected = (
            self.cubes.extract(expression[0])[0].data
            - 0.5 * self.cubes.extract(expression[4])[0].data
        ) >= 0.0
        self.assertArrayEqual(result, expected)


class Test_find_all_routes(IrisTest):

    """Test the find_all_routes method ."""

    def setUp(self):
        """ Setup testing graph """
        self.test_graph = {
            "start_node": ["success_1", "fail_0"],
            "success_1": ["success_1_1", "fail_1_0"],
            "fail_0": ["success_0_1", 3],
            "success_1_1": [1, 2],
            "fail_1_0": [2, 4],
            "success_0_1": [5, 1],
        }

    def test_basic(self):
        """Test find_all_routes returns a list of expected nodes."""
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(self.test_graph, "start_node", 3)
        expected_nodes = [["start_node", "fail_0", 3]]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_multiple_routes(self):
        """Test finds multiple routes."""
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(self.test_graph, "start_node", 1)
        expected_nodes = [
            ["start_node", "success_1", "success_1_1", 1],
            ["start_node", "fail_0", "success_0_1", 1],
        ]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_omit_nodes_top_node(self):
        """Test find_all_routes where omit node is top node."""
        omit_nodes = {"start_node": "success_1"}
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(
            self.test_graph, "start_node", 1, omit_nodes=omit_nodes,
        )
        expected_nodes = [["success_1", "success_1_1", 1]]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_omit_nodes_midtree(self):
        """Test find_all_routes where omit node is mid tree."""
        omit_nodes = {"success_1": "success_1_1"}
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(
            self.test_graph, "start_node", 1, omit_nodes=omit_nodes,
        )
        expected_nodes = [
            ["start_node", "success_1_1", 1],
            ["start_node", "fail_0", "success_0_1", 1],
        ]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_omit_nodes_blocked(self):
        """Test find_all_routes where omitted node is no longer accessible."""
        omit_nodes = {"fail_0": 3}
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(
            self.test_graph, "start_node", 5, omit_nodes=omit_nodes,
        )
        expected_nodes = []
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_omit_nodes_multi(self):
        """Test find_all_routes where multiple omitted nodes."""
        omit_nodes = {"fail_0": 3, "success_1": "success_1_1"}
        plugin = WeatherSymbols()
        result = plugin.find_all_routes(
            self.test_graph, "start_node", 1, omit_nodes=omit_nodes,
        )
        expected_nodes = [["start_node", "success_1_1", 1]]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)


class Test_create_symbol_cube(IrisTest):

    """Test the create_symbol_cube method ."""

    def setUp(self):
        """Set up cube."""
        data = np.array(
            [
                [[0.1, 0.3, 0.4], [0.2, 0.6, 0.7], [0.4, 0.2, 0.1]],
                [[0.2, 0.2, 0.5], [0.1, 0.3, 0.9], [0.8, 0.5, 0.3]],
                [[0.6, 0.3, 0.5], [0.6, 0.8, 0.2], [0.8, 0.1, 0.2]],
            ],
            dtype=np.float32,
        )
        self.cube = set_up_probability_cube(
            data, np.array([288, 290, 292], dtype=np.float32)
        )
        self.cube.attributes["mosg__model_configuration"] = "uk_det uk_ens"
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())

    def test_basic(self):
        """Test cube is constructed with appropriate metadata without
        model_id_attr attribute"""
        result = WeatherSymbols().create_symbol_cube([self.cube])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes["weather_code"], self.wxcode)
        self.assertEqual(result.attributes["weather_code_meaning"], self.wxmeaning)
        self.assertNotIn("mosg__model_configuration", result.attributes)
        self.assertTrue((result.data.mask).all())

    def test_model_id_attr(self):
        """Test cube is constructed with appropriate metadata with
        model_id_attr attribute"""
        result = WeatherSymbols(
            model_id_attr="mosg__model_configuration"
        ).create_symbol_cube([self.cube])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes["weather_code"], self.wxcode)
        self.assertEqual(result.attributes["weather_code_meaning"], self.wxmeaning)
        self.assertArrayEqual(
            result.attributes["mosg__model_configuration"], "uk_det uk_ens"
        )
        self.assertTrue((result.data.mask).all())

    def test_removes_bounds(self):
        """Test bounds are removed from time and forecast period coordinate"""
        self.cube.coord("time").bounds = np.array(
            [
                self.cube.coord("time").points[0] - 3600,
                self.cube.coord("time").points[0],
            ],
            dtype=np.int64,
        )
        self.cube.coord("forecast_period").bounds = np.array(
            [
                self.cube.coord("forecast_period").points[0] - 3600,
                self.cube.coord("forecast_period").points[0],
            ],
            dtype=np.int32,
        )
        result = WeatherSymbols().create_symbol_cube([self.cube])
        self.assertIsNone(result.coord("time").bounds)
        self.assertIsNone(result.coord("forecast_period").bounds)


class Test_compare_to_threshold(IrisTest):
    """Test the compare_to_threshold method ."""

    def test_array(self):
        """Test that compare_to_threshold produces the correct array of
        booleans."""
        arr = np.array([0, 1, 2], dtype=np.int32)
        plugin = WeatherSymbols()
        test_case_map = {
            "<": [True, False, False],
            "<=": [True, True, False],
            ">": [False, False, True],
            ">=": [False, True, True],
        }
        for item in test_case_map:
            result = plugin.compare_array_to_threshold(arr, item, 1)
            self.assertArrayEqual(result, test_case_map[item])

    def test_error_on_unexpected_comparison(self):
        """Test that an error is raised if the comparison operator is not
        one of the expected strings."""
        arr = np.array([0, 1, 2], dtype=np.int32)
        plugin = WeatherSymbols()
        msg = (
            "Invalid comparator: !=. ",
            "Comparator must be one of '<', '>', '<=', '>='.",
        )
        with self.assertRaises(ValueError, msg=msg):
            plugin.compare_array_to_threshold(arr, "!=", 1)


class Test_process(Test_WXCode):

    """Test the find_all_routes method ."""

    def setUp(self):
        """ Set up wxcubes for testing. """
        super().setUp()
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())
        self.expected_wxcode = np.array([[1, 29, 5], [6, 7, 8], [10, 11, 12]])
        self.expected_wxcode_night = np.array([[0, 28, 5], [6, 7, 8], [9, 11, 12]])
        self.expected_wxcode_no_lightning = np.array(
            [[1, 3, 5], [6, 7, 8], [10, 11, 12]]
        )
        self.expected_wxcode_alternate = np.array(
            [[14, 15, 17], [18, 23, 24], [26, 27, 27]]
        )

    def test_basic(self):
        """Test process returns a weather code cube with right values and type.
        """
        plugin = WeatherSymbols()
        result = plugin.process(self.cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes["weather_code"], self.wxcode)
        self.assertEqual(result.attributes["weather_code_meaning"], self.wxmeaning)
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode)
        self.assertEqual(result.dtype, np.int32)

    def test_day_night(self):
        """Test process returns the right values for night. """
        plugin = WeatherSymbols()
        for i, cube in enumerate(self.cubes):
            self.cubes[i].coord("time").points = cube.coord("time").points + 3600 * 12
        result = plugin.process(self.cubes)
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode_night)

    def test_no_lightning(self):
        """Test process returns right values if no lightning. """
        plugin = WeatherSymbols()
        cubes = self.cubes.extract(self.uk_no_lightning)
        result = plugin.process(cubes)
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode_no_lightning)

    def test_lightning(self):
        """Test process returns right values if all lightning. """
        plugin = WeatherSymbols()
        data_lightning = np.ones((3, 3))
        cubes = self.cubes
        cubes[7].data = data_lightning
        result = plugin.process(self.cubes)
        expected_wxcode = np.ones((3, 3)) * 30
        expected_wxcode[0, 1] = 29
        expected_wxcode[1, 1:] = 29
        expected_wxcode[2, 0] = 29
        self.assertArrayAndMaskEqual(result.data, expected_wxcode)

    def test_masked_precip(self):
        """Test process returns right values when precipitation data are fully masked
        (e.g. nowcast-only). The only possible non-masked result is a lightning code as
        these do not include precipitation in any of their decision routes."""
        plugin = WeatherSymbols()
        data_precip = np.ma.masked_all_like(self.cubes[8].data)
        cubes = self.cubes
        cubes[8].data = data_precip
        result = plugin.process(self.cubes)
        expected_wxcode = np.ma.masked_all_like(self.expected_wxcode)
        expected_wxcode[0, 1] = self.expected_wxcode[0, 1]
        self.assertArrayAndMaskEqual(result.data, expected_wxcode)

    def test_weather_data(self):
        """Test process returns the right weather values with a different
        set of data to walk the tree differently."""
        plugin = WeatherSymbols()
        data_snow = np.array(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.1]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        data_sleet = np.array(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        data_rain = np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        # pylint: disable=no-member
        data_precip = np.maximum.reduce([data_snow, data_sleet, data_rain])
        data_precipv = np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        data_cloud = np.array(
            [
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        data_cloud_texture = np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32,
        )

        data_cld_low = np.zeros((3, 3))
        data_vis = np.zeros((2, 3, 3))
        data_lightning = np.zeros((3, 3))

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
        cubes[9].data = data_cloud_texture
        result = plugin.process(cubes)
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode_alternate)

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
        data_cloud_texture = np.zeros_like(self.cubes[9].data)
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
        cubes[9].data = data_cloud_texture
        result = plugin.process(cubes)
        self.assertArrayAndMaskEqual(result.data, expected)

    def test_basic_global(self):
        """Test process returns a wxcode cube with right values for global. """
        plugin = WeatherSymbols(wxtree="global")
        cubes = self.cubes.extract(self.gbl)
        result = plugin.process(cubes)
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode_no_lightning)

    def test_weather_data_global(self):
        """Test process returns the right weather values global part2 """
        plugin = WeatherSymbols(wxtree="global")

        data_snow = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.1]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        data_sleet = np.array(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        data_rain = np.array(
            [
                [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        data_cloud = np.array(
            [
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        data_cld_low = np.zeros((3, 3), dtype=np.float32)
        data_vis = np.zeros((2, 3, 3), dtype=np.float32)
        data_precip = np.max(np.array([data_snow, data_sleet, data_rain]), axis=0)
        data_convective_ratio = np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32,
        )
        cubes = self.cubes.extract(self.gbl)
        cubes[0].data = data_snow
        cubes[1].data = data_sleet
        cubes[2].data = data_rain
        cubes[3].data = data_cloud
        cubes[4].data = data_cld_low
        cubes[5].data = data_vis
        cubes[6].data = data_precip
        cubes[7].data = data_convective_ratio
        result = plugin.process(cubes)
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode_alternate)


if __name__ == "__main__":
    unittest.main()
