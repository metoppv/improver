# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
from datetime import timedelta

import iris
import numpy as np
from cf_units import Unit
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.metadata.probabilistic import (
    find_threshold_coordinate,
    get_threshold_coord_name_from_probability_name,
)
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube
from improver.wxcode.utilities import WX_DICT
from improver.wxcode.weather_symbols import WeatherSymbols

from . import wxcode_decision_tree


class Test_WXCode(IrisTest):

    """Test class for the WX code tests, setting up inputs."""

    def setUp(self):
        """Set up cubes and constraints required for Weather Symbols."""

        time = dt(2017, 10, 10, 12, 0)
        frt = dt(2017, 10, 10, 6, 0)

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
            time_bounds=[time - timedelta(hours=1), time],
            frt=frt,
        )

        hail = set_up_probability_cube(
            np.zeros((1, 3, 3), dtype=np.float32),
            thresholds,
            variable_name="lwe_graupel_and_hail_fall_rate_in_vicinity",
            threshold_units="m s-1",
            time=time,
            time_bounds=[time - timedelta(hours=1), time],
            frt=frt,
        )

        thresholds = np.array([2.77777778e-07], dtype=np.float32)
        precipmaxrate = set_up_probability_cube(
            np.zeros((1, 3, 3), dtype=np.float32),
            thresholds,
            variable_name="lwe_precipitation_rate_max",
            threshold_units="m s-1",
            time=time,
            time_bounds=[time - timedelta(hours=1), time],
            frt=frt,
        )

        thresholds = np.array([1.0], dtype=np.float32)
        data_shower_condition = np.array(
            [[[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]], dtype=np.float32,
        )

        shower_condition = set_up_probability_cube(
            data_shower_condition,
            thresholds,
            variable_name="shower_condition",
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
                precip_rate,
                shower_condition,
                # Period diagnostics below here.
                lightning,
                hail,
                precipmaxrate,
            ]
        )
        names = [cube.name() for cube in self.cubes]
        self.missing_diagnostic = [name for name in names if "lightning" not in name]
        self.plugin = WeatherSymbols(wxtree=wxcode_decision_tree())

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
        result = str(WeatherSymbols(wxtree=wxcode_decision_tree()))
        msg = "<WeatherSymbols start_node=lightning>"
        self.assertEqual(result, msg)


class Test_prepare_input_cubes(Test_WXCode):

    """Test the prepare_input_cubes method."""

    def test_basic(self):
        """Test prepare_input_cubes method raises no error if the data is OK"""
        plugin = WeatherSymbols(wxtree=wxcode_decision_tree())
        plugin.prepare_input_cubes(self.cubes)

    def test_no_lightning(self):
        """Test prepare_input_cubes raises no error if lightning missing"""
        cubes = self.cubes.extract(self.missing_diagnostic)
        _, result = self.plugin.prepare_input_cubes(cubes)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertTrue("lightning" in result)

    def test_raises_error_missing_cubes(self):
        """Test prepare_input_cubes method raises error if data is missing"""
        cubes = self.cubes[0:2]
        msg = "Weather Symbols input cubes are missing"
        with self.assertRaisesRegex(IOError, msg):
            self.plugin.prepare_input_cubes(cubes)

    def test_raises_error_missing_threshold(self):
        """Test prepare_input_cubes method raises error if data is missing"""
        cubes = self.cubes
        cubes[0] = cubes[0][0]
        msg = "Weather Symbols input cubes are missing"
        with self.assertRaisesRegex(IOError, msg):
            self.plugin.prepare_input_cubes(cubes)

    def test_incorrect_units(self):
        """Test that prepare_input_cubes method raises an error if the units are
        incompatible between the input cube and the decision tree."""
        msg = "Unable to convert from"
        threshold_coord = find_threshold_coordinate(self.cubes[0])
        self.cubes[0].coord(threshold_coord).units = Unit("mm kg-1")
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.prepare_input_cubes(self.cubes)

    def test_returns_used_cubes(self):
        """Test that prepare_input_cubes method returns a list of cubes that is
        reduced to include only those diagnostics and thresholds that are used
        in the decision tree. Rain, sleet and snow all have a redundant
        threshold in the input cubes. The test below ensures that for all other
        diagnostics all thresholds are returned, but for rain, sleet and snow
        the extra threshold is omitted."""

        expected = []
        unexpected = []
        for cube in self.cubes:
            threshold_name = get_threshold_coord_name_from_probability_name(cube.name())
            threshold_values = cube.coord(threshold_name).points
            if (
                "rain" in threshold_name
                or "sleet" in threshold_name
                or "snow" in threshold_name
            ):
                unexpected.append(
                    iris.Constraint(
                        coord_values={
                            threshold_name: lambda cell: 2.7e-08 < cell < 2.8e-08
                        }
                    )
                )
                threshold_values = threshold_values[0::2]
            for value in threshold_values:
                expected.append(iris.Constraint(coord_values={threshold_name: value}))

        result, _ = self.plugin.prepare_input_cubes(self.cubes)

        for constraint in expected:
            self.assertTrue(len(result.extract(constraint)) > 0)
        for constraint in unexpected:
            self.assertEqual(len(result.extract(constraint)), 0)


class Test_invert_condition(IrisTest):

    """Test the invert condition method."""

    def test_basic(self):
        """Test that the invert_condition method returns a tuple of strings."""
        plugin = WeatherSymbols(wxtree=wxcode_decision_tree())
        tree = plugin.queries
        result = plugin.invert_condition(tree[list(tree.keys())[0]])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], str)

    def test_invert_thresholds_correctly(self):
        """Test invert_condition inverts thresholds correctly."""
        plugin = WeatherSymbols(wxtree=wxcode_decision_tree())
        node = {"threshold_condition": ">=", "condition_combination": ""}
        possible_inputs = [">=", "<=", "<", ">"]
        inverse_outputs = ["<", ">", ">=", "<="]
        for i, val in enumerate(possible_inputs):
            node["threshold_condition"] = val
            result = plugin.invert_condition(node)
            self.assertEqual(result[0], inverse_outputs[i])

    def test_invert_combination_correctly(self):
        """Test invert_condition inverts combination correctly."""
        plugin = WeatherSymbols(wxtree=wxcode_decision_tree())
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
        plugin = WeatherSymbols(wxtree=wxcode_decision_tree())
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
                "if_true": "heavy_precipitation",
                "if_false": "any_precipitation",
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
        test_condition = self.dummy_queries["significant_precipitation"]
        for t in test_condition["diagnostic_thresholds"]:
            t.convert_units("m s-1")
        thresholds = [t.points.item() for t in test_condition["diagnostic_thresholds"]]
        result = self.plugin.create_condition_chain(test_condition)
        expected = [
            [
                [
                    iris.Constraint(
                        name="probability_of_rainfall_rate_above_threshold",
                        rainfall_rate=lambda cell: np.isclose(
                            cell.point,
                            thresholds[0],
                            rtol=self.plugin.float_tolerance,
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
                            rtol=self.plugin.float_tolerance,
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
        self.plugin.coord_named_threshold = True
        test_condition = self.dummy_queries["significant_precipitation"]
        for t in test_condition["diagnostic_thresholds"]:
            t.convert_units("m s-1")
        thresholds = [t.points.item() for t in test_condition["diagnostic_thresholds"]]
        result = self.plugin.create_condition_chain(test_condition)
        expected = [
            [
                [
                    iris.Constraint(
                        name="probability_of_rainfall_rate_above_threshold",
                        threshold=lambda cell: np.isclose(
                            cell.point,
                            thresholds[0],
                            rtol=self.plugin.float_tolerance,
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
                            rtol=self.plugin.float_tolerance,
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
        test_condition = query["rain_or_snow"]
        for t in (
            test_condition["diagnostic_thresholds"][0]
            + test_condition["diagnostic_thresholds"][1]
        ):
            t.convert_units("m s-1")
        result = self.plugin.create_condition_chain(test_condition)
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
                                rtol=self.plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                        "-",
                        iris.Constraint(
                            name="probability_of_rainfall_rate_above_threshold",
                            rainfall_rate=lambda cell: np.isclose(
                                cell.point,
                                thresholds[1],
                                rtol=self.plugin.float_tolerance,
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
                                rtol=self.plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                        "-",
                        iris.Constraint(
                            name="probability_of_lwe_snowfall_rate_above_threshold",
                            lwe_snowfall_rate=lambda cell: np.isclose(
                                cell.point,
                                thresholds[3],
                                rtol=self.plugin.float_tolerance,
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
        diagnostic = "probability_of_rainfall_rate_above_threshold"
        threshold = AuxCoord(0.03, units="mm hr-1")
        threshold.convert_units("m s-1")
        result = self.plugin.construct_extract_constraint(diagnostic, threshold, False)
        expected = iris.Constraint(
            name="probability_of_rainfall_rate_above_threshold",
            rainfall_rate=lambda cell: np.isclose(
                cell.point,
                threshold.points[0],
                rtol=self.plugin.float_tolerance,
                atol=0,
            ),
        )
        self.assertIsInstance(result, iris.Constraint)
        self.assertArrayEqual(
            self.cubes.extract(result)[0].data, self.cubes.extract(expected)[0].data
        )

    def test_old_naming_convention(self):
        """Test construct_extract_constraint can return a constraint with a
        "threshold" coordinate"""
        diagnostic = "probability_of_rainfall_rate_above_threshold"
        threshold = AuxCoord(0.03, units="mm hr-1")
        result = self.plugin.construct_extract_constraint(diagnostic, threshold, True)
        self.assertIsInstance(result, iris.Constraint)
        self.assertTrue("threshold" in result.__dict__["_coord_values"])

    def test_zero_threshold(self):
        """Test construct_extract_constraint when threshold is zero."""
        diagnostic = (
            "probability_of_number_of_lightning_flashes"
            + "_per_unit_area_in_vicinity_above_threshold"
        )
        threshold = AuxCoord(0.0, units="m-2")
        result = self.plugin.construct_extract_constraint(diagnostic, threshold, False)
        expected = iris.Constraint(
            name=diagnostic,
            number_of_lightning_flashes_per_unit_area=lambda cell: np.isclose(
                cell.point,
                threshold.points[0],
                rtol=0,
                atol=self.plugin.float_abs_tolerance,
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
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        expression = [
            iris.Constraint(
                name="probability_of_lwe_sleetfall_rate_above_threshold",
                lwe_sleetfall_rate=lambda cell: np.isclose(
                    cell.point, t.points[0], rtol=self.plugin.float_tolerance, atol=0,
                ),
            ),
            "-",
            0.5,
            "*",
            iris.Constraint(
                name="probability_of_rainfall_rate_above_threshold",
                rainfall_rate=lambda cell: np.isclose(
                    cell.point, t.points[0], rtol=self.plugin.float_tolerance, atol=0,
                ),
            ),
        ]
        result = self.plugin.evaluate_extract_expression(self.cubes, expression)
        expected = (
            self.cubes.extract(expression[0])[0].data
            - 0.5 * self.cubes.extract(expression[4])[0].data
        )
        self.assertArrayEqual(result, expected)

    def test_sub_expresssions(self):
        """Test evaluating an expression containing sub-expressions."""
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        expression = [
            0.5,
            "*",
            iris.Constraint(
                name="probability_of_lwe_sleetfall_rate_above_threshold",
                lwe_sleetfall_rate=lambda cell: np.isclose(
                    cell.point, t.points[0], rtol=self.plugin.float_tolerance, atol=0,
                ),
            ),
            "+",
            [
                iris.Constraint(
                    name="probability_of_rainfall_rate_above_threshold",
                    rainfall_rate=lambda cell: np.isclose(
                        cell.point,
                        t.points[0],
                        rtol=self.plugin.float_tolerance,
                        atol=0,
                    ),
                ),
                "-",
                iris.Constraint(
                    name="probability_of_lwe_snowfall_rate_above_threshold",
                    lwe_snowfall_rate=lambda cell: np.isclose(
                        cell.point,
                        t.points[0],
                        rtol=self.plugin.float_tolerance,
                        atol=0,
                    ),
                ),
            ],
        ]
        expected = 0.5 * self.cubes.extract(expression[2])[0].data + (
            self.cubes.extract(expression[4][0])[0].data
            - self.cubes.extract(expression[4][2])[0].data
        )
        result = self.plugin.evaluate_extract_expression(self.cubes, expression)
        self.assertArrayEqual(result, expected)


class Test_evaluate_condition_chain(Test_WXCode):
    """Test the evaluate_condition_chain method ."""

    def test_basic(self):
        """Test a simple condition chain with 2 simple expressions joined by "OR"."""
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        chain = [
            [
                [
                    iris.Constraint(
                        name="probability_of_lwe_sleetfall_rate_above_threshold",
                        lwe_sleetfall_rate=lambda cell: np.isclose(
                            cell.point,
                            t.points[0],
                            rtol=self.plugin.float_tolerance,
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
                            rtol=self.plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
            ],
            "OR",
        ]
        result = self.plugin.evaluate_condition_chain(self.cubes, chain)
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
        chain = [
            [
                [
                    iris.Constraint(
                        name="probability_of_lwe_sleetfall_rate_above_threshold",
                        lwe_sleetfall_rate=lambda cell: np.isclose(
                            cell.point,
                            t.points[0],
                            rtol=self.plugin.float_tolerance,
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
                            rtol=self.plugin.float_tolerance,
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
            "Invalid condition chain found. First element has length > 1 "
            "but second element is not 'AND' or 'OR'."
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            self.plugin.evaluate_condition_chain(self.cubes, chain)

    def test_with_operators(self):
        """Test a condition chain where the expressions contain operators."""
        t = AuxCoord(0.1, units="mm hr-1")
        t.convert_units("m s-1")
        chain = [
            [
                [
                    [
                        iris.Constraint(
                            name="probability_of_rainfall_rate_above_threshold",
                            rainfall_rate=lambda cell: np.isclose(
                                cell.point,
                                t.points[0],
                                rtol=self.plugin.float_tolerance,
                                atol=0,
                            ),
                        ),
                        "-",
                        iris.Constraint(
                            name="probability_of_lwe_snowfall_rate_above_threshold",
                            lwe_snowfall_rate=lambda cell: np.isclose(
                                cell.point,
                                t.points[0],
                                rtol=self.plugin.float_tolerance,
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
                            rtol=self.plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
            ],
            "OR",
        ]
        result = self.plugin.evaluate_condition_chain(self.cubes, chain)
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
                                    rtol=self.plugin.float_tolerance,
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
                                    rtol=self.plugin.float_tolerance,
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
                            rtol=self.plugin.float_tolerance,
                            atol=0,
                        ),
                    ),
                    ">=",
                    0.5,
                ],
            ],
            "AND",
        ]
        result = self.plugin.evaluate_condition_chain(self.cubes, chain)
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
        chain = [
            [
                [
                    [
                        iris.Constraint(
                            name="probability_of_lwe_sleetfall_rate_above_threshold",
                            lwe_sleetfall_rate=lambda cell: np.isclose(
                                cell.point,
                                t.points[0],
                                rtol=self.plugin.float_tolerance,
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
                                rtol=self.plugin.float_tolerance,
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
        result = self.plugin.evaluate_condition_chain(self.cubes, chain)
        expression = chain[0][0][0]
        expected = (
            self.cubes.extract(expression[0])[0].data
            - 0.5 * self.cubes.extract(expression[4])[0].data
        ) >= 0.0
        self.assertArrayEqual(result, expected)


class Test_remove_optional_missing(IrisTest):

    """Test the rewriting of the decision tree on-the-fly to take into account
    allowed missing diagnostics."""

    def setUp(self):
        """Setup a decision tree for testing."""
        # self.tree = wxcode_decision_tree()
        self.plugin = WeatherSymbols(wxtree=wxcode_decision_tree())

    def test_first_node(self):
        """Test that if the first node, lightning, is missing, the start_node
        progresses to its "if_diagnostic_missing" option."""

        missing_diagnostic = "lightning"
        target = self.plugin.queries[missing_diagnostic]["if_diagnostic_missing"]
        expected = self.plugin.queries[missing_diagnostic][target]

        self.plugin.remove_optional_missing([missing_diagnostic])

        self.assertEqual(self.plugin.start_node, expected)

    def test_intermediate_node(self):
        """Test that if a node other than the first node is missing, is gets
        cut out of the possible paths through the decision tree. In this case
        it means that the hail "if_false" path skips "heavy_precipitation"
        and instead targets its "if_diagnostic_missing" option, which is
        "heavy_precipitation_cloud"."""

        missing_diagnostics = "heavy_precipitation"
        target = self.plugin.queries[missing_diagnostics]["if_diagnostic_missing"]
        expected = self.plugin.queries[missing_diagnostics][target]

        self.plugin.remove_optional_missing([missing_diagnostics])

        self.assertEqual(self.plugin.queries["hail"]["if_false"], expected)

    def test_sequential_missing_nodes(self):
        """Test that if the diagnostics for two nodes, that are both allowed to
        be missing, are absent, the start node skips both of them."""

        missing_diagnostics = ["lightning", "hail"]

        target = self.plugin.queries[missing_diagnostics[-1]]["if_diagnostic_missing"]
        expected = self.plugin.queries[missing_diagnostics[-1]][target]

        self.plugin.remove_optional_missing(missing_diagnostics)

        self.assertEqual(self.plugin.start_node, expected)

    def test_nonsequential_missing_nodes(self):
        """Test that if the diagnostics for two non-sequential nodes, that are
        both allowed to be missing, are absent, the tree is modified as
        expected. In this case lightning and heavy_precipitation_cloud are missing.

        Route being tested is:

           lighting -> hail -> heavy_precipitation -> heavy_precipitation_cloud
           -> heavy_snow_shower
        """

        missing_diagnostics = ["lightning", "heavy_precipitation_cloud"]

        # Start node should be that targeted by lightning, the first missing
        # diagnostic
        target = self.plugin.queries["lightning"]["if_diagnostic_missing"]
        expected_start = self.plugin.queries["lightning"][target]
        # Expected subsequent step from the resulting first node (hail) due to
        # a missing target (heavy precipitation). Note this is not an expected
        # route through the tree but has been engineered for the test.
        target = self.plugin.queries["heavy_precipitation_cloud"][
            "if_diagnostic_missing"
        ]
        expected_next = self.plugin.queries["heavy_precipitation_cloud"][target]

        self.plugin.remove_optional_missing(missing_diagnostics)

        # Check hail has been made the first node
        self.assertEqual(self.plugin.start_node, expected_start)

        # Exract the heavy precipitation node.
        test_node = self.plugin.queries["heavy_precipitation"]

        # Check it's "if_true" path that targetted "heavy_precipitation_cloud"
        # now targets "heavy_snow_shower"
        self.assertEqual(test_node["if_true"], expected_next)
        # Check the "if_false" path is unmodified as that target diagnostic
        # is not missing
        self.assertEqual(test_node["if_false"], "precipitation_in_vicinity")


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
        self.plugin = WeatherSymbols(wxtree=wxcode_decision_tree())

    def test_basic(self):
        """Test find_all_routes returns a list of expected nodes."""
        result = self.plugin.find_all_routes(self.test_graph, "start_node", 3)
        expected_nodes = [["start_node", "fail_0", 3]]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)

    def test_multiple_routes(self):
        """Test finds multiple routes."""
        result = self.plugin.find_all_routes(self.test_graph, "start_node", 1)
        expected_nodes = [
            ["start_node", "success_1", "success_1_1", 1],
            ["start_node", "fail_0", "success_0_1", 1],
        ]
        self.assertIsInstance(result, list)
        self.assertListEqual(result, expected_nodes)


class Test_check_coincidence(Test_WXCode):
    """Test the check_coincidence method."""

    def setUp(self):
        """Set up cubes for testing"""
        super().setUp()
        (lightning,) = self.cubes.extract(
            "probability_of_number_of_lightning_flashes_per_unit_area_in_"
            "vicinity_above_threshold"
        )
        self.expected_bounds = lightning.coord("time").bounds

    def test_basic(self):
        """Test that a template cube is set as a global on the plugin and that
        no exception is raised."""

        self.plugin.check_coincidence(self.cubes)
        self.assertIn("lwe_precipitation_rate_max", self.plugin.template_cube.name())
        self.assertTrue(
            (
                self.plugin.template_cube.coord("time").bounds == self.expected_bounds
            ).all()
        )

    def test_multiple_matching_periods(self):
        """Test that the last cube in the input list with a period is set as
        the global template cube if multiple period diagnostics with matching
        periods are provided."""

        (shower_cube,) = self.cubes.extract(
            "probability_of_shower_condition_above_threshold"
        )
        shower_cube.coord("time").bounds = self.expected_bounds
        self.cubes.append(shower_cube)

        self.plugin.check_coincidence(self.cubes)
        self.assertIn("shower_condition", self.plugin.template_cube.name())

    def test_unmatched_validity_times(self):
        """Test that an exception is raised if the input cubes do not all share
        the same valdity time."""

        cubes = [cube for cube in self.cubes if "lightning" not in cube.name()]
        cubes[-1].coord("time").points = cubes[-1].coord("time").points + 3600

        msg = (
            "Weather symbol input cubes are valid at different times; \n"
            "\\['probability_of_lwe_snowfall_rate_above_threshold: 1507636800', "
            "'probability_of_lwe_sleetfall_rate_above_threshold: 1507636800', "
            "'probability_of_rainfall_rate_above_threshold: 1507636800', "
            "'probability_of_lwe_precipitation_rate_in_vicinity_above_threshold: 1507636800', "
            "'probability_of_low_and_medium_type_cloud_area_fraction_above_threshold: 1507636800', "
            "'probability_of_low_type_cloud_area_fraction_above_threshold: 1507636800', "
            "'probability_of_visibility_in_air_below_threshold: 1507636800', "
            "'probability_of_lwe_precipitation_rate_above_threshold: 1507636800', "
            "'probability_of_shower_condition_above_threshold: 1507636800', "
            "'probability_of_lwe_graupel_and_hail_fall_rate_in_vicinity_above_threshold: 1507636800', "  # noqa: E501
            "'probability_of_lwe_precipitation_rate_max_above_threshold: 1507640400'\\]"
        )
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.check_coincidence(cubes)

    def test_unmatched_periods(self):
        """Test that an exception is raised if multiple period cubes are
        provided that do not have matching periods."""

        (shower_cube,) = self.cubes.extract(
            "probability_of_shower_condition_above_threshold"
        )
        shower_cube.coord("time").bounds = [
            self.expected_bounds[0][0] - 3600,
            self.expected_bounds[0][1],
        ]
        self.cubes.append(shower_cube)

        msg = (
            "Period diagnostics with different periods have been provided as "
            "input to the weather symbols code. Period diagnostics must all "
            "describe the same period to be used together.\n"
            "\\['probability_of_shower_condition_above_threshold: 7200', "
            "'probability_of_number_of_lightning_flashes_per_unit_area_in_"
            "vicinity_above_threshold: 3600', 'probability_of_lwe_graupel_and_"
            "hail_fall_rate_in_vicinity_above_threshold: 3600', "
            "'probability_of_lwe_precipitation_rate_max_above_threshold: 3600', "
            "'probability_of_shower_condition_above_threshold: 7200'\\]"
        )
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.check_coincidence(self.cubes)

    def test_target_period_mismatch(self):
        """Test that an exception is raised if the diagnostic periods do not
        match the user specified target_period."""

        plugin = WeatherSymbols(wxtree=wxcode_decision_tree(), target_period=10800)
        msg = (
            "Diagnostic periods \\(3600\\) do not match "
            "the user specified target_period \\(10800\\)."
        )
        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_coincidence(self.cubes)

    def test_no_period_diagnostics(self):
        """Test that the first cube in the diagnostic cube list is set as the
        global template cube if there are no period diagnostics."""

        cubes = self.cubes[:8]
        expected = cubes[0]
        self.plugin.check_coincidence(cubes)
        self.assertEqual(self.plugin.template_cube, expected)


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
        self.cube.attributes[
            "mosg__model_run"
        ] = "uk_det:20171109T2300Z:0.500\nuk_ens:20171109T2100Z:0.500"
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())
        self.plugin = WeatherSymbols(wxtree=wxcode_decision_tree())

    def test_basic(self):
        """Test cube is constructed with appropriate metadata without setting
        the model_id_attr or record_run attributes"""
        self.plugin.template_cube = self.cube
        result = self.plugin.create_symbol_cube([self.cube])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes["weather_code"], self.wxcode)
        self.assertEqual(result.attributes["weather_code_meaning"], self.wxmeaning)
        self.assertNotIn("mosg__model_configuration", result.attributes)
        self.assertNotIn("mosg__model_run", result.attributes)
        self.assertTrue((result.data.mask).all())

    def test_model_id_attr(self):
        """Test cube is constructed with appropriate metadata with just the
        model_id_attr attribute set"""
        self.plugin.template_cube = self.cube
        self.plugin.model_id_attr = "mosg__model_configuration"
        result = self.plugin.create_symbol_cube([self.cube])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes["weather_code"], self.wxcode)
        self.assertEqual(result.attributes["weather_code_meaning"], self.wxmeaning)
        self.assertArrayEqual(
            result.attributes["mosg__model_configuration"], "uk_det uk_ens"
        )
        self.assertNotIn("mosg__model_run", result.attributes)
        self.assertTrue((result.data.mask).all())

    def test_record_run_attr(self):
        """Test cube is constructed with appropriate metadata when setting both
        the model_id_attr and record_run attributes. Weights should be discarded
        from the record_run attribute."""
        self.plugin.template_cube = self.cube
        self.plugin.model_id_attr = "mosg__model_configuration"
        self.plugin.record_run_attr = "mosg__model_run"
        result = self.plugin.create_symbol_cube([self.cube])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes["weather_code"], self.wxcode)
        self.assertEqual(result.attributes["weather_code_meaning"], self.wxmeaning)
        self.assertArrayEqual(
            result.attributes["mosg__model_run"],
            "uk_det:20171109T2300Z:\nuk_ens:20171109T2100Z:",
        )
        self.assertArrayEqual(
            result.attributes["mosg__model_configuration"], "uk_det uk_ens"
        )
        self.assertTrue((result.data.mask).all())

    def test_record_run_attr_multiple_inputs(self):
        """Test cube is constructed with appropriate record_run_attr with multiple
        source cubes. Weights should be discarded from the record_run attribute."""
        self.plugin.template_cube = self.cube
        self.plugin.model_id_attr = "mosg__model_configuration"
        self.plugin.record_run_attr = "mosg__model_run"
        cube1 = self.cube.copy()
        cube1.attributes["mosg__model_run"] = "gl_ens:20171109T1800Z:1.000"
        cube1.attributes["mosg__model_configuration"] = "gl_ens"

        result = self.plugin.create_symbol_cube([self.cube, cube1])

        self.assertArrayEqual(
            result.attributes["mosg__model_run"],
            "gl_ens:20171109T1800Z:\nuk_det:20171109T2300Z:\nuk_ens:20171109T2100Z:",
        )
        self.assertArrayEqual(
            result.attributes["mosg__model_configuration"], "gl_ens uk_det uk_ens"
        )
        self.assertTrue((result.data.mask).all())

    def test_bounds_preserved_if_present(self):
        """Test bounds are used to indicate a symbol representing a period has
        been created."""

        expected_time = [
            self.cube.coord("time").points[0] - 3600,
            self.cube.coord("time").points[0],
        ]
        expected_fp = [
            self.cube.coord("forecast_period").points[0] - 3600,
            self.cube.coord("forecast_period").points[0],
        ]

        self.cube.coord("time").bounds = np.array(expected_time, dtype=np.int64,)
        self.cube.coord("forecast_period").bounds = np.array(
            expected_fp, dtype=np.int32,
        )
        self.plugin.template_cube = self.cube
        result = self.plugin.create_symbol_cube([self.cube])
        self.assertTrue((result.coord("time").bounds == expected_time).all())
        self.assertTrue((result.coord("forecast_period").bounds == expected_fp).all())

    def test_no_bounds_for_instantaneous_inputs(self):
        """Test no time bounds are present on the weather symbols cube if the
        inputs are all instantaneous."""

        self.plugin.template_cube = self.cube
        result = self.plugin.create_symbol_cube([self.cube])
        self.assertIsNone(result.coord("time").bounds)
        self.assertIsNone(result.coord("forecast_period").bounds)

    def test_manual_title(self):
        """Test cube is constructed with appropriate metadata including a
        user provided title attribute."""

        target_title = "Weather Symbols"
        plugin = WeatherSymbols(wxtree=wxcode_decision_tree(), title=target_title)
        plugin.template_cube = self.cube

        result = plugin.create_symbol_cube([self.cube])
        self.assertEqual(result.attributes["title"], target_title)


class Test_compare_to_threshold(IrisTest):
    """Test the compare_to_threshold method ."""

    def test_array(self):
        """Test that compare_to_threshold produces the correct array of
        booleans."""
        arr = np.array([0, 1, 2], dtype=np.int32)
        plugin = WeatherSymbols(wxtree=wxcode_decision_tree())
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
        plugin = WeatherSymbols(wxtree=wxcode_decision_tree())
        msg = "Invalid comparator: !=. Comparator must be one of '<', '>', '<=', '>='."

        with self.assertRaisesRegex(ValueError, msg):
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
        result = self.plugin.process(self.cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes["weather_code"], self.wxcode)
        self.assertEqual(result.attributes["weather_code_meaning"], self.wxmeaning)
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode)
        self.assertEqual(result.dtype, np.int32)

    def test_day_night(self):
        """Test codes for night-time weather symbols are returned."""
        twelve_hours = 3600 * 12
        for i, cube in enumerate(self.cubes):
            time_coord = cube.coord("time")
            self.cubes[i].coord("time").points = time_coord.points + twelve_hours
            if time_coord.has_bounds():
                bounds = time_coord.bounds
                bounds = [bound + twelve_hours for bound in bounds]
                self.cubes[i].coord("time").bounds = bounds
        result = self.plugin.process(self.cubes)
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode_night)

    def test_no_lightning(self):
        """Test process returns right values if no lightning. This is equivalent
        to a global weather symbols call as there is currently no lightning
        field."""
        cubes = self.cubes.extract(self.missing_diagnostic)
        result = self.plugin.process(cubes)
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode_no_lightning)

    def test_lightning(self):
        """Test process returns right values if all lightning. """
        data_lightning = np.ones((3, 3))
        cubes = self.cubes
        cubes[9].data = data_lightning
        result = self.plugin.process(self.cubes)
        expected_wxcode = np.ones((3, 3)) * 30
        expected_wxcode[0, 1] = 29
        expected_wxcode[1, 1:] = 29
        expected_wxcode[2, 0] = 29
        self.assertArrayAndMaskEqual(result.data, expected_wxcode)

    def test_lightning_but_missing_next_node(self):
        """Test process returns right values if a lightning input is provided,
        but the next node, hail, is missing. Lightning is set in one corner
        only, all other resulting values should match default expected values.
        This indicates that the missing hail node has been omitted successfully."""
        cubes = self.cubes[:10] + [self.cubes[-1]]
        cubes[9].data[0, 0] = 1
        result = self.plugin.process(cubes)
        self.expected_wxcode[0, 0] = 30
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode)

    def test_masked_precip(self):
        """Test process returns right values when precipitation data are fully masked
        (e.g. nowcast-only). The only possible non-masked result is a lightning code as
        these do not include precipitation in any of their decision routes."""
        data_precip = np.ma.masked_all_like(self.cubes[7].data)
        cubes = self.cubes
        cubes[7].data = data_precip
        result = self.plugin.process(self.cubes)
        expected_wxcode = np.ma.masked_all_like(self.expected_wxcode)
        expected_wxcode[0, 1] = self.expected_wxcode[0, 1]
        self.assertArrayAndMaskEqual(result.data, expected_wxcode)

    def test_weather_data(self):
        """Test process returns the right weather values with a different
        set of data to walk the tree differently."""
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
        data_shower_condition = np.array(
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
        cubes[7].data = data_precip
        cubes[8].data = data_shower_condition
        cubes[9].data = data_lightning
        result = self.plugin.process(cubes)
        self.assertArrayAndMaskEqual(result.data, self.expected_wxcode_alternate)

    def test_sleet(self):
        """Test process returns the sleet weather code."""
        data_snow = np.zeros_like(self.cubes[0].data)
        data_sleet = np.ones_like(self.cubes[0].data)
        data_rain = np.zeros_like(self.cubes[0].data)
        data_precip = np.maximum.reduce([data_snow, data_sleet, data_rain])
        data_precipv = np.ones_like(self.cubes[0].data)
        data_cloud = np.ones_like(self.cubes[4].data)
        data_cld_low = np.ones_like(self.cubes[5].data)
        data_vis = np.zeros_like(self.cubes[6].data)
        data_lightning = np.zeros_like(self.cubes[9].data)
        data_shower_condition = np.zeros_like(self.cubes[8].data)
        expected = np.ones_like(self.expected_wxcode_alternate) * 18

        cubes = self.cubes
        cubes[0].data = data_snow
        cubes[1].data = data_sleet
        cubes[2].data = data_rain
        cubes[3].data = data_precipv
        cubes[4].data = data_cloud
        cubes[5].data = data_cld_low
        cubes[6].data = data_vis
        cubes[7].data = data_precip
        cubes[8].data = data_shower_condition
        cubes[9].data = data_lightning
        result = self.plugin.process(cubes)
        self.assertArrayAndMaskEqual(result.data, expected)


if __name__ == "__main__":
    unittest.main()
