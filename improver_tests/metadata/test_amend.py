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
"""Tests for the improver.metadata.amend module"""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
import pytest
from iris.coords import CellMethod
from iris.tests import IrisTest

from improver.metadata.amend import (
    amend_attributes,
    set_history_attribute,
    update_diagnostic_name,
    update_model_id_attr_attribute,
    update_stage_v110_metadata,
)
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)


def create_cube_with_threshold(data=None, threshold_values=None):
    """
    Create a cube with threshold coord.  Data and threshold values MUST be
    provided as float32 (not float64), or cube setup will fail.
    """
    if threshold_values is None:
        threshold_values = np.array([1.0], dtype=np.float32)

    if data is None:
        data = np.zeros((len(threshold_values), 2, 2, 2), dtype=np.float32)
        data[:, 0, :, :] = 0.5
        data[:, 1, :, :] = 0.6

    cube = set_up_probability_cube(
        data[:, 0, :, :],
        threshold_values,
        variable_name="rainfall_rate",
        threshold_units="m s-1",
        time=dt(2015, 11, 19, 1, 30),
        frt=dt(2015, 11, 18, 22, 0),
    )

    time_points = [dt(2015, 11, 19, 0, 30), dt(2015, 11, 19, 1, 30)]
    cube = add_coordinate(
        cube, time_points, "time", order=[1, 0, 2, 3], is_datetime=True
    )

    cube.attributes["attribute_to_update"] = "first_value"

    cube.data = data
    return cube


class Test_update_stage_v110_metadata(IrisTest):
    """Test the update_stage_v110_metadata function"""

    def setUp(self):
        """Set up variables for use in testing."""
        data = 275.0 * np.ones((3, 3), dtype=np.float32)
        self.cube = set_up_variable_cube(data)

    def test_basic(self):
        """Test that cube is unchanged and function returns False"""
        result = self.cube.copy()
        update_stage_v110_metadata(result)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.data, self.cube.data)
        self.assertEqual(result.attributes, self.cube.attributes)

    def test_update_ukv(self):
        """Test that cube attributes from ukv 1.1.0 are updated"""
        self.cube.attributes["grid_id"] = "ukvx_standard_v1"
        update_stage_v110_metadata(self.cube)
        self.assertTrue("mosg__grid_type" in self.cube.attributes.keys())
        self.assertTrue("mosg__model_configuration" in self.cube.attributes.keys())
        self.assertTrue("mosg__grid_domain" in self.cube.attributes.keys())
        self.assertTrue("mosg__grid_version" in self.cube.attributes.keys())
        self.assertFalse("grid_id" in self.cube.attributes.keys())
        self.assertEqual("standard", self.cube.attributes["mosg__grid_type"])
        self.assertEqual("uk_det", self.cube.attributes["mosg__model_configuration"])
        self.assertEqual("uk_extended", self.cube.attributes["mosg__grid_domain"])
        self.assertEqual("1.1.0", self.cube.attributes["mosg__grid_version"])


class Test_amend_attributes(IrisTest):
    """Test the amend_attributes method."""

    def setUp(self):
        """Set up a cube and dict"""
        self.cube = set_up_variable_cube(
            280 * np.ones((3, 3), dtype=np.float32),
            attributes={
                "mosg__grid_version": "1.3.0",
                "mosg__model_configuration": "uk_det",
            },
        )
        self.metadata_dict = {
            "mosg__grid_version": "remove",
            "source": "IMPROVER unit tests",
            "mosg__model_configuration": "other_model",
        }

    def test_basic(self):
        """Test function adds, removes and modifies attributes as expected"""
        expected_attributes = {
            "source": "IMPROVER unit tests",
            "mosg__model_configuration": "other_model",
        }
        amend_attributes(self.cube, self.metadata_dict)
        self.assertDictEqual(self.cube.attributes, expected_attributes)


class Test_set_history_attribute(IrisTest):
    """Test the set_history_attribute function."""

    def test_add_history(self):
        """Test that a history attribute has been added."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        set_history_attribute(cube, "Nowcast")
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])

    def test_history_already_exists(self):
        """Test that the history attribute is overwritten, if it
        already exists."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        old_history = "2018-09-13T11:28:29: StaGE"
        cube.attributes["history"] = old_history
        set_history_attribute(cube, "Nowcast")
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])
        self.assertFalse(old_history in cube.attributes["history"])

    def test_history_append(self):
        """Test that the history attribute can be updated."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        old_history = "2018-09-13T11:28:29: StaGE"
        cube.attributes["history"] = old_history
        set_history_attribute(cube, "Nowcast", append=True)
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])
        self.assertTrue(old_history in cube.attributes["history"])

    def test_history_append_no_existing(self):
        """Test the "append" option doesn't crash when no history exists."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        set_history_attribute(cube, "Nowcast", append=True)
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])


class Test_update_model_id_attr_attribute(IrisTest):

    """Test the update_model_id_attr_attribute function."""

    def setUp(self):
        """Set up cube."""
        self.cube = set_up_probability_cube(
            np.zeros((2, 2, 2), dtype=np.float32),
            np.array([288, 290], dtype=np.float32),
        )
        self.model_id_attr = "mosg__model_configuration"

    def test_one_input_attribute(self):
        """Test handling of model_id_attr attribute for one input."""
        self.cube.attributes["mosg__model_configuration"] = "uk_ens"
        result = update_model_id_attr_attribute([self.cube], self.model_id_attr)
        self.assertArrayEqual(result["mosg__model_configuration"], "uk_ens")

    def test_two_matching_input_attributes(self):
        """Test handling of model_id_attr attribute for two matching inputs."""
        self.cube.attributes["mosg__model_configuration"] = "uk_ens"
        self.cube1 = self.cube.copy()
        self.cube2 = self.cube.copy()
        result = update_model_id_attr_attribute(
            [self.cube1, self.cube2], self.model_id_attr
        )
        self.assertArrayEqual(result["mosg__model_configuration"], "uk_ens")

    def test_two_different_input_attributes(self):
        """Test handling of model_id_attr attribute for two different inputs."""
        self.cube1 = self.cube.copy()
        self.cube2 = self.cube.copy()
        self.cube1.attributes["mosg__model_configuration"] = "uk_ens"
        self.cube2.attributes["mosg__model_configuration"] = "nc_det"
        result = update_model_id_attr_attribute(
            [self.cube1, self.cube2], self.model_id_attr
        )
        self.assertArrayEqual(result["mosg__model_configuration"], "nc_det uk_ens")

    def test_compound_attributes(self):
        """Test handling of compound attributes."""
        self.cube1 = self.cube.copy()
        self.cube2 = self.cube.copy()
        self.cube1.attributes["mosg__model_configuration"] = "uk_det uk_ens"
        self.cube2.attributes["mosg__model_configuration"] = "nc_det uk_det uk_ens"
        result = update_model_id_attr_attribute(
            [self.cube1, self.cube2], self.model_id_attr
        )
        self.assertArrayEqual(
            result["mosg__model_configuration"], "nc_det uk_det uk_ens"
        )

    def test_attribute_mismatch(self):
        """Test a mismatch in the presence of the model_id_attr attribute."""
        self.cube1 = self.cube.copy()
        self.cube2 = self.cube.copy()
        self.cube1.attributes["mosg__model_configuration"] = "uk_ens"
        msg = "Expected to find mosg__model_configuration attribute on all cubes"
        with self.assertRaisesRegex(AttributeError, msg):
            update_model_id_attr_attribute([self.cube1, self.cube2], self.model_id_attr)


@pytest.mark.parametrize("cell_method", (True, False))
@pytest.mark.parametrize("probability_data", (True, False))
@pytest.mark.parametrize("in_vicinity", (True, False))
def test_update_diagnostic_name(cell_method, in_vicinity, probability_data):
    """Make sure that the update_diagnostic_name method makes the expected changes in the expected
    situations. Checks that cell_method comments are updated to match the cube name, if present;
    that "in_vicinity" can be present on the cube name but not the threshold coord name;
    that both probabilistic and non-probabilistic meta-data are handled correctly."""
    input_name_suffix = ""
    if in_vicinity:
        input_name_suffix = "_in_vicinity"
    source_name = "lwe_thickness_of_precipitation_amount"
    if probability_data:
        cube = set_up_probability_cube(
            np.zeros((2, 2, 2), dtype=np.float32),
            [0, 1],
            f"{source_name}{input_name_suffix}",
            "mm",
        )
    else:
        cube = set_up_percentile_cube(
            np.zeros((2, 2, 2), dtype=np.float32),
            [40, 60],
            name=f"{source_name}{input_name_suffix}",
            units="mm",
        )
    if cell_method:
        cube.add_cell_method(CellMethod("mean", "time", comments=f"of {source_name}"))
    new_base_name = "lwe_thickness_of_precipitation_amount"
    new_name = f"{new_base_name}_in_variable_vicinity"
    if probability_data:
        expected_long_name = f"probability_of_{new_name}_above_threshold"
        expected_name = new_base_name
    else:
        expected_long_name = new_name
        expected_name = source_name
    expected_cm_comment = f"of {new_base_name}"
    update_diagnostic_name(cube, new_name, cube)
    assert cube.long_name == expected_long_name
    if probability_data:
        assert cube.coord(var_name="threshold").name() == expected_name
    else:
        assert "threshold" not in [coord.var_name for coord in cube.coords()]
    if cell_method:
        assert expected_cm_comment in cube.cell_methods[0].comments


if __name__ == "__main__":
    unittest.main()
