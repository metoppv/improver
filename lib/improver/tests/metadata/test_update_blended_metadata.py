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
"""Tests for the improver.metadata.update_blended_metadata module"""

import unittest
import numpy as np
from datetime import datetime as dt

from cf_units import date2num
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.metadata.update_blended_metadata import update_blended_metadata
from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, set_up_probability_cube)


def construct_frt_coord(frt_dts):
    """Construct a forecast_reference_time coordinate from a list
    of datetime.datetime instances"""
    TIME_UNIT = "seconds since 1970-01-01 00:00:00"
    CALENDAR = "gregorian"

    frt_points = []
    for frt in frt_dts:
        frt_points.append(np.round(
            date2num(frt, TIME_UNIT, CALENDAR)).astype(np.int64))

    return AuxCoord(
        frt_points, "forecast_reference_time", units=TIME_UNIT)


class Test_update_blended_metadata(IrisTest):
    """Tests for the update_blended_metadata function"""

    def setUp(self):
        """Set up inputs to metadata update function"""
        attributes = {"source": "Met Office Unified Model",
                      "history": "Post-Processing"}
        self.cycle_blended_cube = set_up_probability_cube(
            np.full((3, 3, 3), 0.5, dtype=np.float32),
            np.array([278, 279, 280], dtype=np.float32),
            time=dt(2015, 11, 23, 7, 0), frt=dt(2015, 11, 23, 3, 0),
            attributes=attributes)

        frts = [dt(2015, 11, 23, 2, 0),
                dt(2015, 11, 23, 3, 0),
                dt(2015, 11, 23, 4, 0)]
        self.frt_coord = construct_frt_coord(frts)

        self.model_blended_cube = self.cycle_blended_cube.copy()
        self.model_blended_cube.add_aux_coord(
            AuxCoord([1500], long_name='model_id', units='no_unit'))
        self.model_blended_cube.add_aux_coord(
            AuxCoord(['uk_det; uk_ens'], long_name='model_configuration',
                     units='no_unit'))

    def test_basic_cycle_blend(self):
        """Test that forecast_reference_time and forecast_period coordinates
        are correctly updated for cycle blending."""
        update_blended_metadata(
            self.cycle_blended_cube, "forecast_reference_time", self.frt_coord)
        self.assertEqual(
            self.cycle_blended_cube.coord("forecast_reference_time").points,
            np.max(self.frt_coord.points))
        self.assertEqual(
            self.cycle_blended_cube.coord("forecast_period").points, [3*3600])

    def test_basic_model_blend(self):
        """Test that forecast_reference_time and forecast_period coordinates
        are correctly updated for model blending, and scalar model coordinates
        are removed."""
        update_blended_metadata(
            self.model_blended_cube, "model_id", self.frt_coord)
        self.assertEqual(
            self.model_blended_cube.coord("forecast_reference_time").points,
            [self.frt_coord.points[-1]])
        self.assertEqual(
            self.model_blended_cube.coord("forecast_period").points, [3*3600])
        for coord_name in ["model_id", "model_configuration"]:
            self.assertNotIn(coord_name, [
                coord.name() for coord in self.model_blended_cube.coords()])

    def test_no_forecast_period(self):
        """Test that a forecast period coordinate is added if not present"""
        self.cycle_blended_cube.remove_coord("forecast_period")
        update_blended_metadata(
            self.cycle_blended_cube, "forecast_reference_time", self.frt_coord)
        self.assertIn("forecast_period", [
            coord.name() for coord in self.cycle_blended_cube.coords()])

    def test_cycletime(self):
        """Test the cycletime argument can be used to set a different
        forecast reference time"""
        update_blended_metadata(
            self.cycle_blended_cube, "forecast_reference_time",
            self.frt_coord, cycletime="20151123T0600Z")
        self.assertSequenceEqual(
            self.cycle_blended_cube.coord("forecast_reference_time").points,
            [self.frt_coord.points[-1] + 2*3600])
        self.assertSequenceEqual(
            self.cycle_blended_cube.coord("forecast_period").points, [3600])

    def test_realization_collapse(self):
        """Test behaviour when cube has been blended over realizations"""
        realization = AuxCoord(
            np.array([6], dtype=np.int32), long_name="realization", units="1")
        cube = set_up_variable_cube(
            275*np.ones((3, 3), dtype=np.float32),
            include_scalar_coords=[realization])
        update_blended_metadata(cube, "realization", None)
        self.assertNotIn(
            "realization", [coord.name() for coord in cube.coords()])

    def test_forecast_coordinate_bounds_removal(self):
        """Test that if a cube has bounds on the forecast period and reference
        time, that these are removed"""
        cube = self.cycle_blended_cube.copy()
        cube.coord("forecast_period").bounds = np.array(
            [[x-0.5, x+0.5] for x in cube.coord(
                "forecast_period").points])
        cube.coord("forecast_reference_time").bounds = np.array(
            [[x-0.5, x+0.5] for x in cube.coord(
                "forecast_reference_time").points])
        cube.coord("forecast_period").bounds = np.array(
            [[x-0.5, x+0.5] for x in cube.coord(
                "forecast_period").points])
        cube.coord("forecast_reference_time").bounds = np.array(
            [[x-0.5, x+0.5] for x in cube.coord(
                "forecast_reference_time").points])
        update_blended_metadata(
            cube, "forecast_reference_time", self.frt_coord)
        self.assertIsNone(cube.coord("forecast_reference_time").bounds)
        self.assertIsNone(cube.coord("forecast_period").bounds)

    def test_attributes_dict(self):
        """Test attributes can be added, updated, and removed from the cube"""
        attribute_changes = {"history": "remove",
                             "source": "IMPROVER",
                             "title": "IMPROVER Multi-Model Blend"}
        expected_attributes = {"source": "IMPROVER",
                               "title": "IMPROVER Multi-Model Blend"}
        update_blended_metadata(
            self.cycle_blended_cube, "forecast_reference_time",
            self.frt_coord, attributes_dict=attribute_changes)
        self.assertDictEqual(
            self.cycle_blended_cube.attributes, expected_attributes)


if __name__ == '__main__':
    unittest.main()
