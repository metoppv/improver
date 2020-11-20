# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Unit tests for TimezoneExtraction plugin."""

import pytest
from datetime import datetime, timedelta

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
from iris.time import PartialDateTime

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.temporal import TimezoneExtraction
from improver.utilities.warnings_handler import ManageWarnings
from improver.metadata.check_datatypes import check_mandatory_standards
from improver.metadata.constants.time_types import TIME_COORDS


@pytest.mark.parametrize(
    "utc_times",
    (
        [datetime(2017, 11, 9, 12, 0)],
        [datetime(2017, 11, 9, 12, 0), datetime(2017, 11, 10, 12, 0)],
    ),
)
def test_create_output_cube(utc_times):
    """Tests that the create_output_cube method builds a cube with appropriate
    meta-data"""
    data_shape = [3, 3, 3]
    cube = set_up_variable_cube(
        np.zeros(data_shape).astype(np.float32),
        standard_grid_metadata="gl_ens",
        attributes={
            "institution": "unknown",
            "source": "IMPROVER",
            "title": "Unit test",
        },
    )
    cube = add_coordinate(
        cube,
        [datetime(2017, 11, 10, 4, 0) + timedelta(hours=h) for h in range(3)],
        "time",
        coord_units=TIME_COORDS["time"].units,
        dtype=TIME_COORDS["time"].dtype,
        is_datetime=True,
    )
    plugin = TimezoneExtraction()
    plugin.create_output_cube(cube, utc_times)
    result = plugin.output_cube
    assert isinstance(result, Cube)
    assert result.name() == cube.name()
    assert result.units == cube.units
    if len(utc_times) == 1:
        expected_shape = data_shape
    else:
        expected_shape = [len(utc_times)] + data_shape
    assert result.shape == tuple(expected_shape)
    assert result.attributes == cube.attributes
    check_mandatory_standards(result)
    result_time = plugin.time_points
    assert result_time.shape == (3, 3)
    assert np.ma.is_masked(result_time)
    assert result_time.mask.all()
    result_utc = result.coord("utc")
    assert [cell.point for cell in result_utc.cells()] == utc_times
    # assert result.coord_dims("time") == plugin.get_xy_dims(result)
