# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
"""Test utilities to support weighted blending"""

from datetime import datetime

import iris
import numpy as np
import pytest

from improver.blending import MODEL_BLEND_COORD, MODEL_NAME_COORD
from improver.blending.utilities import (
    find_blend_dim_coord,
    get_coords_to_remove,
    set_record_run_attr,
    update_blended_metadata,
)
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube


@pytest.fixture(name="cycle_cube")
def cycle_cube_fixture():
    """Set up a cube for cycle blending"""
    thresholds = [10, 20]
    data = np.ones((2, 2, 2), dtype=np.float32)
    frt_list = [
        datetime(2017, 11, 10, 0),
        datetime(2017, 11, 10, 1),
        datetime(2017, 11, 10, 2),
    ]
    cycle_cubes = iris.cube.CubeList([])
    for frt in frt_list:
        cycle_cubes.append(
            set_up_probability_cube(
                data,
                thresholds,
                spatial_grid="equalarea",
                time=datetime(2017, 11, 10, 4, 0),
                frt=frt,
            )
        )
    return cycle_cubes.merge_cube()


@pytest.fixture(name="model_cube")
def model_cube_fixture():
    """Set up a cube for model blending"""
    thresholds = [10, 20]
    data = np.ones((2, 2, 2), dtype=np.float32)
    model_ids = [0, 1000]
    model_names = ["uk_det", "uk_ens"]
    model_cubes = iris.cube.CubeList([])
    for id, name in zip(model_ids, model_names):
        model_id_coord = iris.coords.AuxCoord([id], long_name=MODEL_BLEND_COORD)
        model_name_coord = iris.coords.AuxCoord([name], long_name=MODEL_NAME_COORD)
        model_cubes.append(
            set_up_probability_cube(
                data,
                thresholds,
                spatial_grid="equalarea",
                time=datetime(2017, 11, 10, 4),
                frt=datetime(2017, 11, 10, 1),
                include_scalar_coords=[model_id_coord, model_name_coord],
            )
        )
    return model_cubes.merge_cube()


@pytest.mark.parametrize(
    "input_coord_name", ("forecast_reference_time", "forecast_period")
)
def test_find_blend_dim_coord_noop(cycle_cube, input_coord_name):
    """Test no impact and returns correctly if called on dimension"""
    result = find_blend_dim_coord(cycle_cube, input_coord_name)
    assert result == "forecast_reference_time"


def test_find_blend_dim_coord_error_no_dim(cycle_cube):
    """Test error if blend coordinate has no dimension"""
    cube = next(cycle_cube.slices_over("forecast_reference_time"))
    with pytest.raises(ValueError, match="no associated dimension"):
        find_blend_dim_coord(cube, "forecast_reference_time")


def test_get_coords_to_remove(model_cube):
    """Test correct coordinates are identified for removal"""
    result = get_coords_to_remove(model_cube, MODEL_BLEND_COORD)
    assert set(result) == {MODEL_BLEND_COORD, MODEL_NAME_COORD}


def test_get_coords_to_remove_noop(cycle_cube):
    """Test time coordinates are not removed"""
    result = get_coords_to_remove(cycle_cube, "forecast_reference_time")
    assert not result


def test_update_blended_metadata(model_cube):
    """Test blended metadata is as expected"""
    blend_coord = "model_id"
    expected_attributes = {
        "mosg__model_configuration": "uk_det uk_ens",
        "title": "blend",
        "institution": MANDATORY_ATTRIBUTE_DEFAULTS["institution"],
        "source": MANDATORY_ATTRIBUTE_DEFAULTS["source"],
    }

    collapsed_cube = model_cube.collapsed(blend_coord, iris.analysis.MEAN)
    update_blended_metadata(
        collapsed_cube,
        blend_coord,
        coords_to_remove=[MODEL_BLEND_COORD, MODEL_NAME_COORD],
        cycletime="20171110T0200Z",
        model_id_attr="mosg__model_configuration",
        attributes_dict={"title": "blend"},
    )
    coord_names = [coord.name() for coord in collapsed_cube.coords()]
    assert MODEL_BLEND_COORD not in coord_names
    assert MODEL_NAME_COORD not in coord_names
    assert collapsed_cube.attributes == expected_attributes
    # check frt has been updated via fp proxy - input had 3 hours lead time,
    # output has 2 hours lead time relative to current cycle time
    assert collapsed_cube.coord("forecast_period").points[0] == 2 * 3600


@pytest.mark.parametrize(
    "indices, expected",
    (
        ([0, 1], "uk_det:20171110T0100Z:\nuk_ens:20171110T0100Z:"),
        ([0, 0], "uk_det:20171110T0100Z:"),
    ),
)
def test_set_record_run_attr_basic(model_cube, indices, expected):
    """Test use case where the record_run_attr is constructed from other
    information on the cubes. There are two tests here:

      - cubes with unique attributes that are combined
      - cubes with identical attributes attributes that are combined such that
        only a single entry is returned."""

    record_run_attr = "mosg__model_run"
    model_id_attr = "mosg__model_configuration"
    cubes = [model_cube[index] for index in indices]
    for cube in cubes:
        cube.attributes = {model_id_attr: cube.coord("model_configuration").points[0]}

    set_record_run_attr(cubes, record_run_attr, model_id_attr)

    for cube in cubes:
        assert cube.attributes[record_run_attr] == expected


@pytest.mark.parametrize(
    "existing, expected",
    (
        (
            ["uk_det:20171110T0100Z:", "uk_ens:20171110T0100Z:"],
            "uk_det:20171110T0100Z:\nuk_ens:20171110T0100Z:",
        ),
        (
            [
                "uk_det:20171110T0100Z:\nuk_ens:20171110T0100Z:",
                "uk_ens:20171110T0100Z:",
            ],
            "uk_det:20171110T0100Z:\nuk_ens:20171110T0100Z:",
        ),
        (
            ["uk_det:20171110T0100Z:", "uk_det:20171110T0100Z:"],
            "uk_det:20171110T0100Z:",
        ),
    ),
)
def test_set_record_run_attr_existing_attribute(model_cube, existing, expected):
    """Test the case in which the cubes already have record_run_attr entries
    and these must be combined to create a new shared attribute. There are three
    tests here:

      - cubes with unique model_run attributes that are combined
      - cubes with distinct but overlapping model_run attributes from which the
        elements are combined without duplicates
      - cubes with identical model_run attributes that are combined such that
        only a single entry is returned.

    The test cubes are only vehicles for the attributes in this test, such that
    the attributes imposed do not necessarily match the other cube metadata."""

    record_run_attr = "mosg__model_run"
    model_id_attr = "mosg__model_configuration"
    cubes = [model_cube[0], model_cube[1]]
    for run_attr, cube in zip(existing, cubes):
        cube.attributes.update({record_run_attr: run_attr})

    set_record_run_attr(cubes, record_run_attr, model_id_attr)

    for cube in cubes:
        assert cube.attributes[record_run_attr] == expected


def test_set_record_run_attr_mixed_inputs(model_cube):
    """Test use case where the record_run_attr is constructed from one cube
    with an existing record_run_attr, and one where other information on the
    cube is used."""

    record_run_attr = "mosg__model_run"
    model_id_attr = "mosg__model_configuration"
    cubes = [model_cube[0], model_cube[1]]
    cubes[0].attributes.update({record_run_attr: "uk_det:20171110T0000Z:"})
    cubes[1].attributes = {
        model_id_attr: cubes[1].coord("model_configuration").points[0]
    }
    expected = "uk_det:20171110T0000Z:\nuk_ens:20171110T0100Z:"

    set_record_run_attr(cubes, record_run_attr, model_id_attr)

    for cube in cubes:
        assert cube.attributes[record_run_attr] == expected


def test_set_record_run_attr_exception_model_id_unset(model_cube):
    """Test an exception is raised if the model_id_attr argument provided is
    none and the input cubes do not all have an existing record_run_attr."""
    record_run_attr = "mosg__model_run"
    model_id_attr = None
    cubes = [model_cube[0], model_cube[1]]

    with pytest.raises(Exception, match="Not all input cubes contain an existing"):
        set_record_run_attr(cubes, record_run_attr, model_id_attr)


def test_set_record_run_attr_exception_model_id(model_cube):
    """Test an exception is raised if no model_id_attr is set on the input
    cubes and no existing record_run_attr was present."""
    record_run_attr = "mosg__model_run"
    model_id_attr = "mosg__model_configuration"
    cubes = [model_cube[0], model_cube[1]]

    with pytest.raises(Exception, match="Failure to record run information"):
        set_record_run_attr(cubes, record_run_attr, model_id_attr)


def test_set_record_run_attr_exception_blended(model_cube):
    """Test an exception is raised if an input cube has been through model
    blending but has no record_run attribute. It is not possible to create a
    record_run attribute in this case as the source cycle information has been
    lost on this input cube."""
    record_run_attr = "mosg__model_run"
    model_id_attr = "mosg__model_configuration"
    cubes = [model_cube[0], model_cube[1]]
    blend_time = cubes[0].coord("forecast_reference_time").copy()
    blend_time.rename("blend_time")
    cubes[0].add_aux_coord(blend_time)

    with pytest.raises(Exception, match="This cube has been through model blending"):
        set_record_run_attr(cubes, record_run_attr, model_id_attr)
