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
"""Test utilities to support weighted blending"""

from datetime import datetime
from typing import List, Union

import iris
import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver.blending import (
    MODEL_BLEND_COORD,
    MODEL_NAME_COORD,
    RECORD_COORD,
    WEIGHT_FORMAT,
)
from improver.blending.utilities import (
    find_blend_dim_coord,
    get_coords_to_remove,
    record_run_coord_to_attr,
    store_record_run_as_coord,
    update_blended_metadata,
    update_record_run_weights,
)
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube


def setup_cycle_cube() -> Cube:
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
                attributes={"mosg__model_configuration": "uk_det"},
            )
        )
    return cycle_cubes.merge_cube()


@pytest.fixture
def cycle_cube() -> Cube:
    """Return a cube suitable for cycle blending."""
    return setup_cycle_cube()


@pytest.fixture
def cycle_cube_with_blend_record() -> Cube:
    """Return a cube suitable for cycle blending which includes a blend
    record auxiliary coordinate. This is used to construct a record_run
    attribute."""
    cubes = setup_cycle_cube()
    updated_cubes = CubeList()
    for cube in cubes.slices_over("forecast_reference_time"):
        time = (
            cube.coord("forecast_reference_time").cell(0).point.strftime("%Y%m%dT%H%MZ")
        )

        blend_record_coord = AuxCoord(
            [f"uk_det:{time}:{1:{WEIGHT_FORMAT}}"], long_name=RECORD_COORD
        )
        cube.add_aux_coord(blend_record_coord)
        updated_cubes.append(cube)
    return updated_cubes.merge_cube()


def create_weights_cube(
    cube: Cube, blending_coord: str, weights: Union[List, ndarray]
) -> Cube:
    """Creates a weights cube using the provided weights. These weights are
    associated with the provided blending_coord."""
    weights_cube = next(cube.slices(blending_coord))
    weights_cube.attributes = None
    blending_dim = cube.coord_dims(blending_coord)
    defunct_coords = [
        crd.name()
        for crd in cube.coords(dim_coords=True)
        if not cube.coord_dims(crd) == blending_dim
    ]
    for crd in defunct_coords:
        weights_cube.remove_coord(crd)
    weights_cube.data = weights
    weights_cube.rename("weights")
    weights_cube.units = 1

    return weights_cube


@pytest.fixture
def cycle_blending_weights(weights: Union[List, ndarray]) -> Cube:
    """Using a template cube that is constructed for cycle blending,
    create a weights cube that can be applied to it. The weights
    within the cube are those provided."""
    cube = setup_cycle_cube()
    blending_coord = "forecast_reference_time"
    return create_weights_cube(cube, blending_coord, weights)


def setup_model_cube() -> Cube:
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


@pytest.fixture
def model_cube() -> Cube:
    """Return a cube suitable for model blending."""
    return setup_model_cube()


def model_blend_record_template() -> List[str]:
    """Return blend_record template entries for a cycle blended uk_det cube
    and a cycle blended uk_ens cube. Two template entries are returned that
    can be formatted with the required weights."""
    return [
        "uk_det:20171110T0000Z:{uk_det_weight:{WEIGHT_FORMAT}}\n"
        "uk_det:20171110T0100Z:{uk_det_weight:{WEIGHT_FORMAT}}",
        "uk_ens:20171109T2300Z:{uk_ens_weight:{WEIGHT_FORMAT}}\n"
        "uk_ens:20171110T0000Z:{uk_ens_weight:{WEIGHT_FORMAT}}\n"
        "uk_ens:20171110T0100Z:{uk_ens_weight:{WEIGHT_FORMAT}}",
    ]


@pytest.fixture
def model_cube_with_blend_record() -> Cube:
    """Return a cube suitable for model blending which includes a blend
    record auxiliary coordinate. This is used to construct a record_run
    attribute."""
    cube = setup_model_cube()
    points = [
        item.format(uk_det_weight=0.5, uk_ens_weight=1 / 3, WEIGHT_FORMAT=WEIGHT_FORMAT)
        for item in model_blend_record_template()
    ]
    blend_record_coord = AuxCoord(points, long_name=RECORD_COORD)
    cube.add_aux_coord(blend_record_coord, 0)
    return cube


@pytest.fixture
def model_blending_weights(weights: Union[List, ndarray]) -> Cube:
    """Using a template cube that is constructed for model blending,
    create a weights cube that can be applied to it. The weights
    within the cube are those provided."""
    cube = setup_model_cube()
    blending_coord = MODEL_BLEND_COORD
    return create_weights_cube(cube, blending_coord, weights)


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


def test_get_coords_to_remove(model_cube_with_blend_record):
    """Test correct coordinates are identified for removal for a cube with
    a non-scalar blend coordinate. This should include the temporary
    RECORD_COORD which is added to test its identification."""
    result = get_coords_to_remove(model_cube_with_blend_record, MODEL_BLEND_COORD)
    assert set(result) == {RECORD_COORD, MODEL_BLEND_COORD, MODEL_NAME_COORD}


def test_get_coords_to_remove_noop(cycle_cube):
    """Test time coordinates are not removed"""
    result = get_coords_to_remove(cycle_cube, "forecast_reference_time")
    assert not result


def test_get_coords_to_remove_scalar(model_cube, model_cube_with_blend_record):
    """Test correct coordinates are identified for removal for a cube with
    a scalar blend coordinate. If a RECORD_COORD exists this must also be
    listed for removal."""

    # Scalar cube, model-blending, no RECORD_COORD
    result = get_coords_to_remove(model_cube[0], MODEL_BLEND_COORD)
    assert result == [MODEL_BLEND_COORD, MODEL_NAME_COORD]
    # Scalar cube, non-model-blending, no RECORD_COORD
    result = get_coords_to_remove(model_cube[0], "time")
    assert result == []

    # Scalar cube, model blending, RECORD_COORD
    result = get_coords_to_remove(model_cube_with_blend_record[0], MODEL_BLEND_COORD)
    assert result == [RECORD_COORD, MODEL_BLEND_COORD, MODEL_NAME_COORD]
    # Scalar cube, non-model-blending, RECORD_COORD
    result = get_coords_to_remove(model_cube_with_blend_record[0], "time")
    assert result == [RECORD_COORD]


@pytest.mark.parametrize("forecast_period", (True, False))
def test_update_blended_metadata(model_cube, forecast_period):
    """Test blended metadata is as expected. Parameterizes to test that if
    the input has no forecast_period coordinate, none is created by the
    method."""
    blend_coord = "model_id"
    expected_attributes = {
        "mosg__model_configuration": "uk_det uk_ens",
        "title": "blend",
        "institution": MANDATORY_ATTRIBUTE_DEFAULTS["institution"],
        "source": MANDATORY_ATTRIBUTE_DEFAULTS["source"],
    }

    if not forecast_period:
        model_cube.remove_coord("forecast_period")

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
    if forecast_period:
        assert collapsed_cube.coord("forecast_period").points[0] == 2 * 3600


def test_store_record_run_as_coord_basic(model_cube):
    """Test use case where the record_run_attr is constructed from other
    information on the cubes. The resulting cubes have an additional
    auxiliary RECORD_COORD that stores the cycle and model information."""

    record_run_attr = "mosg__model_run"
    model_id_attr = "mosg__model_configuration"
    cubes = [model_cube[0], model_cube[1]]
    for cube in cubes:
        cube.attributes = {model_id_attr: cube.coord("model_configuration").points[0]}

    expected_points = [["uk_det:20171110T0100Z:1.000"], ["uk_ens:20171110T0100Z:1.000"]]

    store_record_run_as_coord(cubes, record_run_attr, model_id_attr)

    for cube, expected in zip(cubes, expected_points):
        assert cube.coord(RECORD_COORD).points == expected


@pytest.mark.parametrize(
    "attributes",
    (
        [
            ["uk_det:20171110T0100Z:0.500", "uk_ens:20171110T0100Z:0.500"],
            [
                "uk_det:20171110T0100Z:0.500\nuk_ens:20171110T0100Z:0.500",
                "uk_ens:20171110T0100Z:1.000",
            ],
        ]
    ),
)
def test_store_record_run_as_coord_existing_attribute(model_cube, attributes):
    """Test the case in which the cubes already have record_run_attr entries.
    In this case the existing attribute is simply stored as it is within the
    RECORD_COORD.

    The test cubes are only vehicles for the attributes in this test, such that
    the attributes imposed do not necessarily match the other cube metadata."""

    record_run_attr = "mosg__model_run"
    model_id_attr = "mosg__model_configuration"
    cubes = [model_cube[0], model_cube[1]]
    for run_attr, cube in zip(attributes, cubes):
        cube.attributes.update({record_run_attr: run_attr})

    store_record_run_as_coord(cubes, record_run_attr, model_id_attr)

    for cube, expected in zip(cubes, attributes):
        assert cube.coord(RECORD_COORD).points == expected


def test_store_record_run_as_coord_mixed_inputs(model_cube):
    """Test use case where the record_run_attr is constructed from one cube
    with an existing record_run_attr, and one where other information on the
    cube is used."""

    record_run_attr = "mosg__model_run"
    model_id_attr = "mosg__model_configuration"
    cubes = [model_cube[0], model_cube[1]]
    cubes[0].attributes.update({record_run_attr: "uk_det:20171110T0000Z:1.000"})
    cubes[1].attributes = {
        model_id_attr: cubes[1].coord("model_configuration").points[0]
    }
    expected_points = [["uk_det:20171110T0000Z:1.000"], ["uk_ens:20171110T0100Z:1.000"]]

    store_record_run_as_coord(cubes, record_run_attr, model_id_attr)

    for cube, expected in zip(cubes, expected_points):
        assert cube.coord(RECORD_COORD).points == expected


def test_store_record_run_as_coord_exception_model_id_unset(model_cube):
    """Test an exception is raised if the model_id_attr argument provided is
    none and the input cubes do not all have an existing record_run_attr."""
    record_run_attr = "mosg__model_run"
    model_id_attr = None
    cubes = [model_cube[0], model_cube[1]]

    with pytest.raises(Exception, match="Not all input cubes contain an existing"):
        store_record_run_as_coord(cubes, record_run_attr, model_id_attr)


def test_store_record_run_as_coord_exception_model_id(model_cube):
    """Test an exception is raised if no model_id_attr is set on the input
    cubes and no existing record_run_attr was present."""
    record_run_attr = "mosg__model_run"
    model_id_attr = "mosg__model_configuration"
    cubes = [model_cube[0], model_cube[1]]

    with pytest.raises(Exception, match="Failure to record run information"):
        store_record_run_as_coord(cubes, record_run_attr, model_id_attr)


@pytest.mark.parametrize("input_type", (Cube, list, CubeList))
def test_record_run_coord_to_attr_basic(
    model_cube, model_cube_with_blend_record, input_type
):
    """Test that the apply record method can take a RECORD_COORD from a source
    cube, or RECORD_COORDs from a list of cubes, and construct a record run
    attribute on a target cube."""
    expected = (
        "uk_det:20171110T0000Z:0.500\nuk_det:20171110T0100Z:0.500\n"
        "uk_ens:20171109T2300Z:0.333\nuk_ens:20171110T0000Z:0.333\n"
        "uk_ens:20171110T0100Z:0.333"
    )

    record_run_attr = "mosg__model_run"
    if input_type is Cube:
        input_object = model_cube_with_blend_record
    else:
        input_object = input_type(
            model_cube_with_blend_record.slices_over(MODEL_BLEND_COORD)
        )
    record_run_coord_to_attr(model_cube, input_object, record_run_attr)
    assert model_cube.attributes[record_run_attr] == expected


def test_record_run_coord_to_attr_discard_weights(
    model_cube, model_cube_with_blend_record
):
    """Test that the apply record method can remove weights from the resulting
    attribute if so desired. This is for composite diagnostics like weather
    symbols where the weights have little meaning."""

    expected = (
        "uk_det:20171110T0000Z:\nuk_det:20171110T0100Z:\n"
        "uk_ens:20171109T2300Z:\nuk_ens:20171110T0000Z:\n"
        "uk_ens:20171110T0100Z:"
    )
    record_run_attr = "mosg__model_run"
    record_run_coord_to_attr(
        model_cube, model_cube_with_blend_record, record_run_attr, discard_weights=True
    )

    assert model_cube.attributes[record_run_attr] == expected


def test_record_run_coord_to_attr_discard_weights_no_duplicates(
    model_cube, model_cube_with_blend_record
):
    """Test that the apply record method can remove weights from the resulting
    attribute if so desired. If this results in any duplicates (as weights were
    the differentiating feature) these should be removed."""

    record_run_attr = "mosg__model_run"
    cube = model_cube_with_blend_record
    cube.coord(RECORD_COORD).points = [
        "uk_det:20171110T0000Z:0.750",
        "uk_det:20171110T0000Z:0.250",
    ]

    expected = "uk_det:20171110T0000Z:"

    record_run_coord_to_attr(model_cube, cube, record_run_attr, discard_weights=True)

    assert model_cube.attributes[record_run_attr] == expected


@pytest.mark.parametrize(
    "weights",
    [
        [1 / 3, 1 / 3, 1 / 3],  # Evenly weighted
        [0.25, 0.25, 0.5],  # Unevenly weighted
        [1, 1, 1]  # Overweighted. This method does nothing to prevent this; relies on
        # sensible input weights.
    ],
)
def test_update_record_run_weights_cycle(
    cycle_cube_with_blend_record, cycle_blending_weights, weights
):
    """Test that weights are updated as expected in a virgin RECORD_COORD
    in which all weights are 1. The returned cube should have otherwise identical
    data and metadata to the input cube."""

    frts = [
        cube.coord("forecast_reference_time").cell(0).point.strftime("%Y%m%dT%H%MZ")
        for cube in cycle_cube_with_blend_record.slices_over("forecast_reference_time")
    ]

    expected = [
        f"uk_det:{frt}:{weight:{WEIGHT_FORMAT}}" for frt, weight in zip(frts, weights)
    ]

    result = update_record_run_weights(
        cycle_cube_with_blend_record, cycle_blending_weights, "forecast_reference_time"
    )
    assert isinstance(result, Cube)
    # Check weights updates as expected.
    assert list(result.coord(RECORD_COORD).points) == expected
    # Check data, metadata, and other coords are unmodified.
    assert (cycle_cube_with_blend_record.data == result.data).all()
    assert cycle_cube_with_blend_record.metadata == result.metadata
    for coord in result.coords():
        if coord.name() == RECORD_COORD:
            continue
        assert coord == cycle_cube_with_blend_record.coord(coord.name())


@pytest.mark.parametrize(
    "weights",
    [
        [0.5, 0.5],  # Evenly weighted (uk_det=0.5, uk_ens=0.5)
        [0.25, 0.75],  # Unevenly weighted (uk_det=0.25, uk_ens=0.75)
        [0.0001, 0.9999],  # A contribution so low that we end up with a zero weight
        # (uk_det=0.0001, uk_ens=0.9999)
    ],
)
def test_update_record_run_weights_model(
    model_cube_with_blend_record, model_blending_weights, weights
):
    """Test that weights are updated as expected in a model blend cube where
    the RECORD_COORD has been constructed from the record_run attributes of
    the separate model inputs. The uk_det input is a blend of two cycles with
    equal weights (0.5 each), and the uk_ens input is a blend of three cycles
    with equal weights (1/3 each).

    The final test demonstrates that it is possbile to end up with a zero
    weight in the attribute if the contribution falls below what is
    representable at the attribute precision."""

    uk_det_final_weight = 0.5 * weights[0]
    uk_ens_final_weight = (1 / 3) * weights[1]

    expected = [
        item.format(
            uk_det_weight=uk_det_final_weight,
            uk_ens_weight=uk_ens_final_weight,
            WEIGHT_FORMAT=WEIGHT_FORMAT,
        )
        for item in model_blend_record_template()
    ]

    result = update_record_run_weights(
        model_cube_with_blend_record, model_blending_weights, MODEL_BLEND_COORD
    )

    assert isinstance(result, Cube)
    # Check weights updates as expected.
    assert list(result.coord(RECORD_COORD).points) == expected
    # Check data, metadata, and other coords are unmodified.
    assert (model_cube_with_blend_record.data == result.data).all()
    assert model_cube_with_blend_record.metadata == result.metadata
    for coord in result.coords():
        if coord.name() == RECORD_COORD:
            continue
        assert coord == model_cube_with_blend_record.coord(coord.name())
