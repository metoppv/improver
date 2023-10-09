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
"""Fixtures for blending tests"""

from datetime import datetime, timedelta
from typing import List, Tuple, Union
import numpy as np
from numpy import ndarray
import pytest

import iris
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    construct_scalar_time_coords,
)

from improver.blending import (
    MODEL_BLEND_COORD,
    MODEL_NAME_COORD,
    RECORD_COORD,
    WEIGHT_FORMAT,
)


LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "mandatory title",
    "source": "mandatory_source",
    "institution": "mandatory_institution",
}


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


@pytest.fixture
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
def model_cube_with_blend_record(model_blend_record_template) -> Cube:
    """Return a cube suitable for model blending which includes a blend
    record auxiliary coordinate. This is used to construct a record_run
    attribute."""
    cube = setup_model_cube()
    points = [
        item.format(uk_det_weight=0.5, uk_ens_weight=1 / 3, WEIGHT_FORMAT=WEIGHT_FORMAT)
        for item in model_blend_record_template
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


def spot_coords(n_sites):
    """Define a set of coordinates for use in creating spot forecast or
    ancillary inputs.

    Args:
        n_sites:
            The number of sites described by the coordinates.
    Returns:
        tuple:
            Containing a tuple and dict.
            The tuple contains the altitude, latitude, longitude, and
            wmo_id coordinate values.
            The dict contains the kwargs to use with the build_spotdata_cube
            function.
    """

    altitudes = np.arange(0, n_sites, 1, dtype=np.float32)
    latitudes = np.arange(0, n_sites * 10, 10, dtype=np.float32)
    longitudes = np.arange(0, n_sites * 20, 20, dtype=np.float32)
    wmo_ids = np.arange(1000, (1000 * n_sites) + 1, 1000)
    kwargs = {
        "unique_site_id": wmo_ids,
        "unique_site_id_key": "met_office_site_id",
        "grid_attributes": ["x_index", "y_index", "vertical_displacement"],
        "neighbour_methods": ["nearest"],
    }
    return (altitudes, latitudes, longitudes, wmo_ids), kwargs


def threshold_coord(diagnostic_name, thresholds, units):
    """Defined a threshold coordinate with the given name,
     threshold values, and units. Assumes a greater than
     threshold.

     Args:
         diagnostic_name:
             The name of the diagnostic, e.g. air_temperature
         thresholds:
             The threshold values as a list or array.
         units:
             The units of the threshold values.

     Returns:
         Threshold dimension coordinate.
     """

    crd = iris.coords.DimCoord(
        np.array(thresholds, dtype=np.float32),
        standard_name=diagnostic_name,
        units=units,
        var_name="threshold",
        attributes={"spp__relative_to_threshold": "greater_than_or_equal_to"},
    )
    return crd


def make_threshold_cube(n_sites, name, units, data, threshold_values, time, frt, model):
    """Create a spot threshold cube.

    Args:
        n_sites:
            Number of sites to create in the spot cube.
        name:
            The diagnostic name.
        data:
            The data to populate the cube with.
        threshold_values:
            The threshold values.
        time:
            The validity time of the data.
        frt:
            The forecast reference time of the data.
        model:
            A model identifier attribute.

    Returns:
        A spot data cube.
    """
    args, kwargs = spot_coords(n_sites)
    kwargs.pop("neighbour_methods")
    kwargs.pop("grid_attributes")
    threshold = threshold_coord(name, threshold_values, units)

    time_coords = construct_scalar_time_coords(time, frt=frt)

    cube_name = f"probability_of_{name}_above_threshold"
    cube_units = 1

    spot_data_cube = build_spotdata_cube(
        np.array(data, dtype=np.float32),
        cube_name,
        cube_units,
        *args,
        **kwargs,
        scalar_coords=[item[0] for item in time_coords],
        additional_dims=[threshold],
    )
    spot_data_cube.attributes = LOCAL_MANDATORY_ATTRIBUTES
    if model is not None:
        spot_data_cube.attributes.update({"mosg__model_configuration": model})
    return spot_data_cube


def make_neighbour_cube(n_sites, data):
    """Create a spot neighbour cube.

    Args:
        n_sites:
            Number of sites to create in the spot cube.
        data:
            The data to populate the cube with. This is grid point
            indices and vertical displacements.

    Returns:
        A spot neighbour cube.
    """
    args, kwargs = spot_coords(n_sites)
    cube_name = "grid_neighbours"
    cube_units = 1

    spot_neighbour_cube = build_spotdata_cube(
        np.array(data, dtype=np.float32), cube_name, cube_units, *args, **kwargs,
    )
    return spot_neighbour_cube


def create_spot_cubes(
    n_sites,
    ref_filter,
    mismatch_filter,
    time=datetime(2017, 11, 10, 4, 0),
    frt=datetime(2017, 11, 10, 0, 0),
    model="uk_det",
) -> Tuple[Cube, Cube, Cube]:
    """Set up a spot data cube with n_sites from a given model."""

    name = "air_temperature"
    units = "K"
    threshold_values = [273.15, 275.15]

    data = np.linspace(0, 0.9, 2 * n_sites, dtype=np.float32).reshape(2, n_sites)
    cube = make_threshold_cube(
        n_sites, name, units, data, threshold_values, time=time, frt=frt, model=model
    )

    data = np.ones(3 * n_sites).reshape(1, 3, n_sites)
    neighbours = make_neighbour_cube(n_sites, data)

    return cube[:, ref_filter], cube[:, mismatch_filter], neighbours[:, :, ref_filter]


@pytest.fixture
def spot_cubes(n_sites, ref_filter, mismatch_filter) -> Tuple[Cube, Cube, Cube]:
    """Call function to return spot data cubes with n_sites."""
    return create_spot_cubes(n_sites, ref_filter, mismatch_filter)


@pytest.fixture
def default_cubes() -> Tuple[Cube, Cube, Cube]:
    """Call function to return spot data cubes with n_sites."""
    return create_spot_cubes(5, slice(None), slice(None))


@pytest.fixture
def cycle_blend_spot_cubes(data) -> Tuple[Cube, Cube, Cube]:
    """Call function to return spot data cubes suitable for testing
    cycle blending.

    The input data must be a list of arrays shaped as (threshold, site_index).
    """

    name = "air_temperature"
    units = "K"
    time=datetime(2017, 11, 10, 6, 0)
    model = "uk_det"

    cubes = CubeList()
    frts = []
    for index, cube_data in enumerate(data):
        n_thresholds, n_sites = cube_data.shape
        threshold_values = np.arange(273.15, 273.15 + n_thresholds, 1)
        frts.append(time - timedelta(hours=index + 6))
        cube = make_threshold_cube(
                n_sites, name, units, cube_data, threshold_values, time=time, frt=frts[-1], model=model
            )
        if n_thresholds == 1:
            cube = iris.util.squeeze(cube)
        cubes.append(cube)

    return cubes, frts


@pytest.fixture
def model_blend_spot_cubes(models, leadtime) -> Tuple[Cube, Cube, Cube]:
    """Call function to return spot data cubes suitable for testing
    model blending.
    """

    name = "air_temperature"
    units = "K"
    frt = datetime(2017, 11, 10, 0, 0)
    time = frt + timedelta(hours=leadtime)

    cubes = CubeList()
    model_data = {
        "nc_det": 0.6,
        "uk_det": 0.4,
        "uk_ens": 0.2,
    }
    threshold_values = [273.15]
    n_sites = 3

    for model in models:
        data = np.full((1, n_sites), model_data[model])
        cube = make_threshold_cube(
                n_sites, name, units, data, threshold_values, time=time, frt=frt, model=model
            )
        cube = iris.util.squeeze(cube)
        cubes.append(cube)

    return cubes