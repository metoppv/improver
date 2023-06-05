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
"""Unit tests for psychrometric_calculations PrecipPhaseProbability plugin."""

import operator

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube

from improver.psychrometric_calculations.precip_phase_probability import (
    PrecipPhaseProbability,
)
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_variable_cube,
)

LOCAL_MANDATORY_ATTRIBUTES = {
    "title": "mandatory title",
    "source": "mandatory_source",
    "institution": "mandatory_institution",
}

DIM_LENGTH = 3
ALTITUDES = np.full((DIM_LENGTH, DIM_LENGTH), 75, dtype=np.float32)
FALLING_LEVEL_DATA = np.array(
    [np.full((DIM_LENGTH, DIM_LENGTH), 50), np.full((DIM_LENGTH, DIM_LENGTH), 100)],
    dtype=np.float32,
)
FALLING_LEVEL_DATA[0, 0, 0] = 75
FALLING_LEVEL_DATA[1, 0, 0] = 25


def check_metadata(result, phase):
    """
    Checks that the meta-data of the cube "result" are as expected.
    Args:
        result (iris.cube.Cube):
        phase (str):
            Used to construct the expected diagnostic name of the form
            probability_of_{phase}_at_surface

    """
    assert isinstance(result, iris.cube.Cube)
    assert result.name() == f"probability_of_{phase}_at_surface"
    assert result.units == Unit("1")
    assert result.dtype == np.int8
    assert result.attributes == LOCAL_MANDATORY_ATTRIBUTES


def spot_coords():
    """Define a set of coordinates for use in creating spot forecast or
    ancillary inputs."""
    n_sites = DIM_LENGTH * DIM_LENGTH
    altitudes = ALTITUDES.flatten()
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


@pytest.fixture
def gridded_falling_level_cube(ptype, phase) -> Cube:
    """Set up a (percentile), y, x cube of falling-level data. If ptype is
    deterministic the first percentile alone is returned."""

    falling_level_cube = set_up_percentile_cube(
        FALLING_LEVEL_DATA,
        [20, 80],
        units="m",
        spatial_grid="equalarea",
        name=f"altitude_of_{phase}_falling_level",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )

    if ptype == "deterministic":
        falling_level_cube = falling_level_cube[0]
        falling_level_cube.remove_coord("percentile")

    return falling_level_cube


@pytest.fixture
def spot_falling_level_cube(ptype, phase) -> Cube:
    """Set up a (percentile), y, x cube of falling-level data. If ptype is
    deterministic no percentile coordinate is created."""

    falling_level_data = FALLING_LEVEL_DATA.reshape((2, DIM_LENGTH * DIM_LENGTH))

    if ptype == "percentile":
        additional_dims = [DimCoord([20, 80], long_name="percentile", units="%")]
    else:
        additional_dims = None
        falling_level_data = falling_level_data[0]

    args, kwargs = spot_coords()
    kwargs.pop("neighbour_methods")
    kwargs.pop("grid_attributes")

    falling_level_cube = build_spotdata_cube(
        falling_level_data,
        f"altitude_of_{phase}_falling_level",
        "m",
        *args,
        **kwargs,
        additional_dims=additional_dims,
    )
    falling_level_cube.attributes = LOCAL_MANDATORY_ATTRIBUTES
    return falling_level_cube


@pytest.fixture
def gridded_altitude_cube() -> Cube:
    """Set up a cube of gridded altitudes that is equivalent to a gridded
    orography ancillary."""

    altitude_cube = set_up_variable_cube(
        ALTITUDES,
        name="surface_altitude",
        units="m",
        spatial_grid="equalarea",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    return altitude_cube


@pytest.fixture
def spot_altitude_cube() -> Cube:
    """Set up a cube of site specific altitudes that is equivalent to a spot
    neighbour ancillary."""

    n_sites = DIM_LENGTH * DIM_LENGTH
    neighbours = np.array(
        [[np.arange(0, n_sites), np.arange(0, n_sites), np.zeros(n_sites)]],
        dtype=np.float32,
    )
    args, kwargs = spot_coords()

    altitude_cube = build_spotdata_cube(
        neighbours, "grid_neighbours", 1, *args, **kwargs
    )
    return altitude_cube


@pytest.fixture(params=["spot", "gridded"])
def cube_inputs(
    request,
    gridded_falling_level_cube,
    gridded_altitude_cube,
    spot_falling_level_cube,
    spot_altitude_cube,
) -> tuple:
    """Return a tuple of cubes suitable for testing either spot or
    gridded forecasts depending upon the value of the parameter."""
    if request.param == "spot":
        return spot_falling_level_cube, spot_altitude_cube
    if request.param == "gridded":
        return gridded_falling_level_cube, gridded_altitude_cube


@pytest.mark.parametrize("ptype", ("deterministic", "percentile"))
@pytest.mark.parametrize("altitude_units", ("m", "metres", "feet"))
@pytest.mark.parametrize("phase", ("rain", "snow", "rain_from_hail"))
def test_probabilities(cube_inputs, phase, altitude_units, ptype):
    """Test that process returns a cube with the right name, shape, units and
    values when working with gridded or spot data. The altitudes are converted
    into various units to demonstrate that the plugin's unit conversion is
    working, returing the same result regardless. Included in these tests is
    the use of both 'm' and 'metres' to demonstrate that the use of these
    synonyms has no unexpected impact."""

    falling_level_cube, altitudes = cube_inputs

    # Construct expected values and then modify altitude units
    if phase == "snow":
        comparator = operator.le
        index = 1
    else:
        comparator = operator.gt
        index = 0

    if falling_level_cube.coords("wmo_id"):
        expected = comparator(
            falling_level_cube.data, altitudes.coord("altitude").points
        )
        altitudes.coord("altitude").convert_units(altitude_units)
    else:
        expected = comparator(falling_level_cube.data, altitudes.data)
        altitudes.convert_units(altitude_units)

    expected_shape = falling_level_cube.shape
    if ptype == "percentile":
        expected = expected[index]
        expected_shape = falling_level_cube[0].shape

    # Call plugin and check returned cube
    cubes = iris.cube.CubeList([falling_level_cube, altitudes])
    result = PrecipPhaseProbability().process(cubes)

    check_metadata(result, phase)
    assert result.attributes == LOCAL_MANDATORY_ATTRIBUTES
    assert (result.data == expected).all()
    assert result.shape == expected_shape


@pytest.mark.parametrize("ptype", ("deterministic", "percentile"))
@pytest.mark.parametrize("phase", ["rain"])
def test_multi_realizations(cube_inputs, phase, ptype):
    """Test that process returns a cube with the right name, shape, units and
    values when working with multi-realization gridded or spot data. In this
    case we expect the realization coordinate to be preserved. Only one phase
    is tested here as the behaviour is the same across phases."""

    falling_level_cube, altitudes = cube_inputs

    # Construct a multi-realization cube
    cubes = iris.cube.CubeList()
    for i in [0, 1]:
        cube = falling_level_cube.copy()
        cube.add_aux_coord(DimCoord([i], "realization", units=1))
        cubes.append(cube)
    falling_level_cube = cubes.merge_cube()

    # Construct expected values for rain phase
    comparator = operator.gt
    index = 0

    if falling_level_cube.coords("wmo_id"):
        expected = comparator(
            falling_level_cube.data, altitudes.coord("altitude").points
        )
    else:
        expected = comparator(falling_level_cube.data, altitudes.data)

    expected_shape = falling_level_cube.shape
    if ptype == "percentile":
        expected = expected[:, index]
        expected_shape = falling_level_cube[:, 0].shape

    # Call plugin and check returned cube
    cubes = iris.cube.CubeList([falling_level_cube, altitudes])
    result = PrecipPhaseProbability().process(cubes)

    check_metadata(result, phase)
    assert result.attributes == LOCAL_MANDATORY_ATTRIBUTES
    assert (result.data == expected).all()
    assert result.shape == expected_shape


@pytest.mark.parametrize("phase", ["kittens"])
@pytest.mark.parametrize("ptype", ["deterministic"])
def test_bad_phase_cube(gridded_falling_level_cube, gridded_altitude_cube):
    """Test that process raises an exception when the input phase cube is
    incorrectly named."""
    msg = "Could not extract a rain, rain from hail or snow falling-level cube from"
    cubes = iris.cube.CubeList([gridded_falling_level_cube, gridded_altitude_cube])

    with pytest.raises(ValueError, match=msg):
        PrecipPhaseProbability().process(cubes)


@pytest.mark.parametrize("phase", ["snow"])
@pytest.mark.parametrize("ptype", ["deterministic"])
def test_bad_orography_cube(gridded_falling_level_cube, gridded_altitude_cube):
    """Test that process raises an exception when the input orography
    cube is incorrectly named."""
    gridded_altitude_cube.rename("altitude_of_kittens")
    cubes = iris.cube.CubeList([gridded_falling_level_cube, gridded_altitude_cube])

    msg = "Could not extract surface_altitude cube from"
    with pytest.raises(ValueError, match=msg):
        PrecipPhaseProbability().process(cubes)


@pytest.mark.parametrize("phase", ["rain"])
@pytest.mark.parametrize("ptype", ["deterministic"])
def test_bad_units(gridded_falling_level_cube, gridded_altitude_cube):
    """Test that process raises an exception when the input cubes cannot
    be coerced into the same units."""
    gridded_altitude_cube.units = Unit("seconds")
    cubes = iris.cube.CubeList([gridded_falling_level_cube, gridded_altitude_cube])

    msg = "Unable to convert from "
    with pytest.raises(ValueError, match=msg):
        PrecipPhaseProbability().process(cubes)


@pytest.mark.parametrize("phase", ["rain"])
@pytest.mark.parametrize("ptype", ["deterministic"])
def test_spatial_mismatch(gridded_falling_level_cube):
    """Test that process raises an exception when the input cubes have
    different spatial coordinates."""

    altitude_cube = set_up_variable_cube(
        ALTITUDES,
        name="surface_altitude",
        units="m",
        spatial_grid="latlon",
        attributes=LOCAL_MANDATORY_ATTRIBUTES,
    )
    cubes = iris.cube.CubeList([gridded_falling_level_cube, altitude_cube])

    msg = "Spatial coords mismatch between"
    with pytest.raises(ValueError, match=msg):
        PrecipPhaseProbability().process(cubes)


@pytest.mark.parametrize("phase", ["rain", "snow"])
@pytest.mark.parametrize("ptype", ["percentile"])
def test_incorrect_percentiles(
    gridded_falling_level_cube, gridded_altitude_cube, phase
):
    """Test that process raises an exception when the input falling-level cube
    does not contain the expected percentile for a given phase."""
    if phase == "snow":
        gridded_falling_level_cube.coord("percentile").points = np.array(
            [20, 70], dtype=np.float32
        )
    elif phase == "rain":
        gridded_falling_level_cube.coord("percentile").points = np.array(
            [30, 80], dtype=np.float32
        )

    cubes = iris.cube.CubeList([gridded_falling_level_cube, gridded_altitude_cube])

    msg = f"Cube altitude_of_{phase}_falling_level does not contain the required percentile"
    with pytest.raises(ValueError, match=msg):
        PrecipPhaseProbability().process(cubes)
