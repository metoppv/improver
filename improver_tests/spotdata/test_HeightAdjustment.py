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
"""Unit tests for SpotHeightAdjustment plugin"""

import iris
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.spotdata.height_adjustment import SpotHeightAdjustment
from improver.utilities.cube_manipulation import enforce_coordinate_ordering

name = "cloud_base_height_assuming_only_consider_cloud_area_fraction_greater_than_4p5_oktas"


@pytest.fixture()
def prob_cube() -> Cube:
    """Set up a spot data cube of probabilities"""
    altitude = np.array([256.5, 359.1, 301.8, 406.2])
    latitude = np.linspace(58.0, 59.5, 4)
    longitude = np.linspace(-0.25, 0.5, 4)
    wmo_id = ["03854", "03962", "03142", "03331"]
    threshold_coord = DimCoord(
        points=[50, 100, 1000], var_name="threshold", long_name=name, units="m",
    )

    data = np.asarray([[0.1, 0, 0.2, 0.1], [0.7, 0, 0.3, 0.2], [1, 0.4, 0.4, 0.9]])

    prob_cube = build_spotdata_cube(
        data,
        name="probability_of_" + name + "_below_threshold",
        units="1",
        altitude=altitude,
        latitude=latitude,
        longitude=longitude,
        wmo_id=wmo_id,
        additional_dims=[threshold_coord],
    )
    enforce_coordinate_ordering(prob_cube, ["spot_index", name])
    return prob_cube


@pytest.fixture()
def prob_cube_realizations(prob_cube) -> Cube:
    """Set up a spot data cube of probabilities with a realization coordinate"""
    realization0 = iris.coords.DimCoord([0], standard_name="realization", units=1)
    realization1 = iris.coords.DimCoord([1], standard_name="realization", units=1)

    cube0 = prob_cube.copy()
    cube1 = prob_cube.copy()
    cube0.add_aux_coord(realization0)
    cube1.add_aux_coord(realization1)
    cubes = iris.cube.CubeList([cube0, cube1])
    cube = cubes.merge_cube()
    return cube


@pytest.fixture()
def realization_cube() -> Cube:
    """Set up a spot data cube with a realization coordinate. The units of this cube
    are set to feet so unit conversion can be tested within the plugin"""
    altitude = np.array([256.5, 359.1, 301.8, 406.2])
    latitude = np.linspace(58.0, 59.5, 4)
    longitude = np.linspace(-0.25, 0.5, 4)
    wmo_id = ["03854", "03962", "03142", "03331"]
    realization_coord = DimCoord(points=[0, 1, 2], var_name="realization", units="1")

    data = np.asarray(
        [[1000, 4000, -200, 100], [3000, 10000, 0, 200], [4000, 11000, 30, 150]]
    )

    rea_cube = build_spotdata_cube(
        data,
        name=name,
        units="ft",
        altitude=altitude,
        latitude=latitude,
        longitude=longitude,
        wmo_id=wmo_id,
        additional_dims=[realization_coord],
    )
    return rea_cube


@pytest.fixture()
def percentile_cube() -> Cube:
    """Set up a spot data cube with a percentile coordinate"""
    altitude = np.array([256.5, 359.1, 301.8, 406.2])
    latitude = np.linspace(58.0, 59.5, 4)
    longitude = np.linspace(-0.25, 0.5, 4)
    wmo_id = ["03854", "03962", "03142", "03331"]
    percentile_coord = DimCoord(points=[25, 50, 75], var_name="percentile", units="%")

    data = np.asarray(
        [[1400, 4000, -500, 100], [2000, 10000, 0, 100], [3000, 11000, 30, 150]]
    )

    perc_cube = build_spotdata_cube(
        data,
        name=name,
        units="m",
        altitude=altitude,
        latitude=latitude,
        longitude=longitude,
        wmo_id=wmo_id,
        additional_dims=[percentile_coord],
    )
    return perc_cube


@pytest.fixture()
def neighbour_cube() -> Cube:
    """Set up a neighbour cube with vertical displacement coordinate"""
    neighbours = np.array([[[0.0, -100.0, 0.0, 100.0]]])

    altitudes = np.array([0, 1, 3, 2])
    latitudes = np.array([10, 10, 20, 20])
    longitudes = np.array([10, 10, 20, 20])
    wmo_ids = np.arange(4)
    neighbour_cube = build_spotdata_cube(
        neighbours,
        "grid_neighbours",
        1,
        altitudes,
        latitudes,
        longitudes,
        wmo_ids,
        neighbour_methods=["nearest"],
        grid_attributes=["vertical_displacement"],
    )
    return neighbour_cube


@pytest.mark.parametrize("order", (True, False))
@pytest.mark.parametrize(
    "cube_name, expected",
    (
        (
            "prob_cube",
            [
                [0.1, 0.7, 1],
                [0, 0, 0.35555556],
                [0.2, 0.3, 0.4],
                [0.23888889, 0.27777778, 0.9],
            ],
        ),
        (
            "prob_cube_realizations",
            [
                [
                    [0.1, 0.7, 1],
                    [0, 0, 0.35555556],
                    [0.2, 0.3, 0.4],
                    [0.23888889, 0.27777778, 0.9],
                ],
                [
                    [0.1, 0.7, 1],
                    [0, 0, 0.35555556],
                    [0.2, 0.3, 0.4],
                    [0.23888889, 0.27777778, 0.9],
                ],
            ],
        ),
        (
            "realization_cube",
            [
                [1000, 3671.91601, -200.0, 428.08399],
                [3000.0, 9671.91601, 0.0, 528.08399],
                [4000.0, 10671.91601, 30.0, 478.08399],
            ],
        ),
        (
            "percentile_cube",
            [[1400, 3900, -500, 200], [2000, 9900, 0, 200], [3000, 10900, 30, 250]],
        ),
    ),
)
def test_cube(cube_name, expected, neighbour_cube, order, request):
    """Tests plugin produces correct metadata and results for different input cubes"""
    cube = request.getfixturevalue(cube_name)
    coords_original = [c.name() for c in cube.dim_coords]
    if order:
        enforce_coordinate_ordering(
            cube, [name, "realization", "percentile", "spot_index"]
        )

    coords_enforced = [c.name() for c in cube.dim_coords]
    cube_units = cube.units
    cube_title = cube.name()
    try:
        threshold_coord = cube.coord(name)
    except CoordinateNotFoundError:
        pass
    result = SpotHeightAdjustment()(cube, neighbour_cube)
    coords_result = [c.name() for c in result.dim_coords]

    if order:
        enforce_coordinate_ordering(result, coords_original)
    assert coords_result == coords_enforced
    assert result.units == cube_units
    assert result.name() == cube_title
    if cube_name == ("prob_cube" or "prob_cube_realizations"):
        assert result.coord(name) == threshold_coord
    np.testing.assert_allclose(result.data, expected)


def test_prob_cube_threshold_unit(prob_cube, neighbour_cube):
    """Tests that correct units and data are returned if probability cubes threshold is converted
    to metres"""
    prob_cube.coord(name).units = "km"
    expected = [
        [0.1, 0.7, 1.0],
        [0, 0, 0.3999556],
        [0.2, 0.3, 0.4],
        [0.1002, 0.2000778, 0.9],
    ]
    cube_units = prob_cube.coord(name).units

    result = SpotHeightAdjustment()(prob_cube, neighbour_cube)
    assert result.coord(name).units == cube_units
    assert result.coord(name).units == "km"
    np.testing.assert_almost_equal(result.data, expected)


def test_insufficient_thresholds(prob_cube, neighbour_cube):
    """Tests an error is raised if there are insufficient thresholds"""
    cube = next(prob_cube.slices_over(name))
    with pytest.raises(ValueError, match="There are fewer than 2 thresholds"):
        SpotHeightAdjustment()(cube, neighbour_cube)
