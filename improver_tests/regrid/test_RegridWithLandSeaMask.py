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
"""Unit tests for the RegridWithLandSeaMask class"""
# set up a special data set and corresponding land-sea mask info
# set up target grid and its land-sea mask info
# it is designed to cover different scenarios for regridding with land-sea
# the regridding reference results are manually checked for different methods


import iris
from iris.cube import Cube
from iris.coords import DimCoord
import numpy as np
import improver.cli as imcli

# function for creating cube from data, lats, lons
def create_cube(cube_array, lats, lons, name, unit):
    """
    create a simple cube from data and lat/lon coords
    args:
        cube_array (numpy.ndarray):
            data value (two dimensional)
        lats ( numpy.ndarray):
            latitude array
        lons ( numpy.ndarray):
            longitude array
        name (str):
            variable name
        unit (str):
            variable unit
    return:
         cube_v(iris.cube.Cube):
             new-created cube
    """
    cube_v = Cube(cube_array)
    # assume name
    cube_v.var_name = name
    # cube_v.standard_name= name
    cube_v.units = unit

    coord_lat = DimCoord(
        lats,
        standard_name="latitude",
        units="degrees",
        coord_system=iris.coord_systems.GeogCS(6371229),
    )
    coord_lon = DimCoord(
        lons,
        standard_name="longitude",
        units="degrees",
        coord_system=iris.coord_systems.GeogCS(6371229),
    )

    cube_v.add_dim_coord(coord_lat, 0)
    cube_v.add_dim_coord(coord_lon, 1)

    return cube_v


def define_source_target_grid_data():
    """ define cube_in, cube_in_mask,cube_out_mask using assumed data """
    # source (input) grid
    in_lats = np.linspace(0, 15, 4)
    in_lons = np.linspace(0, 40, 5)

    # target (output) grid
    out_lats = np.linspace(0, 14, 8)
    out_lons = np.linspace(5, 35, 11)

    # assume a set of nwp data
    data = np.arange(20).reshape(4, 5)

    # input grid mask info
    in_mask = np.empty((4, 5), dtype=int)
    in_mask[:, :] = 1
    in_mask[0, 2] = 0
    in_mask[2, 2:4] = 0
    in_mask[3, 2:4] = 0

    # output grid mask info
    out_mask = np.empty((8, 11), dtype=int)
    out_mask[:, :] = 1
    out_mask[0, 4:7] = 0
    out_mask[1, 5] = 0
    out_mask[5:9, 4:10] = 0
    out_mask[6, 6] = 1

    cube_in = create_cube(data, in_lats, in_lons, "air_temperature", "Celsius")
    cube_in_mask = create_cube(in_mask, in_lats, in_lons, "Land_Binary_Mask", "1")
    cube_out_mask = create_cube(out_mask, out_lats, out_lons, "Land_Binary_Mask", "1")

    return cube_in, cube_out_mask, cube_in_mask


def test_regrid_nearest_2():
    """Test nearest neighbour regridding"""

    cube_in, cube_out_mask, _ = define_source_target_grid_data()
    regrid_nearest = imcli.regrid.process(
        cube=cube_in,
        target_grid=cube_out_mask,
        regrid_mode="nearest-2",
        regridded_title="regridding with nearest neighbouring method",
    )
    expected_results = np.array(
        [
            [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            [5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8],
            [5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8],
            [10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13],
            [10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13],
            [10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13],
            [15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 18],
        ]
    )

    np.testing.assert_allclose(regrid_nearest.data, expected_results, atol=1e-3)


def test_regrid_bilinear_2():
    """Test bilinear regridding"""

    cube_in, cube_out_mask, _ = define_source_target_grid_data()
    regrid_bilinear = imcli.regrid.process(
        cube=cube_in,
        target_grid=cube_out_mask,
        regrid_mode="bilinear-2",
        regridded_title="regridding with bilinear method",
    )
    expected_results = np.array(
        [
            [0.50, 0.80, 1.10, 1.40, 1.70, 2.00, 2.30, 2.60, 2.90, 3.20, 3.5],
            [2.50, 2.80, 3.10, 3.40, 3.70, 4.00, 4.30, 4.60, 4.90, 5.20, 5.50],
            [4.50, 4.80, 5.10, 5.40, 5.70, 6.00, 6.30, 6.60, 6.90, 7.20, 7.50],
            [6.50, 6.80, 7.10, 7.40, 7.70, 8.00, 8.30, 8.60, 8.90, 9.20, 9.50],
            [8.50, 8.80, 9.10, 9.40, 9.70, 10.00, 10.30, 10.60, 10.90, 11.20, 11.50],
            [10.5, 10.80, 11.10, 11.40, 11.70, 12.0, 12.30, 12.60, 12.90, 13.20, 13.5],
            [
                12.50,
                12.80,
                13.10,
                13.40,
                13.70,
                14.00,
                14.30,
                14.60,
                14.90,
                15.20,
                15.50,
            ],
            [
                14.50,
                14.80,
                15.10,
                15.40,
                15.70,
                16.00,
                16.30,
                16.60,
                16.90,
                17.20,
                17.50,
            ],
        ]
    )

    np.testing.assert_allclose(regrid_bilinear.data, expected_results, atol=1e-3)


def test_regrid_nearest_with_mask_2():
    """Test nearest-with-mask-2 regridding"""

    cube_in, cube_out_mask, cube_in_mask = define_source_target_grid_data()
    regrid_nearest_with_mask = imcli.regrid.process(
        cube=cube_in,
        target_grid=cube_out_mask,
        land_sea_mask=cube_in_mask,
        regrid_mode="nearest-with-mask-2",
        regridded_title="regridding with nearest neighbouring with mask method",
    )

    expected_results = np.array(
        [
            [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            [0, 1, 1, 1, 7, 2, 7, 3, 3, 3, 3],
            [5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8],
            [5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9],
            [10, 11, 11, 11, 7, 7, 7, 8, 8, 8, 14],
            [10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14],
            [10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14],
            [15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19],
        ]
    )

    np.testing.assert_allclose(
        regrid_nearest_with_mask.data, expected_results, atol=1e-3
    )


def test_regrid_bilinear_with_mask_2():
    """Test bilinear-with-mask-2 regridding """

    cube_in, cube_out_mask, cube_in_mask = define_source_target_grid_data()
    regrid_bilinear_with_mask = imcli.regrid.process(
        cube=cube_in,
        target_grid=cube_out_mask,
        land_sea_mask=cube_in_mask,
        regrid_mode="bilinear-with-mask-2",
        regridded_title="regridding with bilinear-with-mask-2 method",
    )

    expected_results = np.array(
        [
            [
                0.5,
                0.80,
                1.40096027,
                3.29159665,
                2.0,
                2.0,
                2.0,
                4.94332907,
                3.25585669,
                3.20,
                3.5,
            ],
            [
                2.50,
                2.80,
                3.10,
                3.40,
                5.48911114,
                2.76267353,
                6.32926057,
                4.60,
                4.90,
                5.20,
                5.50,
            ],
            [4.50, 4.80, 5.10, 5.40, 5.70, 7.01540265, 6.30, 6.60, 6.90, 7.20, 7.50],
            [
                6.50,
                6.80,
                7.10,
                7.40,
                7.70,
                7.00,
                7.19032955,
                7.66809729,
                7.66179522,
                9.20,
                9.50,
            ],
            [
                8.50,
                8.80,
                9.10,
                9.40,
                8.106325,
                7.0,
                7.0,
                7.62914938,
                7.21672498,
                9.1143364,
                10.52362841,
            ],
            [
                10.5,
                10.80,
                11.00012407,
                11.01183078,
                13.15439245,
                12.0,
                12.30,
                12.60,
                12.90,
                13.71286288,
                15.74504042,
            ],
            [
                12.50,
                12.80,
                12.23411396,
                13.25881261,
                14.14155376,
                14.00,
                14.30,
                14.60,
                14.90,
                14.96331972,
                16.33340055,
            ],
            [
                14.50,
                14.80,
                15.09969542,
                14.22658959,
                15.50904876,
                16.00,
                16.30,
                16.60,
                16.90,
                16.91113956,
                17.03773564,
            ],
        ]
    )

    np.testing.assert_allclose(
        regrid_bilinear_with_mask.data, expected_results, atol=1e-3
    )
