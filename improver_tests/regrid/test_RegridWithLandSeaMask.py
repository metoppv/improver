# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube

import improver.cli as imcli
from improver.regrid.bilinear import basic_indexes
from improver.regrid.grid import convert_from_projection_to_latlons, latlon_from_cube


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
    out_mask[7, 6] = 1
    out_mask[1, 0] = 0

    cube_in = create_cube(data, in_lats, in_lons, "air_temperature", "Celsius")
    cube_in_mask = create_cube(in_mask, in_lats, in_lons, "Land_Binary_Mask", "1")
    cube_out_mask = create_cube(out_mask, out_lats, out_lons, "Land_Binary_Mask", "1")

    return cube_in, cube_out_mask, cube_in_mask


def define_source_target_grid_data_same_domain():
    """ define cube_in, cube_in_mask,cube_out_mask, assume the same domain  """
    # source (input) grid
    in_lats = np.linspace(0, 15, 4)
    in_lons = np.linspace(0, 40, 5)

    # target (output) grid
    out_lats = np.linspace(0, 15, 7)
    out_lons = np.linspace(5, 40, 9)

    # assume a set of nwp data
    data = np.arange(20).reshape(4, 5)

    # input grid mask info
    in_mask = np.empty((4, 5), dtype=int)
    in_mask[:, :] = 1
    in_mask[0, 2] = 0
    in_mask[2, 2:4] = 0
    in_mask[3, 2:4] = 0

    # output grid mask info
    out_mask = np.empty((7, 9), dtype=int)
    out_mask[:, :] = 1
    out_mask[0, 3:6] = 0
    out_mask[1, 4] = 0
    out_mask[4:9, 4:8] = 0
    out_mask[6, 6] = 1
    out_mask[1, 0] = 0

    cube_in = create_cube(data, in_lats, in_lons, "air_temperature", "Celsius")
    cube_in_mask = create_cube(in_mask, in_lats, in_lons, "Land_Binary_Mask", "1")
    cube_out_mask = create_cube(out_mask, out_lats, out_lons, "Land_Binary_Mask", "1")

    return cube_in, cube_out_mask, cube_in_mask


def create_cube_lambert():
    """ define a cube with LambertAzimuthalEqualArea system """
    cube_data = np.zeros([3, 3])
    cube_v = Cube(cube_data)
    coord_proj_y = DimCoord(
        np.array([-1036000.0, -1034000.0, -1032000.0], dtype=np.float32),
        bounds=np.array(
            [
                [-1037000.0, -1035000.0],
                [-1035000.0, -1033000.0],
                [-1033000.0, -1031000.0],
            ],
            dtype=np.float32,
        ),
        standard_name="projection_y_coordinate",
        units="metres",
        var_name="projection_y_coordinate",
        coord_system=iris.coord_systems.LambertAzimuthalEqualArea(
            latitude_of_projection_origin=54.9,
            longitude_of_projection_origin=-2.5,
            false_easting=0.0,
            false_northing=0.0,
            ellipsoid=iris.coord_systems.GeogCS(
                semi_major_axis=6378137.0, semi_minor_axis=6356752.314140356
            ),
        ),
    )

    coord_proj_x = DimCoord(
        np.array([-1158000.0, -1156000.0, -1154000.0], dtype=np.float32),
        bounds=np.array(
            [
                [-1159000.0, -1157000.0],
                [-1157000.0, -1155000.0],
                [-1155000.0, -1153000.0],
            ],
            dtype=np.float32,
        ),
        standard_name="projection_x_coordinate",
        units="metres",
        var_name="projection_x_coordinate",
        coord_system=iris.coord_systems.LambertAzimuthalEqualArea(
            latitude_of_projection_origin=54.9,
            longitude_of_projection_origin=-2.5,
            false_easting=0.0,
            false_northing=0.0,
            ellipsoid=iris.coord_systems.GeogCS(
                semi_major_axis=6378137.0, semi_minor_axis=6356752.314140356
            ),
        ),
    )

    cube_v.add_dim_coord(coord_proj_y, 0)
    cube_v.add_dim_coord(coord_proj_x, 1)

    return cube_v


def test_convert_from_projection_to_latlons():
    """Test convert_from_projection_to_latlons """
    cube_in, _, _ = define_source_target_grid_data()
    cube_lambert = create_cube_lambert()
    out_latlons = convert_from_projection_to_latlons(cube_lambert, cube_in)
    expected_results = np.array(
        [
            [44.517, -17.117],
            [44.520, -17.092],
            [44.524, -17.068],
            [44.534, -17.122],
            [44.538, -17.097],
            [44.542, -17.072],
            [44.552, -17.127],
            [44.556, -17.102],
            [44.560, -17.077],
        ]
    )

    np.testing.assert_allclose(out_latlons.data, expected_results, atol=1e-3)


def test_basic_indexes():
    """Test basic_indexes for identical source and target domain case """
    cube_in, cube_out_mask, _ = define_source_target_grid_data_same_domain()
    in_latlons = latlon_from_cube(cube_in)
    out_latlons = latlon_from_cube(cube_out_mask)
    in_lons_size = cube_in.coord(axis="x").shape[0]
    indexes = basic_indexes(out_latlons, in_latlons, in_lons_size)
    test_results = indexes[58:63, :]
    expected_results = np.array(
        [
            [12, 17, 18, 13],
            [12, 17, 18, 13],
            [13, 18, 19, 14],
            [13, 18, 19, 14],
            [13, 18, 19, 14],
        ]
    )
    np.testing.assert_array_equal(test_results, expected_results)


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
            [12.5, 12.8, 13.1, 13.4, 13.7, 14.0, 14.3, 14.6, 14.9, 15.2, 15.5],
            [14.5, 14.8, 15.1, 15.4, 15.7, 16.0, 16.3, 16.6, 16.9, 17.2, 17.5],
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
        # big grid size. just give a huge number for skipping search limit
        land_sea_mask_vicinity=250000000,
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
            [10, 11, 11, 11, 12, 12, 7, 13, 13, 13, 14],
            [15, 16, 16, 16, 17, 17, 7, 18, 18, 18, 19],
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
        # big grid size. just give a huge number for skipping search limit
        land_sea_mask_vicinity=250000000,
        regridded_title="regridding with bilinear-with-mask-2 method",
    )

    expected_results = np.array(
        [
            [
                0.5,
                0.80000001,
                1.40096027,
                3.29159665,
                2.0,
                2.0,
                2.0,
                4.94332907,
                3.25585669,
                3.20000005,
                3.5,
            ],
            [
                2.50000004,
                2.79999994,
                3.10000011,
                3.40000007,
                5.48911114,
                2.76267353,
                6.32926057,
                4.60000011,
                4.90000018,
                5.19999988,
                5.50000013,
            ],
            [
                4.50000007,
                4.79999989,
                5.09999994,
                5.40000008,
                5.69999993,
                7.01540265,
                6.29999994,
                6.6000001,
                6.89999992,
                7.19999984,
                7.50000011,
            ],
            [
                6.5000001,
                6.79999985,
                7.09999997,
                7.40000011,
                7.69999996,
                7.0,
                7.19032955,
                7.66809729,
                7.66179522,
                9.20000014,
                9.50000015,
            ],
            [
                8.50000028,
                8.7999998,
                9.10000034,
                9.4000003,
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
                10.80000016,
                11.00012407,
                11.01183078,
                13.15439245,
                12.0,
                12.30000001,
                12.60000038,
                12.89999971,
                13.71286288,
                15.74504042,
            ],
            [
                12.50000034,
                12.79999971,
                12.23411396,
                13.25881261,
                14.14155376,
                14.00000039,
                8.07328124,
                14.59999996,
                14.90000051,
                14.96331972,
                16.33340055,
            ],
            [
                14.50000022,
                14.79999967,
                15.09969542,
                14.22658959,
                15.50904876,
                16.00000024,
                9.87329583,
                16.59999963,
                16.90000057,
                16.91113956,
                17.03773564,
            ],
        ]
    )

    np.testing.assert_allclose(
        regrid_bilinear_with_mask.data, expected_results, atol=1e-3
    )
