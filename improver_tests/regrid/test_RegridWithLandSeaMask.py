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
"""Unit tests for the RegridWithLandSeaMask class"""
# set up a special data set and corresponding land-sea mask info
# set up target grid and its land-sea mask info
# it is designed to cover different scenarios for regridding with land-sea
# the regridding reference results are manually checked for different methods
# not using "set_up_variable_cube" because of different spacing at lat/lon

import numpy as np
import pytest

from improver.regrid.bilinear import basic_indexes
from improver.regrid.grid import calculate_input_grid_spacing, latlon_from_cube
from improver.regrid.landsea import RegridLandSea
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.pad_spatial import pad_cube_with_halo


def modify_cube_coordinate_value(cube, coord_x, coord_y):
    """modify x(longitude) & y(latitude) andcoordinates for a cube"""
    cube.coord(axis="x").points = coord_x
    cube.coord(axis="x").bounds = None
    cube.coord(axis="x").guess_bounds()
    cube.coord(axis="y").points = coord_y
    cube.coord(axis="y").bounds = None
    cube.coord(axis="y").guess_bounds()
    return cube


def define_source_target_grid_data():
    """ define cube_in, cube_in_mask,cube_out_mask using assumed data """
    # source (input) grid
    in_lats = np.linspace(0, 15, 4, dtype=np.float32)
    in_lons = np.linspace(0, 40, 5, dtype=np.float32)

    # target (output) grid
    out_lats = np.linspace(0, 14, 8, dtype=np.float32)
    out_lons = np.linspace(5, 35, 11, dtype=np.float32)

    # assume a set of nwp data
    data = np.arange(20).reshape(4, 5).astype(np.float32)

    # input grid mask info
    in_mask = np.empty((4, 5), dtype=np.int)
    in_mask[:, :] = 1
    in_mask[0, 2] = 0
    in_mask[2, 2:4] = 0
    in_mask[3, 2:4] = 0

    # output grid mask info
    out_mask = np.empty((8, 11), dtype=np.int)
    out_mask[:, :] = 1
    out_mask[0, 4:7] = 0
    out_mask[1, 5] = 0
    out_mask[5:9, 4:10] = 0

    out_mask[6, 6] = 1
    out_mask[7, 6] = 1
    out_mask[1, 0] = 0

    # create cube with default spacing
    cube_in = set_up_variable_cube(data, "air_temperature", "Celsius")
    cube_in_mask = set_up_variable_cube(in_mask, "Land_Binary_Mask", "1")
    cube_out_mask = set_up_variable_cube(out_mask, "Land_Binary_Mask", "1")

    # modify cube coordinates to the designed value
    cube_in = modify_cube_coordinate_value(cube_in, in_lons, in_lats)
    cube_in_mask = modify_cube_coordinate_value(cube_in_mask, in_lons, in_lats)
    cube_out_mask = modify_cube_coordinate_value(cube_out_mask, out_lons, out_lats)

    return cube_in, cube_out_mask, cube_in_mask


def define_source_target_grid_data_same_domain():
    """ define cube_in, cube_in_mask,cube_out_mask, assume the same domain  """
    # source (input) grid
    in_lats = np.linspace(0, 15, 4, dtype=np.float32)
    in_lons = np.linspace(0, 40, 5, dtype=np.float32)

    # target (output) grid
    out_lats = np.linspace(0, 15, 7, dtype=np.float32)
    out_lons = np.linspace(0, 40, 9, dtype=np.float32)

    # assume a set of nwp data
    data = np.arange(20).reshape(4, 5).astype(np.float32)

    # input grid mask info
    in_mask = np.empty((4, 5), dtype=np.int)
    in_mask[:, :] = 1
    in_mask[0, 2] = 0
    in_mask[2, 2:4] = 0
    in_mask[3, 2:4] = 0

    # output grid mask info
    out_mask = np.empty((7, 9), dtype=np.int)
    out_mask[:, :] = 1
    out_mask[0, 3:6] = 0
    out_mask[1, 4] = 0
    out_mask[4:9, 4:8] = 0
    out_mask[6, 6] = 1
    out_mask[1, 0] = 0

    # create cube with default spacing
    cube_in = set_up_variable_cube(data, "air_temperature", "Celsius")
    cube_in_mask = set_up_variable_cube(in_mask, "Land_Binary_Mask", "1")
    cube_out_mask = set_up_variable_cube(out_mask, "Land_Binary_Mask", "1")

    # modify cube coordinates to the designed value
    cube_in = modify_cube_coordinate_value(cube_in, in_lons, in_lats)
    cube_in_mask = modify_cube_coordinate_value(cube_in_mask, in_lons, in_lats)
    cube_out_mask = modify_cube_coordinate_value(cube_out_mask, out_lons, out_lats)

    return cube_in, cube_out_mask, cube_in_mask


def test_basic_indexes():
    """Test basic_indexes for identical source and target domain case """
    cube_in, cube_out_mask, _ = define_source_target_grid_data_same_domain()
    in_latlons = latlon_from_cube(cube_in)
    out_latlons = latlon_from_cube(cube_out_mask)
    in_lons_size = cube_in.coord(axis="x").shape[0]
    lat_spacing, lon_spacing = calculate_input_grid_spacing(cube_in)
    indexes = basic_indexes(
        out_latlons, in_latlons, in_lons_size, lat_spacing, lon_spacing
    )
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
    """Test nearest neighbour regridding option 'nearest-2'"""

    cube_in, cube_out_mask, _ = define_source_target_grid_data()
    regrid_nearest = RegridLandSea(regrid_mode="nearest-2",)(cube_in, cube_out_mask)
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
    """Test bilinear regridding option 'bilinear-2'"""

    cube_in, cube_out_mask, _ = define_source_target_grid_data()
    regrid_bilinear = RegridLandSea(regrid_mode="bilinear-2",)(cube_in, cube_out_mask)

    expected_results = np.array(
        [
            [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5],
            [2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5],
            [4.5, 4.8, 5.1, 5.4, 5.7, 6.0, 6.3, 6.6, 6.9, 7.2, 7.5],
            [6.5, 6.8, 7.1, 7.4, 7.7, 8.0, 8.3, 8.6, 8.9, 9.2, 9.5],
            [8.5, 8.8, 9.1, 9.4, 9.7, 10.0, 10.3, 10.6, 10.9, 11.2, 11.5],
            [10.5, 10.8, 11.1, 11.4, 11.7, 12.0, 12.3, 12.6, 12.9, 13.2, 13.5],
            [12.5, 12.8, 13.1, 13.4, 13.7, 14.0, 14.3, 14.6, 14.9, 15.2, 15.5],
            [14.5, 14.8, 15.1, 15.4, 15.7, 16.0, 16.3, 16.6, 16.9, 17.2, 17.5],
        ]
    )

    np.testing.assert_allclose(regrid_bilinear.data, expected_results, atol=1e-3)


def test_regrid_nearest_with_mask_2():
    """Test nearest-with-mask-2 regridding"""

    cube_in, cube_out_mask, cube_in_mask = define_source_target_grid_data()
    regrid_nearest_with_mask = RegridLandSea(
        regrid_mode="nearest-with-mask-2",
        landmask=cube_in_mask,
        landmask_vicinity=250000000,
    )(cube_in, cube_out_mask)

    expected_results = np.array(
        [
            [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            [0, 1, 1, 1, 7, 2, 7, 3, 3, 3, 3],
            [5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8],
            [5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8],
            [10, 11, 11, 11, 7, 7, 7, 8, 8, 8, 14],
            [10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14],
            [10, 11, 11, 11, 12, 12, 7, 13, 13, 13, 14],
            [15, 16, 16, 16, 17, 17, 7, 18, 18, 18, 19],
        ]
    )

    np.testing.assert_allclose(
        regrid_nearest_with_mask.data, expected_results, atol=1e-3
    )

    # consider constant field
    cube_in.data = np.repeat(1.0, 20).reshape(4, 5).astype(np.float32)
    regrid_nearest_with_mask = RegridLandSea(
        regrid_mode="nearest-with-mask-2",
        landmask=cube_in_mask,
        landmask_vicinity=250000000,
    )(cube_in, cube_out_mask)

    expected_results = np.repeat(1.0, 88).reshape(8, 11).astype(np.float32)
    np.testing.assert_allclose(
        regrid_nearest_with_mask.data, expected_results, atol=1e-3
    )


def test_regrid_bilinear_with_mask_2():
    """Test bilinear-with-mask-2 regridding """

    cube_in, cube_out_mask, cube_in_mask = define_source_target_grid_data()
    regrid_bilinear_with_mask = RegridLandSea(
        regrid_mode="bilinear-with-mask-2",
        landmask=cube_in_mask,
        landmask_vicinity=250000000,
    )(cube_in, cube_out_mask)

    expected_results = np.array(
        [
            [0.5, 0.8, 1.401, 3.292, 2.0, 2.0, 2.0, 4.943, 3.256, 3.2, 3.5],
            [2.5, 2.8, 3.1, 3.4, 5.489, 2.763, 6.329, 4.6, 4.9, 5.2, 5.5],
            [4.5, 4.8, 5.1, 5.4, 5.7, 6.985, 6.3, 6.6, 6.9, 7.2, 7.5],
            [6.5, 6.8, 7.1, 7.4, 7.7, 7.0, 7.19, 7.668, 7.662, 9.2, 9.5],
            [8.5, 8.8, 9.1, 9.4, 8.106, 7.0, 7.0, 7.629, 7.217, 9.114, 10.524],
            [10.5, 10.8, 11.0, 11.012, 13.154, 12.0, 12.3, 12.6, 12.9, 13.713, 15.745],
            [
                12.5,
                12.8,
                12.234,
                13.259,
                14.142,
                14.0,
                8.073,
                14.6,
                14.9,
                14.963,
                16.333,
            ],
            [14.5, 14.8, 15.1, 14.227, 15.509, 16.0, 9.873, 16.6, 16.9, 16.911, 17.038],
        ]
    )

    np.testing.assert_allclose(
        regrid_bilinear_with_mask.data, expected_results, atol=1e-3
    )

    # consider constant field
    cube_in.data = np.repeat(1.0, 20).reshape(4, 5).astype(np.float32)
    regrid_bilinear_with_mask = RegridLandSea(
        regrid_mode="bilinear-with-mask-2",
        landmask=cube_in_mask,
        landmask_vicinity=250000000,
    )(cube_in, cube_out_mask)

    expected_results = np.repeat(1.0, 88).reshape(8, 11).astype(np.float32)

    np.testing.assert_allclose(
        regrid_bilinear_with_mask.data, expected_results, atol=1e-3
    )


@pytest.mark.parametrize("regridder", ("nearest", "bilinear"))
@pytest.mark.parametrize("landmask", (True, False))
@pytest.mark.parametrize("maskedinput", (True, False))
def test_target_domain_bigger_than_source_domain(regridder, landmask, maskedinput):
    """Test regridding when target domain is bigger than source domain"""

    # set up source cube, target cube and land-sea mask cube
    cube_in, cube_out_mask, cube_in_mask = define_source_target_grid_data_same_domain()

    # add a circle of grid points so that output domain is much bigger than input domain
    width_x, width_y = 2, 4  # lon,lat
    cube_out_mask_pad = pad_cube_with_halo(cube_out_mask, width_x, width_y)

    if landmask:
        with_mask = "-with-mask"
    else:
        with_mask = ""
        cube_in_mask = None
    regrid_mode = f"{regridder}{with_mask}-2"

    if maskedinput:
        # convert the input data to a masked array with no values covered by the mask
        cube_in_masked_data = np.ma.masked_array(cube_in.data, mask=False)
        cube_in.data = cube_in_masked_data

    # run the regridding
    regridderLandSea = RegridLandSea(
        regrid_mode=regrid_mode, landmask=cube_in_mask, landmask_vicinity=250000000,
    )
    regrid_out = regridderLandSea(cube_in, cube_out_mask)
    regrid_out_pad = regridderLandSea(cube_in, cube_out_mask_pad)

    # check that results inside the padding matches the same regridding without padding
    np.testing.assert_allclose(
        regrid_out.data, regrid_out_pad.data[width_y:-width_y, width_x:-width_x],
    )

    # check results in the padded area
    if maskedinput:
        # masked array input should result in masked array output
        assert hasattr(regrid_out_pad.data, "mask")
        assert regrid_out_pad.dtype == np.float32
        regrid_out_pad.data.mask[width_y:-width_y, width_x:-width_x] = True
        np.testing.assert_array_equal(
            regrid_out_pad.data.mask,
            np.full_like(regrid_out_pad.data, True, dtype=np.bool),
        )
    else:
        assert not hasattr(regrid_out_pad.data, "mask")
        assert regrid_out_pad.dtype == np.float32
        # fill the area inside the padding with NaNs
        regrid_out_pad.data[width_y:-width_y, width_x:-width_x] = np.nan
        # this should result in the whole grid being NaN
        np.testing.assert_array_equal(
            regrid_out_pad.data, np.full_like(regrid_out_pad.data, np.nan)
        )
