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
"""Module for saving netcdf cubes with desired attribute types."""

import os
import warnings
from typing import Optional, Union

import cf_units
import iris
from iris.cube import Cube, CubeList

from improver.metadata.check_datatypes import check_mandatory_standards


def _order_cell_methods(cube: Cube) -> None:
    """
    Sorts the cell methods on a cube such that if there are multiple methods
    they are always written in a consistent order in the output cube. The
    input cube is modified.

    Args:
        cube:
            The cube on which the cell methods are to be sorted.
    """
    cell_methods = tuple(sorted(cube.cell_methods))
    cube.cell_methods = cell_methods


def _check_metadata(cube: Cube) -> None:
    """
    Checks cube metadata that needs to be correct to guarantee data integrity

    Args:
        cube:
            Cube to be checked

    Raises:
        ValueError: if time coordinates do not have the required datatypes
            and units; needed because values may be wrong
        ValueError: if numerical datatypes are other than 32-bit (except
            where specified); needed because values may be wrong
        ValueError: if cube dataset has unknown units; because this may cause
            misinterpretation on "load"
    """
    check_mandatory_standards(cube)
    if cf_units.Unit(cube.units).is_unknown():
        raise ValueError("{} has unknown units".format(cube.name()))


def save_netcdf(
    cubelist: Union[Cube, CubeList],
    filename: str,
    compression_level: int = 1,
    least_significant_digit: Optional[int] = None,
) -> None:
    """Save the input Cube or CubeList as a NetCDF file and check metadata
    where required for integrity.

    Uses the functionality provided by iris.fileformats.netcdf.save with
    local_keys to record non-global attributes as data attributes rather than
    global attributes.

    Args:
        cubelist:
            Cube or list of cubes to be saved
        filename:
            Filename to save input cube(s)
        compression_level:
            1-9 to specify compression level, or 0 to not compress (default compress
            with complevel 1)
        least_significant_digit:
            If specified will truncate the data to a precision given by
            10**(-least_significant_digit), e.g. if least_significant_digit=2, then the data will
            be quantized to a precision of 0.01 (10**(-2)). See
            http://www.esrl.noaa.gov/psd/data/gridded/conventions/cdc_netcdf_standard.shtml
            for details. When used with `compression level`, this will result in lossy
            compression.

    Raises:
        warning if cubelist contains cubes of varying dimensions.
    """
    if isinstance(cubelist, iris.cube.Cube):
        cubelist = iris.cube.CubeList([cubelist])
    elif not isinstance(cubelist, iris.cube.CubeList):
        cubelist = iris.cube.CubeList(cubelist)

    for cube in cubelist:
        _order_cell_methods(cube)
        _check_metadata(cube)
        # iris.fileformats.netcdf.save will add a new "least_significant_digit"
        # attribute, but will not update an existing attribute when saving with
        # different precision.  Therefore we remove the "least_significant_digit"
        # attribute if present.
        cube.attributes.pop("least_significant_digit", None)

    # If all xy slices are the same shape, use this to determine
    # the chunksize for the netCDF (eg. 1, 1, 970, 1042)
    chunksizes = None
    if len({cube.shape[:2] for cube in cubelist}) == 1:
        cube = cubelist[0]
        if cube.ndim >= 2:
            xy_chunksizes = [cube.shape[-2], cube.shape[-1]]
            chunksizes = tuple([1] * (cube.ndim - 2) + xy_chunksizes)
    else:
        msg = "Chunksize not set as cubelist " "contains cubes of varying dimensions"
        warnings.warn(msg)

    global_keys = [
        "title",
        "um_version",
        "grid_id",
        "source",
        "Conventions",
        "institution",
        "history",
    ]
    global_keys.extend([key for key in cube.attributes.keys() if "mosg__" in key])

    local_keys = {
        key
        for cube in cubelist
        for key in cube.attributes.keys()
        if key not in global_keys
    }

    if compression_level not in range(10):
        raise ValueError(
            "Compression level must be an integer value between 0 and 9 (0 to disable compression)"
        )

    # save atomically by writing to a temporary file and then renaming
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ftmp = str(filename) + ".tmp"
    iris.fileformats.netcdf.save(
        cubelist,
        ftmp,
        local_keys=local_keys,
        complevel=compression_level,
        shuffle=True,
        zlib=compression_level > 0,
        chunksizes=chunksizes,
        least_significant_digit=least_significant_digit,
    )
    os.rename(ftmp, filename)
