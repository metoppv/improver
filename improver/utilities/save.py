# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module for saving netcdf cubes with desired attribute types."""

import os
import tempfile
import warnings
from typing import Union

import cf_units
import iris
import iris.fileformats
from iris.cube import Cube, CubeAttrsDict, CubeList

from improver.metadata.check_datatypes import check_mandatory_standards
from improver.utilities.common_input_handle import as_cubelist


def _order_cell_methods(cube: Cube) -> None:
    """
    Sorts the cell methods on a cube such that if there are multiple methods
    they are always written in a consistent order in the output cube. The
    input cube is modified. Ensure that if there are any identical duplicate
    cell methods, only one of these is included in the outputs.

    Args:
        cube:
            The cube on which the cell methods are to be sorted.
    """
    cell_methods = set(cube.cell_methods)
    cell_methods = tuple(sorted(cell_methods))
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
    **kwargs,
) -> None:
    """
    Save the input Cube or CubeList as a NetCDF file and check metadata
    where required for integrity.

    Uses the functionality provided by iris.fileformats.netcdf.save with
    local_keys to record non-global attributes as data attributes rather than
    global attributes.  The save is made with
    iris.FUTURE.context(save_split_attrs=True).  The following argument
    defaults for save deviate from iris.fileformats.netcdf.save defaults:
    - chunksizes are set to (1, 1, x, y) if all cubes have the same xy shape
      and the chunksizes are not specified in kwargs.
    - complevel default is 1 (iris default is 4)
    - zlib is set to True if complevel > 0 (iris default is False)
    - shuffle is set to True (iris default is False)

    Args:
        cubelist:
            Cube or CubeList to be saved.
        filename:
            Filename to save input cube(s)
        **kwargs:
            Additional keyword arguments to pass to iris.fileformats.netcdf.save.

    Raises:
        ValueError:
            If compression_level is not between 0 and 9.

    Warns:
        If cubelist contains cubes of varying dimensions.
    """
    cubelist = as_cubelist(cubelist)

    if "compression_level" in kwargs:
        warnings.warn(
            "The 'compression_level' argument is deprecated and will be removed in a future release. "
            "Please use 'complevel' instead.",
            DeprecationWarning,
        )
    complevel = kwargs.pop("compression_level", None)
    complevel = kwargs.pop("complevel", complevel)
    if complevel is None:
        complevel = 1
    else:
        complevel = int(complevel)
        if complevel not in range(10):
            # iris does no validation of the compression level, so we do it here
            raise ValueError(
                "Compression level must be an integer value between 0 and 9 (0 to disable compression)"
            )
    zlib = kwargs.pop("zlib", complevel > 0 if complevel is not None else False)
    shuffle = kwargs.pop("shuffle", True)

    for cube in cubelist:
        _order_cell_methods(cube)
        _check_metadata(cube)
        # iris.fileformats.netcdf.save will add a new "least_significant_digit"
        # attribute, but will not update an existing attribute when saving with
        # different precision. Therefore, we remove the "least_significant_digit"
        # attribute if present.
        cube.attributes.pop("least_significant_digit", None)
        _cube_attributes_for_save(cube)

    chunksizes = kwargs.pop("chunksizes", None)
    if chunksizes is None:
        if len({cube.shape[:2] for cube in cubelist}) == 1:
            cube = cubelist[0]
            if cube.ndim >= 2:
                # If all xy slices are the same shape, use this to determine
                # the chunksize for the netCDF (eg. 1, 1, 970, 1042)
                xy_chunksizes = [cube.shape[-2], cube.shape[-1]]
                chunksizes = tuple([1] * (cube.ndim - 2) + xy_chunksizes)
        else:
            msg = "Chunksize not set as cubelist contains cubes of varying dimensions"
            warnings.warn(msg)

    # save atomically by writing to a unique temporary file of the form <filename>-<unique>.tmp
    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(filename),
        prefix=os.path.basename(filename) + "-",
        suffix=".tmp",
    ) as tmp_file:
        tmp_filename = tmp_file.name
        with iris.FUTURE.context(save_split_attrs=True):
            iris.fileformats.netcdf.save(
                cubelist,
                tmp_filename,
                complevel=complevel,
                shuffle=shuffle,
                zlib=zlib,
                chunksizes=chunksizes,
                **kwargs,
            )
        os.rename(tmp_filename, filename)
        os.chmod(filename, 0o644)


def _cube_attributes_for_save(cube: Cube):
    """
    Separate global and local attributes for saving with iris by ensuring a CubeAttrsDict
    represents all attributes.

    Args:
        cube:
            The cube for which the attributes are to be separated.
    """
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
    global_attributes = {k: v for k, v in cube.attributes.items() if k in global_keys}
    local_attributes = {
        k: v for k, v in cube.attributes.items() if k not in global_keys
    }
    attributes = CubeAttrsDict(locals=local_attributes, globals=global_attributes)
    cube.attributes = attributes
