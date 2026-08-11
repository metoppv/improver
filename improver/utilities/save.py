# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module for saving netcdf cubes with desired attribute types."""

import tempfile
import warnings
from pathlib import Path
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


def _horizontal_grid(cube):
    x = cube.coord(axis="x", dim_coords=True)
    xdim = cube.coord_dims(x)[0]
    y = cube.coord(axis="y", dim_coords=True)
    ydim = cube.coord_dims(y)[0]
    return xdim, ydim, x, y


def _derive_chunksizes(cubelist):
    derive_chunksize = True
    rcube = cubelist[0]
    try:
        rxdim, rydim, rx, ry = _horizontal_grid(rcube)
    except iris.exceptions.CoordinateNotFoundError:
        return None

    if derive_chunksize and len(cubelist) > 1:
        for cube in cubelist[1:]:
            # check that chunksizes can apply to the full cubelist
            try:
                xdim, ydim, x, y = _horizontal_grid(cube)
            except iris.exceptions.CoordinateNotFoundError:
                return None
            if (
                cube.ndim != rcube.ndim  # same dimensionality
                or xdim != rxdim  # same x dimension mapping
                or ydim != rydim  # same y dimension mapping
                or cube.shape[xdim] != rcube.shape[rxdim]  # same x dimension size
                or cube.shape[ydim] != rcube.shape[rydim]  # same y dimension size
            ):
                derive_chunksize = False
                msg = "Chunksize not set as cubelist contains cubes of varying x-y shape/mapping"
                warnings.warn(msg)
                break

    if derive_chunksize and rcube.ndim >= 2:
        # If all xy slices are the same shape, use this to determine
        # the chunksize for the netCDF (eg. 1, 1, 970, 1042)
        chunksizes = [1] * rcube.ndim
        chunksizes[rxdim] = rcube.shape[rxdim]
        chunksizes[rydim] = rcube.shape[rydim]
    return tuple(chunksizes) if derive_chunksize else None


def save_netcdf(
    cubelist: Union[Cube, CubeList],
    filename: str | Path,
    complevel: int = 1,
    zlib: bool | None = None,
    shuffle: bool = True,
    chunksizes: tuple | None = None,
    **kwargs,
) -> None:
    """
    Save the input Cube or CubeList as a NetCDF file and check metadata
    where required for integrity.

    Uses the functionality provided by iris.fileformats.netcdf.save with
    local_keys to record non-global attributes as data attributes rather than
    global attributes.  The save is made with
    iris.FUTURE.context(save_split_attrs=True).
    iris.fileformats.netcdf.save will add a new "least_significant_digit"
    attribute, but will not update an existing attribute when saving with
    different precision. Therefore, we remove the "least_significant_digit"
    attribute if present.

    We further deviate from iris.fileformats.netcdf.save default behaviour
    as per keyword arguments.

    Args:
        cubelist:
            Cube or CubeList to be saved.
        filename:
            Filename to save input cube(s)
        complevel:
            Compression level for the NetCDF file. Must be an integer between 0 and 9
            where 0 disables compression. Default is 1 (iris default is 4).
        zlib:
            Whether to use zlib compression. If None (default), set to True
            if complevel > 0, otherwise False. iris default is False.
        shuffle:
            Whether to use HDF5 shuffle filter. Default is True (iris default is False).
        chunksizes:
            Tuple defining chunk sizes for the output file. If None (default),
            automatically determined as a 1 for all dimensions except the x and y
            dimensions, which are set to the full size of the x and y dimensions.
        **kwargs:
            Additional keyword arguments to pass to iris.fileformats.netcdf.save.

    Raises:
        ValueError:
            If complevel is not between 0 and 9.


    Warns:
        If compression_level is passed via kwargs (deprecated, use complevel instead).
        If cubelist contains cubes of varying dimensions.
    """
    cubelist = as_cubelist(cubelist)

    # Handle deprecated compression_level argument
    if "compression_level" in kwargs:
        warnings.warn(
            "The 'compression_level' argument is deprecated and will be removed in a future release. "
            "Please use 'complevel' instead.  Overriding 'complevel' with 'compression_level' if both "
            "are provided.",
            FutureWarning,
            stacklevel=2,
        )
        complevel = kwargs.pop("compression_level", None)

    if complevel is None:
        complevel = 1
    else:
        # iris does no validation of the compression level, so we do it here
        try:
            old_complevel = complevel
            complevel = int(complevel)
            if old_complevel != complevel or complevel not in range(10):
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(
                "Compression level must be an integer value between 0 and 9 (0 to disable compression)"
            )

    if zlib is None:
        zlib = complevel > 0

    for cube in cubelist:
        _order_cell_methods(cube)
        _check_metadata(cube)
        cube.attributes.pop("least_significant_digit", None)
        _cube_attributes_for_save(cube)

    if chunksizes is None:
        chunksizes = _derive_chunksizes(cubelist)

    filename = Path(filename)

    # save atomically by writing to a unique temporary file of the form <filename>-<unique>.tmp
    with tempfile.NamedTemporaryFile(
        dir=filename.parent,
        prefix=filename.name + "-",
        suffix=".tmp",
    ) as tmp_file:
        tmp_filename = Path(tmp_file.name)
        with iris.FUTURE.context(save_split_attrs=True):
            iris.fileformats.netcdf.save(
                cubelist,
                str(tmp_filename),
                complevel=complevel,
                shuffle=shuffle,
                zlib=zlib,
                chunksizes=chunksizes,
                **kwargs,
            )
        tmp_filename.replace(filename)
        filename.chmod(0o644)


def _cube_attributes_for_save(cube: Cube):
    """
    Separate global and local attributes for saving with iris by ensuring a CubeAttrsDict
    represents all attributes.

    Args:
        cube:
            The cube for which the attributes are to be separated.
    """
    global_keys = [
        "Conventions",
        "grid_id",
        "history",
        "institution",
        "name_netcdf_out_vers",
        "name_version",
        "reference",
        "source",
        "title",
        "um_version",
    ]
    global_keys.extend([key for key in cube.attributes.keys() if "mosg__" in key])
    global_attributes = {k: v for k, v in cube.attributes.items() if k in global_keys}
    local_attributes = {
        k: v for k, v in cube.attributes.items() if k not in global_keys
    }
    attributes = CubeAttrsDict(locals=local_attributes, globals=global_attributes)
    cube.attributes = attributes
