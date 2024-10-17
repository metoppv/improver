# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Utilities for mandatory datatype and units checking"""

from typing import List, Optional, Union

import iris
import numpy as np
from cf_units import Unit
from iris.coords import Coord
from iris.cube import Cube, CubeList
from numpy import dtype

from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.constants.time_types import TIME_COORDS


def _is_time_coord(obj: Union[Cube, Coord]) -> bool:
    """
    Checks whether the supplied object is an iris.coords.Coord and has a name
    that matches a known time coord.

    Args:
        obj:
            Object to be tested

    Returns:
        True if obj is a recognised time coord.
    """
    return isinstance(obj, iris.coords.Coord) and obj.name() in TIME_COORDS


def get_required_dtype(obj: Union[Cube, Coord]) -> dtype:
    """
    Returns the appropriate dtype for the supplied object. This includes
    special dtypes for time coordinates.

    Args:
        obj:
            Object to be tested

    Returns:
        The mandatory dtype corresponding to the object supplied.
    """
    if _is_time_coord(obj):
        return np.dtype(TIME_COORDS[obj.name()].dtype)
    if np.issubdtype(obj.dtype, np.floating):
        return np.dtype(FLOAT_DTYPE)
    if np.issubdtype(obj.dtype, np.integer):
        # pass back same dtype - all ints are acceptable if not a time coord
        return obj.dtype
    # Assume everything else is correct (to allow string-type objects, bool)
    return obj.dtype


def check_dtype(obj: Union[Cube, Coord]) -> bool:
    """
    Finds the mandatory dtype for obj and checks that it is correctly
    applied. If obj is a coord, any bounds are checked too.

    Args:
        obj:
            Object to be tested

    Returns:
        True if obj is of the mandated dtype.
    """
    # if coord, acts on coord.points
    req_dtype = get_required_dtype(obj)
    dtype_ok = obj.dtype == req_dtype

    if isinstance(obj, iris.coords.Coord) and obj.has_bounds():
        # check bounds - want the same dtype as the points
        bounds_dtype_ok = obj.bounds.dtype == req_dtype
        dtype_ok = dtype_ok and bounds_dtype_ok
    return dtype_ok


def enforce_dtype(
    operation: str, inputs: Union[List[Cube], CubeList], result: Cube
) -> None:
    """
    Ensures that result has not been automatically promoted to float64.

    Args:
        operation:
            The operation that was performed (for the error message)
        inputs:
            The Numpy arrays or cubes that the operation was performed on (for the
            error message)
        result:
            The result of the operation

    Raises:
        TypeError:
            If result.dtype does not match the meta-data standard.
    """
    if not check_dtype(result):
        unique_cube_types = set([c.dtype for c in inputs])
        raise TypeError(
            f"Operation {operation} on types {unique_cube_types} results in "
            "float64 data which cannot be safely coerced to float32 (Hint: "
            "combining int8 and float32 works)"
        )


def get_required_units(obj: Union[Cube, Coord]) -> Optional[str]:
    """
    Returns the mandatory units for the supplied obj. Only time coords have
    these.

    Args:
        obj:
            Object to be tested

    Returns:
        The mandatory units corresponding to the object supplied or None
        if there are no specific requirements for the object.
    """
    if _is_time_coord(obj):
        return TIME_COORDS[obj.name()].units
    return None


def check_units(obj: Union[Cube, Coord]) -> bool:
    """
    Checks if the supplied object complies with the relevant mandatory units.

    Args:
        obj:
            Object to be tested

    Returns:
        True if obj meets the mandatory units requirements or has no
        mandatory units requirement.
    """
    req_units = get_required_units(obj)
    if req_units is None:
        return True
    # check object and string representation to get consistent output
    # (e.g Unit('second') == Unit('seconds') == Unit('s'))
    return Unit(obj.units) == Unit(req_units)


def check_mandatory_standards(cube: Cube) -> None:
    """
    Checks for mandatory dtype and unit standards on a cube and raises a
    useful exception if any non-compliance is found.

    Args:
        cube:
            The cube to be checked for conformance with standards.

    Raises:
        ValueError:
            If the cube fails to meet any mandatory dtype and units standards
    """

    def check_dtype_and_units(obj: Union[Cube, Coord]) -> List[str]:
        """
        Check object meets the mandatory dtype and units.

        Args:
            obj:
                The object to be checked.

        Returns:
            Contains formatted strings describing each conformance breach.
        """
        dtype_ok = check_dtype(obj)
        units_ok = check_units(obj)

        errors = []
        if not dtype_ok:
            req_dtype = get_required_dtype(obj)
            msg = (
                f"{obj.name()} of type {type(obj)} does not have "
                f"required dtype.\n"
                f"Expected: {req_dtype}, "
            )
            if isinstance(obj, iris.coords.Coord):
                msg += f"Actual (points): {obj.points.dtype}"
                if obj.has_bounds():
                    msg += f", Actual (bounds): {obj.bounds.dtype}"
            else:
                msg += f"Actual: {obj.dtype}"
            errors.append(msg)
        if not units_ok:
            req_units = get_required_units(obj)
            msg = (
                f"{obj.name()} of type {type(obj)} does not have "
                f"required units.\n"
                f"Expected: {req_units}, Actual: {obj.units}"
            )
            errors.append(msg)
        return errors

    error_list = []
    error_list.extend(check_dtype_and_units(cube))
    for coord in cube.coords():
        error_list.extend(check_dtype_and_units(coord))
    if error_list:
        raise ValueError("\n".join(error_list))
