# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Utilities for datatype checking"""

import iris
import numpy as np
from cf_units import Unit

from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.constants.time_types import TIME_COORDS


def _is_time_coord(obj):
    return isinstance(obj, iris.coords.Coord) and obj.name() in TIME_COORDS


def get_required_dtype(obj):
    if _is_time_coord(obj):
        return np.dtype(TIME_COORDS[obj.name()].dtype)
    if np.issubdtype(obj.dtype, np.floating):
        return np.dtype(FLOAT_DTYPE)
    if np.issubdtype(obj.dtype, np.integer):
        # pass back same dtype - all ints are acceptable if not a time coord
        return obj.dtype
    # Assume everything else is correct (to allow string-type objects, bool)
    return obj.dtype


def check_dtype(obj):
    # if coord, acts on coord.points
    req_dtype = get_required_dtype(obj)
    dtype_ok = obj.dtype == req_dtype

    if isinstance(obj, iris.coords.Coord) and obj.has_bounds():
        # check bounds - want the same dtype as the points
        bounds_dtype_ok = obj.bounds.dtype == req_dtype
        dtype_ok = dtype_ok and bounds_dtype_ok
    return dtype_ok


def get_required_units(obj):
    if _is_time_coord(obj):
        return TIME_COORDS[obj.name()].units
    # No special requirement on non-time coordinate objects
    return None


def check_units(obj):
    if hasattr(obj, 'units'):
        req_units = get_required_units(obj)
        if req_units is not None:
            return Unit(obj.units) == Unit(req_units)
    return True


def check_datatypes(cube):
    def check_dtype_and_units(obj):
        dtype_ok = check_dtype(obj)
        units_ok = check_units(obj)

        errors = []
        if not dtype_ok:
            req_dtype = get_required_dtype(obj)
            msg = (f"{obj.name()} of type {type(obj)} does not have "
                   f"required dtype.\n"
                   f"Expected: {req_dtype}, ")
            if isinstance(obj, iris.coords.Coord):
                msg += f"Actual (points): {obj.points.dtype}"
                if obj.has_bounds():
                    msg += f", Actual (bounds): {obj.bounds.dtype}"
            else:
                msg += f"Actual: {obj.dtype}"
            errors.append(msg)
        if not units_ok:
            req_units = get_required_units(obj)
            msg = (f"{obj.name()} of type {type(obj)} does not have "
                   f"required units.\n"
                   f"Expected: {req_units}, Actual: {obj.units}")
        return errors

    errors = []
    errors.extend(check_dtype_and_units(cube))
    for coord in cube.coords():
        errors.extend(check_dtype_and_units(coord))
    if errors:
        raise ValueError('\n'.join(errors))
