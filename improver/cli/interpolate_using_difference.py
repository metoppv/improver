#!/usr/bin/env python
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
"""Script to fill masked regions in a field using interpolation of the
difference between it and a reference field."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            reference_cube: cli.inputcube,
            limit: cli.inputcube = None,
            *,
            limit_as_maximum=True):
    """
    Uses interpolation to fill masked regions in the data contained within the
    input cube. This is achieved by calculating the difference between the
    input cube and a complete (i.e. complete across the whole domain) reference
    cube. The difference between the data in regions where they overlap is
    calculated and this difference field is then interpolated across the
    domain. Any masked regions in the input cube data are then filled with data
    calculated as the reference cube data minus the interpolated difference
    field.

    Args:
        cube (iris.cube.Cube):
            A cube containing data in which there are masked regions to be
            filled.
        reference_cube (iris.cube.Cube):
            A cube containing data in the same units as the cube of data to be
            interpolated. The data in this cube must be complete across the
            entire domain.
        limit (iris.cube.Cube):
            A cube of limiting values to apply to the cube that is being filled
            in. This can be used to ensure that the resulting values do not
            fall below / exceed the limiting values; whether the limit values
            should be used as minima or maxima is determined by the
            limit_as_maximum option.
        limit_as_maximum (bool):
            If True the limit values are treated as maxima for the data in the
            interpolated regions. If False the limit values are treated as
            minima.
    Returns:
        iris.cube.Cube:
            Processed cube with the masked regions filled in through
            interpolation.
    """
    from improver.utilities.interpolation import InterpolateUsingDifference

    result = InterpolateUsingDifference()(
        cube, reference_cube, limit=limit, limit_as_maximum=limit_as_maximum)
    return result
