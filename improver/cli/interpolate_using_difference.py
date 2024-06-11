#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to fill masked regions in a field using interpolation of the
difference between it and a reference field."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    reference_cube: cli.inputcube,
    limit: cli.inputcube = None,
    *,
    limit_as_maximum=True,
):
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

    result = InterpolateUsingDifference(limit_as_maximum=limit_as_maximum)(
        cube, reference_cube=reference_cube, limit=limit,
    )
    return result
