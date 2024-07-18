#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Script to run weighted blending across adjacent points"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube_nolazy,
    coordinate,
    central_point: float,
    units=None,
    width: float = None,
    calendar="gregorian",
    blend_time_using_forecast_period=False,
):
    """Runs weighted blending across adjacent points.

    Uses the TriangularWeightedBlendAcrossAdjacentPoints to blend across
    the dimension of a particular coordinate. It does not collapse the
    coordinate, but instead blends across adjacent points and puts the
    blended values back in the original coordinate, with adjusted bounds.

    Args:
        cubes (list of iris.cube.Cube):
            A list of cubes including and surrounding the central point.
        coordinate (str):
            The coordinate over which the blending will be applied.
        central_point (float):
            Central point at which the output from the triangular weighted
            blending will be calculated. This should be in the units of the
            units argument that is passed in. This value should be a point
            on the coordinate for blending over.
        units (str):
            Units of the central_point and width.
        width (float):
            The width from the triangle’s centre point, in units of the units
            argument, which will determine the triangular weighting function
            used to blend that specified point with its adjacent points. Beyond
            this width the weighting drops to zero.
        calendar (str)
            Calendar for parameter_unit if required.
        blend_time_using_forecast_period (bool):
            If True, we are blending over time but using the forecast
            period coordinate as a proxy. Note, this should only be used when
            time and forecast_period share a dimension: i.e when all cubes
            provided are from the same forecast cycle.

    Returns:
        iris.cube.Cube:
            A processed cube, with the same coordinates as the input
            central_cube. The points in one dimension corresponding to
            the specified coordinate will be blended with the adjacent
            points based on a triangular weighting function of the
            specified width.

    Raises:
        ValueError:
            If coordinate has "time" in it.
        ValueError:
            If blend_time_forecast_period is not used with forecast_period
            coordinate.

    """
    from cf_units import Unit
    from iris.cube import CubeList

    from improver.blending.blend_across_adjacent_points import (
        TriangularWeightedBlendAcrossAdjacentPoints,
    )
    from improver.utilities.cube_manipulation import MergeCubes

    # TriangularWeightedBlendAcrossAdjacentPoints can't currently handle
    # blending over times where iris reads the coordinate points as datetime
    # objects. Fail here to avoid unhelpful errors downstream.
    if "time" in coordinate:
        msg = (
            "Cannot blend over {} coordinate (points encoded as datetime "
            "objects)".format(coordinate)
        )
        raise ValueError(msg)

    # This is left as a placeholder for when we have this capability
    if coordinate == "time":
        units = Unit(units, calendar)

    cubes = CubeList(cubes)

    if blend_time_using_forecast_period and coordinate == "forecast_period":
        cube = MergeCubes()(cubes, check_time_bounds_ranges=True)
    elif blend_time_using_forecast_period:
        msg = (
            '"--blend-time-using-forecast-period" can only be used with '
            '"forecast_period" coordinate'
        )
        raise ValueError(msg)
    else:
        cube = MergeCubes()(cubes)

    blending_plugin = TriangularWeightedBlendAcrossAdjacentPoints(
        coordinate, central_point, units, width
    )
    result = blending_plugin(cube)
    return result
