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

"""Script to run weighted blending across adjacent points"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube,
            coordinate,
            central_point: float,
            units=None,
            width: float = None,
            calendar='gregorian',
            blend_time_using_forecast_period=False):
    """Runs weighted blending across adjacent points.

    Uses the TriangularWeightedBlendAcrossAdjacentPoints to blend across
    a particular coordinate. It does not collapse the coordinate, but
    instead blends across adjacent points and puts the blended values back
    in the original coordinate, with adjusted bounds.

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
            Units of the central_point and width
        width (float):
            Width of the triangular weighting function used in the blending,
            in the units of the units argument.
        calendar (str)
            Calendar for parameter_unit if required.
        blend_time_using_forecast_period (bool):
            If True, we are blending over time but using the forecast
            period coordinate as a proxy. Note, this should only be used when
            time and forecast_period share a dimension: i.e when all cubes
            provided are from the same forecast cycle.

    Returns:
        iris.cube.Cube:
            A processed Cube

    Raises:
        ValueError:
            If coordinate has "time" in it.
        ValueError:
            If blend_time_forecast_period is not used with forecast_period
            coordinate.

    """
    from cf_units import Unit
    from iris.cube import CubeList

    from improver.blending.blend_across_adjacent_points import \
        TriangularWeightedBlendAcrossAdjacentPoints
    from improver.utilities.cube_manipulation import MergeCubes

    # TriangularWeightedBlendAcrossAdjacentPoints can't currently handle
    # blending over times where iris reads the coordinate points as datetime
    # objects. Fail here to avoid unhelpful errors downstream.
    if "time" in coordinate:
        msg = ("Cannot blend over {} coordinate (points encoded as datetime "
               "objects)".format(coordinate))
        raise ValueError(msg)

    # This is left as a placeholder for when we have this capability
    if coordinate == 'time':
        units = Unit(units, calendar)

    cubes = CubeList(cubes)

    if blend_time_using_forecast_period and coordinate == 'forecast_period':
        cube = MergeCubes().process(cubes, check_time_bounds_ranges=True)
    elif blend_time_using_forecast_period:
        msg = ('"--blend-time-using-forecast-period" can only be used with '
               '"forecast_period" coordinate')
        raise ValueError(msg)
    else:
        cube = MergeCubes().process(cubes)

    blending_plugin = TriangularWeightedBlendAcrossAdjacentPoints(
        coordinate, central_point, units, width)
    result = blending_plugin(cube)
    return result
