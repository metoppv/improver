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

from cf_units import Unit

from improver.argparser import ArgParser
from improver.blending.blend_across_adjacent_points import \
    TriangularWeightedBlendAcrossAdjacentPoints
from improver.utilities.cube_manipulation import MergeCubes
from improver.utilities.load import load_cubelist
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments and ensure they are set correctly.
       Then run Triangular weighted blending across the given coordinate."""
    parser = ArgParser(
        description='Use the TriangularWeightedBlendAcrossAdjacentPoints to '
                    'blend across a particular coordinate. It does not '
                    'collapse the coordinate, but instead blends across '
                    'adjacent points and puts the blended values back in the '
                    'original coordinate, with adjusted bounds.')
    parser.add_argument('coordinate', type=str,
                        metavar='COORDINATE_TO_BLEND_OVER',
                        help='The coordinate over which the blending '
                             'will be applied.')
    parser.add_argument('central_point', metavar='CENTRAL_POINT', type=float,
                        help='Central point at which the output from the '
                             'triangular weighted blending will be '
                             'calculated. This should be in the units of the '
                             'units argument that is passed in. '
                             'This value should be a point on the '
                             'coordinate for blending over.')
    parser.add_argument('--units', metavar='UNIT_STRING', required=True,
                        help='Units of the central_point and width.')
    parser.add_argument('--calendar', metavar='CALENDAR',
                        default='gregorian',
                        help='Calendar for parameter_unit if required. '
                             'Default=gregorian')
    parser.add_argument('--width', metavar='TRIANGLE_WIDTH', type=float,
                        required=True,
                        help='Width of the triangular weighting function used '
                             'in the blending, in the units of the '
                             'units argument passed in.')
    parser.add_argument('--blend_time_using_forecast_period',
                        default=False, action='store_true', help='Flag that '
                        'we are blending over time but using the forecast '
                        'period coordinate as a proxy.  Note this should only '
                        'be used when time and forecast_period share a '
                        'dimension: ie when all files provided are from the '
                        'same forecast cycle.')
    parser.add_argument('input_filepaths', metavar='INPUT_FILES', nargs="+",
                        help='Paths to input NetCDF files including and '
                             'surrounding the central_point.')
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed NetCDF.')

    args = parser.parse_args(args=argv)

    # Load Cubelist
    cubelist = load_cubelist(args.input_filepaths)
    # Process Cube
    result = process(cubelist, args.coordinate, args.central_point,
                     args.units, args.width, args.calendar,
                     args.blend_time_using_forecast_period)
    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(cubelist, coordinate, central_point, units, width,
            calendar='gregorian', blend_time_using_forecast_period=False):
    """Runs weighted blending across adjacent points.

    Uses the TriangularWeightedBlendAcrossAdjacentPoints to blend across
    a particular coordinate. It does not collapse the coordinate, but
    instead blends across adjacent points and puts the blended values back
    in the original coordinate, with adjusted bounds.

    Args:
        cubelist (iris.cube.CubeList):
            CubeList including and surrounding the central point.
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
            Default is 'gregorian'.
        blend_time_using_forecast_period (bool):
            If True, we are blending over time but using the forecast
            period coordinate as a proxy. Note, this should only be used when
            time and forecast_period share a dimension: i.e when all cubes
            provided are from the same forecast cycle.
            Default is False.

    Returns:
        result (iris.cube.Cube):
            A processed Cube

    Raises:
        ValueError:
            If coordinate has "time" in it.
        ValueError:
            If blend_time_forecast_period is not used with forecast_period
            coordinate.

    """
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

    if blend_time_using_forecast_period and coordinate == 'forecast_period':
        cube = MergeCubes().process(cubelist, check_time_bounds_ranges=True)
    elif blend_time_using_forecast_period:
        msg = ('"--blend_time_using_forecast_period" can only be used with '
               '"forecast_period" coordinate')
        raise ValueError(msg)
    else:
        cube = MergeCubes().process(cubelist)

    blending_plugin = TriangularWeightedBlendAcrossAdjacentPoints(
        coordinate, central_point, units, width)
    result = blending_plugin.process(cube)
    return result


if __name__ == "__main__":
    main()
