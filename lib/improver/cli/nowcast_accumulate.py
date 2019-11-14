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
"""Script to produce nowcast accumulations."""

import numpy as np
from iris.cube import CubeList

from improver.argparser import ArgParser
from improver.nowcasting.accumulation import Accumulation
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Calculate optical flow advection velocities"""

    parser = ArgParser(
        description="Calculate optical flow components from input fields.")
    parser.add_argument("input_filepath", metavar="INPUT_FILEPATH",
                        type=str, help="Path to input NetCDF file.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILEPATH",
                        help="The output path for the resulting NetCDF")
    parser.add_argument("--max_lead_time", type=int, default=360,
                        help="Maximum lead time required (mins).")
    parser.add_argument("--lead_time_interval", type=int, default=15,
                        help="Interval between required lead times (mins).")
    parser.add_argument(
        "--accumulation_period", type=int, default=15,
        help="The period over which the accumulation is calculated (mins). "
        "Only full accumulation periods will be computed. At lead times "
        "that are shorter than the accumulation period, no accumulation "
        "output will be produced.")
    parser.add_argument(
        "--accumulation_units", type=str, default='m',
        help="Desired units in which the accumulations should be expressed,"
        "e.g. mm")

    args = parser.parse_args(args=argv)

    # Load Cubes

    u_cube = load_cube(args.u_and_v_filepath,
                       "precipitation_advection_x_velocity", allow_none=True)
    v_cube = load_cube(args.u_and_v_filepath,
                       "precipitation_advection_y_velocity", allow_none=True)

    # Process
    result = process(u_cube, v_cube, args.max_lead_time,
                     args.lead_time_interval, args.accumulation_period,
                     args.accumulation_units)

    # Save Cubes
    save_netcdf(result, args.output_filepath)


def process(u_cube, v_cube, max_lead_time, lead_time_interval,
            accumulation_period, accumulation_units):
    """

    Args:
        u_cube:
        v_cube:
        max_lead_time:
        lead_time_interval:
        accumulation_period:
        accumulation_units:

    Returns:

    """

    lead_times = (
        np.arange(lead_time_interval, max_lead_time + 1,
                  lead_time_interval))
    plugin = Accumulation(
        accumulation_units=accumulation_units,
        accumulation_period=accumulation_period * 60,
        forecast_periods=lead_times * 60)
    accumulation_cubes = plugin.process(CubeList([u_cube, v_cube]))
    return accumulation_cubes


if __name__ == "__main__":
    main()
