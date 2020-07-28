#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""CLI to generate metadata cube for acceptance tests."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *,
    name="air_temperature",
    units=None,
    spatial_grid="latlon",
    time="20171110T0400Z",
    frt="20171110T0400Z",
    ensemble_members=8,
    attributes: cli.inputjson = None,
    resolution=None,
    domain_corner: cli.comma_separated_list = None,
    npoints=71,
    height_levels=None,
):
    """ Generate a cube with metadata only.

    Args:
        name (str):
            Output variable name.
        units (Optional[str]):
            Output variable units.
        spatial_grid (Optional[str]):
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        time (Optional[str]):
            Single cube validity time. Datetime string of format YYYYMMDDTHHMMZ.
        frt (Optional[str]):
            Single cube forecast reference time. Datetime string of format YYYYMMDDTHHMMZ.
        ensemble_members (Optional[int]):
            Number of ensemble members.
        attributes (Optional[Dict]):
            Dictionary of additional metadata attributes.
        resolution (Optional[float]):
            Resolution of grid (metres or degrees).
        domain_corner (Optional[Tuple[float, float]]):
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for equalarea).
        npoints (Optional[int]):
            Number of points along a single axis.
        height_levels (Optional[List[float]]):
            List of altitude/pressure levels.

    Returns:
        iris.cube.Cube:
            Output of generate_metadata()
    """
    from improver.synthetic_data.generate_metadata import generate_metadata
    from improver.utilities.temporal import cycletime_to_datetime

    if domain_corner is not None:
        try:
            domain_corner = (float(domain_corner[0]), float(domain_corner[1]))
        except IndexError:
            raise TypeError("Domain corner must be a list of length 2")

    if resolution is not None:
        resolution = float(resolution)

    if height_levels is not None:
        height_levels = [float(h) for h in height_levels]

    # Convert str time and frt to datetime
    time = cycletime_to_datetime(time)
    frt = cycletime_to_datetime(frt)

    return generate_metadata(
        name,
        units=units,
        spatial_grid=spatial_grid,
        time=time,
        frt=frt,
        ensemble_members=ensemble_members,
        attributes=attributes,
        resolution=resolution,
        domain_corner=domain_corner,
        npoints=npoints,
        height_levels=height_levels,
    )
