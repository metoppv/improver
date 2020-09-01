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
    name="air_pressure_at_sea_level",
    units=None,
    spatial_grid="latlon",
    time="20171110T0400Z",
    time_period: int = None,
    frt="20171110T0000Z",
    ensemble_members: int = 8,
    leading_dimension: cli.comma_separated_list_of_float = None,
    leading_dimension_json: cli.inputjson = None,
    percentile=False,
    probability=False,
    spp__relative_to_threshold="above",
    attributes: cli.inputjson = None,
    resolution: float = None,
    domain_corner: cli.comma_separated_list_of_float = None,
    npoints: int = 71,
    height_levels: cli.comma_separated_list_of_float = None,
    height_levels_json: cli.inputjson = None,
    pressure=False,
):
    """ Generate a cube with metadata only.

    Args:
        name (Optional[str]):
            Output variable name.
        units (Optional[str]):
            Output variable units.
        spatial_grid (Optional[str]):
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        time (Optional[str]):
            Single cube validity time. Datetime string of format YYYYMMDDTHHMMZ. If time period given, time is used as the upper time bound.
        time_period (Optional[int]):
            The period in minutes between the time bounds. This is used to calculate the lower time bound.
        frt (Optional[str]):
            Single cube forecast reference time. Datetime string of format YYYYMMDDTHHMMZ.
        ensemble_members (Optional[int]):
            Number of ensemble members. Default 8, unless percentile or probability set to True. Will not be used if leading_dimension or leading_dimension_json provided.
        leading_dimension (Optional[List[float]]):
            List of realizations, percentiles or thresholds.
        leading_dimension_json (Optional[Dict]):
            Dictionary containing a list of realizations, percentiles or thresholds. If both leading_dimension and leading_dimension_json are provided, leading_dimension is used.
        percentile (Optional[bool]):
            Flag to indicate whether the leading dimension is percentile values. If True, a percentile cube is created.
        probability (Optional[bool]):
            Flag to indicate whether the leading dimension is threshold values. If True, a probability cube is created.
        spp__relative_to_threshold (Optional[str]):
            Value of the attribute "spp__relative_to_threshold" which is required for IMPROVER probability cubes.
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
        height_levels_json (Optional[Dict]):
            Dictionary containing a list of height levels. If both height_levels and height_levels_json are provided, height_levels is used.
        pressure (Optional[bool]):
            Flag to indicate whether the height levels are specified as pressure, in Pa. If False, use height in metres.

    Returns:
        iris.cube.Cube:
            Output of generate_metadata()
    """
    from improver.synthetic_data.generate_metadata import generate_metadata
    from improver.utilities.temporal import cycletime_to_datetime

    if leading_dimension is None and leading_dimension_json is not None:
        if percentile is True:
            leading_dimension = leading_dimension_json["percentiles"]
        elif probability is True:
            leading_dimension = leading_dimension_json["thresholds"]
        else:
            leading_dimension = leading_dimension_json["realizations"]

    if height_levels is None and height_levels_json is not None:
        height_levels = height_levels_json["height_levels"]

    # Convert str time and frt to datetime
    time = cycletime_to_datetime(time)
    frt = cycletime_to_datetime(frt)

    return generate_metadata(
        name=name,
        units=units,
        spatial_grid=spatial_grid,
        time=time,
        time_period=time_period,
        frt=frt,
        ensemble_members=ensemble_members,
        leading_dimension=leading_dimension,
        percentile=percentile,
        probability=probability,
        spp__relative_to_threshold=spp__relative_to_threshold,
        attributes=attributes,
        resolution=resolution,
        domain_corner=domain_corner,
        npoints=npoints,
        height_levels=height_levels,
        pressure=pressure,
    )
