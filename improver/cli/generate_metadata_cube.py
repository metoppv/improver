#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
    mandatory_attributes_json: cli.inputjson,
    *,
    name="air_pressure_at_sea_level",
    units=None,
    spatial_grid="latlon",
    time_period: int = None,
    json_input: cli.inputjson = None,
    ensemble_members: int = 8,
    x_grid_spacing: float = None,
    y_grid_spacing: float = None,
    domain_corner: cli.comma_separated_list_of_float = None,
    npoints: int = 71,
):
    """ Generate a cube with metadata only.

    Args:
        mandatory_attributes_json (Dict):
            Specifies the values of the mandatory attributes, title, institution and
            source.
        name (Optional[str]):
            Output variable name, or if creating a probability cube the name of the
            underlying variable to which the probability field applies.
        units (Optional[str]):
            Output variable units, or if creating a probability cube the units of the
            underlying variable / threshold.
        spatial_grid (Optional[str]):
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        time_period (Optional[int]):
            The period in minutes between the time bounds. This is used to calculate
            the lower time bound. If unset the diagnostic will be instantaneous, i.e.
            without time bounds.
        json_input (Optional[Dict]):
            Dictionary containing values for one or more of: "name", "units", "time",
            "time_bounds", "frt", "spp__relative_to_threshold", "attributes"
            (dictionary of additional metadata attributes) and "coords" (dictionary).
            "coords" can contain "height_levels" (list of height/pressure level values),
            and one of "realizations", "percentiles" or "thresholds" (list of dimension
            values).
        ensemble_members (Optional[int]):
            Number of ensemble members. Default 8. Will not be used if "realizations",
            "percentiles" or "thresholds" provided in json_input.
        grid_spacing (Optional[float]):
            Resolution of grid (metres or degrees).
        domain_corner (Optional[Tuple[float, float]]):
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for
            equalarea).
        npoints (Optional[int]):
            Number of points along each of the y and x spatial axes.

    Returns:
        iris.cube.Cube:
            Output of generate_metadata()
    """
    # Set arguments to pass to generate_metadata function and remove json_input for
    # processing contents before adding
    generate_metadata_args = locals()
    for key in ["mandatory_attributes_json", "json_input"]:
        generate_metadata_args.pop(key, None)

    from improver.synthetic_data.generate_metadata import generate_metadata
    from improver.synthetic_data.utilities import (
        get_height_levels,
        get_leading_dimension,
    )
    from improver.utilities.temporal import cycletime_to_datetime

    if json_input is not None:
        # Get leading dimension and height/pressure data from json_input
        if "coords" in json_input:
            coord_data = json_input["coords"]

            (
                json_input["leading_dimension"],
                json_input["cube_type"],
            ) = get_leading_dimension(coord_data)
            json_input["height_levels"], json_input["pressure"] = get_height_levels(
                coord_data
            )

            json_input.pop("coords", None)

        # Convert str time, frt and time_bounds to datetime
        if "time" in json_input:
            json_input["time"] = cycletime_to_datetime(json_input["time"])

        if "frt" in json_input:
            json_input["frt"] = cycletime_to_datetime(json_input["frt"])

        if "time_bounds" in json_input:
            time_bounds = []
            for tb in json_input["time_bounds"]:
                time_bounds.append(cycletime_to_datetime(tb))
            json_input["time_bounds"] = time_bounds

        # Update generate_metadata_args with the json_input data
        generate_metadata_args.update(json_input)
    return generate_metadata(mandatory_attributes_json, **generate_metadata_args)
