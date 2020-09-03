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

DEFAULT_CUBE_VALUES = {
    "name": "air_pressure_at_sea_level",
    "units": None,
    "time": "20171110T0400Z",
    "frt": "20171110T0000Z",
    "time_bounds": None,
    "spp__relative_to_threshold": "above",
    "attributes": None,
}


class InputData:
    def __init__(self, data):
        self.data = data

    def get_data_from_dictionary(self, value, key):
        """If value not provided, gets data from input json if included, otherwise 
        returns default value."""
        if value is None:
            if self.data is not None and key in self.data:
                value = self.data[key]
            else:
                value = DEFAULT_CUBE_VALUES[key]

        return value

    def _error_more_than_one_leading_dimension(self):
        """Raises an error to inform the user that only one leading dimension can be
        provided in the input data."""
        raise ValueError(
            'Only one of "realization", "percentile" or "probability" dimensions should be provided.'
        )

    def get_leading_dimension_from_dictionary(self, leading_dimension, cube_type):
        """Gets leading dimension values from coords nested dictionary and sets cube
        type based on what dimension key is used."""
        if "realizations" in self.data:
            leading_dimension = self.data["realizations"]

        if "percentiles" in self.data:
            if leading_dimension is not None:
                self._error_more_than_one_leading_dimension()

            leading_dimension = self.data["percentiles"]
            cube_type = "percentile"

        if "thresholds" in self.data:
            if leading_dimension is not None:
                self._error_more_than_one_leading_dimension()

            leading_dimension = self.data["thresholds"]
            cube_type = "probability"

        return leading_dimension, cube_type

    def get_height_levels_from_dictionary(self, height_levels, pressure):
        """Gets height level values from coords nested dictionary and sets pressure
        value based on whether heights or pressures key is used."""
        if "heights" in self.data:
            height_levels = self.data["heights"]
        elif "pressures" in self.data:
            height_levels = self.data["pressures"]
            pressure = True

        return height_levels, pressure


def _check_domain_corner(domain_corner):
    """Checks that domain corner has a length of two int or float values, raises error
    if not"""
    if domain_corner is not None and len(domain_corner) != 2:
        raise ValueError("Domain corner must be a list or tuple of length 2.")


@cli.clizefy
@cli.with_output
def process(
    *,
    name=None,
    units=None,
    spatial_grid="latlon",
    time=None,
    time_period: int = None,
    frt=None,
    json_input: cli.inputjson = None,
    ensemble_members: int = 8,
    grid_spacing: float = None,
    domain_corner: cli.comma_separated_list_of_float = None,
    npoints: int = 71,
):
    """ Generate a cube with metadata only.

    Args:
        name (Optional[str]):
            Output variable name, or if creating a probability cube the name of the
            underlying variable to which the probability field applies. Default:
            "air_pressure_at_sea_level".
        units (Optional[str]):
            Output variable units, or if creating a probability cube the units of the
            underlying variable / threshold.
        spatial_grid (Optional[str]):
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        time (Optional[str]):
            Single cube validity time. Datetime string of format YYYYMMDDTHHMMZ. If
            time period given, time is used as the upper time bound. Default:
            "20171110T0400Z".
        time_period (Optional[int]):
            The period in minutes between the time bounds. This is used to calculate
            the lower time bound.
        frt (Optional[str]):
            Single cube forecast reference time. Datetime string of format
            YYYYMMDDTHHMMZ. Default: "20171110T0000Z".
        json_input (Optional[Dict]):
            Dictionary containing values for one or more of: "name", "units", "time",
            "time_bounds", "frt", "spp__relative_to_threshold", "attributes"
            (dictionary of additional metadata attributes) and "coords" (dictionary).
            "coords" can contain "height_levels" (list of height/pressure level values),
            and one of "realizations", "percentiles" or "thresholds" (list of dimension
            values).
        ensemble_members (Optional[int]):
            Number of ensemble members. Default 8, unless percentile or probability set
            to True. Will not be used if leading_dimension or leading_dimension_json
            provided.
        grid_spacing (Optional[float]):
            Resolution of grid (metres or degrees).
        domain_corner (Optional[Tuple[float, float]]):
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for
            equalarea).
        npoints (Optional[int]):
            Number of points along a single axis.

    Returns:
        iris.cube.Cube:
            Output of generate_metadata()
    """
    from improver.synthetic_data.generate_metadata import generate_metadata
    from improver.utilities.temporal import cycletime_to_datetime

    # Check domain corner is list of length 2 if provided
    _check_domain_corner(domain_corner)

    # Initialize variables that are only set from input json file
    time_bounds = None
    attributes = None
    spp__relative_to_threshold = None
    leading_dimension = None
    cube_type = "variable"
    height_levels = None
    pressure = False

    input_data = InputData(json_input)

    name = input_data.get_data_from_dictionary(name, "name")
    units = input_data.get_data_from_dictionary(units, "units")
    time = input_data.get_data_from_dictionary(time, "time")
    time_bounds = input_data.get_data_from_dictionary(time_period, "time_bounds")
    frt = input_data.get_data_from_dictionary(frt, "frt")
    spp__relative_to_threshold = input_data.get_data_from_dictionary(
        spp__relative_to_threshold, "spp__relative_to_threshold"
    )
    attributes = input_data.get_data_from_dictionary(attributes, "attributes")

    if json_input is not None and "coords" in json_input:
        coords = json_input["coords"]
        coord_data = InputData(coords)

        (
            leading_dimension,
            cube_type,
        ) = coord_data.get_leading_dimension_from_dictionary(
            leading_dimension, cube_type
        )
        height_levels, pressure = coord_data.get_height_levels_from_dictionary(
            height_levels, pressure
        )

    # Convert str time and frt to datetime
    time = cycletime_to_datetime(time)
    frt = cycletime_to_datetime(frt)

    # Set arguments to pass to generate_metadata function
    generate_metadata_args = locals()
    generate_metadata_args.pop("leading_dimension_json", None)
    generate_metadata_args.pop("height_levels_json", None)
    generate_metadata_args.pop("coord_data", None)
    generate_metadata_args.pop("coords", None)
    generate_metadata_args.pop("input_data", None)
    generate_metadata_args.pop("json_input", None)
    generate_metadata_args.pop("cycletime_to_datetime", None)
    generate_metadata_args.pop("generate_metadata", None)

    return generate_metadata(**generate_metadata_args)
