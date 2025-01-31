#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
    """Generate a cube with metadata only.

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
            "coords" can contain "vertical_levels" (list of height/pressure level values),
            and one of "realizations", "percentiles" or "thresholds" (list of dimension
            values).
        ensemble_members (Optional[int]):
            Number of ensemble members. Default 8. Will not be used if "realizations",
            "percentiles" or "thresholds" provided in json_input.
        x_grid_spacing (Optional[float]):
            Resolution of grid along the x-axis (metres or degrees).
        y_grid_spacing (Optional[float]):
            Resolution of grid along the y-axis (metres or degrees).
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
        get_leading_dimension,
        get_vertical_levels,
    )
    from improver.utilities.temporal import cycletime_to_datetime

    if json_input is not None:
        # Get leading dimension and height/pressure data from json_input
        if "coords" in json_input:
            coord_data = json_input["coords"]

            (json_input["leading_dimension"], json_input["cube_type"]) = (
                get_leading_dimension(coord_data)
            )
            json_input["vertical_levels"], json_input["pressure"], json_input["height"] = get_vertical_levels(
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
