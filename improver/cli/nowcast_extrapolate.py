#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to extrapolate input data given advection velocity fields."""

from improver import cli

# Creates the value_converter that clize needs.
inputadvection = cli.create_constrained_inputcubelist_converter(
    lambda cube: cube.name()
    in ["precipitation_advection_x_velocity", "grid_eastward_wind"],
    lambda cube: cube.name()
    in ["precipitation_advection_y_velocity", "grid_northward_wind"],
)


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    advection_velocity: inputadvection,
    orographic_enhancement: cli.inputcube = None,
    *,
    attributes_config: cli.inputjson = None,
    max_lead_time: int = 360,
    lead_time_interval: int = 15,
):
    """Module to extrapolate input cubes given advection velocity fields.

    Args:
        cube (iris.cube.Cube):
            The data to be advected.
        advection_velocity (iris.cube.CubeList):
            Advection cubes of U and V.
            These must have the names of either:
            precipitation_advection_x_velocity or grid_eastward_wind
            precipitation_advection_y_velocity or grid_northward_wind
        orographic_enhancement (iris.cube.Cube):
            Cube containing orographic enhancement forecasts for the lead times
            at which an extrapolation nowcast is required.
        attributes_config (dict):
            Dictionary containing the required changes to the attributes.
        max_lead_time (int):
            Maximum lead time required (mins).
        lead_time_interval (int):
            Interval between required lead times (mins).

    Returns:
        iris.cube.CubeList:
            New cubes with updated time and extrapolated data.
    """
    from improver.nowcasting.pysteps_advection import PystepsExtrapolate
    from improver.utilities.cube_manipulation import MergeCubes

    u_cube, v_cube = advection_velocity

    # extrapolate input data to required lead times
    forecast_plugin = PystepsExtrapolate(lead_time_interval, max_lead_time)
    forecast_cubes = forecast_plugin(
        cube, u_cube, v_cube, orographic_enhancement, attributes_dict=attributes_config
    )

    return MergeCubes()(forecast_cubes)
