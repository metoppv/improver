#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to accumulate input data given advection velocity fields."""

from typing import Callable, List

from improver import cli

# The accumulation frequency in minutes.
ACCUMULATION_FIDELITY = 1


def name_constraint(names: List[str]) -> Callable:
    """
    Generates a callable constraint for matching cube names.

    The callable constraint will realise the data of those cubes matching the
    constraint.

    Args:
        names:
            List of cube names to constrain our cubes.

    Returns:
        A callable which when called, returns True or False for the provided cube,
        depending on whether it matches the names provided.  A matching cube
        will also have its data realised by the callable.
    """

    def constraint(cube):
        ret = False
        if cube.name() in names:
            ret = True
            cube.data
        return ret

    return constraint


# Creates the value_converter that clize needs.
inputadvection = cli.create_constrained_inputcubelist_converter(
    name_constraint(["precipitation_advection_x_velocity", "grid_eastward_wind"]),
    name_constraint(["precipitation_advection_y_velocity", "grid_northward_wind"]),
)


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube_nolazy,
    advection_velocity: inputadvection,
    orographic_enhancement: cli.inputcube_nolazy,
    *,
    attributes_config: cli.inputjson = None,
    max_lead_time=360,
    lead_time_interval=15,
    accumulation_period=15,
    accumulation_units="m",
):
    """Module to extrapolate and accumulate the weather with 1 min fidelity.

    Args:
        cube (iris.cube.Cube):
            The input Cube to be processed.
        advection_velocity (iris.cube.CubeList):
            Advection cubes of U and V.
            These must have the names of either:
            precipitation_advection_x_velocity or grid_eastward_wind
            precipitation_advection_y_velocity or grid_northward_wind
        orographic_enhancement (iris.cube.Cube):
            Cube containing the orographic enhancement fields. May have data
            for multiple times in the cube.
        attributes_config (dict):
            Dictionary containing the required changes to the attributes.
        max_lead_time (int):
            Maximum lead time required (mins).
        lead_time_interval (int):
            Interval between required lead times (mins).
        accumulation_period (int):
            The period over which the accumulation is calculated (mins).
            Only full accumulation periods will be computed. At lead times
            that are shorter than the accumulation period, no accumulation
            output will be produced.
        accumulation_units (str):
            Desired units in which the accumulations should be expressed.
            e.g. 'mm'

    Returns:
        iris.cube.CubeList:
            New cubes with accumulated data.

    Raises:
        ValueError:
            If advection_velocity doesn't contain x and y velocity.
    """
    import numpy as np

    from improver.nowcasting.accumulation import Accumulation
    from improver.nowcasting.pysteps_advection import PystepsExtrapolate
    from improver.utilities.cube_manipulation import MergeCubes

    u_cube, v_cube = advection_velocity

    if not (u_cube and v_cube):
        raise ValueError("Neither u_cube or v_cube can be None")

    # extrapolate input data to the maximum required lead time
    forecast_plugin = PystepsExtrapolate(ACCUMULATION_FIDELITY, max_lead_time)
    forecast_cubes = forecast_plugin(
        cube, u_cube, v_cube, orographic_enhancement, attributes_dict=attributes_config
    )
    lead_times = np.arange(lead_time_interval, max_lead_time + 1, lead_time_interval)

    # Accumulate high frequency rate into desired accumulation intervals.
    plugin = Accumulation(
        accumulation_units=accumulation_units,
        accumulation_period=accumulation_period * 60,
        forecast_periods=lead_times * 60,
    )
    result = plugin(forecast_cubes)

    return MergeCubes()(result)
