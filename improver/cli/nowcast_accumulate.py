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
"""Script to accumulate input data given advection velocity fields."""

from improver import cli

# The accumulation frequency in minutes.
ACCUMULATION_FIDELITY = 1

# Creates the value_converter that clize needs.
inputadvection = cli.create_constrained_inputcubelist_converter(
    ['precipitation_advection_x_velocity', 'grid_eastward_wind'],
    ['precipitation_advection_y_velocity', 'grid_northward_wind'])


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            advection_velocity: inputadvection,
            orographic_enhancement: cli.inputcube,
            *,
            attributes_config: cli.inputjson = None,
            max_lead_time=360,
            lead_time_interval=15,
            accumulation_period=15,
            accumulation_units='m'):
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
    from improver.nowcasting.forecasting import CreateExtrapolationForecast
    from improver.utilities.cube_manipulation import merge_cubes

    u_cube, v_cube = advection_velocity

    if not (u_cube and v_cube):
        raise ValueError("Neither u_cube or v_cube can be None")

    # extrapolate input data to the maximum required lead time
    forecast_cubes = CreateExtrapolationForecast(
        cube, u_cube, v_cube, orographic_enhancement,
        attributes_dict=attributes_config).process(ACCUMULATION_FIDELITY,
                                                   max_lead_time)

    lead_times = (np.arange(lead_time_interval, max_lead_time + 1,
                            lead_time_interval))

    # Accumulate high frequency rate into desired accumulation intervals.
    result = Accumulation(
        accumulation_units=accumulation_units,
        accumulation_period=accumulation_period * 60,
        forecast_periods=lead_times * 60).process(forecast_cubes)

    return merge_cubes(result)
