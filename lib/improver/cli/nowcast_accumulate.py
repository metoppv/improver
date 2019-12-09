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

inputadvection = cli.create_constrained_inputcubelist_converter(
    'precipitation_advection_x_velocity', 'precipitation_advection_y_velocity')


@cli.clizefy
@cli.with_output
def process(input_cube: cli.inputcube,
            advection_cubes: inputadvection,
            *oe_cube: cli.inputcube,
            attributes_dict: cli.inputjson = None,
            max_lead_time=360,
            lead_time_interval=15,
            accumulation_period=15,
            accumulation_units='m'):
    """Module to extrapolate and accumulate the weather with 1 min fidelity.

    Args:
        input_cube (iris.cube.Cube):
            The input Cube to be processed.
        advection_cubes (iris.cube.CubeList):
            Advection cubes of U and V.
        oe_cube (iris.cube.Cube):
            Cube containing the orographic enhancement fields. May have data
            for multiple times in the cube.
            Default is None.
        attributes_dict (dict):
            Dictionary containing the required changes to the attributes.
            Default is None.
        max_lead_time (int):
            Maximum lead time required (mins).
            Default is 360.
        lead_time_interval (int):
            Interval between required lead times (mins).
            Default is 15.
        accumulation_period (int):
            The period over which the accumulation is calculated (mins).
            Only full accumulation periods will be computed. At lead times
            that are shorter than the accumulation period, no accumulation
            output will be produced.
        accumulation_units (str):
            Desired units in which the accumulations should be expressed.
            e.g. 'mm'
            Default is 'm'.

    Returns:
        iris.cube.CubeList:
            New cubes with accumulated data.

    Raises:
        TypeError:
            If advection_cubes doesn't contain an x velocity and a y velocity.
    """
    from iris import Constraint
    from iris.cube import CubeList
    import numpy as np
    
    from improver.nowcasting.accumulation import Accumulation
    from improver.nowcasting.forecasting import CreateExtrapolationForecast
    from improver.utilities.cube_manipulation import merge_cubes

    # The accumulation frequency in minutes.
    ACCUMULATION_FIDELITY = 1

    u_cube = advection_cubes.extract(
        Constraint("precipitation_advection_x_velocity"), True)
    v_cube = advection_cubes.extract(
        Constraint("precipitation_advection_y_velocity"), True)

    if not (u_cube and v_cube):
        raise TypeError(
            "Neither u_cube or v_cube can be none")
    oe_cube = merge_cubes(CubeList(oe_cube))

    # extrapolate input data to required lead times
    forecast_cubes = CreateExtrapolationForecast(
        input_cube, u_cube, v_cube, orographic_enhancement_cube=oe_cube,
        attributes_dict=attributes_dict).process(ACCUMULATION_FIDELITY,
                                                 max_lead_time)

    lead_times = (np.arange(lead_time_interval, max_lead_time + 1,
                            lead_time_interval))

    result = Accumulation(
        accumulation_units=accumulation_units,
        accumulation_period=accumulation_period * 60,
        forecast_periods=lead_times * 60).process(forecast_cubes)

    return merge_cubes(result)
