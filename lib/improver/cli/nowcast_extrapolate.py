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
"""Script to extrapolate input data given advection velocity fields."""


from improver import cli

# Creates the value_converter that clize needs.
inputadvection = cli.create_constrained_inputcubelist_converter(
    'precipitation_advection_x_velocity', 'precipitation_advection_y_velocity')


@cli.clizefy
@cli.with_output
def process(input_cube: cli.inputcube,
            advection_cubes: inputadvection,
            orographic_enhancement_cube: cli.inputcube = None,
            *,
            attributes_dict: cli.inputjson = None,
            max_lead_time: int = 360, lead_time_interval: int = 15):
    """Module  to extrapolate input cubes given advection velocity fields.

    Args:
        input_cube (iris.cube.Cube):
            The data to be advected.
        advection_cubes (iris.cube.CubeList):
            Advection cubes of U and V.
            These must have the names of.
            precipitation_advection_x_velocity
            precipitation_advection_y_velocity
        orographic_enhancement_cube (iris.cube.Cube):
            Cube containing orographic enhancement forecasts for the lead times
            at which an extrapolation nowcast is required.
        attributes_dict (dict):
            Dictionary containing the required changes to the attributes.
        max_lead_time (int):
            Maximum lead time required (mins).
        lead_time_interval (int):
            Interval between required lead times (mins).

    Returns:
        iris.cube.CubeList:
            New cubes with updated time and extrapolated data.
    """
    from iris import Constraint

    from improver.nowcasting.forecasting import CreateExtrapolationForecast
    from improver.utilities.cube_manipulation import merge_cubes

    u_cube = advection_cubes.extract(
        Constraint("precipitation_advection_x_velocity"), True)
    v_cube = advection_cubes.extract(
        Constraint("precipitation_advection_y_velocity"), True)

    # extrapolate input data to required lead times
    forecast_plugin = CreateExtrapolationForecast(
        input_cube, u_cube, v_cube,
        orographic_enhancement_cube=orographic_enhancement_cube,
        attributes_dict=attributes_dict)
    forecast_cubes = forecast_plugin.process(lead_time_interval, max_lead_time)

    return merge_cubes(forecast_cubes)
