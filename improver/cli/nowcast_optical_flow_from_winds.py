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
"""Script to calculate optical flow components as perturbations from model
steering flow"""

from improver import cli

# Creates the value_converter that clize needs.
inputflow = cli.create_constrained_inputcubelist_converter(
    "grid_eastward_wind", "grid_northward_wind",
)


@cli.clizefy
@cli.with_output
def process(
    steering_flow: inputflow,
    orographic_enhancement: cli.inputcube,
    *cubes: cli.inputcube,
):
    """Calculate optical flow components as perturbations from the model
    steering flow.  Advects the older of the two input radar observations to
    the validity time of the newer observvation, then calculates the velocity
    required to adjust this forecast to match the observation.  Sums the
    steering flow and perturbation values to give advection components for
    extrapolation nowcasting.

    Args:
        steering_flow (iris.cube.CubeList):
            Model steering flow as u- and v- wind components.  These must
            have names: "grid_eastward_wind" and "grid_northward_wind".
        orographic_enhancement (iris.cube.Cube):
            Cube containing the orographic enhancement fields.
        cubes (tuple of iris.cube.Cube):
            Two radar precipitation observation cubes.

    Returns:
        iris.cube.CubeList:
            List of u- and v- advection velocities
    """
    import numpy as np
    import iris
    from iris.cube import CubeList

    from improver.nowcasting.optical_flow import OpticalFlow
    from improver.nowcasting.pysteps_advection import PystepsExtrapolate
    from improver.nowcasting.utilities import ApplyOrographicEnhancement

    if len(cubes) != 2:
        raise ValueError("Expected 2 radar cubes - got {}".format(len(cubes)))

    # cannot sort in place as cubes is tuple...
    cubes = sorted(cubes, key=lambda x: x.coord("time").points[0])

    lead_time_seconds = (
        cubes[1].coord("time").cell(0).point - cubes[0].coord("time").cell(0).point
    ).total_seconds()
    lead_time_minutes = int(lead_time_seconds / 60)

    # advect earlier cube forward to match time of later cube, using steering flow
    advected_cube = PystepsExtrapolate(lead_time_minutes, lead_time_minutes)(
        cubes[0], *steering_flow, orographic_enhancement
    )[-1]

    # calculate velocity perturbations required to match forecast to later cube
    cube_list = ApplyOrographicEnhancement("subtract")(
        [advected_cube, cubes[1]], orographic_enhancement
    )
    perturbations = OpticalFlow()(*cube_list)

    # sum perturbations and original flow field to get advection velocities
    for flow, adj in zip(steering_flow, perturbations):
        flow.convert_units(adj.units)
        perturbed_field = np.where(
            np.isfinite(adj.data), adj.data + flow.data, flow.data
        )
        adj.data = perturbed_field.astype(adj.dtype)

    return CubeList(perturbations)
