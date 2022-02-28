#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
    the validity time of the newer observation, then calculates the velocity
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
    from iris.cube import CubeList

    from improver.nowcasting.optical_flow import (
        generate_advection_velocities_from_winds,
    )

    if len(cubes) != 2:
        raise ValueError("Expected 2 radar cubes - got {}".format(len(cubes)))

    advection_velocities = generate_advection_velocities_from_winds(
        CubeList(cubes), steering_flow, orographic_enhancement
    )

    return advection_velocities
