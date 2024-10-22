#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate optical flow components as perturbations from model
steering flow"""

from improver import cli

# Creates the value_converter that clize needs.
inputflow = cli.create_constrained_inputcubelist_converter(
    "grid_eastward_wind", "grid_northward_wind"
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
