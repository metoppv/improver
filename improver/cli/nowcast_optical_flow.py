#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate optical flow advection velocities with option to
extrapolate."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    orographic_enhancement: cli.inputcube, *cubes: cli.inputcube,
):
    """Calculate optical flow components from input fields.

    Args:
        orographic_enhancement (iris.cube.Cube):
            Cube containing the orographic enhancement fields.
        cubes (iris.cube.CubeList):
            Cubes from which to calculate optical flow velocities.
            These three cubes will be sorted by their time coords.

    Returns:
        iris.cube.CubeList:
            List of the umean and vmean cubes.

    """
    from iris.cube import CubeList

    from improver.nowcasting.optical_flow import generate_optical_flow_components
    from improver.nowcasting.utilities import ApplyOrographicEnhancement

    original_cube_list = CubeList(cubes)
    # order input files by validity time
    original_cube_list.sort(key=lambda x: x.coord("time").points[0])

    # subtract orographic enhancement
    cube_list = ApplyOrographicEnhancement("subtract")(
        original_cube_list, orographic_enhancement
    )

    # calculate optical flow velocities from T-1 to T and T-2 to T-1, and
    # average to produce the velocities for use in advection
    u_mean, v_mean = generate_optical_flow_components(
        cube_list, ofc_box_size=30, smart_smoothing_iterations=100
    )

    return CubeList([u_mean, v_mean])
