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
"""Script to calculate optical flow advection velocities with option to
extrapolate."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(orographic_enhancement: cli.inputcube,
            *cubes: cli.inputcube,
            ofc_box_size: int = 30,
            smart_smoothing_iterations: int = 100):
    """Calculate optical flow components from input fields.

    Args:
        orographic_enhancement (iris.cube.Cube):
            Cube containing the orographic enhancement fields.
        cubes (iris.cube.CubeList):
            Cubes from which to calculate optical flow velocities.
            These three cubes will be sorted by their time coords.
        ofc_box_size (int):
            Size of square 'box' (in grid spaces) within which to solve
            the optical flow equations.
        smart_smoothing_iterations (int):
            Number of iterations to perform in enforcing smoothness constraint
            for optical flow velocities.

    Returns:
        iris.cube.CubeList:
            List of the umean and vmean cubes.

    """
    from iris.cube import CubeList

    from improver.nowcasting.optical_flow import \
        generate_optical_flow_components
    from improver.nowcasting.utilities import ApplyOrographicEnhancement

    original_cube_list = CubeList(cubes)
    # order input files by validity time
    original_cube_list.sort(key=lambda x: x.coord("time").points[0])

    # subtract orographic enhancement
    cube_list = ApplyOrographicEnhancement("subtract").process(
            original_cube_list, orographic_enhancement)

    # calculate optical flow velocities from T-1 to T and T-2 to T-1, and
    # average to produce the velocities for use in advection
    u_mean, v_mean = generate_optical_flow_components(
        cube_list, ofc_box_size, smart_smoothing_iterations)

    return CubeList([u_mean, v_mean])
