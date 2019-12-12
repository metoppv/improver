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
def process(radar_t: cli.inputcube,
            radar_t1: cli.inputcube,
            radarT2: cli.inputcube,
            orographic_enhancement_cube: cli.inputcube = None,
            *,
            attributes_dict: cli.inputjson = None,
            ofc_box_size: int = 30,
            smart_smoothing_iterations: int = 100):
    """Calculate optical flow components from input fields.

    Args:
        radar_t (iris.cube.CubeList):
            Cube from which to calculate optical flow velocities.
            These three cubes can be any order.
        radar_t1 (iris.cube.CubeList):
            Cube from which to calculate optical flow velocities.
            These three cubes can be any order.
        radarT2 (iris.cube.CubeList):
            Cube from which to calculate optical flow velocities.
            These three cubes can be any order.
        orographic_enhancement_cube (iris.cube.Cube):
            Cube containing the orographic enhancement fields.
            Default is None.
        attributes_dict (dict):
            Dictionary containing required changes to the attributes.
            Every output file will have the attributes_dict applied.
            Default is None.
        ofc_box_size (int):
            Size of square 'box' (in grid spaces) within which to solve
            the optical flow equations.
            Default is 30.
        smart_smoothing_iterations (int):
            Number of iterations to perform in enforcing smoothness constraint
            for optical flow velocities.
            Default is 100.

    Returns:
        iris.cube.CubeList:
            List of the umean and vmean cubes.

    Raises:
        ValueError:
            If there is no oe_cube but a cube is called 'precipitation_rate'.

    """
    from iris.cube import CubeList

    from improver.nowcasting.optical_flow import \
        generate_optical_flow_components
    from improver.nowcasting.utilities import ApplyOrographicEnhancement

    original_cube_list = CubeList([radar_t, radar_t1, radarT2])
    # order input files by validity time
    original_cube_list.sort(key=lambda x: x.coord("time").points[0])

    # subtract orographic enhancement
    if orographic_enhancement_cube:
        cube_list = ApplyOrographicEnhancement("subtract").process(
            original_cube_list, orographic_enhancement_cube)
    else:
        cube_list = original_cube_list
        if any("precipitation_rate" in cube.name() for cube in cube_list):
            cube_names = [cube.name() for cube in cube_list]
            msg = ("For precipitation fields, orographic enhancement "
                   "filepaths must be supplied. The names of the cubes "
                   "supplied were: {}".format(cube_names))
            raise ValueError(msg)

    # calculate optical flow velocities from T-1 to T and T-2 to T-1, and
    # average to produce the velocities for use in advection
    u_mean, v_mean = generate_optical_flow_components(
        cube_list, ofc_box_size, smart_smoothing_iterations, attributes_dict)

    return CubeList([u_mean, v_mean])
