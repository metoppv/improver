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
"""Script to calculate whether precipitation is showery."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube):
    """
    Determine the shower condition from global or UK data depending
    on input fields
    Args:
        cloud (iris.cube.Cube or None):
            Probability of total cloud amount above threshold
        cloud_texture (iris.cube.Cube or None):
            Probability of texture of total cloud amount above threshold
        conv_ratio (iris.cube.Cube or None):
            Probability of convective ratio above threshold
    Returns:
        iris.cube.Cube:
            Binary (0/1) "precipitation is showery"

    """
    from iris.cube import CubeList

    cubes = CubeList(cubes)
    cloud, = cubes.extract("probability_of_low_and_medium_type_cloud_area_fraction_above_threshold") or [None]
    conv_ratio, = cubes.extract("probability_of_convective_ratio_above_threshold") or [None]
    cloud_texture, = cubes.extract("probability_of_texture_of_low_and_medium_type_cloud_area_fraction_above_threshold") or [None]

    from improver.precipitation_type.shower_condition import ShowerCondition
    shower_condition = ShowerCondition()(cloud=cloud, conv_ratio=conv_ratio, cloud_texture=cloud_texture)
    return shower_condition
