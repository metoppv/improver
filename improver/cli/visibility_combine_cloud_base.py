#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""CLI to combine visibility and cloud bases"""
from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcubelist,
    initial_scaling_value: float,
    first_unscaled_threshold: float,
):
    """Combine the probability of visibility above or below a threshold with the probability of
    cloud base at ground level.

    Before combining with a visibility threshold, the cube of cloud base at ground level is
    multiplied by a scalar. The scalar is determined differently depending on the corresponding
    visibility threshold:

    1) If the threshold is greater than or equal to first_unscaled_threshold then the scaling factor
    is 1.
    2) If the threshold is less than first_unscaled_thresholds then a scaling factor is calculated
    by inputting the threshold into a negative fourth level polynomial. The constants in this curve
    have been defined such that a threshold equal to first_unscaled_threshold gives a scaling factor
    of 1.0 and a threshold of 0m gives a scaling factor equal to initial_scaling_value.

    The maximum probability is then taken of the scaled cloud base at ground level and the
    visibility threshold.

    From experimentation the current best known parameter values for combining visibility with a
    cloud base at ground level of 4.5 oktas of cloud or greater are initial_scaling_value=0.6
    and first_unscaled_threshold=5000.0

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            containing:
                visibility (iris.cube.Cube):
                    Cube of probability of visibility relative to thresholds
                cloud base at ground level (iris.cube.Cube):
                    Cube of probability of cloud base at ground level. This cube should only
                    have spatial dimensions (e.g. spot_index or x,y coordinates).
        initial_scaling_value (float):
            The scaling factor for a visibility threshold of 0m
        first_unscaled_threshold (float):
            The first visibility threshold with a corresponding scaling factor of 1.
    Returns:
        iris.cube.Cube:
            Cube of probability of visibility combined with scaled cloud bases
    """

    from improver.visibility.visibility_combine_cloud_base import (
        VisibilityCombineCloudBase,
    )

    return VisibilityCombineCloudBase(initial_scaling_value, first_unscaled_threshold)(
        cubes
    )
