# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Utilities for use by precipitation_type plugins / functions."""

from iris.cube import Cube

from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.probabilistic import find_threshold_coordinate


def make_shower_condition_cube(cube: Cube, in_place: bool = False) -> Cube:
    """
    Modify a cloud or precipitation rate texture cube to become a shower
    condition proxy. This will modify the threshold coordinate and diagnostic
    name to match those produced using the ShowerConditionProbability plugin.
    This modification enables these two proxies to be blended to get a smooth
    transition in the areas classified as showery when transitioning from
    high resolution to coarse resolution models. """

    if not in_place:
        cube = cube.copy()

    shower_condition_name = "shower_condition"
    cube.rename(f"probability_of_{shower_condition_name}_above_threshold")
    shower_threshold = find_threshold_coordinate(cube)

    # We introduce an implied threshold of shower conditions.
    # Above 50% conditions are showery.
    cube.coord(shower_threshold).rename(shower_condition_name)
    cube.coord(shower_condition_name).var_name = "threshold"
    cube.coord(shower_condition_name).points = FLOAT_DTYPE(0.5)

    return cube
